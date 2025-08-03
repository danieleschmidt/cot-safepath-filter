"""
Repository pattern implementations for data access.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_
from sqlalchemy.exc import IntegrityError

from .models import FilterOperation, SafetyDetection, FilterRule, User, SystemMetrics, AuditLog
from ..exceptions import ValidationError, ConfigurationError
from ..models import FilterRequest, FilterResult, SafetyScore


class BaseRepository:
    """Base repository with common CRUD operations."""
    
    def __init__(self, session: Session):
        self.session = session
        self.model = None  # Override in subclasses
    
    def create(self, **kwargs) -> Any:
        """Create a new record."""
        try:
            instance = self.model(**kwargs)
            self.session.add(instance)
            self.session.flush()  # Get ID without committing
            return instance
        except IntegrityError as e:
            self.session.rollback()
            raise ValidationError(f"Failed to create {self.model.__name__}: {e}")
    
    def get_by_id(self, record_id: int) -> Optional[Any]:
        """Get record by ID."""
        return self.session.query(self.model).filter(self.model.id == record_id).first()
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """Get all records with pagination."""
        return (
            self.session.query(self.model)
            .offset(offset)
            .limit(limit)
            .all()
        )
    
    def update(self, record_id: int, **kwargs) -> Optional[Any]:
        """Update a record by ID."""
        instance = self.get_by_id(record_id)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            self.session.flush()
        return instance
    
    def delete(self, record_id: int) -> bool:
        """Delete a record by ID."""
        instance = self.get_by_id(record_id)
        if instance:
            self.session.delete(instance)
            self.session.flush()
            return True
        return False
    
    def count(self) -> int:
        """Get total count of records."""
        return self.session.query(self.model).count()


class FilterOperationRepository(BaseRepository):
    """Repository for filter operation records."""
    
    def __init__(self, session: Session):
        super().__init__(session)
        self.model = FilterOperation
    
    def create_from_request_result(
        self, 
        request: FilterRequest, 
        result: FilterResult,
        user_id: str = None,
        session_id: str = None
    ) -> FilterOperation:
        """Create filter operation record from request and result."""
        return self.create(
            request_id=request.request_id,
            session_id=session_id,
            user_id=user_id,
            input_hash=self._hash_content(request.content),
            content_length=len(request.content),
            safety_score=result.safety_score.overall_score,
            was_filtered=result.was_filtered,
            filter_reasons=result.filter_reasons,
            processing_time_ms=result.processing_time_ms,
            cache_hit=False,  # TODO: Track cache hits
            safety_level=request.safety_level.value,
            filter_threshold=0.7,  # TODO: Get from config
            metadata=request.metadata,
            context=request.context,
        )
    
    def get_by_request_id(self, request_id: str) -> Optional[FilterOperation]:
        """Get operation by request ID."""
        return (
            self.session.query(FilterOperation)
            .filter(FilterOperation.request_id == request_id)
            .first()
        )
    
    def get_by_content_hash(self, content_hash: str) -> Optional[FilterOperation]:
        """Get operation by content hash for deduplication."""
        return (
            self.session.query(FilterOperation)
            .filter(FilterOperation.input_hash == content_hash)
            .order_by(desc(FilterOperation.created_at))
            .first()
        )
    
    def get_user_operations(
        self, 
        user_id: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[FilterOperation]:
        """Get operations for a specific user."""
        return (
            self.session.query(FilterOperation)
            .filter(FilterOperation.user_id == user_id)
            .order_by(desc(FilterOperation.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )
    
    def get_operations_by_timeframe(
        self,
        start_time: datetime,
        end_time: datetime,
        was_filtered: bool = None
    ) -> List[FilterOperation]:
        """Get operations within a timeframe."""
        query = self.session.query(FilterOperation).filter(
            and_(
                FilterOperation.created_at >= start_time,
                FilterOperation.created_at <= end_time
            )
        )
        
        if was_filtered is not None:
            query = query.filter(FilterOperation.was_filtered == was_filtered)
        
        return query.order_by(desc(FilterOperation.created_at)).all()
    
    def get_safety_score_statistics(self, days: int = 7) -> Dict[str, float]:
        """Get safety score statistics for the last N days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = (
            self.session.query(
                func.avg(FilterOperation.safety_score).label('avg_score'),
                func.min(FilterOperation.safety_score).label('min_score'),
                func.max(FilterOperation.safety_score).label('max_score'),
                func.count(FilterOperation.id).label('total_operations'),
                func.sum(
                    func.case([(FilterOperation.was_filtered == True, 1)], else_=0)
                ).label('filtered_count')
            )
            .filter(FilterOperation.created_at >= cutoff_date)
            .first()
        )
        
        return {
            'average_safety_score': float(result.avg_score or 0),
            'minimum_safety_score': float(result.min_score or 0),
            'maximum_safety_score': float(result.max_score or 0),
            'total_operations': int(result.total_operations or 0),
            'filtered_operations': int(result.filtered_count or 0),
            'filter_rate': float(result.filtered_count or 0) / max(result.total_operations or 1, 1)
        }
    
    def get_performance_metrics(self, days: int = 7) -> Dict[str, float]:
        """Get performance metrics for the last N days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = (
            self.session.query(
                func.avg(FilterOperation.processing_time_ms).label('avg_time'),
                func.percentile_cont(0.95).within_group(
                    FilterOperation.processing_time_ms.asc()
                ).label('p95_time'),
                func.max(FilterOperation.processing_time_ms).label('max_time'),
                func.count(FilterOperation.id).label('total_requests')
            )
            .filter(FilterOperation.created_at >= cutoff_date)
            .first()
        )
        
        return {
            'average_processing_time_ms': float(result.avg_time or 0),
            'p95_processing_time_ms': float(result.p95_time or 0),
            'max_processing_time_ms': float(result.max_time or 0),
            'total_requests': int(result.total_requests or 0)
        }
    
    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """Clean up old filter operation records."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        deleted_count = (
            self.session.query(FilterOperation)
            .filter(FilterOperation.created_at < cutoff_date)
            .delete()
        )
        
        return deleted_count
    
    def _hash_content(self, content: str) -> str:
        """Hash content for storage."""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()


class SafetyDetectionRepository(BaseRepository):
    """Repository for safety detection records."""
    
    def __init__(self, session: Session):
        super().__init__(session)
        self.model = SafetyDetection
    
    def create_from_detection_result(
        self,
        operation_id: int,
        detector_name: str,
        confidence: float,
        patterns: List[str],
        severity: str,
        is_harmful: bool,
        reasoning: str = None,
        metadata: Dict[str, Any] = None
    ) -> SafetyDetection:
        """Create detection record from detector result."""
        return self.create(
            operation_id=operation_id,
            detector_name=detector_name,
            confidence=confidence,
            detected_patterns=patterns,
            severity=severity,
            is_harmful=is_harmful,
            reasoning=reasoning,
            metadata=metadata or {}
        )
    
    def get_detections_by_operation(self, operation_id: int) -> List[SafetyDetection]:
        """Get all detections for a filter operation."""
        return (
            self.session.query(SafetyDetection)
            .filter(SafetyDetection.operation_id == operation_id)
            .order_by(desc(SafetyDetection.confidence))
            .all()
        )
    
    def get_detections_by_detector(
        self, 
        detector_name: str, 
        limit: int = 100
    ) -> List[SafetyDetection]:
        """Get recent detections by detector name."""
        return (
            self.session.query(SafetyDetection)
            .filter(SafetyDetection.detector_name == detector_name)
            .order_by(desc(SafetyDetection.created_at))
            .limit(limit)
            .all()
        )
    
    def get_detector_performance(self, detector_name: str, days: int = 7) -> Dict[str, Any]:
        """Get performance metrics for a specific detector."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = (
            self.session.query(
                func.count(SafetyDetection.id).label('total_detections'),
                func.sum(
                    func.case([(SafetyDetection.is_harmful == True, 1)], else_=0)
                ).label('harmful_detections'),
                func.avg(SafetyDetection.confidence).label('avg_confidence'),
                func.max(SafetyDetection.confidence).label('max_confidence')
            )
            .filter(
                and_(
                    SafetyDetection.detector_name == detector_name,
                    SafetyDetection.created_at >= cutoff_date
                )
            )
            .first()
        )
        
        return {
            'detector_name': detector_name,
            'total_detections': int(result.total_detections or 0),
            'harmful_detections': int(result.harmful_detections or 0),
            'detection_rate': float(result.harmful_detections or 0) / max(result.total_detections or 1, 1),
            'average_confidence': float(result.avg_confidence or 0),
            'max_confidence': float(result.max_confidence or 0)
        }


class FilterRuleRepository(BaseRepository):
    """Repository for filter rule management."""
    
    def __init__(self, session: Session):
        super().__init__(session)
        self.model = FilterRule
    
    def get_active_rules(self) -> List[FilterRule]:
        """Get all active filter rules."""
        return (
            self.session.query(FilterRule)
            .filter(FilterRule.enabled == True)
            .order_by(desc(FilterRule.priority), FilterRule.name)
            .all()
        )
    
    def get_by_name(self, name: str) -> Optional[FilterRule]:
        """Get rule by name."""
        return (
            self.session.query(FilterRule)
            .filter(FilterRule.name == name)
            .first()
        )
    
    def get_by_category(self, category: str) -> List[FilterRule]:
        """Get rules by category."""
        return (
            self.session.query(FilterRule)
            .filter(FilterRule.category == category)
            .order_by(desc(FilterRule.priority))
            .all()
        )
    
    def increment_usage(self, rule_id: int) -> None:
        """Increment usage count for a rule."""
        rule = self.get_by_id(rule_id)
        if rule:
            rule.usage_count += 1
            rule.last_triggered = datetime.utcnow()
            self.session.flush()
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about filter rules."""
        result = (
            self.session.query(
                func.count(FilterRule.id).label('total_rules'),
                func.sum(
                    func.case([(FilterRule.enabled == True, 1)], else_=0)
                ).label('active_rules'),
                func.sum(FilterRule.usage_count).label('total_usage')
            )
            .first()
        )
        
        return {
            'total_rules': int(result.total_rules or 0),
            'active_rules': int(result.active_rules or 0),
            'inactive_rules': int(result.total_rules or 0) - int(result.active_rules or 0),
            'total_rule_usage': int(result.total_usage or 0)
        }


class UserRepository(BaseRepository):
    """Repository for user management."""
    
    def __init__(self, session: Session):
        super().__init__(session)
        self.model = User
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return (
            self.session.query(User)
            .filter(User.username == username)
            .first()
        )
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return (
            self.session.query(User)
            .filter(User.email == email)
            .first()
        )
    
    def get_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        return (
            self.session.query(User)
            .filter(
                and_(
                    User.api_key == api_key,
                    User.api_key_active == True,
                    User.is_active == True
                )
            )
            .first()
        )
    
    def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp."""
        user = self.get_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()
            self.session.flush()
    
    def update_last_request(self, user_id: str) -> None:
        """Update user's last request timestamp and increment counter."""
        user = self.get_by_id(user_id)
        if user:
            user.last_request = datetime.utcnow()
            user.request_count += 1
            self.session.flush()
    
    def get_active_users(self, days: int = 30) -> List[User]:
        """Get users active in the last N days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        return (
            self.session.query(User)
            .filter(
                and_(
                    User.is_active == True,
                    or_(
                        User.last_login >= cutoff_date,
                        User.last_request >= cutoff_date
                    )
                )
            )
            .order_by(desc(User.last_request))
            .all()
        )