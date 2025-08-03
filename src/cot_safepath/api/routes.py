"""
API routes for CoT SafePath Filter.
"""

import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .dependencies import get_current_user, get_database_session, get_redis_client
from .schemas import (
    FilterRequest, FilterResponse, FilterOperationResponse,
    SafetyDetectionResponse, UserResponse, RuleResponse,
    FilterStatsResponse, SystemStatsResponse
)
from ..core import SafePathFilter
from ..models import FilterConfig, FilterRequest as CoreFilterRequest, SafetyLevel
from ..database.repositories import (
    FilterOperationRepository, SafetyDetectionRepository, 
    FilterRuleRepository, UserRepository
)
from ..cache import FilterResultCache, RateLimitCache
from ..exceptions import ValidationError, RateLimitError


router = APIRouter()
security = HTTPBearer(auto_error=False)


# Filter endpoints
@router.post("/filter", response_model=FilterResponse, tags=["Filtering"])
async def filter_content(
    request: FilterRequest,
    db: Session = Depends(get_database_session),
    user = Depends(get_current_user),
    redis_client = Depends(get_redis_client)
):
    """
    Filter chain-of-thought content for safety.
    
    This endpoint analyzes the provided content and applies multi-stage filtering
    to detect and sanitize harmful or deceptive reasoning patterns.
    """
    start_time = time.time()
    
    try:
        # Check rate limits
        rate_limiter = RateLimitCache(redis_client)
        user_id = user.id if user else request.client_ip or "anonymous"
        
        rate_limit_result = rate_limiter.check_rate_limit(
            identifier=user_id,
            limit=1000,  # TODO: Get from user settings
            window_seconds=3600
        )
        
        if not rate_limit_result['allowed']:
            raise RateLimitError(
                f"Rate limit exceeded. Try again in {rate_limit_result['reset_time'] - int(time.time())} seconds",
                limit=rate_limit_result['limit']
            )
        
        # Check cache first
        cache = FilterResultCache(redis_client)
        cached_result = cache.get_cached_result(
            content=request.content,
            safety_level=request.safety_level.value if request.safety_level else None,
            context=request.context
        )
        
        if cached_result:
            # Return cached result
            return FilterResponse(
                filtered_content=cached_result.filtered_content,
                safety_score=cached_result.safety_score.overall_score,
                confidence=cached_result.safety_score.confidence,
                was_filtered=cached_result.was_filtered,
                filter_reasons=cached_result.filter_reasons,
                processing_time_ms=cached_result.processing_time_ms,
                request_id=cached_result.request_id,
                cached=True,
                rate_limit=rate_limit_result
            )
        
        # Create filter configuration
        config = FilterConfig(
            safety_level=request.safety_level or SafetyLevel.BALANCED,
            filter_threshold=request.filter_threshold or 0.7,
            enable_caching=True,
            log_filtered=True
        )
        
        # Initialize filter
        filter_engine = SafePathFilter(config)
        
        # Create core filter request
        core_request = CoreFilterRequest(
            content=request.content,
            context=request.context,
            safety_level=config.safety_level,
            metadata=request.metadata or {}
        )
        
        # Perform filtering
        result = filter_engine.filter(core_request)
        
        # Store in database
        filter_repo = FilterOperationRepository(db)
        operation = filter_repo.create_from_request_result(
            request=core_request,
            result=result,
            user_id=user.id if user else None
        )
        
        # Store detections
        detection_repo = SafetyDetectionRepository(db)
        # TODO: Get actual detector results from filter pipeline
        
        # Cache result
        cache.cache_result(
            content=request.content,
            result=result,
            safety_level=config.safety_level.value,
            context=request.context
        )
        
        # Commit database transaction
        db.commit()
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return FilterResponse(
            filtered_content=result.filtered_content,
            safety_score=result.safety_score.overall_score,
            confidence=result.safety_score.confidence,
            was_filtered=result.was_filtered,
            filter_reasons=result.filter_reasons,
            processing_time_ms=processing_time,
            request_id=result.request_id,
            cached=False,
            rate_limit=rate_limit_result
        )
        
    except Exception as e:
        db.rollback()
        if isinstance(e, (ValidationError, RateLimitError)):
            raise HTTPException(status_code=400, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/filter/operations", response_model=List[FilterOperationResponse], tags=["Filtering"])
async def get_filter_operations(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    user_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    was_filtered: Optional[bool] = Query(None),
    db: Session = Depends(get_database_session),
    current_user = Depends(get_current_user)
):
    """Get filter operations with optional filtering."""
    
    filter_repo = FilterOperationRepository(db)
    
    if start_date and end_date:
        operations = filter_repo.get_operations_by_timeframe(
            start_time=start_date,
            end_time=end_date,
            was_filtered=was_filtered
        )
    elif user_id:
        # Check permissions
        if current_user.role != "admin" and current_user.id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        operations = filter_repo.get_user_operations(user_id, limit, offset)
    else:
        # Admin only for all operations
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        operations = filter_repo.get_all(limit, offset)
    
    return [
        FilterOperationResponse(
            id=op.id,
            request_id=op.request_id,
            user_id=op.user_id,
            safety_score=op.safety_score,
            was_filtered=op.was_filtered,
            filter_reasons=op.filter_reasons,
            processing_time_ms=op.processing_time_ms,
            safety_level=op.safety_level,
            created_at=op.created_at
        )
        for op in operations
    ]


@router.get("/filter/stats", response_model=FilterStatsResponse, tags=["Analytics"])
async def get_filter_stats(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_database_session),
    current_user = Depends(get_current_user)
):
    """Get filtering statistics for the last N days."""
    
    # Admin or user can see their own stats
    filter_repo = FilterOperationRepository(db)
    
    if current_user.role == "admin":
        safety_stats = filter_repo.get_safety_score_statistics(days)
        performance_stats = filter_repo.get_performance_metrics(days)
    else:
        # TODO: Implement user-specific stats
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return FilterStatsResponse(
        **safety_stats,
        **performance_stats,
        time_period_days=days
    )


# Rule management endpoints
@router.get("/rules", response_model=List[RuleResponse], tags=["Rules"])
async def get_rules(
    enabled_only: bool = Query(True),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_database_session),
    current_user = Depends(get_current_user)
):
    """Get filter rules."""
    
    rule_repo = FilterRuleRepository(db)
    
    if enabled_only:
        rules = rule_repo.get_active_rules()
    elif category:
        rules = rule_repo.get_by_category(category)
    else:
        rules = rule_repo.get_all()
    
    return [
        RuleResponse(
            id=rule.id,
            name=rule.name,
            description=rule.description,
            pattern=rule.pattern,
            action=rule.action,
            severity=rule.severity,
            enabled=rule.enabled,
            category=rule.category,
            usage_count=rule.usage_count,
            created_at=rule.created_at,
            updated_at=rule.updated_at
        )
        for rule in rules
    ]


@router.post("/rules", response_model=RuleResponse, tags=["Rules"])
async def create_rule(
    name: str = Body(...),
    description: str = Body(None),
    pattern: str = Body(...),
    action: str = Body(...),
    severity: str = Body(...),
    category: str = Body(None),
    enabled: bool = Body(True),
    db: Session = Depends(get_database_session),
    current_user = Depends(get_current_user)
):
    """Create a new filter rule (admin only)."""
    
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    rule_repo = FilterRuleRepository(db)
    
    # Check if rule name already exists
    existing_rule = rule_repo.get_by_name(name)
    if existing_rule:
        raise HTTPException(status_code=400, detail="Rule name already exists")
    
    try:
        rule = rule_repo.create(
            name=name,
            description=description,
            pattern=pattern,
            action=action,
            severity=severity,
            category=category,
            enabled=enabled,
            created_by=current_user.id
        )
        
        db.commit()
        
        return RuleResponse(
            id=rule.id,
            name=rule.name,
            description=rule.description,
            pattern=rule.pattern,
            action=rule.action,
            severity=rule.severity,
            enabled=rule.enabled,
            category=rule.category,
            usage_count=rule.usage_count,
            created_at=rule.created_at,
            updated_at=rule.updated_at
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


# User management endpoints
@router.get("/users", response_model=List[UserResponse], tags=["Users"])
async def get_users(
    active_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_database_session),
    current_user = Depends(get_current_user)
):
    """Get users (admin only)."""
    
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    user_repo = UserRepository(db)
    
    if active_only:
        users = user_repo.get_active_users(days=30)
    else:
        users = user_repo.get_all(limit, offset)
    
    return [
        UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            request_count=user.request_count,
            last_login=user.last_login,
            created_at=user.created_at
        )
        for user in users
    ]


@router.get("/users/me", response_model=UserResponse, tags=["Users"])
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user information."""
    
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        request_count=current_user.request_count,
        last_login=current_user.last_login,
        created_at=current_user.created_at
    )


# System monitoring endpoints
@router.get("/system/stats", response_model=SystemStatsResponse, tags=["System"])
async def get_system_stats(
    db: Session = Depends(get_database_session),
    redis_client = Depends(get_redis_client),
    current_user = Depends(get_current_user)
):
    """Get system statistics (admin only)."""
    
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Database stats
    filter_repo = FilterOperationRepository(db)
    rule_repo = FilterRuleRepository(db)
    user_repo = UserRepository(db)
    
    filter_stats = filter_repo.get_safety_score_statistics(7)
    rule_stats = rule_repo.get_rule_statistics()
    
    # Redis stats
    cache = FilterResultCache(redis_client)
    cache_stats = cache.get_cache_stats()
    
    # System info
    redis_info = redis_client.info()
    
    return SystemStatsResponse(
        database_stats={
            "total_operations": filter_stats['total_operations'],
            "filtered_operations": filter_stats['filtered_operations'],
            "average_safety_score": filter_stats['average_safety_score'],
            "total_rules": rule_stats['total_rules'],
            "active_rules": rule_stats['active_rules']
        },
        cache_stats={
            "total_keys": cache_stats.get('total_keys', 0),
            "hit_rate": cache_stats.get('hit_rate', 0.0),
            "memory_usage_mb": cache_stats.get('estimated_memory_bytes', 0) / 1024 / 1024
        },
        system_stats={
            "uptime_seconds": redis_info.get('uptime_in_seconds', 0),
            "redis_version": redis_info.get('redis_version', 'unknown'),
            "connected_clients": redis_info.get('connected_clients', 0)
        }
    )


# Batch filtering endpoint
@router.post("/filter/batch", response_model=List[FilterResponse], tags=["Filtering"])
async def filter_batch(
    requests: List[FilterRequest],
    db: Session = Depends(get_database_session),
    user = Depends(get_current_user),
    redis_client = Depends(get_redis_client)
):
    """
    Filter multiple pieces of content in batch.
    Limited to 10 requests per batch.
    """
    
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 requests per batch")
    
    results = []
    
    for request in requests:
        try:
            # Reuse the single filter endpoint logic
            result = await filter_content(request, db, user, redis_client)
            results.append(result)
        except Exception as e:
            # Add error result for failed requests
            results.append(FilterResponse(
                filtered_content=request.content,
                safety_score=0.0,
                confidence=0.0,
                was_filtered=False,
                filter_reasons=[f"Processing error: {str(e)}"],
                processing_time_ms=0,
                request_id=None,
                cached=False,
                error=str(e)
            ))
    
    return results