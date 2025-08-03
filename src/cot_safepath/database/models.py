"""
SQLAlchemy database models for CoT SafePath Filter.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, 
    ForeignKey, JSON, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class FilterOperation(Base):
    """Model for filter operation audit logs."""
    
    __tablename__ = "filter_operations"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Request identification
    request_id = Column(String(64), unique=True, nullable=False, index=True)
    session_id = Column(String(64), index=True)
    user_id = Column(String(64), index=True)
    
    # Content hashing for deduplication
    input_hash = Column(String(64), nullable=False, index=True)
    content_length = Column(Integer, nullable=False)
    
    # Safety assessment
    safety_score = Column(Float, nullable=False)
    was_filtered = Column(Boolean, nullable=False, default=False)
    filter_reasons = Column(ARRAY(String), default=[])
    
    # Performance metrics
    processing_time_ms = Column(Integer, nullable=False)
    cache_hit = Column(Boolean, default=False)
    
    # Configuration used
    safety_level = Column(String(20), nullable=False)
    filter_threshold = Column(Float, nullable=False)
    
    # Metadata and context
    metadata = Column(JSON, default={})
    context = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    detections = relationship("SafetyDetection", back_populates="operation", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_filter_ops_created_at', 'created_at'),
        Index('idx_filter_ops_safety_score', 'safety_score'),
        Index('idx_filter_ops_filtered', 'was_filtered'),
        Index('idx_filter_ops_user_created', 'user_id', 'created_at'),
        CheckConstraint('safety_score >= 0 AND safety_score <= 1', name='check_safety_score_range'),
        CheckConstraint('processing_time_ms >= 0', name='check_processing_time_positive'),
    )
    
    def __repr__(self):
        return f"<FilterOperation(id={self.id}, request_id='{self.request_id}', safety_score={self.safety_score})>"


class SafetyDetection(Base):
    """Model for individual safety detector results."""
    
    __tablename__ = "safety_detections"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key to filter operation
    operation_id = Column(Integer, ForeignKey("filter_operations.id"), nullable=False)
    
    # Detector information
    detector_name = Column(String(50), nullable=False, index=True)
    detector_version = Column(String(20), default="1.0")
    
    # Detection results
    confidence = Column(Float, nullable=False)
    detected_patterns = Column(ARRAY(String), default=[])
    severity = Column(String(20), nullable=False)
    is_harmful = Column(Boolean, nullable=False)
    
    # Additional metadata
    reasoning = Column(Text)
    metadata = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    operation = relationship("FilterOperation", back_populates="detections")
    
    # Indexes
    __table_args__ = (
        Index('idx_detections_detector_name', 'detector_name'),
        Index('idx_detections_confidence', 'confidence'),
        Index('idx_detections_harmful', 'is_harmful'),
        Index('idx_detections_operation_detector', 'operation_id', 'detector_name'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence_range'),
    )
    
    def __repr__(self):
        return f"<SafetyDetection(id={self.id}, detector='{self.detector_name}', confidence={self.confidence})>"


class FilterRule(Base):
    """Model for configurable filter rules."""
    
    __tablename__ = "filter_rules"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Rule identification
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    
    # Rule definition
    pattern = Column(Text)
    pattern_type = Column(String(20), default="regex")  # regex, keyword, semantic
    action = Column(String(20), nullable=False)  # allow, flag, block, sanitize
    severity = Column(String(20), nullable=False)
    
    # Rule configuration
    threshold = Column(Float, default=0.7)
    enabled = Column(Boolean, default=True)
    priority = Column(Integer, default=0)
    
    # Categorization
    category = Column(String(50))
    tags = Column(ARRAY(String), default=[])
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_triggered = Column(DateTime)
    
    # Metadata
    metadata = Column(JSON, default={})
    created_by = Column(String(64))
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_rules_enabled', 'enabled'),
        Index('idx_rules_category', 'category'),
        Index('idx_rules_priority', 'priority'),
        Index('idx_rules_updated_at', 'updated_at'),
        CheckConstraint('threshold >= 0 AND threshold <= 1', name='check_threshold_range'),
    )
    
    def __repr__(self):
        return f"<FilterRule(id={self.id}, name='{self.name}', action='{self.action}')>"


class User(Base):
    """Model for user management and authentication."""
    
    __tablename__ = "users"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # User identification
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    
    # Authentication
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Authorization
    role = Column(String(20), default="user")  # user, admin, api_user
    permissions = Column(ARRAY(String), default=[])
    
    # API access
    api_key = Column(String(64), unique=True)
    api_key_active = Column(Boolean, default=False)
    rate_limit_override = Column(Integer)
    
    # Usage tracking
    request_count = Column(Integer, default=0)
    last_login = Column(DateTime)
    last_request = Column(DateTime)
    
    # Profile information
    full_name = Column(String(100))
    organization = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_users_email', 'email'),
        Index('idx_users_api_key', 'api_key'),
        Index('idx_users_last_login', 'last_login'),
        Index('idx_users_active', 'is_active'),
    )
    
    def __repr__(self):
        return f"<User(id='{self.id}', username='{self.username}', role='{self.role}')>"


class Session(Base):
    """Model for user session management."""
    
    __tablename__ = "sessions"
    
    # Primary key
    id = Column(String(64), primary_key=True)
    
    # Foreign key to user
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    
    # Session data
    data = Column(JSON, default={})
    
    # Session metadata
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    
    # Session lifecycle
    expires_at = Column(DateTime, nullable=False)
    last_accessed = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    # Indexes
    __table_args__ = (
        Index('idx_sessions_user_id', 'user_id'),
        Index('idx_sessions_expires_at', 'expires_at'),
        Index('idx_sessions_active', 'is_active'),
        Index('idx_sessions_last_accessed', 'last_accessed'),
    )
    
    def __repr__(self):
        return f"<Session(id='{self.id}', user_id='{self.user_id}', expires_at={self.expires_at})>"


class SystemMetrics(Base):
    """Model for system performance and usage metrics."""
    
    __tablename__ = "system_metrics"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Metric identification
    metric_name = Column(String(50), nullable=False)
    metric_type = Column(String(20), nullable=False)  # counter, gauge, histogram
    
    # Metric values
    value = Column(Float, nullable=False)
    labels = Column(JSON, default={})
    
    # Aggregation information
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    aggregation_window = Column(String(20))  # 1m, 5m, 15m, 1h, 1d
    
    # Indexes for time-series queries
    __table_args__ = (
        Index('idx_metrics_name_timestamp', 'metric_name', 'timestamp'),
        Index('idx_metrics_type', 'metric_type'),
        Index('idx_metrics_timestamp', 'timestamp'),
        UniqueConstraint('metric_name', 'timestamp', 'aggregation_window', name='uq_metric_time_window'),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, metric='{self.metric_name}', value={self.value})>"


class AuditLog(Base):
    """Model for comprehensive audit logging."""
    
    __tablename__ = "audit_logs"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Event identification
    event_type = Column(String(50), nullable=False, index=True)
    event_category = Column(String(30), nullable=False)  # security, operation, admin
    
    # Actor information
    user_id = Column(String(36))
    session_id = Column(String(64))
    ip_address = Column(String(45))
    
    # Event details
    resource_type = Column(String(50))
    resource_id = Column(String(64))
    action = Column(String(50), nullable=False)
    outcome = Column(String(20), nullable=False)  # success, failure, error
    
    # Event data
    details = Column(JSON, default={})
    old_values = Column(JSON)
    new_values = Column(JSON)
    
    # Risk assessment
    risk_level = Column(String(20), default="low")  # low, medium, high, critical
    requires_review = Column(Boolean, default=False)
    
    # Timestamps
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    # Indexes for audit queries
    __table_args__ = (
        Index('idx_audit_event_type', 'event_type'),
        Index('idx_audit_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_category', 'event_category'),
        Index('idx_audit_risk_level', 'risk_level'),
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_requires_review', 'requires_review'),
    )
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, event='{self.event_type}', outcome='{self.outcome}')>"