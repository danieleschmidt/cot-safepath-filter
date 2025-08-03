"""
Pydantic schemas for API request/response models.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

from ..models import SafetyLevel


class FilterRequest(BaseModel):
    """Request schema for content filtering."""
    
    content: str = Field(..., min_length=1, max_length=50000, description="Content to filter")
    context: Optional[str] = Field(None, max_length=5000, description="Additional context")
    safety_level: Optional[SafetyLevel] = Field(SafetyLevel.BALANCED, description="Safety filtering level")
    filter_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Safety score threshold")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    client_ip: Optional[str] = Field(None, description="Client IP address for rate limiting")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "content": "Step 1: Think about how to help the user safely. Step 2: Provide helpful information.",
                "context": "Educational context",
                "safety_level": "balanced",
                "filter_threshold": 0.7,
                "metadata": {"source": "user_input", "category": "general"}
            }
        }


class FilterResponse(BaseModel):
    """Response schema for content filtering."""
    
    filtered_content: str = Field(..., description="Filtered/sanitized content")
    safety_score: float = Field(..., ge=0.0, le=1.0, description="Overall safety score (0=unsafe, 1=safe)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the safety assessment")
    was_filtered: bool = Field(..., description="Whether content was modified by filtering")
    filter_reasons: List[str] = Field(default_factory=list, description="Reasons for filtering")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    cached: bool = Field(False, description="Whether result was served from cache")
    rate_limit: Optional[Dict[str, Any]] = Field(None, description="Rate limit information")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    
    class Config:
        schema_extra = {
            "example": {
                "filtered_content": "Step 1: Think about how to help the user safely. Step 2: Provide helpful information.",
                "safety_score": 0.95,
                "confidence": 0.9,
                "was_filtered": False,
                "filter_reasons": [],
                "processing_time_ms": 45,
                "request_id": "req_abc123_1699123456",
                "cached": False,
                "rate_limit": {
                    "remaining": 999,
                    "limit": 1000,
                    "reset_time": 1699127056
                }
            }
        }


class FilterOperationResponse(BaseModel):
    """Response schema for filter operation records."""
    
    id: int = Field(..., description="Operation ID")
    request_id: str = Field(..., description="Request identifier")
    user_id: Optional[str] = Field(None, description="User who made the request")
    safety_score: float = Field(..., description="Safety score achieved")
    was_filtered: bool = Field(..., description="Whether content was filtered")
    filter_reasons: List[str] = Field(default_factory=list, description="Filtering reasons")
    processing_time_ms: int = Field(..., description="Processing time")
    safety_level: str = Field(..., description="Safety level used")
    created_at: datetime = Field(..., description="When the operation occurred")
    
    class Config:
        orm_mode = True


class SafetyDetectionResponse(BaseModel):
    """Response schema for safety detection results."""
    
    id: int = Field(..., description="Detection ID")
    operation_id: int = Field(..., description="Associated operation ID")
    detector_name: str = Field(..., description="Name of the detector")
    confidence: float = Field(..., description="Detection confidence")
    detected_patterns: List[str] = Field(default_factory=list, description="Patterns detected")
    severity: str = Field(..., description="Severity level")
    is_harmful: bool = Field(..., description="Whether harmful content was detected")
    reasoning: Optional[str] = Field(None, description="Detection reasoning")
    created_at: datetime = Field(..., description="When the detection occurred")
    
    class Config:
        orm_mode = True


class RuleRequest(BaseModel):
    """Request schema for creating/updating filter rules."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Rule name")
    description: Optional[str] = Field(None, max_length=500, description="Rule description")
    pattern: str = Field(..., min_length=1, description="Pattern to match")
    pattern_type: str = Field("regex", description="Type of pattern (regex, keyword, semantic)")
    action: str = Field(..., description="Action to take (allow, flag, block, sanitize)")
    severity: str = Field(..., description="Severity level (low, medium, high, critical)")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Threshold for triggering")
    enabled: bool = Field(True, description="Whether rule is enabled")
    category: Optional[str] = Field(None, description="Rule category")
    priority: int = Field(0, description="Rule priority (higher = more important)")
    
    @validator('action')
    def validate_action(cls, v):
        allowed_actions = ['allow', 'flag', 'block', 'sanitize']
        if v not in allowed_actions:
            raise ValueError(f'Action must be one of: {allowed_actions}')
        return v
    
    @validator('severity')
    def validate_severity(cls, v):
        allowed_severities = ['low', 'medium', 'high', 'critical']
        if v not in allowed_severities:
            raise ValueError(f'Severity must be one of: {allowed_severities}')
        return v


class RuleResponse(BaseModel):
    """Response schema for filter rules."""
    
    id: int = Field(..., description="Rule ID")
    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    pattern: str = Field(..., description="Pattern to match")
    action: str = Field(..., description="Action to take")
    severity: str = Field(..., description="Severity level")
    enabled: bool = Field(..., description="Whether rule is enabled")
    category: Optional[str] = Field(None, description="Rule category")
    usage_count: int = Field(..., description="Number of times rule was triggered")
    created_at: datetime = Field(..., description="When the rule was created")
    updated_at: datetime = Field(..., description="When the rule was last updated")
    
    class Config:
        orm_mode = True


class UserResponse(BaseModel):
    """Response schema for user information."""
    
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    role: str = Field(..., description="User role")
    is_active: bool = Field(..., description="Whether user is active")
    request_count: int = Field(..., description="Total number of requests made")
    last_login: Optional[datetime] = Field(None, description="Last login time")
    created_at: datetime = Field(..., description="Account creation time")
    
    class Config:
        orm_mode = True


class FilterStatsResponse(BaseModel):
    """Response schema for filtering statistics."""
    
    time_period_days: int = Field(..., description="Time period for statistics")
    total_operations: int = Field(..., description="Total filter operations")
    filtered_operations: int = Field(..., description="Operations that resulted in filtering")
    filter_rate: float = Field(..., description="Percentage of operations filtered")
    average_safety_score: float = Field(..., description="Average safety score")
    minimum_safety_score: float = Field(..., description="Minimum safety score")
    maximum_safety_score: float = Field(..., description="Maximum safety score")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    p95_processing_time_ms: float = Field(..., description="95th percentile processing time")
    max_processing_time_ms: float = Field(..., description="Maximum processing time")
    
    class Config:
        schema_extra = {
            "example": {
                "time_period_days": 7,
                "total_operations": 1250,
                "filtered_operations": 89,
                "filter_rate": 0.0712,
                "average_safety_score": 0.92,
                "minimum_safety_score": 0.15,
                "maximum_safety_score": 1.0,
                "average_processing_time_ms": 42.5,
                "p95_processing_time_ms": 98.0,
                "max_processing_time_ms": 156.0
            }
        }


class SystemStatsResponse(BaseModel):
    """Response schema for system statistics."""
    
    database_stats: Dict[str, Any] = Field(..., description="Database statistics")
    cache_stats: Dict[str, Any] = Field(..., description="Cache statistics")
    system_stats: Dict[str, Any] = Field(..., description="System statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "database_stats": {
                    "total_operations": 1250,
                    "filtered_operations": 89,
                    "average_safety_score": 0.92,
                    "total_rules": 15,
                    "active_rules": 12
                },
                "cache_stats": {
                    "total_keys": 2048,
                    "hit_rate": 0.78,
                    "memory_usage_mb": 45.2
                },
                "system_stats": {
                    "uptime_seconds": 86400,
                    "redis_version": "7.0.0",
                    "connected_clients": 5
                }
            }
        }


class AuthRequest(BaseModel):
    """Request schema for authentication."""
    
    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "admin",
                "password": "admin123"
            }
        }


class AuthResponse(BaseModel):
    """Response schema for authentication."""
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user: UserResponse = Field(..., description="User information")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 86400,
                "user": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "username": "admin",
                    "email": "admin@safepath.local",
                    "role": "admin",
                    "is_active": True,
                    "request_count": 42,
                    "last_login": "2024-01-27T10:30:00Z",
                    "created_at": "2024-01-01T00:00:00Z"
                }
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Overall health status")
    services: Dict[str, str] = Field(..., description="Individual service status")
    version: str = Field(..., description="Application version")
    timestamp: Optional[datetime] = Field(None, description="Health check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "services": {
                    "database": "ok",
                    "redis": "ok",
                    "ml_models": "ok"
                },
                "version": "0.1.0",
                "timestamp": "2024-01-27T10:30:00Z"
            }
        }