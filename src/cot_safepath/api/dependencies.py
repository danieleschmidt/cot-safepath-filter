"""
FastAPI dependency injection for common services.
"""

import os
import jwt
from typing import Optional, Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ..database import get_database_session as _get_db_session, User
from ..database.repositories import UserRepository
from ..cache import get_redis_client as _get_redis_client
from ..exceptions import ValidationError


security = HTTPBearer(auto_error=False)


def get_database_session() -> Generator[Session, None, None]:
    """Dependency for database session."""
    with _get_db_session() as session:
        try:
            yield session
        finally:
            session.close()


def get_redis_client():
    """Dependency for Redis client."""
    return _get_redis_client()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_database_session)
) -> Optional[User]:
    """
    Dependency for getting current authenticated user.
    
    Returns None if no authentication provided (for public endpoints).
    Raises HTTPException if authentication is invalid.
    """
    if not credentials:
        return None
    
    try:
        # Decode JWT token
        secret_key = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key-change-in-production")
        algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        
        payload = jwt.decode(credentials.credentials, secret_key, algorithms=[algorithm])
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
        user_repo = UserRepository(db)
        user = user_repo.get_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update last request time
        user_repo.update_last_request(user.id)
        
        return user
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_authenticated_user(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """
    Dependency that requires authentication.
    Raises HTTPException if user is not authenticated.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return current_user


async def require_admin_user(
    current_user: User = Depends(require_authenticated_user)
) -> User:
    """
    Dependency that requires admin role.
    Raises HTTPException if user is not admin.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


async def get_current_user_or_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_database_session)
) -> Optional[User]:
    """
    Dependency for getting user via JWT token or API key.
    
    First tries JWT authentication, then falls back to API key.
    """
    if not credentials:
        return None
    
    token = credentials.credentials
    
    # First try JWT authentication
    try:
        user = await get_current_user(credentials, db)
        if user:
            return user
    except HTTPException:
        # JWT failed, try API key
        pass
    
    # Try API key authentication
    try:
        user_repo = UserRepository(db)
        user = user_repo.get_by_api_key(token)
        
        if user:
            user_repo.update_last_request(user.id)
            return user
        
        # Invalid API key
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


class RateLimitInfo:
    """Information about rate limiting for the current request."""
    
    def __init__(self, limit: int, remaining: int, reset_time: int):
        self.limit = limit
        self.remaining = remaining
        self.reset_time = reset_time


async def get_rate_limit_info(
    current_user: Optional[User] = Depends(get_current_user),
    redis_client = Depends(get_redis_client)
) -> RateLimitInfo:
    """
    Dependency for getting rate limit information.
    """
    from ..cache import RateLimitCache
    
    # Get user-specific or default limits
    if current_user and current_user.rate_limit_override:
        limit = current_user.rate_limit_override
    else:
        limit = 1000  # Default limit
    
    # Get identifier
    identifier = f"user:{current_user.id}" if current_user else "anonymous"
    
    # Check current rate limit status
    rate_limiter = RateLimitCache(redis_client)
    status = rate_limiter.get_rate_limit_status(identifier, 3600)  # 1 hour window
    
    remaining = max(0, limit - status['current_count'])
    reset_time = int(status.get('window_end', 0)) + 3600
    
    return RateLimitInfo(
        limit=limit,
        remaining=remaining,
        reset_time=reset_time
    )


def validate_request_size(max_size: int = 10485760):  # 10MB default
    """
    Dependency factory for validating request size.
    """
    async def _validate_request_size(request):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request size exceeds maximum allowed size of {max_size} bytes"
            )
        return True
    
    return _validate_request_size


def validate_content_type(allowed_types: list = None):
    """
    Dependency factory for validating content type.
    """
    if allowed_types is None:
        allowed_types = ["application/json"]
    
    async def _validate_content_type(request):
        content_type = request.headers.get("content-type", "")
        if not any(content_type.startswith(allowed_type) for allowed_type in allowed_types):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Content-Type must be one of: {', '.join(allowed_types)}"
            )
        return True
    
    return _validate_content_type


class PaginationParams:
    """Pagination parameters for list endpoints."""
    
    def __init__(self, limit: int = 100, offset: int = 0):
        self.limit = min(max(1, limit), 1000)  # Between 1 and 1000
        self.offset = max(0, offset)  # Non-negative


async def get_pagination_params(
    limit: int = 100,
    offset: int = 0
) -> PaginationParams:
    """Dependency for pagination parameters."""
    return PaginationParams(limit=limit, offset=offset)


class FilterParams:
    """Filtering parameters for filter endpoints."""
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        safety_level: Optional[str] = None,
        was_filtered: Optional[bool] = None,
        min_safety_score: Optional[float] = None,
        max_safety_score: Optional[float] = None
    ):
        self.user_id = user_id
        self.safety_level = safety_level
        self.was_filtered = was_filtered
        self.min_safety_score = min_safety_score
        self.max_safety_score = max_safety_score


async def get_filter_params(
    user_id: Optional[str] = None,
    safety_level: Optional[str] = None,
    was_filtered: Optional[bool] = None,
    min_safety_score: Optional[float] = None,
    max_safety_score: Optional[float] = None
) -> FilterParams:
    """Dependency for filter parameters."""
    
    # Validate safety level
    if safety_level:
        from ..models import SafetyLevel
        valid_levels = [level.value for level in SafetyLevel]
        if safety_level not in valid_levels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid safety level. Must be one of: {valid_levels}"
            )
    
    # Validate safety score range
    if min_safety_score is not None and (min_safety_score < 0 or min_safety_score > 1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="min_safety_score must be between 0 and 1"
        )
    
    if max_safety_score is not None and (max_safety_score < 0 or max_safety_score > 1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_safety_score must be between 0 and 1"
        )
    
    if (min_safety_score is not None and max_safety_score is not None and 
        min_safety_score > max_safety_score):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="min_safety_score cannot be greater than max_safety_score"
        )
    
    return FilterParams(
        user_id=user_id,
        safety_level=safety_level,
        was_filtered=was_filtered,
        min_safety_score=min_safety_score,
        max_safety_score=max_safety_score
    )