"""
Custom middleware for the SafePath API.
"""

import time
import logging
import json
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import HTTPException

from ..cache import RateLimitCache, get_redis_manager
from ..exceptions import RateLimitError


logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log response
            logger.info(
                f"Response: {response.status_code} "
                f"in {process_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"in {process_time:.3f}s - {str(e)}"
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for API rate limiting."""
    
    def __init__(self, app, default_limit: int = 1000, window_seconds: int = 3600):
        super().__init__(app)
        self.default_limit = default_limit
        self.window_seconds = window_seconds
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks and static endpoints
        if request.url.path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        try:
            # Get Redis client
            redis_manager = get_redis_manager()
            rate_limiter = RateLimitCache(redis_manager)
            
            # Get identifier (IP address or user ID from auth)
            identifier = self._get_identifier(request)
            
            # Check rate limit
            rate_limit_result = rate_limiter.check_rate_limit(
                identifier=identifier,
                limit=self.default_limit,
                window_seconds=self.window_seconds
            )
            
            if not rate_limit_result['allowed']:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "type": "RateLimitExceeded",
                            "message": "Rate limit exceeded",
                            "details": {
                                "limit": rate_limit_result['limit'],
                                "current_count": rate_limit_result['current_count'],
                                "reset_time": rate_limit_result['reset_time'],
                                "retry_after": rate_limit_result['reset_time'] - int(time.time())
                            }
                        }
                    },
                    headers={
                        "X-RateLimit-Limit": str(rate_limit_result['limit']),
                        "X-RateLimit-Remaining": str(rate_limit_result['remaining']),
                        "X-RateLimit-Reset": str(rate_limit_result['reset_time']),
                        "Retry-After": str(rate_limit_result['reset_time'] - int(time.time()))
                    }
                )
            
            # Continue with request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(rate_limit_result['limit'])
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_result['remaining'])
            response.headers["X-RateLimit-Reset"] = str(rate_limit_result['reset_time'])
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # On error, allow the request (fail open)
            return await call_next(request)
    
    def _get_identifier(self, request: Request) -> str:
        """Get rate limiting identifier from request."""
        # Try to get user ID from authorization header
        auth_header = request.headers.get("authorization")
        if auth_header:
            try:
                # Extract user ID from JWT token (simplified)
                # In real implementation, decode and validate JWT
                token = auth_header.replace("Bearer ", "")
                # For now, use token as identifier
                return f"user:{token[:16]}"
            except:
                pass
        
        # Fall back to IP address
        client_ip = "unknown"
        if request.client:
            client_ip = request.client.host
        
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            client_ip = real_ip
        
        return f"ip:{client_ip}"


class SafetyMiddleware(BaseHTTPMiddleware):
    """Middleware for additional safety checks."""
    
    def __init__(self, app, max_request_size: int = 10485760):  # 10MB
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "type": "RequestTooLarge",
                        "message": f"Request size exceeds maximum allowed size of {self.max_request_size} bytes"
                    }
                }
            )
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                if request.url.path.startswith("/api/"):
                    return JSONResponse(
                        status_code=415,
                        content={
                            "error": {
                                "type": "UnsupportedMediaType",
                                "message": "Content-Type must be application/json"
                            }
                        }
                    )
        
        # Check for suspicious patterns in URL
        if self._contains_suspicious_patterns(str(request.url)):
            logger.warning(f"Suspicious request detected: {request.url}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "BadRequest",
                        "message": "Invalid request"
                    }
                }
            )
        
        # Add security headers to response
        response = await call_next(request)
        
        # Security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    def _contains_suspicious_patterns(self, url: str) -> bool:
        """Check for suspicious patterns in URL."""
        suspicious_patterns = [
            "../",  # Path traversal
            "..\\",  # Windows path traversal
            "<script",  # XSS attempt
            "javascript:",  # JavaScript injection
            "data:",  # Data URI
            "vbscript:",  # VBScript injection
            "%3c",  # Encoded <
            "%3e",  # Encoded >
            "union select",  # SQL injection
            "or 1=1",  # SQL injection
            "drop table",  # SQL injection
        ]
        
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in suspicious_patterns)


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with additional validation."""
    
    def __init__(
        self, 
        app, 
        allowed_origins: list = None,
        allowed_methods: list = None,
        allowed_headers: list = None
    ):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or ["*"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response(status_code=200)
        else:
            response = await call_next(request)
        
        # Add CORS headers
        if origin:
            if self._is_origin_allowed(origin):
                response.headers["Access-Control-Allow-Origin"] = origin
            else:
                logger.warning(f"Origin not allowed: {origin}")
                response.headers["Access-Control-Allow-Origin"] = "null"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "3600"
        
        return response
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if "*" in self.allowed_origins:
            return True
        
        return origin in self.allowed_origins


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting request metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.request_duration_sum = 0.0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            self.request_count += 1
            self.request_duration_sum += duration
            
            # Add metrics headers
            response.headers["X-Request-Count"] = str(self.request_count)
            response.headers["X-Average-Duration"] = str(self.request_duration_sum / self.request_count)
            
            # Log metrics (could be sent to Prometheus)
            logger.debug(
                f"Metrics: request_count={self.request_count}, "
                f"duration={duration:.3f}s, "
                f"status={response.status_code}"
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Request failed in {duration:.3f}s: {e}")
            raise