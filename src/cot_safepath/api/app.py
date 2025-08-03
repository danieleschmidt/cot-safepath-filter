"""
FastAPI application factory and configuration.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from .routes import router
from .middleware import SafetyMiddleware, RateLimitMiddleware, LoggingMiddleware
from ..database import init_database
from ..cache import init_redis
from ..exceptions import SafePathError


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting CoT SafePath Filter API...")
    
    try:
        # Initialize database
        db_manager = init_database()
        db_manager.check_connection()
        logger.info("Database connection established")
        
        # Initialize Redis
        redis_manager = init_redis()
        redis_manager.ping()
        logger.info("Redis connection established")
        
        # Run database migrations if needed
        # db_manager.run_migrations()
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down CoT SafePath Filter API...")


def create_app(
    debug: bool = None,
    testing: bool = False,
    database_url: str = None,
    redis_url: str = None
) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        debug: Enable debug mode
        testing: Enable testing mode
        database_url: Database connection URL
        redis_url: Redis connection URL
        
    Returns:
        Configured FastAPI application
    """
    
    # Configure debug mode
    if debug is None:
        debug = os.getenv("DEBUG", "false").lower() == "true"
    
    # Create FastAPI app
    app = FastAPI(
        title="CoT SafePath Filter API",
        description="Real-time middleware that intercepts and sanitizes chain-of-thought reasoning to prevent harmful or deceptive reasoning patterns",
        version="0.1.0",
        debug=debug,
        lifespan=lifespan if not testing else None,
        docs_url="/docs" if debug else None,
        redoc_url="/redoc" if debug else None,
        openapi_url="/openapi.json" if debug else None,
    )
    
    # Configure CORS
    allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(SafetyMiddleware)
    
    # Include routers
    app.include_router(router, prefix="/api/v1")
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        try:
            # Check database
            from ..database import get_database_manager
            db_manager = get_database_manager()
            db_ok = db_manager.check_connection()
            
            # Check Redis
            from ..cache import get_redis_manager
            redis_manager = get_redis_manager()
            redis_ok = redis_manager.ping()
            
            status = "healthy" if db_ok and redis_ok else "unhealthy"
            
            return {
                "status": status,
                "services": {
                    "database": "ok" if db_ok else "error",
                    "redis": "ok" if redis_ok else "error"
                },
                "version": "0.1.0"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "version": "0.1.0"
                }
            )
    
    # Metrics endpoint
    @app.get("/metrics", tags=["Monitoring"])
    async def metrics():
        """Prometheus metrics endpoint."""
        try:
            from ..monitoring import PrometheusMetrics
            metrics_collector = PrometheusMetrics()
            return metrics_collector.generate_metrics()
        except Exception as e:
            logger.error(f"Metrics generation failed: {e}")
            raise HTTPException(status_code=500, detail="Metrics unavailable")
    
    # Error handlers
    @app.exception_handler(SafePathError)
    async def safepath_error_handler(request: Request, exc: SafePathError):
        """Handle SafePath-specific errors."""
        logger.error(f"SafePath error: {exc.message}")
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": exc.__class__.__name__,
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details
                }
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "ValidationError",
                    "message": "Request validation failed",
                    "details": exc.errors()
                }
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "HTTPError",
                    "message": exc.detail,
                    "status_code": exc.status_code
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle unexpected errors."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "InternalServerError",
                    "message": "An unexpected error occurred"
                }
            }
        )
    
    return app


# Create default app instance
app = create_app()


def main():
    """Run the application with uvicorn."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if workers > 1:
        # Multi-worker mode
        uvicorn.run(
            "cot_safepath.api.app:app",
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            access_log=True
        )
    else:
        # Single worker mode (supports reload)
        uvicorn.run(
            "cot_safepath.api.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )


if __name__ == "__main__":
    main()