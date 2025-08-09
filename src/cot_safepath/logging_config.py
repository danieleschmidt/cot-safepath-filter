"""
Advanced logging configuration for CoT SafePath Filter.
"""

import os
import sys
import json
import logging
import logging.config
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import structlog


class SecuritySafeFormatter(logging.Formatter):
    """Custom formatter that redacts sensitive information."""
    
    SENSITIVE_KEYS = {
        'password', 'api_key', 'token', 'secret', 'key',
        'auth', 'credential', 'private', 'confidential'
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redacted_value = "[REDACTED]"
    
    def format(self, record):
        # Redact sensitive information from the record
        if hasattr(record, 'args') and record.args:
            record.args = self._redact_sensitive_data(record.args)
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._redact_sensitive_strings(record.msg)
        
        return super().format(record)
    
    def _redact_sensitive_data(self, data):
        """Recursively redact sensitive data from log arguments."""
        if isinstance(data, dict):
            return {
                key: self.redacted_value if self._is_sensitive_key(key) 
                else self._redact_sensitive_data(value)
                for key, value in data.items()
            }
        elif isinstance(data, (list, tuple)):
            return [self._redact_sensitive_data(item) for item in data]
        elif isinstance(data, str):
            return self._redact_sensitive_strings(data)
        else:
            return data
    
    def _is_sensitive_key(self, key):
        """Check if a key name indicates sensitive data."""
        if not isinstance(key, str):
            return False
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self.SENSITIVE_KEYS)
    
    def _redact_sensitive_strings(self, text):
        """Redact sensitive patterns in strings."""
        import re
        
        # Patterns for common sensitive data
        patterns = [
            (r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\\s]+)["\']?', r'api_key="[REDACTED]"'),
            (r'token["\']?\s*[:=]\s*["\']?([^"\'\\s]+)["\']?', r'token="[REDACTED]"'),
            (r'password["\']?\s*[:=]\s*["\']?([^"\'\\s]+)["\']?', r'password="[REDACTED]"'),
            (r'sk-[a-zA-Z0-9]{32,}', r'sk-[REDACTED]'),  # API keys starting with sk-
            (r'Bearer\\s+([a-zA-Z0-9_.-]+)', r'Bearer [REDACTED]'),  # Bearer tokens
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text


class StructuredLogProcessor:
    """Processor for structlog to add consistent fields."""
    
    def __init__(self, service_name: str = "safepath"):
        self.service_name = service_name
    
    def __call__(self, logger, method_name, event_dict):
        # Add consistent fields to all log entries
        event_dict['service'] = self.service_name
        event_dict['timestamp'] = datetime.utcnow().isoformat()
        event_dict['level'] = method_name.upper()
        
        # Add request context if available
        import contextvars
        try:
            request_id = contextvars.copy_context().get('request_id', None)
            if request_id:
                event_dict['request_id'] = request_id
        except:
            pass
        
        return event_dict


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json: bool = False,
    enable_console: bool = True,
    service_name: str = "safepath"
) -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_json: Whether to use JSON formatting
        enable_console: Whether to log to console
        service_name: Name of the service for structured logging
    """
    
    # Ensure log directory exists if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Base logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(funcName)s() - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'json': {
                'format': '%(message)s',
                'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
            },
            'security': {
                '()': SecuritySafeFormatter,
                'format': '%(asctime)s [%(levelname)s] SECURITY %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {},
        'loggers': {
            'cot_safepath': {
                'level': level,
                'handlers': [],
                'propagate': False
            },
            'cot_safepath.security': {
                'level': 'INFO',
                'handlers': ['security_file'],
                'propagate': False
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': [],
                'propagate': False
            },
            'fastapi': {
                'level': 'INFO', 
                'handlers': [],
                'propagate': False
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': []
        }
    }
    
    # Add console handler
    if enable_console:
        config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'level': level,
            'formatter': 'json' if enable_json else 'standard',
            'stream': 'ext://sys.stdout'
        }
        config['loggers']['cot_safepath']['handlers'].append('console')
        config['loggers']['uvicorn']['handlers'].append('console')
        config['loggers']['fastapi']['handlers'].append('console')
        config['root']['handlers'].append('console')
    
    # Add file handler
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'json' if enable_json else 'detailed',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        }
        config['loggers']['cot_safepath']['handlers'].append('file')
        config['loggers']['uvicorn']['handlers'].append('file')
        config['loggers']['fastapi']['handlers'].append('file')
        config['root']['handlers'].append('file')
        
        # Security-specific log file
        security_log_file = str(Path(log_file).with_suffix('.security.log'))
        config['handlers']['security_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'security',
            'filename': security_log_file,
            'maxBytes': 5242880,  # 5MB
            'backupCount': 10,
            'encoding': 'utf8'
        }
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Setup structlog
    structlog.configure(
        processors=[
            StructuredLogProcessor(service_name),
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if enable_json else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Test logging
    logger = logging.getLogger('cot_safepath')
    logger.info(f"Logging initialized - Level: {level}, JSON: {enable_json}, File: {log_file}")


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def setup_request_context(request_id: str) -> None:
    """Setup request context for logging."""
    import contextvars
    ctx = contextvars.copy_context()
    ctx['request_id'] = request_id


class LoggingMiddleware:
    """Middleware to add request context to all logs."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Generate request ID
            import uuid
            request_id = str(uuid.uuid4())
            
            # Add to scope for access in handlers
            scope["request_id"] = request_id
            
            # Setup logging context
            setup_request_context(request_id)
        
        await self.app(scope, receive, send)


def configure_production_logging():
    """Configure logging for production environment."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_file = os.getenv('LOG_FILE', '/var/log/safepath/application.log')
    enable_json = os.getenv('LOG_FORMAT', 'json').lower() == 'json'
    
    setup_logging(
        level=log_level,
        log_file=log_file,
        enable_json=enable_json,
        enable_console=True,
        service_name="safepath-prod"
    )


def configure_development_logging():
    """Configure logging for development environment."""
    setup_logging(
        level='DEBUG',
        log_file='logs/safepath-dev.log',
        enable_json=False,
        enable_console=True,
        service_name="safepath-dev"
    )


def configure_testing_logging():
    """Configure logging for testing environment."""
    setup_logging(
        level='WARNING',
        log_file=None,
        enable_json=False,
        enable_console=False,  # Quiet during tests
        service_name="safepath-test"
    )


# Auto-configure based on environment
def auto_configure_logging():
    """Automatically configure logging based on environment."""
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        configure_production_logging()
    elif env == 'testing':
        configure_testing_logging()
    else:
        configure_development_logging()


# Initialize logging when module is imported
if __name__ != "__main__":
    auto_configure_logging()