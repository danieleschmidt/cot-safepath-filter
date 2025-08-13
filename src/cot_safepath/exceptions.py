"""
Enhanced error handling and custom exceptions for CoT SafePath Filter.
"""

import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class SafePathError(Exception):
    """Enhanced base exception for SafePath operations."""
    
    def __init__(
        self, 
        message: str, 
        code: str = None, 
        details: Dict[str, Any] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.code = code or "SAFEPATH_ERROR"
        self.details = details or {}
        self.cause = cause
        self.recoverable = recoverable
        self.timestamp = datetime.utcnow()
        
        # Log the error when it's created
        self._log_error()
    
    def _log_error(self) -> None:
        """Log the error with appropriate level."""
        error_level = self.details.get('log_level', 'error')
        
        log_data = {
            'error_code': self.code,
            'error_message': self.message,
            'details': self.details,
            'recoverable': self.recoverable,
            'timestamp': self.timestamp.isoformat()
        }
        
        if self.cause:
            log_data['caused_by'] = str(self.cause)
        
        if error_level == 'critical':
            logger.critical(f"SafePath Error [{self.code}]: {self.message}", extra=log_data)
        elif error_level == 'warning':
            logger.warning(f"SafePath Error [{self.code}]: {self.message}", extra=log_data)
        else:
            logger.error(f"SafePath Error [{self.code}]: {self.message}", extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'code': self.code,
            'message': self.message,
            'details': self.details,
            'recoverable': self.recoverable,
            'timestamp': self.timestamp.isoformat(),
            'cause': str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code='{self.code}', message='{self.message}')"


class FilterError(SafePathError):
    """Exception raised during filtering operations."""
    
    def __init__(self, message: str, filter_name: str = None, stage: str = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'filter_name': filter_name,
            'stage': stage
        })
        super().__init__(message, code="FILTER_ERROR", details=details, **kwargs)
        self.filter_name = filter_name
        self.stage = stage


class DetectorError(SafePathError):
    """Exception raised by safety detectors."""
    
    def __init__(self, message: str, detector_name: str = None, confidence: float = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'detector_name': detector_name,
            'confidence': confidence
        })
        super().__init__(message, code="DETECTOR_ERROR", details=details, **kwargs)
        self.detector_name = detector_name
        self.confidence = confidence


class ModelError(SafePathError):
    """Exception raised by ML models."""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        super().__init__(message, code="MODEL_ERROR", **kwargs)
        self.model_name = model_name


class ValidationError(SafePathError):
    """Exception raised during input validation."""
    
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'field': field,
            'invalid_value': str(value) if value is not None else None
        })
        super().__init__(message, code="VALIDATION_ERROR", details=details, **kwargs)
        self.field = field
        self.value = value


class ConfigurationError(SafePathError):
    """Exception raised for configuration issues."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(message, code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key


class RateLimitError(SafePathError):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, message: str, limit: int = None, **kwargs):
        super().__init__(message, code="RATE_LIMIT_ERROR", **kwargs)
        self.limit = limit


class TimeoutError(SafePathError):
    """Exception raised when operations timeout."""
    
    def __init__(self, message: str, timeout_ms: int = None, elapsed_ms: int = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'timeout_ms': timeout_ms,
            'elapsed_ms': elapsed_ms
        })
        super().__init__(message, code="TIMEOUT_ERROR", details=details, **kwargs)
        self.timeout_ms = timeout_ms
        self.elapsed_ms = elapsed_ms


class SecurityError(SafePathError):
    """Exception raised for security violations."""
    
    def __init__(self, message: str, threat_type: str = None, source_ip: str = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'threat_type': threat_type,
            'source_ip': source_ip,
            'log_level': 'critical'  # Security errors are critical
        })
        super().__init__(
            message, 
            code="SECURITY_ERROR", 
            details=details, 
            recoverable=False,  # Security errors are not recoverable
            **kwargs
        )
        self.threat_type = threat_type
        self.source_ip = source_ip


class IntegrationError(SafePathError):
    """Exception raised during framework integration."""
    
    def __init__(self, message: str, framework: str = None, version: str = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'framework': framework,
            'version': version
        })
        super().__init__(message, code="INTEGRATION_ERROR", details=details, **kwargs)
        self.framework = framework
        self.version = version


class CapacityError(SafePathError):
    """Exception raised when system capacity is exceeded."""
    
    def __init__(self, message: str, resource: str = None, current: int = None, limit: int = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'resource': resource,
            'current_usage': current,
            'limit': limit,
            'utilization': f"{(current/limit*100):.1f}%" if current and limit else None
        })
        super().__init__(message, code="CAPACITY_ERROR", details=details, **kwargs)
        self.resource = resource
        self.current = current
        self.limit = limit


class CacheError(SafePathError):
    """Exception raised during cache operations."""
    
    def __init__(self, message: str, cache_type: str = None, operation: str = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'cache_type': cache_type,
            'operation': operation
        })
        super().__init__(message, code="CACHE_ERROR", details=details, **kwargs)
        self.cache_type = cache_type
        self.operation = operation


class DatabaseError(SafePathError):
    """Exception raised during database operations."""
    
    def __init__(self, message: str, table: str = None, operation: str = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'table': table,
            'operation': operation
        })
        super().__init__(message, code="DATABASE_ERROR", details=details, **kwargs)
        self.table = table
        self.operation = operation


class DeploymentError(SafePathError):
    """Exception raised during deployment operations."""
    
    def __init__(self, message: str, region: str = None, component: str = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'region': region,
            'component': component
        })
        super().__init__(message, code="DEPLOYMENT_ERROR", details=details, **kwargs)
        self.region = region
        self.component = component


class RegionalComplianceError(SafePathError):
    """Exception raised for regional compliance violations."""
    
    def __init__(self, message: str, region: str = None, framework: str = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'region': region,
            'compliance_framework': framework,
            'log_level': 'critical'  # Compliance errors are critical
        })
        super().__init__(
            message, 
            code="COMPLIANCE_ERROR", 
            details=details, 
            recoverable=False,  # Compliance errors are not recoverable
            **kwargs
        )
        self.region = region
        self.framework = framework