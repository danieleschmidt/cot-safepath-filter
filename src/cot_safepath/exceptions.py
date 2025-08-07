"""
Custom exceptions for the CoT SafePath Filter.
"""


class SafePathError(Exception):
    """Base exception for SafePath operations."""
    
    def __init__(self, message: str, code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code or "SAFEPATH_ERROR"
        self.details = details or {}


class FilterError(SafePathError):
    """Exception raised during filtering operations."""
    
    def __init__(self, message: str, filter_name: str = None, **kwargs):
        super().__init__(message, code="FILTER_ERROR", **kwargs)
        self.filter_name = filter_name


class DetectorError(SafePathError):
    """Exception raised by safety detectors."""
    
    def __init__(self, message: str, detector_name: str = None, **kwargs):
        super().__init__(message, code="DETECTOR_ERROR", **kwargs)
        self.detector_name = detector_name


class ModelError(SafePathError):
    """Exception raised by ML models."""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        super().__init__(message, code="MODEL_ERROR", **kwargs)
        self.model_name = model_name


class ValidationError(SafePathError):
    """Exception raised during input validation."""
    
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(message, code="VALIDATION_ERROR", **kwargs)
        self.field = field


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
    
    def __init__(self, message: str, timeout_ms: int = None, **kwargs):
        super().__init__(message, code="TIMEOUT_ERROR", **kwargs)
        self.timeout_ms = timeout_ms


class IntegrationError(SafePathError):
    """Exception raised during framework integration."""
    
    def __init__(self, message: str, framework: str = None, **kwargs):
        super().__init__(message, code="INTEGRATION_ERROR", **kwargs)
        self.framework = framework


class SecurityError(SafePathError):
    """Exception raised for security-related issues."""
    
    def __init__(self, message: str, threat_level: str = None, **kwargs):
        super().__init__(message, code="SECURITY_ERROR", **kwargs)
        self.threat_level = threat_level