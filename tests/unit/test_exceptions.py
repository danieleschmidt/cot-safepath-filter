"""Unit tests for custom exception classes."""

import pytest
from typing import Dict, Any

from cot_safepath.exceptions import (
    SafePathError, FilterError, DetectorError, ModelError, ValidationError,
    ConfigurationError, RateLimitError, TimeoutError, IntegrationError
)


class TestSafePathError:
    """Test the base SafePathError exception."""
    
    def test_safepath_error_basic(self):
        """Test basic SafePathError creation."""
        error = SafePathError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.code == "SAFEPATH_ERROR"
        assert error.details == {}
    
    def test_safepath_error_with_code(self):
        """Test SafePathError with custom code."""
        error = SafePathError("Test message", code="CUSTOM_CODE")
        
        assert error.message == "Test message"
        assert error.code == "CUSTOM_CODE"
        assert error.details == {}
    
    def test_safepath_error_with_details(self):
        """Test SafePathError with details dictionary."""
        details = {"field": "content", "value": "invalid", "reason": "too_long"}
        error = SafePathError("Test message", details=details)
        
        assert error.message == "Test message"
        assert error.code == "SAFEPATH_ERROR"
        assert error.details == details
    
    def test_safepath_error_full_parameters(self):
        """Test SafePathError with all parameters."""
        details = {"context": "testing", "severity": "high"}
        error = SafePathError(
            message="Full error message",
            code="FULL_ERROR",
            details=details
        )
        
        assert error.message == "Full error message"
        assert error.code == "FULL_ERROR"
        assert error.details == details
    
    def test_safepath_error_inheritance(self):
        """Test SafePathError inheritance from Exception."""
        error = SafePathError("Test message")
        
        assert isinstance(error, Exception)
        assert isinstance(error, SafePathError)
    
    def test_safepath_error_string_representation(self):
        """Test string representation of SafePathError."""
        error = SafePathError("String representation test")
        
        assert str(error) == "String representation test"
        assert repr(error)  # Should not raise exception


class TestFilterError:
    """Test FilterError exception."""
    
    def test_filter_error_basic(self):
        """Test basic FilterError creation."""
        error = FilterError("Filter processing failed")
        
        assert str(error) == "Filter processing failed"
        assert error.message == "Filter processing failed"
        assert error.code == "FILTER_ERROR"
        assert error.filter_name is None
    
    def test_filter_error_with_filter_name(self):
        """Test FilterError with filter name."""
        error = FilterError("Token filter failed", filter_name="token_filter")
        
        assert error.message == "Token filter failed"
        assert error.filter_name == "token_filter"
        assert error.code == "FILTER_ERROR"
    
    def test_filter_error_with_details(self):
        """Test FilterError with details."""
        details = {"stage": "preprocessing", "input_length": 1000}
        error = FilterError("Filter error", filter_name="preprocessor", details=details)
        
        assert error.filter_name == "preprocessor"
        assert error.details == details
    
    def test_filter_error_inheritance(self):
        """Test FilterError inheritance."""
        error = FilterError("Test message")
        
        assert isinstance(error, SafePathError)
        assert isinstance(error, FilterError)
        assert error.code == "FILTER_ERROR"


class TestDetectorError:
    """Test DetectorError exception."""
    
    def test_detector_error_basic(self):
        """Test basic DetectorError creation."""
        error = DetectorError("Detector analysis failed")
        
        assert str(error) == "Detector analysis failed"
        assert error.message == "Detector analysis failed"
        assert error.code == "DETECTOR_ERROR"
        assert error.detector_name is None
    
    def test_detector_error_with_detector_name(self):
        """Test DetectorError with detector name."""
        error = DetectorError("Deception detection failed", detector_name="deception_detector")
        
        assert error.message == "Deception detection failed"
        assert error.detector_name == "deception_detector"
        assert error.code == "DETECTOR_ERROR"
    
    def test_detector_error_with_details(self):
        """Test DetectorError with additional details."""
        details = {"confidence_threshold": 0.7, "patterns_analyzed": 5}
        error = DetectorError(
            "Pattern matching failed",
            detector_name="pattern_detector",
            details=details
        )
        
        assert error.detector_name == "pattern_detector"
        assert error.details == details
    
    def test_detector_error_inheritance(self):
        """Test DetectorError inheritance."""
        error = DetectorError("Test message")
        
        assert isinstance(error, SafePathError)
        assert isinstance(error, DetectorError)
        assert error.code == "DETECTOR_ERROR"


class TestModelError:
    """Test ModelError exception."""
    
    def test_model_error_basic(self):
        """Test basic ModelError creation."""
        error = ModelError("ML model inference failed")
        
        assert str(error) == "ML model inference failed"
        assert error.message == "ML model inference failed"
        assert error.code == "MODEL_ERROR"
        assert error.model_name is None
    
    def test_model_error_with_model_name(self):
        """Test ModelError with model name."""
        error = ModelError("BERT model failed", model_name="bert-base-uncased")
        
        assert error.message == "BERT model failed"
        assert error.model_name == "bert-base-uncased"
        assert error.code == "MODEL_ERROR"
    
    def test_model_error_with_details(self):
        """Test ModelError with model-specific details."""
        details = {
            "model_version": "1.2.3",
            "input_shape": [1, 512],
            "memory_usage": "2GB"
        }
        error = ModelError(
            "Model loading failed",
            model_name="safety_classifier",
            details=details
        )
        
        assert error.model_name == "safety_classifier"
        assert error.details == details
    
    def test_model_error_inheritance(self):
        """Test ModelError inheritance."""
        error = ModelError("Test message")
        
        assert isinstance(error, SafePathError)
        assert isinstance(error, ModelError)
        assert error.code == "MODEL_ERROR"


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_validation_error_basic(self):
        """Test basic ValidationError creation."""
        error = ValidationError("Input validation failed")
        
        assert str(error) == "Input validation failed"
        assert error.message == "Input validation failed"
        assert error.code == "VALIDATION_ERROR"
        assert error.field is None
    
    def test_validation_error_with_field(self):
        """Test ValidationError with field name."""
        error = ValidationError("Content too long", field="content")
        
        assert error.message == "Content too long"
        assert error.field == "content"
        assert error.code == "VALIDATION_ERROR"
    
    def test_validation_error_with_details(self):
        """Test ValidationError with validation details."""
        details = {
            "max_length": 50000,
            "actual_length": 75000,
            "validation_rule": "max_length_check"
        }
        error = ValidationError(
            "Content exceeds maximum length",
            field="content",
            details=details
        )
        
        assert error.field == "content"
        assert error.details == details
        assert error.details["max_length"] == 50000
    
    def test_validation_error_inheritance(self):
        """Test ValidationError inheritance."""
        error = ValidationError("Test message")
        
        assert isinstance(error, SafePathError)
        assert isinstance(error, ValidationError)
        assert error.code == "VALIDATION_ERROR"


class TestConfigurationError:
    """Test ConfigurationError exception."""
    
    def test_configuration_error_basic(self):
        """Test basic ConfigurationError creation."""
        error = ConfigurationError("Invalid configuration")
        
        assert str(error) == "Invalid configuration"
        assert error.message == "Invalid configuration"
        assert error.code == "CONFIG_ERROR"
        assert error.config_key is None
    
    def test_configuration_error_with_config_key(self):
        """Test ConfigurationError with configuration key."""
        error = ConfigurationError("Invalid threshold value", config_key="filter_threshold")
        
        assert error.message == "Invalid threshold value"
        assert error.config_key == "filter_threshold"
        assert error.code == "CONFIG_ERROR"
    
    def test_configuration_error_with_details(self):
        """Test ConfigurationError with configuration details."""
        details = {
            "expected_type": "float",
            "actual_type": "str",
            "valid_range": [0.0, 1.0],
            "provided_value": "invalid"
        }
        error = ConfigurationError(
            "Configuration type mismatch",
            config_key="threshold",
            details=details
        )
        
        assert error.config_key == "threshold"
        assert error.details == details
    
    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inheritance."""
        error = ConfigurationError("Test message")
        
        assert isinstance(error, SafePathError)
        assert isinstance(error, ConfigurationError)
        assert error.code == "CONFIG_ERROR"


class TestRateLimitError:
    """Test RateLimitError exception."""
    
    def test_rate_limit_error_basic(self):
        """Test basic RateLimitError creation."""
        error = RateLimitError("Rate limit exceeded")
        
        assert str(error) == "Rate limit exceeded"
        assert error.message == "Rate limit exceeded"
        assert error.code == "RATE_LIMIT_ERROR"
        assert error.limit is None
    
    def test_rate_limit_error_with_limit(self):
        """Test RateLimitError with limit value."""
        error = RateLimitError("Too many requests", limit=100)
        
        assert error.message == "Too many requests"
        assert error.limit == 100
        assert error.code == "RATE_LIMIT_ERROR"
    
    def test_rate_limit_error_with_details(self):
        """Test RateLimitError with rate limiting details."""
        details = {
            "requests_per_minute": 60,
            "current_count": 85,
            "window_start": "2023-01-01T12:00:00Z",
            "retry_after": 30
        }
        error = RateLimitError(
            "Rate limit exceeded for user",
            limit=60,
            details=details
        )
        
        assert error.limit == 60
        assert error.details == details
        assert error.details["current_count"] == 85
    
    def test_rate_limit_error_inheritance(self):
        """Test RateLimitError inheritance."""
        error = RateLimitError("Test message")
        
        assert isinstance(error, SafePathError)
        assert isinstance(error, RateLimitError)
        assert error.code == "RATE_LIMIT_ERROR"


class TestTimeoutError:
    """Test TimeoutError exception."""
    
    def test_timeout_error_basic(self):
        """Test basic TimeoutError creation."""
        error = TimeoutError("Operation timed out")
        
        assert str(error) == "Operation timed out"
        assert error.message == "Operation timed out"
        assert error.code == "TIMEOUT_ERROR"
        assert error.timeout_ms is None
    
    def test_timeout_error_with_timeout(self):
        """Test TimeoutError with timeout value."""
        error = TimeoutError("Processing timed out", timeout_ms=5000)
        
        assert error.message == "Processing timed out"
        assert error.timeout_ms == 5000
        assert error.code == "TIMEOUT_ERROR"
    
    def test_timeout_error_with_details(self):
        """Test TimeoutError with timeout details."""
        details = {
            "operation": "semantic_analysis",
            "start_time": "2023-01-01T12:00:00Z",
            "timeout_threshold": 10000,
            "actual_duration": 12000
        }
        error = TimeoutError(
            "Semantic analysis timed out",
            timeout_ms=10000,
            details=details
        )
        
        assert error.timeout_ms == 10000
        assert error.details == details
        assert error.details["actual_duration"] == 12000
    
    def test_timeout_error_inheritance(self):
        """Test TimeoutError inheritance."""
        error = TimeoutError("Test message")
        
        assert isinstance(error, SafePathError)
        assert isinstance(error, TimeoutError)
        assert error.code == "TIMEOUT_ERROR"


class TestIntegrationError:
    """Test IntegrationError exception."""
    
    def test_integration_error_basic(self):
        """Test basic IntegrationError creation."""
        error = IntegrationError("Framework integration failed")
        
        assert str(error) == "Framework integration failed"
        assert error.message == "Framework integration failed"
        assert error.code == "INTEGRATION_ERROR"
        assert error.framework is None
    
    def test_integration_error_with_framework(self):
        """Test IntegrationError with framework name."""
        error = IntegrationError("FastAPI integration failed", framework="fastapi")
        
        assert error.message == "FastAPI integration failed"
        assert error.framework == "fastapi"
        assert error.code == "INTEGRATION_ERROR"
    
    def test_integration_error_with_details(self):
        """Test IntegrationError with integration details."""
        details = {
            "framework_version": "0.68.0",
            "python_version": "3.9.0",
            "error_location": "middleware_setup",
            "dependency_issue": "missing_module"
        }
        error = IntegrationError(
            "Middleware integration failed",
            framework="starlette",
            details=details
        )
        
        assert error.framework == "starlette"
        assert error.details == details
        assert error.details["framework_version"] == "0.68.0"
    
    def test_integration_error_inheritance(self):
        """Test IntegrationError inheritance."""
        error = IntegrationError("Test message")
        
        assert isinstance(error, SafePathError)
        assert isinstance(error, IntegrationError)
        assert error.code == "INTEGRATION_ERROR"


class TestExceptionHierarchy:
    """Test the exception hierarchy and relationships."""
    
    def test_all_exceptions_inherit_from_safepath_error(self):
        """Test that all custom exceptions inherit from SafePathError."""
        exception_classes = [
            FilterError, DetectorError, ModelError, ValidationError,
            ConfigurationError, RateLimitError, TimeoutError, IntegrationError
        ]
        
        for exception_class in exception_classes:
            error = exception_class("Test message")
            assert isinstance(error, SafePathError)
            assert isinstance(error, Exception)
    
    def test_exception_codes_are_unique(self):
        """Test that exception codes are unique."""
        exceptions = [
            SafePathError("test"),
            FilterError("test"),
            DetectorError("test"),
            ModelError("test"),
            ValidationError("test"),
            ConfigurationError("test"),
            RateLimitError("test"),
            TimeoutError("test"),
            IntegrationError("test")
        ]
        
        codes = [exc.code for exc in exceptions]
        assert len(codes) == len(set(codes))  # All codes should be unique
    
    def test_exception_code_patterns(self):
        """Test that exception codes follow expected patterns."""
        code_mappings = {
            SafePathError: "SAFEPATH_ERROR",
            FilterError: "FILTER_ERROR",
            DetectorError: "DETECTOR_ERROR",
            ModelError: "MODEL_ERROR",
            ValidationError: "VALIDATION_ERROR",
            ConfigurationError: "CONFIG_ERROR",
            RateLimitError: "RATE_LIMIT_ERROR",
            TimeoutError: "TIMEOUT_ERROR",
            IntegrationError: "INTEGRATION_ERROR"
        }
        
        for exception_class, expected_code in code_mappings.items():
            error = exception_class("test message")
            assert error.code == expected_code


class TestExceptionUsagePatterns:
    """Test common exception usage patterns."""
    
    def test_raising_and_catching_specific_exceptions(self):
        """Test raising and catching specific exception types."""
        def raise_filter_error():
            raise FilterError("Filter processing failed", filter_name="test_filter")
        
        def raise_validation_error():
            raise ValidationError("Invalid input", field="content")
        
        # Test FilterError catching
        with pytest.raises(FilterError) as exc_info:
            raise_filter_error()
        
        assert exc_info.value.filter_name == "test_filter"
        assert exc_info.value.code == "FILTER_ERROR"
        
        # Test ValidationError catching
        with pytest.raises(ValidationError) as exc_info:
            raise_validation_error()
        
        assert exc_info.value.field == "content"
        assert exc_info.value.code == "VALIDATION_ERROR"
    
    def test_catching_base_exception(self):
        """Test catching base SafePathError for any custom exception."""
        def raise_various_errors(error_type):
            if error_type == "filter":
                raise FilterError("Filter error")
            elif error_type == "detector":
                raise DetectorError("Detector error")
            elif error_type == "validation":
                raise ValidationError("Validation error")
        
        # All should be catchable as SafePathError
        for error_type in ["filter", "detector", "validation"]:
            with pytest.raises(SafePathError):
                raise_various_errors(error_type)
    
    def test_exception_chaining(self):
        """Test exception chaining with cause."""
        def inner_function():
            raise ValueError("Original error")
        
        def outer_function():
            try:
                inner_function()
            except ValueError as e:
                raise FilterError("Filter failed due to internal error") from e
        
        with pytest.raises(FilterError) as exc_info:
            outer_function()
        
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "Original error"
    
    def test_exception_with_complex_details(self):
        """Test exceptions with complex details dictionaries."""
        complex_details = {
            "timestamp": "2023-01-01T12:00:00Z",
            "request_id": "req_123456",
            "user_context": {
                "user_id": "user_789",
                "session_id": "session_abc"
            },
            "processing_info": {
                "stage": "semantic_analysis",
                "model": "bert-base-uncased",
                "confidence_threshold": 0.7,
                "detected_patterns": ["pattern1", "pattern2"]
            },
            "system_info": {
                "memory_usage": "1.2GB",
                "processing_time_ms": 150,
                "cpu_usage": "45%"
            }
        }
        
        error = DetectorError(
            "Complex semantic analysis failed",
            detector_name="semantic_detector",
            details=complex_details
        )
        
        assert error.detector_name == "semantic_detector"
        assert error.details == complex_details
        assert error.details["user_context"]["user_id"] == "user_789"
        assert error.details["processing_info"]["model"] == "bert-base-uncased"
        assert len(error.details["processing_info"]["detected_patterns"]) == 2
    
    def test_exception_serialization_compatibility(self):
        """Test that exceptions can be serialized (for logging/monitoring)."""
        error = FilterError(
            "Serialization test error",
            filter_name="test_filter",
            details={"key": "value", "number": 42}
        )
        
        # Test that exception attributes are accessible
        error_dict = {
            "message": error.message,
            "code": error.code,
            "filter_name": error.filter_name,
            "details": error.details
        }
        
        assert error_dict["message"] == "Serialization test error"
        assert error_dict["code"] == "FILTER_ERROR"
        assert error_dict["filter_name"] == "test_filter"
        assert error_dict["details"]["key"] == "value"
        assert error_dict["details"]["number"] == 42


class TestExceptionErrorHandling:
    """Test error handling edge cases."""
    
    def test_exception_with_none_values(self):
        """Test exceptions with None values in optional fields."""
        error = FilterError("Test message", filter_name=None)
        
        assert error.message == "Test message"
        assert error.filter_name is None
        assert error.code == "FILTER_ERROR"
        assert error.details == {}
    
    def test_exception_with_empty_details(self):
        """Test exceptions with empty details dictionary."""
        error = ValidationError("Test message", field="test", details={})
        
        assert error.field == "test"
        assert error.details == {}
        assert len(error.details) == 0
    
    def test_exception_string_representations(self):
        """Test string representations of exceptions."""
        errors = [
            SafePathError("SafePath error"),
            FilterError("Filter error", filter_name="test_filter"),
            DetectorError("Detector error", detector_name="test_detector"),
            ValidationError("Validation error", field="content"),
            ConfigurationError("Config error", config_key="threshold"),
            RateLimitError("Rate limit error", limit=100),
            TimeoutError("Timeout error", timeout_ms=5000),
            IntegrationError("Integration error", framework="fastapi")
        ]
        
        for error in errors:
            # Should not raise exception when converted to string
            str_repr = str(error)
            assert isinstance(str_repr, str)
            assert len(str_repr) > 0
            
            # Should not raise exception when getting repr
            repr_str = repr(error)
            assert isinstance(repr_str, str)
            assert len(repr_str) > 0
    
    def test_exception_equality(self):
        """Test exception equality comparisons."""
        error1 = FilterError("Same message", filter_name="filter1")
        error2 = FilterError("Same message", filter_name="filter1")
        error3 = FilterError("Different message", filter_name="filter1")
        error4 = FilterError("Same message", filter_name="filter2")
        
        # Exceptions are objects, so they should not be equal even with same data
        assert error1 is not error2
        assert error1 != error2  # Different instances
        
        # But they should have the same attributes
        assert error1.message == error2.message
        assert error1.filter_name == error2.filter_name
        assert error1.code == error2.code