"""
Error Boundary System - Generation 2 Comprehensive Error Handling.

Manages error propagation, containment, and graceful degradation across the pipeline.
"""

import functools
import traceback
import logging
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, Type, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import sys
import signal
import gc
import weakref

from .models import FilterResult, FilterRequest, SafetyScore, Severity
from .exceptions import FilterError, TimeoutError, DetectorError, ValidationError


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for error classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorBoundaryState(Enum):
    """States of an error boundary."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CIRCUIT_OPEN = "circuit_open"
    RECOVERING = "recovering"


@dataclass
class ErrorContext:
    """Context information about an error."""
    error: Exception
    error_type: str
    component: str
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    traceback_str: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.traceback_str is None:
            self.traceback_str = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_message": str(self.error),
            "error_type": self.error_type,
            "component": self.component,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "traceback": self.traceback_str,
            "metadata": self.metadata
        }


@dataclass
class FallbackResult:
    """Result from a fallback operation."""
    success: bool
    result: Optional[Any] = None
    error: Optional[Exception] = None
    fallback_type: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorClassifier:
    """Classifies errors to determine appropriate handling strategies."""
    
    def __init__(self):
        self.error_severity_map = {
            ValidationError: ErrorSeverity.MEDIUM,
            FilterError: ErrorSeverity.HIGH,
            DetectorError: ErrorSeverity.HIGH,
            TimeoutError: ErrorSeverity.MEDIUM,
            MemoryError: ErrorSeverity.CRITICAL,
            KeyboardInterrupt: ErrorSeverity.CRITICAL,
            SystemExit: ErrorSeverity.CRITICAL,
            OSError: ErrorSeverity.HIGH,
            ConnectionError: ErrorSeverity.MEDIUM,
            ValueError: ErrorSeverity.LOW,
            TypeError: ErrorSeverity.MEDIUM,
            AttributeError: ErrorSeverity.MEDIUM,
            ImportError: ErrorSeverity.HIGH,
            RuntimeError: ErrorSeverity.HIGH,
        }
        
        # Keywords in error messages that indicate severity
        self.severity_keywords = {
            ErrorSeverity.CRITICAL: [
                'memory', 'system', 'segmentation', 'corruption', 
                'security', 'unauthorized', 'permission denied'
            ],
            ErrorSeverity.HIGH: [
                'connection', 'network', 'timeout', 'resource', 
                'unavailable', 'failed to load', 'cannot connect'
            ],
            ErrorSeverity.MEDIUM: [
                'validation', 'format', 'parse', 'decode', 
                'invalid', 'missing', 'configuration'
            ],
            ErrorSeverity.LOW: [
                'warning', 'deprecated', 'fallback', 'retry'
            ]
        }
    
    def classify_error(self, error: Exception, component: str = "unknown") -> ErrorContext:
        """Classify an error and create context."""
        error_type = type(error).__name__
        
        # Base severity from error type
        severity = self.error_severity_map.get(type(error), ErrorSeverity.MEDIUM)
        
        # Adjust severity based on error message
        error_message = str(error).lower()
        for sev_level, keywords in self.severity_keywords.items():
            if any(keyword in error_message for keyword in keywords):
                # Take the more severe classification
                if self._severity_value(sev_level) > self._severity_value(severity):
                    severity = sev_level
                break
        
        return ErrorContext(
            error=error,
            error_type=error_type,
            component=component,
            severity=severity,
            metadata={
                "error_message_length": len(str(error)),
                "has_cause": error.__cause__ is not None,
                "has_context": error.__context__ is not None,
            }
        )
    
    def _severity_value(self, severity: ErrorSeverity) -> int:
        """Get numeric value for severity comparison."""
        return {
            ErrorSeverity.LOW: 1,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 4
        }[severity]


class FallbackProvider:
    """Provides fallback mechanisms for failed operations."""
    
    def __init__(self):
        self.fallback_cache: Dict[str, Any] = {}
        self.default_results = {}
        
    def register_default_result(self, component: str, result_factory: Callable[[], Any]) -> None:
        """Register a default result factory for a component."""
        self.default_results[component] = result_factory
    
    def get_fallback_filter_result(self, request: FilterRequest, 
                                  error_context: ErrorContext) -> FallbackResult:
        """Get a fallback filter result when normal processing fails."""
        try:
            # Create a safe fallback result
            fallback_content = request.content
            
            # Apply basic safety measures based on error severity
            if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                # More conservative approach for serious errors
                safety_score = SafetyScore(
                    overall_score=0.3,  # Lower score indicates more caution
                    confidence=0.1,     # Low confidence
                    is_safe=False,      # Err on the side of caution
                    detected_patterns=["error_fallback"],
                    severity=Severity.HIGH
                )
                was_filtered = True
                filter_reasons = [f"error_fallback:{error_context.error_type}"]
            else:
                # Less severe errors can have more permissive fallbacks
                safety_score = SafetyScore(
                    overall_score=0.7,
                    confidence=0.3,
                    is_safe=True,
                    detected_patterns=["error_fallback"],
                    severity=Severity.LOW
                )
                was_filtered = False
                filter_reasons = [f"error_fallback_safe:{error_context.error_type}"]
            
            result = FilterResult(
                filtered_content=fallback_content,
                safety_score=safety_score,
                was_filtered=was_filtered,
                filter_reasons=filter_reasons,
                original_content=request.content if was_filtered else None,
                processing_time_ms=0,
                request_id=request.request_id
            )
            
            return FallbackResult(
                success=True,
                result=result,
                fallback_type="safe_default",
                metadata={
                    "error_severity": error_context.severity.value,
                    "original_error": str(error_context.error)
                }
            )
            
        except Exception as e:
            logger.error(f"Fallback creation failed: {e}")
            return FallbackResult(
                success=False,
                error=e,
                fallback_type="failed_fallback"
            )
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get a cached result as fallback."""
        return self.fallback_cache.get(cache_key)
    
    def cache_successful_result(self, cache_key: str, result: Any) -> None:
        """Cache a successful result for future fallback use."""
        self.fallback_cache[cache_key] = result
        
        # Limit cache size
        if len(self.fallback_cache) > 1000:
            # Remove oldest entries (simple strategy)
            oldest_keys = list(self.fallback_cache.keys())[:100]
            for key in oldest_keys:
                del self.fallback_cache[key]


class ErrorBoundary:
    """Error boundary that catches and handles errors in a component."""
    
    def __init__(self, component_name: str, max_errors: int = 5, 
                 time_window: int = 300, circuit_open_time: int = 60):
        self.component_name = component_name
        self.max_errors = max_errors
        self.time_window = time_window  # seconds
        self.circuit_open_time = circuit_open_time  # seconds
        
        self.error_history: List[ErrorContext] = []
        self.state = ErrorBoundaryState.HEALTHY
        self.circuit_opened_at: Optional[datetime] = None
        
        self.classifier = ErrorClassifier()
        self.fallback_provider = FallbackProvider()
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Error boundary created for {component_name}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with error boundary."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    @contextmanager
    def protect(self, operation_name: str = "operation"):
        """Context manager for protecting code blocks."""
        try:
            yield self
        except Exception as e:
            error_context = self.classifier.classify_error(e, self.component_name)
            self._handle_error(error_context)
            raise
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function within the error boundary."""
        with self._lock:
            # Check circuit breaker state
            if self.state == ErrorBoundaryState.CIRCUIT_OPEN:
                if self._should_try_recovery():
                    self.state = ErrorBoundaryState.RECOVERING
                    logger.info(f"Error boundary for {self.component_name} attempting recovery")
                else:
                    raise FilterError(f"Circuit breaker open for {self.component_name}")
        
        try:
            result = func(*args, **kwargs)
            
            # On success during recovery, check if we can close the circuit
            if self.state == ErrorBoundaryState.RECOVERING:
                with self._lock:
                    self.state = ErrorBoundaryState.HEALTHY
                    self.circuit_opened_at = None
                    logger.info(f"Error boundary for {self.component_name} recovered")
            
            # Cache successful result for potential fallback use
            if hasattr(result, '__dict__'):
                cache_key = f"{self.component_name}:{hash(str(args))}"
                self.fallback_provider.cache_successful_result(cache_key, result)
            
            return result
            
        except Exception as e:
            error_context = self.classifier.classify_error(e, self.component_name)
            return self._handle_error(error_context, func, *args, **kwargs)
    
    def _handle_error(self, error_context: ErrorContext, 
                     failed_func: Optional[Callable] = None,
                     *args, **kwargs) -> Any:
        """Handle an error that occurred within the boundary."""
        with self._lock:
            # Record the error
            self.error_history.append(error_context)
            
            # Clean old errors outside time window
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.time_window)
            self.error_history = [
                err for err in self.error_history 
                if err.timestamp > cutoff_time
            ]
            
            # Update boundary state based on error frequency and severity
            self._update_boundary_state(error_context)
            
            # Log the error
            self._log_error(error_context)
        
        # Attempt fallback if possible
        if failed_func and args and isinstance(args[0], FilterRequest):
            fallback = self.fallback_provider.get_fallback_filter_result(args[0], error_context)
            if fallback.success:
                logger.info(f"Using fallback result for {self.component_name}")
                return fallback.result
        
        # If no fallback available, re-raise the error
        raise error_context.error
    
    def _update_boundary_state(self, error_context: ErrorContext) -> None:
        """Update boundary state based on error patterns."""
        recent_errors = len(self.error_history)
        
        # Count critical and high severity errors
        critical_errors = sum(
            1 for err in self.error_history 
            if err.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]
        )
        
        # State transition logic
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.state = ErrorBoundaryState.CIRCUIT_OPEN
            self.circuit_opened_at = datetime.utcnow()
        elif recent_errors >= self.max_errors or critical_errors >= 3:
            self.state = ErrorBoundaryState.CIRCUIT_OPEN
            self.circuit_opened_at = datetime.utcnow()
        elif recent_errors >= self.max_errors // 2:
            self.state = ErrorBoundaryState.FAILING
        elif recent_errors > 1:
            self.state = ErrorBoundaryState.DEGRADED
        else:
            # Single error doesn't change state unless critical
            pass
    
    def _should_try_recovery(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self.circuit_opened_at is None:
            return True
        
        time_since_open = (datetime.utcnow() - self.circuit_opened_at).total_seconds()
        return time_since_open >= self.circuit_open_time
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with appropriate level based on severity."""
        error_dict = error_context.to_dict()
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error in {self.component_name}: {error_dict}")
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error in {self.component_name}: {error_dict}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error in {self.component_name}: {error_dict}")
        else:
            logger.info(f"Low severity error in {self.component_name}: {error_dict}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the boundary."""
        with self._lock:
            recent_errors = len(self.error_history)
            critical_errors = sum(
                1 for err in self.error_history 
                if err.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]
            )
            
            return {
                "component": self.component_name,
                "state": self.state.value,
                "recent_errors": recent_errors,
                "critical_errors": critical_errors,
                "circuit_opened_at": self.circuit_opened_at.isoformat() if self.circuit_opened_at else None,
                "error_history_size": len(self.error_history)
            }
    
    def reset(self) -> None:
        """Reset the error boundary to healthy state."""
        with self._lock:
            self.error_history.clear()
            self.state = ErrorBoundaryState.HEALTHY
            self.circuit_opened_at = None
            logger.info(f"Error boundary for {self.component_name} reset")


class GlobalErrorHandler:
    """Global error handler for the entire system."""
    
    def __init__(self):
        self.error_boundaries: Dict[str, ErrorBoundary] = {}
        self.global_error_count = 0
        self.system_state = ErrorBoundaryState.HEALTHY
        self.last_critical_error: Optional[datetime] = None
        
        # Global error tracking
        self.error_statistics = {
            "total_errors": 0,
            "errors_by_severity": {sev.value: 0 for sev in ErrorSeverity},
            "errors_by_component": {},
        }
        
        # Install global exception handler
        self._install_global_handlers()
        
    def register_component(self, component_name: str, **boundary_kwargs) -> ErrorBoundary:
        """Register a component with its own error boundary."""
        if component_name not in self.error_boundaries:
            self.error_boundaries[component_name] = ErrorBoundary(component_name, **boundary_kwargs)
        return self.error_boundaries[component_name]
    
    def get_boundary(self, component_name: str) -> ErrorBoundary:
        """Get error boundary for a component."""
        if component_name not in self.error_boundaries:
            return self.register_component(component_name)
        return self.error_boundaries[component_name]
    
    def handle_global_error(self, error: Exception, component: str = "system") -> None:
        """Handle a global system error."""
        self.global_error_count += 1
        
        classifier = ErrorClassifier()
        error_context = classifier.classify_error(error, component)
        
        # Update statistics
        self.error_statistics["total_errors"] += 1
        self.error_statistics["errors_by_severity"][error_context.severity.value] += 1
        if component not in self.error_statistics["errors_by_component"]:
            self.error_statistics["errors_by_component"][component] = 0
        self.error_statistics["errors_by_component"][component] += 1
        
        # Update system state
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.system_state = ErrorBoundaryState.CIRCUIT_OPEN
            self.last_critical_error = datetime.utcnow()
            logger.critical(f"Critical system error: {error}")
        elif self.global_error_count > 10:  # Threshold for system degradation
            self.system_state = ErrorBoundaryState.DEGRADED
            logger.warning("System state degraded due to error frequency")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        component_health = {}
        for name, boundary in self.error_boundaries.items():
            component_health[name] = boundary.get_health_status()
        
        return {
            "system_state": self.system_state.value,
            "global_error_count": self.global_error_count,
            "last_critical_error": self.last_critical_error.isoformat() if self.last_critical_error else None,
            "error_statistics": self.error_statistics,
            "component_health": component_health,
            "total_components": len(self.error_boundaries)
        }
    
    def _install_global_handlers(self) -> None:
        """Install global exception handlers."""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Allow KeyboardInterrupt to work normally
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            self.handle_global_error(exc_value, "global")
            logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        sys.excepthook = handle_exception
        
        # Install signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self._cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def _cleanup(self) -> None:
        """Cleanup global error handler resources."""
        logger.info("Cleaning up global error handler")
        for boundary in self.error_boundaries.values():
            boundary.fallback_provider.fallback_cache.clear()
        self.error_boundaries.clear()
        gc.collect()


# Global instance
global_error_handler = GlobalErrorHandler()