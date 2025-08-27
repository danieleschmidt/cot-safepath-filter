"""
Robust error handling and recovery system for SafePath Filter - Generation 2.

Comprehensive error handling, circuit breakers, retry mechanisms,
graceful degradation, and failure recovery strategies.
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import traceback
import json

from .exceptions import FilterError, DetectorError, TimeoutError, ValidationError


logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(str, Enum):
    """Recovery actions for different error types."""
    
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    ESCALATE = "escalate"
    IGNORE = "ignore"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    
    error: Exception
    error_type: str
    severity: ErrorSeverity
    timestamp: datetime
    component: str
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    
    def __post_init__(self):
        if self.stack_trace is None:
            self.stack_trace = traceback.format_exc()


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    stop_on: List[Type[Exception]] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    success_threshold: int = 3  # Successes needed to close circuit


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker implementation for robust error handling."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
        
    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            if (self.last_failure_time and 
                datetime.utcnow() - self.last_failure_time >= timedelta(seconds=self.config.recovery_timeout)):
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        
        # HALF_OPEN state
        return True
    
    def on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def on_failure(self, exception: Exception):
        """Handle failed execution."""
        if isinstance(exception, self.config.expected_exception):
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if (self.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN] and
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker opened due to {self.failure_count} failures")


class RetryMechanism:
    """Intelligent retry mechanism with exponential backoff."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check if exception is in stop list
        if any(isinstance(exception, exc_type) for exc_type in self.config.stop_on):
            return False
        
        # Check if exception is in retry list
        return any(isinstance(exception, exc_type) for exc_type in self.config.retry_on)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        delay = self.config.base_delay * (self.config.backoff_factor ** (attempt - 1))
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            import random
            delay *= random.uniform(0.5, 1.5)
        
        return delay
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                return result
                
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(attempt, e):
                    break
                
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt)
                    logger.warning(f"Retry attempt {attempt} failed: {e}. Retrying in {delay:.2f}s")
                    
                    if asyncio.iscoroutinefunction(func):
                        await asyncio.sleep(delay)
                    else:
                        time.sleep(delay)
        
        # All retries exhausted
        raise last_exception or Exception("All retry attempts failed")


class ErrorClassifier:
    """Classifies errors and determines appropriate recovery actions."""
    
    ERROR_MAPPINGS = {
        # Network and I/O errors - usually transient
        ConnectionError: (ErrorSeverity.MEDIUM, RecoveryAction.RETRY),
        TimeoutError: (ErrorSeverity.MEDIUM, RecoveryAction.RETRY),
        OSError: (ErrorSeverity.MEDIUM, RecoveryAction.RETRY),
        
        # Validation errors - usually permanent
        ValidationError: (ErrorSeverity.HIGH, RecoveryAction.ESCALATE),
        ValueError: (ErrorSeverity.MEDIUM, RecoveryAction.FALLBACK),
        
        # Filter-specific errors
        FilterError: (ErrorSeverity.HIGH, RecoveryAction.CIRCUIT_BREAK),
        DetectorError: (ErrorSeverity.MEDIUM, RecoveryAction.FALLBACK),
        
        # System errors
        MemoryError: (ErrorSeverity.CRITICAL, RecoveryAction.ESCALATE),
        RecursionError: (ErrorSeverity.HIGH, RecoveryAction.ESCALATE),
        
        # Generic fallback
        Exception: (ErrorSeverity.LOW, RecoveryAction.RETRY),
    }
    
    @classmethod
    def classify_error(cls, error: Exception) -> tuple[ErrorSeverity, RecoveryAction]:
        """Classify error and determine recovery action."""
        error_type = type(error)
        
        # Check for exact match first
        if error_type in cls.ERROR_MAPPINGS:
            return cls.ERROR_MAPPINGS[error_type]
        
        # Check for inheritance
        for mapped_type, (severity, action) in cls.ERROR_MAPPINGS.items():
            if isinstance(error, mapped_type):
                return severity, action
        
        # Default classification
        return ErrorSeverity.LOW, RecoveryAction.RETRY


class FallbackManager:
    """Manages fallback strategies when primary operations fail."""
    
    def __init__(self):
        self.fallback_strategies: Dict[str, List[Callable]] = {}
        self.default_responses: Dict[str, Any] = {}
    
    def register_fallback(self, operation: str, fallback_func: Callable, priority: int = 0):
        """Register a fallback strategy for an operation."""
        if operation not in self.fallback_strategies:
            self.fallback_strategies[operation] = []
        
        self.fallback_strategies[operation].append((priority, fallback_func))
        self.fallback_strategies[operation].sort(key=lambda x: x[0])  # Sort by priority
    
    def set_default_response(self, operation: str, response: Any):
        """Set a default response when all fallbacks fail."""
        self.default_responses[operation] = response
    
    async def execute_fallback(self, operation: str, error: Exception, *args, **kwargs) -> Any:
        """Execute fallback strategies for an operation."""
        fallbacks = self.fallback_strategies.get(operation, [])
        
        for priority, fallback_func in fallbacks:
            try:
                logger.info(f"Attempting fallback for {operation} with priority {priority}")
                
                if asyncio.iscoroutinefunction(fallback_func):
                    result = await fallback_func(error, *args, **kwargs)
                else:
                    result = fallback_func(error, *args, **kwargs)
                
                logger.info(f"Fallback succeeded for {operation}")
                return result
                
            except Exception as e:
                logger.warning(f"Fallback failed for {operation}: {e}")
                continue
        
        # All fallbacks failed, return default response
        if operation in self.default_responses:
            logger.info(f"Using default response for {operation}")
            return self.default_responses[operation]
        
        # No fallbacks or defaults available
        raise FilterError(f"All fallback strategies failed for {operation}")


class RobustErrorHandler:
    """Main error handling coordinator."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_mechanisms: Dict[str, RetryMechanism] = {}
        self.fallback_manager = FallbackManager()
        self.error_history: List[ErrorContext] = []
        self.max_history_size = 1000
        
        self._setup_default_fallbacks()
    
    def _setup_default_fallbacks(self):
        """Setup default fallback strategies."""
        # Default filter fallback - return safe content
        def safe_filter_fallback(error: Exception, content: str, *args, **kwargs):
            logger.warning(f"Filter fallback activated due to: {error}")
            return "[CONTENT FILTERED: Safety system unavailable]"
        
        self.fallback_manager.register_fallback("filter", safe_filter_fallback)
        
        # Default detection fallback - assume content is safe but log warning
        def safe_detection_fallback(error: Exception, content: str, *args, **kwargs):
            logger.warning(f"Detection fallback activated due to: {error}")
            from .models import DetectionResult, Severity
            return DetectionResult(
                detector_name="fallback_detector",
                confidence=0.0,
                detected_patterns=[],
                severity=Severity.LOW,
                is_harmful=False,
                reasoning=f"Fallback mode due to error: {str(error)[:100]}"
            )
        
        self.fallback_manager.register_fallback("detection", safe_detection_fallback)
    
    def register_circuit_breaker(self, component: str, config: CircuitBreakerConfig):
        """Register circuit breaker for a component."""
        self.circuit_breakers[component] = CircuitBreaker(config)
    
    def register_retry_mechanism(self, component: str, config: RetryConfig):
        """Register retry mechanism for a component."""
        self.retry_mechanisms[component] = RetryMechanism(config)
    
    def get_circuit_breaker(self, component: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for component."""
        return self.circuit_breakers.get(component)
    
    def get_retry_mechanism(self, component: str) -> Optional[RetryMechanism]:
        """Get retry mechanism for component."""
        return self.retry_mechanisms.get(component)
    
    def record_error(self, error: Exception, component: str, operation: str, metadata: Dict[str, Any] = None):
        """Record error for analysis and monitoring."""
        severity, recovery_action = ErrorClassifier.classify_error(error)
        
        error_context = ErrorContext(
            error=error,
            error_type=type(error).__name__,
            severity=severity,
            timestamp=datetime.utcnow(),
            component=component,
            operation=operation,
            metadata=metadata or {}
        )
        
        self.error_history.append(error_context)
        
        # Maintain history size
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size // 2:]
        
        logger.error(
            f"Error recorded: {error_context.error_type} in {component}.{operation} "
            f"(severity: {severity.value}, action: {recovery_action.value})"
        )
        
        return error_context
    
    async def handle_error(
        self, 
        error: Exception, 
        component: str, 
        operation: str, 
        fallback_args: tuple = (), 
        fallback_kwargs: Dict[str, Any] = None
    ) -> Any:
        """Handle error with appropriate recovery strategy."""
        fallback_kwargs = fallback_kwargs or {}
        
        # Record error
        error_context = self.record_error(error, component, operation)
        
        # Determine recovery action
        severity, recovery_action = ErrorClassifier.classify_error(error)
        
        # Execute recovery action
        if recovery_action == RecoveryAction.CIRCUIT_BREAK:
            circuit_breaker = self.get_circuit_breaker(component)
            if circuit_breaker:
                circuit_breaker.on_failure(error)
        
        if recovery_action == RecoveryAction.FALLBACK:
            try:
                return await self.fallback_manager.execute_fallback(
                    operation, error, *fallback_args, **fallback_kwargs
                )
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
                raise error  # Re-raise original error
        
        if recovery_action == RecoveryAction.ESCALATE:
            logger.critical(f"Critical error escalated: {error}")
            # Could implement notification system here
            raise error
        
        # For other actions, re-raise the error
        raise error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {}
        
        # Count errors by type
        error_counts = {}
        severity_counts = {}
        component_counts = {}
        
        for error_context in self.error_history:
            error_counts[error_context.error_type] = error_counts.get(error_context.error_type, 0) + 1
            severity_counts[error_context.severity.value] = severity_counts.get(error_context.severity.value, 0) + 1
            component_counts[error_context.component] = component_counts.get(error_context.component, 0) + 1
        
        # Recent error rate (last hour)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_errors = [e for e in self.error_history if e.timestamp >= one_hour_ago]
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_counts,
            "severity_distribution": severity_counts,
            "component_distribution": component_counts,
            "recent_error_rate": len(recent_errors),
            "circuit_breaker_states": {
                component: breaker.state.value 
                for component, breaker in self.circuit_breakers.items()
            }
        }


# Global error handler instance
_global_error_handler: Optional[RobustErrorHandler] = None


def get_global_error_handler() -> RobustErrorHandler:
    """Get or create global error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = RobustErrorHandler()
    return _global_error_handler


def robust_operation(
    component: str,
    operation: str = None,
    enable_retry: bool = True,
    enable_circuit_breaker: bool = True,
    enable_fallback: bool = True
):
    """Decorator for robust error handling."""
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = get_global_error_handler()
            
            # Check circuit breaker
            if enable_circuit_breaker:
                circuit_breaker = error_handler.get_circuit_breaker(component)
                if circuit_breaker and not circuit_breaker.can_execute():
                    raise FilterError(f"Circuit breaker open for {component}")
            
            # Execute with retry if configured
            if enable_retry:
                retry_mechanism = error_handler.get_retry_mechanism(component)
                if retry_mechanism:
                    try:
                        result = await retry_mechanism.execute_with_retry(func, *args, **kwargs)
                        if circuit_breaker:
                            circuit_breaker.on_success()
                        return result
                    except Exception as e:
                        if enable_fallback:
                            return await error_handler.handle_error(
                                e, component, op_name, args, kwargs
                            )
                        raise
            
            # Execute normally
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                if circuit_breaker:
                    circuit_breaker.on_success()
                return result
                
            except Exception as e:
                if circuit_breaker:
                    circuit_breaker.on_failure(e)
                
                if enable_fallback:
                    return await error_handler.handle_error(
                        e, component, op_name, args, kwargs
                    )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return async_wrapper(*args, **kwargs)
            
            error_handler = get_global_error_handler()
            
            # Check circuit breaker
            if enable_circuit_breaker:
                circuit_breaker = error_handler.get_circuit_breaker(component)
                if circuit_breaker and not circuit_breaker.can_execute():
                    raise FilterError(f"Circuit breaker open for {component}")
            
            try:
                result = func(*args, **kwargs)
                if circuit_breaker:
                    circuit_breaker.on_success()
                return result
                
            except Exception as e:
                if circuit_breaker:
                    circuit_breaker.on_failure(e)
                
                if enable_fallback:
                    # For sync functions, we need to handle async fallbacks
                    loop = None
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    return loop.run_until_complete(
                        error_handler.handle_error(e, component, op_name, args, kwargs)
                    )
                raise
        
        return sync_wrapper
    return decorator


# Convenience functions for common configurations
def configure_robust_filtering():
    """Configure robust error handling for filtering operations."""
    error_handler = get_global_error_handler()
    
    # Configure circuit breaker for core filtering
    filter_circuit_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        expected_exception=FilterError
    )
    error_handler.register_circuit_breaker("filter", filter_circuit_config)
    
    # Configure retry for transient errors
    filter_retry_config = RetryConfig(
        max_attempts=2,
        base_delay=0.5,
        retry_on=[ConnectionError, TimeoutError],
        stop_on=[ValidationError]
    )
    error_handler.register_retry_mechanism("filter", filter_retry_config)
    
    # Configure circuit breaker for detectors
    detector_circuit_config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception=DetectorError
    )
    error_handler.register_circuit_breaker("detector", detector_circuit_config)
    
    logger.info("Robust error handling configured for SafePath Filter")
    
    return error_handler