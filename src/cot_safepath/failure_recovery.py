"""
Failure Recovery System - Generation 1 Auto-Healing Mechanisms.

Automated recovery strategies for different types of pipeline failures.
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from .models import FilterResult, FilterRequest, SafetyScore
from .exceptions import FilterError, TimeoutError, DetectorError, ValidationError
from .self_healing_core import HealthStatus, HealingAction, HealingEvent
from .pipeline_diagnostics import DiagnosticFinding, DiagnosticSeverity, PerformanceIssue


logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Types of recovery strategies available."""
    IMMEDIATE_RETRY = "immediate_retry"
    DELAYED_RETRY = "delayed_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAILOVER = "failover"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    COMPONENT_ISOLATION = "component_isolation"
    CACHE_FALLBACK = "cache_fallback"


class FailureType(Enum):
    """Types of failures that can occur in the pipeline."""
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    DETECTOR_ERROR = "detector_error"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class FailureContext:
    """Context information about a failure."""
    failure_type: FailureType
    component: str
    error: Exception
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    previous_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "component": self.component,
            "error_message": str(self.error),
            "error_type": type(self.error).__name__,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "previous_attempts": self.previous_attempts,
            "metadata": self.metadata
        }


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    failure_context: FailureContext
    strategy: RecoveryStrategy
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = False
    duration_ms: float = 0.0
    result: Optional[Any] = None
    new_error: Optional[Exception] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_context": self.failure_context.to_dict(),
            "strategy": self.strategy.value,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "duration_ms": self.duration_ms,
            "new_error": str(self.new_error) if self.new_error else None
        }


class CircuitBreaker:
    """Circuit breaker pattern implementation for failure isolation."""
    
    def __init__(self, component_name: str, failure_threshold: int = 5, 
                 recovery_timeout: int = 30, success_threshold: int = 3):
        self.component_name = component_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
        
        logger.info(f"Circuit breaker initialized for {component_name}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                logger.info(f"Circuit breaker for {self.component_name} moving to half-open")
            else:
                raise FilterError(f"Circuit breaker open for {self.component_name}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        self.failure_count = 0
        
        if self.state == "half-open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "closed"
                self.success_count = 0
                logger.info(f"Circuit breaker for {self.component_name} reset to closed")
    
    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker for {self.component_name} opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "component": self.component_name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class FailureClassifier:
    """Classifies failures to determine appropriate recovery strategies."""
    
    def __init__(self):
        self.classification_rules = {
            TimeoutError: FailureType.TIMEOUT,
            MemoryError: FailureType.MEMORY_ERROR,
            ValidationError: FailureType.VALIDATION_ERROR,
            DetectorError: FailureType.DETECTOR_ERROR,
            ConnectionError: FailureType.NETWORK_ERROR,
            OSError: FailureType.RESOURCE_EXHAUSTION,
        }
    
    def classify_failure(self, error: Exception, component: str, metadata: Dict[str, Any] = None) -> FailureContext:
        """Classify a failure and create context."""
        failure_type = FailureType.UNKNOWN_ERROR
        
        # Check direct type matches
        for error_type, failure_type_enum in self.classification_rules.items():
            if isinstance(error, error_type):
                failure_type = failure_type_enum
                break
        
        # Check error message patterns
        error_message = str(error).lower()
        if "timeout" in error_message:
            failure_type = FailureType.TIMEOUT
        elif "memory" in error_message:
            failure_type = FailureType.MEMORY_ERROR
        elif "connection" in error_message or "network" in error_message:
            failure_type = FailureType.NETWORK_ERROR
        
        return FailureContext(
            failure_type=failure_type,
            component=component,
            error=error,
            metadata=metadata or {}
        )
    
    def suggest_recovery_strategies(self, failure_context: FailureContext) -> List[RecoveryStrategy]:
        """Suggest appropriate recovery strategies based on failure context."""
        strategies = []
        
        # Base strategy selection based on failure type
        if failure_context.failure_type == FailureType.TIMEOUT:
            strategies.extend([
                RecoveryStrategy.IMMEDIATE_RETRY,
                RecoveryStrategy.DELAYED_RETRY,
                RecoveryStrategy.CIRCUIT_BREAKER
            ])
        elif failure_context.failure_type == FailureType.MEMORY_ERROR:
            strategies.extend([
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.COMPONENT_ISOLATION,
                RecoveryStrategy.CACHE_FALLBACK
            ])
        elif failure_context.failure_type == FailureType.NETWORK_ERROR:
            strategies.extend([
                RecoveryStrategy.EXPONENTIAL_BACKOFF,
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.CIRCUIT_BREAKER
            ])
        elif failure_context.failure_type == FailureType.VALIDATION_ERROR:
            strategies.extend([
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.CACHE_FALLBACK
            ])
        elif failure_context.failure_type == FailureType.DETECTOR_ERROR:
            strategies.extend([
                RecoveryStrategy.IMMEDIATE_RETRY,
                RecoveryStrategy.COMPONENT_ISOLATION,
                RecoveryStrategy.FAILOVER
            ])
        else:
            # Default strategies for unknown errors
            strategies.extend([
                RecoveryStrategy.IMMEDIATE_RETRY,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ])
        
        # Modify strategies based on previous attempts
        if failure_context.previous_attempts > 2:
            # Remove immediate retry after multiple attempts
            strategies = [s for s in strategies if s != RecoveryStrategy.IMMEDIATE_RETRY]
            strategies.insert(0, RecoveryStrategy.CIRCUIT_BREAKER)
        
        return strategies


class RecoveryExecutor:
    """Executes recovery strategies for failed operations."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_cache: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def execute_recovery(self, failure_context: FailureContext, 
                        strategies: List[RecoveryStrategy],
                        operation: Callable, *args, **kwargs) -> RecoveryAttempt:
        """Execute recovery strategies in order until one succeeds."""
        
        for strategy in strategies:
            attempt = self._execute_strategy(strategy, failure_context, operation, *args, **kwargs)
            if attempt.success:
                return attempt
        
        # All strategies failed
        return RecoveryAttempt(
            failure_context=failure_context,
            strategy=strategies[-1] if strategies else RecoveryStrategy.IMMEDIATE_RETRY,
            success=False,
            new_error=Exception("All recovery strategies failed")
        )
    
    def _execute_strategy(self, strategy: RecoveryStrategy, failure_context: FailureContext,
                         operation: Callable, *args, **kwargs) -> RecoveryAttempt:
        """Execute a specific recovery strategy."""
        start_time = time.time()
        attempt = RecoveryAttempt(failure_context=failure_context, strategy=strategy)
        
        try:
            if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
                result = operation(*args, **kwargs)
                attempt.success = True
                attempt.result = result
                
            elif strategy == RecoveryStrategy.DELAYED_RETRY:
                time.sleep(1.0)  # Wait 1 second
                result = operation(*args, **kwargs)
                attempt.success = True
                attempt.result = result
                
            elif strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                wait_time = min(2 ** failure_context.previous_attempts, 30)  # Max 30 seconds
                time.sleep(wait_time)
                result = operation(*args, **kwargs)
                attempt.success = True
                attempt.result = result
                
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                circuit_breaker = self._get_circuit_breaker(failure_context.component)
                result = circuit_breaker.call(operation, *args, **kwargs)
                attempt.success = True
                attempt.result = result
                
            elif strategy == RecoveryStrategy.CACHE_FALLBACK:
                # Try to get result from cache
                cache_key = f"{failure_context.component}:{hash(str(args))}"
                if cache_key in self.recovery_cache:
                    attempt.success = True
                    attempt.result = self.recovery_cache[cache_key]
                else:
                    raise Exception("No cached result available")
                    
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                # Return a degraded but safe result
                attempt.success = True
                attempt.result = self._create_degraded_result(failure_context, *args, **kwargs)
                
            elif strategy == RecoveryStrategy.COMPONENT_ISOLATION:
                # Skip the failing component and continue with partial processing
                attempt.success = True
                attempt.result = self._create_isolated_result(failure_context, *args, **kwargs)
                
            elif strategy == RecoveryStrategy.FAILOVER:
                # This would implement failover to backup components
                # For now, just retry the operation
                result = operation(*args, **kwargs)
                attempt.success = True
                attempt.result = result
                
        except Exception as e:
            attempt.new_error = e
            logger.warning(f"Recovery strategy {strategy} failed: {e}")
        
        attempt.duration_ms = (time.time() - start_time) * 1000
        return attempt
    
    def _get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get or create circuit breaker for component."""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker(component)
        return self.circuit_breakers[component]
    
    def _create_degraded_result(self, failure_context: FailureContext, *args, **kwargs) -> Any:
        """Create a degraded but safe result when normal processing fails."""
        # This would create a safe fallback result
        # For filter operations, return the original content with a warning
        if args and hasattr(args[0], 'content'):
            request = args[0]
            return FilterResult(
                filtered_content=request.content,
                safety_score=SafetyScore(
                    overall_score=0.5,  # Neutral score
                    confidence=0.1,     # Low confidence
                    is_safe=True,       # Assume safe in degraded mode
                    detected_patterns=["degraded_mode"],
                    severity=None
                ),
                was_filtered=False,
                filter_reasons=["degraded_mode_active"],
                original_content=None,
                processing_time_ms=0,
                request_id=getattr(request, 'request_id', None)
            )
        
        return None
    
    def _create_isolated_result(self, failure_context: FailureContext, *args, **kwargs) -> Any:
        """Create result with failing component isolated."""
        # Similar to degraded result but specifically marks component as isolated
        if args and hasattr(args[0], 'content'):
            request = args[0]
            return FilterResult(
                filtered_content=request.content,
                safety_score=SafetyScore(
                    overall_score=0.7,  # Slightly higher score
                    confidence=0.3,     # Low-medium confidence
                    is_safe=True,
                    detected_patterns=[f"component_isolated:{failure_context.component}"],
                    severity=None
                ),
                was_filtered=False,
                filter_reasons=[f"component_isolated:{failure_context.component}"],
                original_content=None,
                processing_time_ms=0,
                request_id=getattr(request, 'request_id', None)
            )
        
        return None
    
    def cache_successful_result(self, component: str, args: Tuple, result: Any) -> None:
        """Cache a successful result for future fallback use."""
        cache_key = f"{component}:{hash(str(args))}"
        self.recovery_cache[cache_key] = result
        
        # Limit cache size
        if len(self.recovery_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.recovery_cache.keys())[:100]
            for key in oldest_keys:
                del self.recovery_cache[key]
    
    def get_circuit_breaker_states(self) -> Dict[str, Dict[str, Any]]:
        """Get the state of all circuit breakers."""
        return {name: cb.get_state() for name, cb in self.circuit_breakers.items()}
    
    def cleanup(self) -> None:
        """Cleanup recovery resources."""
        self.executor.shutdown(wait=True)
        self.circuit_breakers.clear()
        self.recovery_cache.clear()


class FailureRecoveryManager:
    """Central manager for failure recovery operations."""
    
    def __init__(self):
        self.classifier = FailureClassifier()
        self.executor = RecoveryExecutor()
        self.recovery_history: List[RecoveryAttempt] = []
        self.max_history = 1000
        
        # Recovery statistics
        self.recovery_stats = defaultdict(int)
        
        logger.info("Failure recovery manager initialized")
    
    def handle_failure(self, error: Exception, component: str, operation: Callable,
                      request_id: Optional[str] = None, metadata: Dict[str, Any] = None,
                      *args, **kwargs) -> Any:
        """Handle a failure by attempting recovery."""
        
        # Classify the failure
        failure_context = self.classifier.classify_failure(error, component, metadata)
        failure_context.request_id = request_id
        
        # Get suggested recovery strategies
        strategies = self.classifier.suggest_recovery_strategies(failure_context)
        
        # Attempt recovery
        recovery_attempt = self.executor.execute_recovery(
            failure_context, strategies, operation, *args, **kwargs
        )
        
        # Record the attempt
        self.recovery_history.append(recovery_attempt)
        if len(self.recovery_history) > self.max_history:
            self.recovery_history = self.recovery_history[-self.max_history//2:]
        
        # Update statistics
        self.recovery_stats[f"attempts_{recovery_attempt.strategy.value}"] += 1
        if recovery_attempt.success:
            self.recovery_stats[f"successes_{recovery_attempt.strategy.value}"] += 1
            
            # Cache successful result
            self.executor.cache_successful_result(component, args, recovery_attempt.result)
        else:
            self.recovery_stats[f"failures_{recovery_attempt.strategy.value}"] += 1
        
        # Log the outcome
        if recovery_attempt.success:
            logger.info(f"Recovery successful for {component} using {recovery_attempt.strategy}")
        else:
            logger.error(f"Recovery failed for {component}, all strategies exhausted")
        
        if recovery_attempt.success:
            return recovery_attempt.result
        else:
            # Re-raise the last error or original error
            raise recovery_attempt.new_error or error
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        total_attempts = sum(v for k, v in self.recovery_stats.items() if k.startswith("attempts_"))
        total_successes = sum(v for k, v in self.recovery_stats.items() if k.startswith("successes_"))
        
        success_rate = total_successes / max(total_attempts, 1)
        
        return {
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "overall_success_rate": success_rate,
            "strategy_stats": dict(self.recovery_stats),
            "circuit_breakers": self.executor.get_circuit_breaker_states(),
            "cache_size": len(self.executor.recovery_cache),
            "recent_failures": [
                attempt.to_dict() for attempt in self.recovery_history[-10:]
                if not attempt.success
            ]
        }
    
    def cleanup(self) -> None:
        """Cleanup recovery resources."""
        self.executor.cleanup()
        self.recovery_history.clear()
        self.recovery_stats.clear()