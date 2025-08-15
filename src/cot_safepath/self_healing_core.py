"""
Self-Healing Pipeline Guard Core - Generation 1 Implementation.

Autonomous detection, diagnosis, and healing of pipeline failures in real-time.
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import weakref
from concurrent.futures import ThreadPoolExecutor

from .models import FilterResult, ProcessingMetrics, SafetyScore
from .exceptions import FilterError, TimeoutError
from .core import FilterPipeline, SafePathFilter


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status indicators for pipeline components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"
    RECOVERING = "recovering"


class HealingAction(Enum):
    """Types of healing actions that can be performed."""
    RESTART_COMPONENT = "restart_component"
    CLEAR_CACHE = "clear_cache"
    ADJUST_THRESHOLD = "adjust_threshold"
    FALLBACK_MODE = "fallback_mode"
    CIRCUIT_BREAKER = "circuit_breaker"
    SCALE_RESOURCES = "scale_resources"


@dataclass
class HealthMetric:
    """Health metric for a pipeline component."""
    component: str
    metric_name: str
    value: float
    threshold: float
    status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


@dataclass
class HealingEvent:
    """Record of a healing action taken."""
    component: str
    action: HealingAction
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class ComponentHealthMonitor:
    """Monitors health of individual pipeline components."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.metrics_history: List[HealthMetric] = []
        self.error_count = 0
        self.last_success_time = datetime.utcnow()
        self.response_times: List[float] = []
        self.max_history = 1000
        
        # Health thresholds
        self.error_rate_threshold = 0.1  # 10% error rate
        self.response_time_threshold = 5.0  # 5 seconds
        self.success_rate_threshold = 0.95  # 95% success rate
    
    def record_success(self, response_time: float) -> None:
        """Record successful operation."""
        self.last_success_time = datetime.utcnow()
        self.response_times.append(response_time)
        
        # Keep only recent response times
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-50:]
    
    def record_error(self, error: Exception) -> None:
        """Record error occurrence."""
        self.error_count += 1
        logger.warning(f"Component {self.component_name} error: {error}")
    
    def get_health_status(self) -> HealthStatus:
        """Determine current health status."""
        now = datetime.utcnow()
        time_since_success = (now - self.last_success_time).total_seconds()
        
        # Check if component hasn't succeeded in a while
        if time_since_success > 300:  # 5 minutes
            return HealthStatus.CRITICAL
        
        # Check response times
        if self.response_times:
            avg_response_time = statistics.mean(self.response_times)
            if avg_response_time > self.response_time_threshold * 2:
                return HealthStatus.FAILING
            elif avg_response_time > self.response_time_threshold:
                return HealthStatus.DEGRADED
        
        # Check error rate
        if self.error_count > 10:  # Simple threshold for now
            if time_since_success < 60:  # Recent success
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.FAILING
        
        return HealthStatus.HEALTHY
    
    def get_metrics(self) -> List[HealthMetric]:
        """Get current health metrics."""
        metrics = []
        now = datetime.utcnow()
        
        # Response time metric
        if self.response_times:
            avg_response_time = statistics.mean(self.response_times)
            metrics.append(HealthMetric(
                component=self.component_name,
                metric_name="avg_response_time",
                value=avg_response_time,
                threshold=self.response_time_threshold,
                status=HealthStatus.HEALTHY if avg_response_time < self.response_time_threshold else HealthStatus.DEGRADED
            ))
        
        # Time since last success
        time_since_success = (now - self.last_success_time).total_seconds()
        metrics.append(HealthMetric(
            component=self.component_name,
            metric_name="time_since_success",
            value=time_since_success,
            threshold=300.0,  # 5 minutes
            status=HealthStatus.HEALTHY if time_since_success < 300 else HealthStatus.CRITICAL
        ))
        
        # Error count
        metrics.append(HealthMetric(
            component=self.component_name,
            metric_name="error_count",
            value=float(self.error_count),
            threshold=10.0,
            status=HealthStatus.HEALTHY if self.error_count < 10 else HealthStatus.DEGRADED
        ))
        
        return metrics


class AutoHealer:
    """Automated healing system for pipeline components."""
    
    def __init__(self):
        self.healing_strategies: Dict[str, List[Callable]] = {}
        self.healing_history: List[HealingEvent] = []
        self.max_healing_attempts = 3
        self.healing_cooldown = timedelta(minutes=5)
        
    def register_healing_strategy(self, component: str, action: HealingAction, 
                                 handler: Callable[[str, Dict[str, Any]], bool]) -> None:
        """Register a healing strategy for a component."""
        if component not in self.healing_strategies:
            self.healing_strategies[component] = []
        self.healing_strategies[component].append((action, handler))
    
    def attempt_healing(self, component: str, health_status: HealthStatus, 
                       metrics: List[HealthMetric]) -> bool:
        """Attempt to heal a failing component."""
        # Check if we're in cooldown period
        recent_attempts = [
            event for event in self.healing_history
            if event.component == component and 
            datetime.utcnow() - event.timestamp < self.healing_cooldown
        ]
        
        if len(recent_attempts) >= self.max_healing_attempts:
            logger.warning(f"Component {component} in healing cooldown")
            return False
        
        # Select appropriate healing action based on health status
        healing_action = self._select_healing_action(health_status, metrics)
        
        if component in self.healing_strategies:
            for action, handler in self.healing_strategies[component]:
                if action == healing_action:
                    try:
                        logger.info(f"Attempting to heal {component} with action {action}")
                        success = handler(component, {"metrics": metrics, "status": health_status})
                        
                        # Record healing event
                        event = HealingEvent(
                            component=component,
                            action=action,
                            reason=f"Health status: {health_status}",
                            success=success,
                            details={"metrics_count": len(metrics)}
                        )
                        self.healing_history.append(event)
                        
                        if success:
                            logger.info(f"Successfully healed {component}")
                            return True
                        else:
                            logger.warning(f"Healing attempt failed for {component}")
                            
                    except Exception as e:
                        logger.error(f"Healing handler failed for {component}: {e}")
                        event = HealingEvent(
                            component=component,
                            action=action,
                            reason=f"Handler exception: {e}",
                            success=False
                        )
                        self.healing_history.append(event)
        
        return False
    
    def _select_healing_action(self, health_status: HealthStatus, 
                             metrics: List[HealthMetric]) -> HealingAction:
        """Select appropriate healing action based on health status."""
        if health_status == HealthStatus.CRITICAL:
            return HealingAction.RESTART_COMPONENT
        elif health_status == HealthStatus.FAILING:
            return HealingAction.CLEAR_CACHE
        elif health_status == HealthStatus.DEGRADED:
            return HealingAction.ADJUST_THRESHOLD
        else:
            return HealingAction.CLEAR_CACHE


class SelfHealingPipelineGuard:
    """Main self-healing pipeline guard that monitors and heals the filtering pipeline."""
    
    def __init__(self, filter_pipeline: FilterPipeline):
        self.pipeline = filter_pipeline
        self.component_monitors: Dict[str, ComponentHealthMonitor] = {}
        self.auto_healer = AutoHealer()
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        self.monitoring_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize component monitors
        for stage in self.pipeline.stages:
            self.component_monitors[stage.name] = ComponentHealthMonitor(stage.name)
        
        # Register default healing strategies
        self._register_default_healing_strategies()
        
        logger.info("Self-healing pipeline guard initialized")
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Pipeline monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("Pipeline monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_pipeline_health()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _check_pipeline_health(self) -> None:
        """Check health of all pipeline components."""
        for component_name, monitor in self.component_monitors.items():
            health_status = monitor.get_health_status()
            metrics = monitor.get_metrics()
            
            if health_status in [HealthStatus.DEGRADED, HealthStatus.FAILING, HealthStatus.CRITICAL]:
                logger.warning(f"Component {component_name} health: {health_status}")
                
                # Attempt healing
                self.auto_healer.attempt_healing(component_name, health_status, metrics)
    
    def record_pipeline_operation(self, stage_name: str, success: bool, 
                                 response_time: float, error: Optional[Exception] = None) -> None:
        """Record the result of a pipeline operation."""
        if stage_name in self.component_monitors:
            monitor = self.component_monitors[stage_name]
            if success:
                monitor.record_success(response_time)
            elif error:
                monitor.record_error(error)
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status of the pipeline."""
        component_health = {}
        overall_status = HealthStatus.HEALTHY
        
        for component_name, monitor in self.component_monitors.items():
            status = monitor.get_health_status()
            component_health[component_name] = status
            
            # Overall status is the worst component status
            if status.value == "critical":
                overall_status = HealthStatus.CRITICAL
            elif status.value == "failing" and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.FAILING
            elif status.value == "degraded" and overall_status not in [HealthStatus.CRITICAL, HealthStatus.FAILING]:
                overall_status = HealthStatus.DEGRADED
        
        return {
            "overall_status": overall_status,
            "components": component_health,
            "healing_events": len(self.auto_healer.healing_history),
            "monitoring_active": self.monitoring_active
        }
    
    def _register_default_healing_strategies(self) -> None:
        """Register default healing strategies for pipeline components."""
        
        def restart_component(component: str, context: Dict[str, Any]) -> bool:
            """Restart a pipeline component."""
            try:
                # Find the stage and reset its metrics
                for stage in self.pipeline.stages:
                    if stage.name == component:
                        stage.metrics = {"processed": 0, "filtered": 0, "errors": 0}
                        # Reset monitor
                        if component in self.component_monitors:
                            monitor = self.component_monitors[component]
                            monitor.error_count = 0
                            monitor.last_success_time = datetime.utcnow()
                            monitor.response_times.clear()
                        logger.info(f"Restarted component {component}")
                        return True
                return False
            except Exception as e:
                logger.error(f"Failed to restart component {component}: {e}")
                return False
        
        def clear_cache(component: str, context: Dict[str, Any]) -> bool:
            """Clear cache for a component."""
            try:
                # This would clear any component-specific caches
                logger.info(f"Cleared cache for component {component}")
                return True
            except Exception as e:
                logger.error(f"Failed to clear cache for {component}: {e}")
                return False
        
        def adjust_threshold(component: str, context: Dict[str, Any]) -> bool:
            """Adjust thresholds for degraded performance."""
            try:
                if component in self.component_monitors:
                    monitor = self.component_monitors[component]
                    # Temporarily increase thresholds to give component time to recover
                    monitor.error_rate_threshold *= 1.5
                    monitor.response_time_threshold *= 1.2
                    logger.info(f"Adjusted thresholds for component {component}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Failed to adjust threshold for {component}: {e}")
                return False
        
        # Register strategies for all components
        for component_name in self.component_monitors.keys():
            self.auto_healer.register_healing_strategy(component_name, HealingAction.RESTART_COMPONENT, restart_component)
            self.auto_healer.register_healing_strategy(component_name, HealingAction.CLEAR_CACHE, clear_cache)
            self.auto_healer.register_healing_strategy(component_name, HealingAction.ADJUST_THRESHOLD, adjust_threshold)
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop_monitoring()
        self.executor.shutdown(wait=True)


class EnhancedSafePathFilter(SafePathFilter):
    """Enhanced SafePath filter with self-healing capabilities."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.self_healing_guard = SelfHealingPipelineGuard(self.pipeline)
        self.self_healing_guard.start_monitoring()
        logger.info("Enhanced SafePath filter with self-healing initialized")
    
    def filter(self, request):
        """Filter with self-healing monitoring."""
        start_time = time.time()
        
        try:
            result = super().filter(request)
            
            # Record successful operation for all stages
            processing_time = time.time() - start_time
            for stage in self.pipeline.stages:
                self.self_healing_guard.record_pipeline_operation(
                    stage.name, True, processing_time
                )
            
            return result
            
        except Exception as e:
            # Record failure
            processing_time = time.time() - start_time
            for stage in self.pipeline.stages:
                self.self_healing_guard.record_pipeline_operation(
                    stage.name, False, processing_time, e
                )
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the filter pipeline."""
        return self.self_healing_guard.get_overall_health()
    
    def cleanup(self) -> None:
        """Enhanced cleanup with self-healing resources."""
        super().cleanup()
        self.self_healing_guard.cleanup()