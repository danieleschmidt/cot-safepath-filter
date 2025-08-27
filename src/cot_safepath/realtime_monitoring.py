"""
Real-time monitoring and alerting system for SafePath Filter - Generation 1.

Comprehensive monitoring, metrics collection, alerting, and observability
features for production deployment.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import threading
import logging

from .models import FilterResult, SafetyScore, Severity
from .exceptions import FilterError


logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    
    enable_metrics: bool = True
    enable_alerting: bool = True
    metrics_retention_hours: int = 24
    alert_cooldown_minutes: int = 5
    real_time_dashboard: bool = True
    export_prometheus: bool = False
    export_json: bool = True
    webhook_notifications: bool = False
    email_notifications: bool = False
    log_level: str = "INFO"


@dataclass
class MetricPoint:
    """A single metric data point."""
    
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    
    name: str
    condition: str  # e.g., "safety_score < 0.3"
    severity: Severity
    cooldown_minutes: int = 5
    description: str = ""
    enabled: bool = True
    actions: List[str] = None  # ["email", "webhook", "log"]
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = ["log"]


@dataclass
class Alert:
    """An active alert."""
    
    rule_name: str
    severity: Severity
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MetricsCollector:
    """Collects and stores metrics in memory with time-series support."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def record_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Record a counter metric."""
        with self._lock:
            self.counters[name] += value
            self._add_metric(name, self.counters[name], tags)
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric."""
        with self._lock:
            self.gauges[name] = value
            self._add_metric(name, value, tags)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only recent values for memory efficiency
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            self._add_metric(name, value, tags)
    
    def _add_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Add metric point to time series."""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            metric_name=name,
            value=value,
            tags=tags or {}
        )
        self.metrics[name].append(point)
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[MetricPoint]:
        """Get metric history for specified time range."""
        with self._lock:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            return [
                point for point in self.metrics[name] 
                if point.timestamp >= cutoff
            ]
    
    def get_counter_value(self, name: str) -> float:
        """Get current counter value."""
        return self.counters.get(name, 0.0)
    
    def get_gauge_value(self, name: str) -> Optional[float]:
        """Get current gauge value."""
        return self.gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        values = self.histograms.get(name, [])
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "mean": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_all_metrics_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of all current metrics."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: self.get_histogram_stats(name) 
                    for name in self.histograms.keys()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _start_cleanup_task(self):
        """Start background task to clean up old metrics."""
        def cleanup():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self._cleanup_old_metrics()
                except Exception as e:
                    logger.error(f"Metrics cleanup failed: {e}")
        
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()
    
    def _cleanup_old_metrics(self):
        """Remove old metric points to prevent memory growth."""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            for metric_name in list(self.metrics.keys()):
                points = self.metrics[metric_name]
                # Remove old points
                while points and points[0].timestamp < cutoff:
                    points.popleft()


class AlertManager:
    """Manages alert rules and active alerts."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.last_alert_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        
        # Default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        self.add_rule(AlertRule(
            name="high_risk_detection",
            condition="safety_score < 0.3",
            severity=Severity.HIGH,
            description="High risk content detected",
            actions=["log", "webhook"] if self.config.webhook_notifications else ["log"]
        ))
        
        self.add_rule(AlertRule(
            name="excessive_filtering",
            condition="filter_rate > 0.5",
            severity=Severity.MEDIUM,
            description="High filtering rate detected",
            cooldown_minutes=10
        ))
        
        self.add_rule(AlertRule(
            name="processing_latency",
            condition="avg_processing_time > 1000",
            severity=Severity.MEDIUM,
            description="High processing latency detected"
        ))
        
        self.add_rule(AlertRule(
            name="error_rate_spike",
            condition="error_rate > 0.1",
            severity=Severity.HIGH,
            description="Error rate spike detected"
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self._lock:
            self.rules[rule.name] = rule
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
    
    def evaluate_rules(self, metrics: Dict[str, Any]):
        """Evaluate all alert rules against current metrics."""
        if not self.config.enable_alerting:
            return
        
        with self._lock:
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                last_alert = self.last_alert_times.get(rule_name)
                if last_alert and datetime.utcnow() - last_alert < timedelta(minutes=rule.cooldown_minutes):
                    continue
                
                # Evaluate condition
                try:
                    if self._evaluate_condition(rule.condition, metrics):
                        self._trigger_alert(rule, metrics)
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert condition against metrics."""
        # Simple condition evaluator - in production, use a proper expression evaluator
        
        # Extract metric references and values from metrics
        context = {}
        
        # Add counter values
        for name, value in metrics.get("counters", {}).items():
            context[name.replace(".", "_")] = value
        
        # Add gauge values
        for name, value in metrics.get("gauges", {}).items():
            context[name.replace(".", "_")] = value
        
        # Add histogram stats
        for name, stats in metrics.get("histograms", {}).items():
            for stat_name, stat_value in stats.items():
                context[f"{name}_{stat_name}".replace(".", "_")] = stat_value
        
        # Calculate derived metrics
        total_requests = context.get("total_requests", 1)
        filtered_requests = context.get("filtered_requests", 0)
        context["filter_rate"] = filtered_requests / max(total_requests, 1)
        
        error_count = context.get("error_count", 0)
        context["error_rate"] = error_count / max(total_requests, 1)
        
        # Evaluate condition
        try:
            # Simple evaluation - replace with proper parser in production
            for var_name, var_value in context.items():
                condition = condition.replace(var_name, str(var_value))
            
            return eval(condition, {"__builtins__": {}})
        except:
            return False
    
    def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert."""
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=f"Alert: {rule.description}",
            timestamp=datetime.utcnow(),
            metadata={"metrics": metrics, "rule": asdict(rule)}
        )
        
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        self.last_alert_times[rule.name] = alert.timestamp
        
        # Execute alert actions
        self._execute_alert_actions(alert, rule.actions)
        
        logger.warning(f"Alert triggered: {alert.message}")
    
    def _execute_alert_actions(self, alert: Alert, actions: List[str]):
        """Execute alert actions."""
        for action in actions:
            try:
                if action == "log":
                    self._log_alert(alert)
                elif action == "webhook" and self.config.webhook_notifications:
                    self._send_webhook_alert(alert)
                elif action == "email" and self.config.email_notifications:
                    self._send_email_alert(alert)
            except Exception as e:
                logger.error(f"Alert action {action} failed: {e}")
    
    def _log_alert(self, alert: Alert):
        """Log alert to system logs."""
        log_level = getattr(logging, alert.severity.value.upper())
        logger.log(log_level, f"ALERT: {alert.message} (Rule: {alert.rule_name})")
    
    def _send_webhook_alert(self, alert: Alert):
        """Send webhook notification."""
        # Placeholder - implement webhook sending
        webhook_data = {
            "alert": asdict(alert),
            "timestamp": alert.timestamp.isoformat()
        }
        logger.info(f"Webhook alert would be sent: {json.dumps(webhook_data)}")
    
    def _send_email_alert(self, alert: Alert):
        """Send email notification."""
        # Placeholder - implement email sending
        logger.info(f"Email alert would be sent: {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
    
    def resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        with self._lock:
            if rule_name in self.active_alerts:
                alert = self.active_alerts[rule_name]
                alert.resolved = True
                alert.resolution_timestamp = datetime.utcnow()
                del self.active_alerts[rule_name]


class SafePathMonitor:
    """Main monitoring system for SafePath Filter."""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.metrics = MetricsCollector(self.config.metrics_retention_hours)
        self.alerts = AlertManager(self.config)
        self.running = False
        self._monitor_task = None
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Initialize metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize default metrics."""
        self.metrics.record_counter("total_requests", 0)
        self.metrics.record_counter("filtered_requests", 0)
        self.metrics.record_counter("blocked_requests", 0)
        self.metrics.record_counter("error_count", 0)
        self.metrics.record_gauge("safety_score", 1.0)
        self.metrics.record_gauge("avg_processing_time", 0.0)
    
    def start(self):
        """Start the monitoring system."""
        if self.running:
            return
        
        self.running = True
        logger.info("SafePath monitoring system started")
        
        # Start monitoring loop
        if asyncio.get_event_loop().is_running():
            # If event loop is running, create task
            self._monitor_task = asyncio.create_task(self._monitor_loop())
        else:
            # Otherwise start in separate thread
            thread = threading.Thread(target=self._run_monitor_loop, daemon=True)
            thread.start()
    
    def stop(self):
        """Stop the monitoring system."""
        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("SafePath monitoring system stopped")
    
    def _run_monitor_loop(self):
        """Run monitor loop in sync context."""
        asyncio.run(self._monitor_loop())
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Get current metrics snapshot
                metrics_snapshot = self.metrics.get_all_metrics_snapshot()
                
                # Evaluate alert rules
                self.alerts.evaluate_rules(metrics_snapshot)
                
                # Export metrics if configured
                if self.config.export_json:
                    self._export_metrics_json(metrics_snapshot)
                
                if self.config.export_prometheus:
                    self._export_prometheus_metrics(metrics_snapshot)
                
                # Wait before next iteration
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)
    
    def record_filter_operation(self, result: FilterResult):
        """Record metrics from a filter operation."""
        if not self.config.enable_metrics:
            return
        
        # Record basic metrics
        self.metrics.record_counter("total_requests")
        
        if result.was_filtered:
            self.metrics.record_counter("filtered_requests")
            
            if not result.safety_score.is_safe:
                self.metrics.record_counter("blocked_requests")
        
        # Record safety score
        self.metrics.record_gauge("safety_score", result.safety_score.overall_score)
        self.metrics.record_histogram("safety_score_distribution", result.safety_score.overall_score)
        
        # Record processing time
        if result.processing_time_ms:
            self.metrics.record_histogram("processing_time", result.processing_time_ms)
            self.metrics.record_gauge("avg_processing_time", result.processing_time_ms)
        
        # Record by severity
        if result.safety_score.severity:
            severity_counter = f"severity_{result.safety_score.severity.value}"
            self.metrics.record_counter(severity_counter)
        
        # Record detected patterns
        for reason in result.filter_reasons:
            detector_name = reason.split(":")[0] if ":" in reason else "unknown"
            self.metrics.record_counter(f"detector_activation_{detector_name}")
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record an error."""
        self.metrics.record_counter("error_count")
        
        error_type = error.__class__.__name__
        self.metrics.record_counter(f"error_type_{error_type}")
        
        logger.error(f"SafePath error recorded: {error}", extra=context)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for real-time dashboard."""
        metrics_snapshot = self.metrics.get_all_metrics_snapshot()
        active_alerts = self.alerts.get_active_alerts()
        
        # Calculate additional dashboard metrics
        total_requests = metrics_snapshot["counters"].get("total_requests", 0)
        filtered_requests = metrics_snapshot["counters"].get("filtered_requests", 0)
        blocked_requests = metrics_snapshot["counters"].get("blocked_requests", 0)
        error_count = metrics_snapshot["counters"].get("error_count", 0)
        
        filter_rate = filtered_requests / max(total_requests, 1)
        block_rate = blocked_requests / max(total_requests, 1) 
        error_rate = error_count / max(total_requests, 1)
        
        dashboard_data = {
            "overview": {
                "total_requests": total_requests,
                "filter_rate": filter_rate,
                "block_rate": block_rate,
                "error_rate": error_rate,
                "avg_processing_time": metrics_snapshot["gauges"].get("avg_processing_time", 0),
                "current_safety_score": metrics_snapshot["gauges"].get("safety_score", 1.0)
            },
            "alerts": {
                "active_count": len(active_alerts),
                "alerts": [asdict(alert) for alert in active_alerts]
            },
            "metrics": metrics_snapshot,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return dashboard_data
    
    def _export_metrics_json(self, metrics: Dict[str, Any]):
        """Export metrics to JSON file."""
        try:
            export_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics,
                "alerts": [asdict(alert) for alert in self.alerts.get_active_alerts()]
            }
            
            with open("safepath_metrics.json", "w") as f:
                json.dump(export_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
    
    def _export_prometheus_metrics(self, metrics: Dict[str, Any]):
        """Export metrics in Prometheus format."""
        try:
            prometheus_lines = []
            timestamp = int(time.time() * 1000)
            
            # Export counters
            for name, value in metrics["counters"].items():
                prometheus_lines.append(f'safepath_{name}_total {value} {timestamp}')
            
            # Export gauges  
            for name, value in metrics["gauges"].items():
                prometheus_lines.append(f'safepath_{name} {value} {timestamp}')
            
            # Export histogram summaries
            for name, stats in metrics["histograms"].items():
                for stat_name, stat_value in stats.items():
                    prometheus_lines.append(f'safepath_{name}_{stat_name} {stat_value} {timestamp}')
            
            with open("safepath_metrics.prom", "w") as f:
                f.write("\n".join(prometheus_lines))
        
        except Exception as e:
            logger.error(f"Prometheus export failed: {e}")


# Global monitoring instance
_global_monitor: Optional[SafePathMonitor] = None


def get_global_monitor() -> SafePathMonitor:
    """Get or create global monitoring instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SafePathMonitor()
        _global_monitor.start()
    return _global_monitor


def configure_monitoring(config: MonitoringConfig):
    """Configure global monitoring."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop()
    _global_monitor = SafePathMonitor(config)
    _global_monitor.start()
    return _global_monitor


# Decorator for automatic monitoring
def monitor_filter_operation(func):
    """Decorator to automatically monitor filter operations."""
    def wrapper(*args, **kwargs):
        monitor = get_global_monitor()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Record successful operation
            if isinstance(result, FilterResult):
                monitor.record_filter_operation(result)
            
            return result
            
        except Exception as e:
            # Record error
            monitor.record_error(e, {"function": func.__name__})
            raise
    
    return wrapper