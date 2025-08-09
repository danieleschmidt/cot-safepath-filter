"""
Monitoring and metrics collection for SafePath.
"""

import time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes if prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def inc(self, *args): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def observe(self, *args): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args): pass
    def generate_latest(): return b"# Prometheus not available"
import threading


logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time."""
    
    timestamp: float
    total_requests: int
    filtered_requests: int
    average_latency: float
    error_count: int
    cache_hits: int
    cache_misses: int


class PrometheusMetrics:
    """Prometheus metrics collector for SafePath."""
    
    def __init__(self):
        # Request metrics
        self.request_total = Counter(
            'safepath_requests_total',
            'Total number of filter requests',
            ['safety_level', 'was_filtered']
        )
        
        self.request_duration = Histogram(
            'safepath_request_duration_seconds',
            'Request processing duration in seconds',
            ['safety_level']
        )
        
        self.safety_score = Histogram(
            'safepath_safety_score',
            'Safety scores distribution',
            ['safety_level'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Error metrics
        self.errors_total = Counter(
            'safepath_errors_total',
            'Total number of errors',
            ['error_type']
        )
        
        # Cache metrics
        self.cache_operations = Counter(
            'safepath_cache_operations_total',
            'Total cache operations',
            ['operation', 'result']
        )
        
        # System metrics
        self.active_connections = Gauge(
            'safepath_active_connections',
            'Number of active connections'
        )
        
        self.memory_usage = Gauge(
            'safepath_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        # Detector metrics
        self.detector_invocations = Counter(
            'safepath_detector_invocations_total',
            'Total detector invocations',
            ['detector_name', 'result']
        )
        
        self.detector_duration = Histogram(
            'safepath_detector_duration_seconds',
            'Detector processing duration',
            ['detector_name']
        )
    
    def record_request(
        self, 
        safety_level: str, 
        was_filtered: bool, 
        processing_time: float,
        safety_score: float
    ):
        """Record a filter request."""
        self.request_total.labels(
            safety_level=safety_level,
            was_filtered=str(was_filtered).lower()
        ).inc()
        
        self.request_duration.labels(safety_level=safety_level).observe(processing_time)
        self.safety_score.labels(safety_level=safety_level).observe(safety_score)
    
    def record_error(self, error_type: str):
        """Record an error."""
        self.errors_total.labels(error_type=error_type).inc()
    
    def record_cache_operation(self, operation: str, result: str):
        """Record a cache operation."""
        self.cache_operations.labels(operation=operation, result=result).inc()
    
    def record_detector_invocation(
        self, 
        detector_name: str, 
        result: str, 
        duration: float
    ):
        """Record a detector invocation."""
        self.detector_invocations.labels(
            detector_name=detector_name,
            result=result
        ).inc()
        
        self.detector_duration.labels(detector_name=detector_name).observe(duration)
    
    def update_system_metrics(self, connections: int, memory_bytes: int):
        """Update system metrics."""
        self.active_connections.set(connections)
        self.memory_usage.set(memory_bytes)
    
    def generate_metrics(self) -> str:
        """Generate Prometheus metrics format."""
        return generate_latest().decode('utf-8')


class MetricsCollector:
    """Collects and aggregates metrics over time."""
    
    def __init__(self, max_snapshots: int = 1000):
        self.max_snapshots = max_snapshots
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.current_metrics = {
            'total_requests': 0,
            'filtered_requests': 0,
            'total_latency': 0.0,
            'error_count': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self._lock = threading.Lock()
        
        # Detector-specific metrics
        self.detector_metrics = defaultdict(lambda: {
            'invocations': 0,
            'detections': 0,
            'total_duration': 0.0
        })
    
    def record_request(
        self, 
        was_filtered: bool, 
        latency_ms: float,
        safety_score: float
    ):
        """Record a request with its metrics."""
        with self._lock:
            self.current_metrics['total_requests'] += 1
            if was_filtered:
                self.current_metrics['filtered_requests'] += 1
            self.current_metrics['total_latency'] += latency_ms
    
    def record_error(self):
        """Record an error."""
        with self._lock:
            self.current_metrics['error_count'] += 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        with self._lock:
            self.current_metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        with self._lock:
            self.current_metrics['cache_misses'] += 1
    
    def record_detector_metrics(
        self, 
        detector_name: str, 
        detected: bool, 
        duration_ms: float
    ):
        """Record detector-specific metrics."""
        with self._lock:
            metrics = self.detector_metrics[detector_name]
            metrics['invocations'] += 1
            if detected:
                metrics['detections'] += 1
            metrics['total_duration'] += duration_ms
    
    def take_snapshot(self) -> MetricSnapshot:
        """Take a snapshot of current metrics."""
        with self._lock:
            total_requests = self.current_metrics['total_requests']
            average_latency = (
                self.current_metrics['total_latency'] / max(total_requests, 1)
            )
            
            snapshot = MetricSnapshot(
                timestamp=time.time(),
                total_requests=total_requests,
                filtered_requests=self.current_metrics['filtered_requests'],
                average_latency=average_latency,
                error_count=self.current_metrics['error_count'],
                cache_hits=self.current_metrics['cache_hits'],
                cache_misses=self.current_metrics['cache_misses']
            )
            
            self.snapshots.append(snapshot)
            return snapshot
    
    def get_recent_snapshots(self, count: int = 100) -> List[MetricSnapshot]:
        """Get recent metric snapshots."""
        with self._lock:
            return list(self.snapshots)[-count:]
    
    def get_detector_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of detector metrics."""
        with self._lock:
            summary = {}
            for detector_name, metrics in self.detector_metrics.items():
                invocations = metrics['invocations']
                summary[detector_name] = {
                    'invocations': invocations,
                    'detections': metrics['detections'],
                    'detection_rate': metrics['detections'] / max(invocations, 1),
                    'average_duration_ms': metrics['total_duration'] / max(invocations, 1)
                }
            return summary
    
    def get_current_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        with self._lock:
            total_requests = self.current_metrics['total_requests']
            cache_total = self.current_metrics['cache_hits'] + self.current_metrics['cache_misses']
            
            return {
                'total_requests': total_requests,
                'filtered_requests': self.current_metrics['filtered_requests'],
                'filter_rate': self.current_metrics['filtered_requests'] / max(total_requests, 1),
                'average_latency_ms': self.current_metrics['total_latency'] / max(total_requests, 1),
                'error_count': self.current_metrics['error_count'],
                'error_rate': self.current_metrics['error_count'] / max(total_requests, 1),
                'cache_hit_rate': self.current_metrics['cache_hits'] / max(cache_total, 1),
                'cache_total_operations': cache_total,
                'detector_summary': self.get_detector_summary()
            }
    
    def reset_metrics(self):
        """Reset all metrics to zero."""
        with self._lock:
            self.current_metrics = {
                'total_requests': 0,
                'filtered_requests': 0,
                'total_latency': 0.0,
                'error_count': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }
            self.detector_metrics.clear()


class PerformanceMonitor:
    """Monitor performance and detect anomalies."""
    
    def __init__(self, alert_threshold_ms: float = 100.0):
        self.alert_threshold_ms = alert_threshold_ms
        self.recent_latencies: deque = deque(maxlen=100)
        self.alerts: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def record_latency(self, latency_ms: float):
        """Record a latency measurement."""
        with self._lock:
            self.recent_latencies.append(latency_ms)
            
            # Check for performance alerts
            if latency_ms > self.alert_threshold_ms:
                self.alerts.append({
                    'timestamp': time.time(),
                    'type': 'high_latency',
                    'value': latency_ms,
                    'threshold': self.alert_threshold_ms
                })
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        with self._lock:
            if not self.recent_latencies:
                return {
                    'min_latency_ms': 0,
                    'max_latency_ms': 0,
                    'avg_latency_ms': 0,
                    'p95_latency_ms': 0,
                    'p99_latency_ms': 0
                }
            
            sorted_latencies = sorted(self.recent_latencies)
            count = len(sorted_latencies)
            
            return {
                'min_latency_ms': sorted_latencies[0],
                'max_latency_ms': sorted_latencies[-1],
                'avg_latency_ms': sum(sorted_latencies) / count,
                'p95_latency_ms': sorted_latencies[int(count * 0.95)],
                'p99_latency_ms': sorted_latencies[int(count * 0.99)]
            }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance alerts."""
        with self._lock:
            return self.alerts[-limit:] if self.alerts else []
    
    def clear_alerts(self):
        """Clear all alerts."""
        with self._lock:
            self.alerts.clear()


# Global instances
prometheus_metrics = PrometheusMetrics()
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor()


def record_filter_request(
    safety_level: str,
    was_filtered: bool,
    processing_time_ms: float,
    safety_score: float
):
    """Record a filter request across all monitoring systems."""
    processing_time_seconds = processing_time_ms / 1000.0
    
    # Prometheus metrics
    prometheus_metrics.record_request(
        safety_level=safety_level,
        was_filtered=was_filtered,
        processing_time=processing_time_seconds,
        safety_score=safety_score
    )
    
    # Internal metrics collector
    metrics_collector.record_request(
        was_filtered=was_filtered,
        latency_ms=processing_time_ms,
        safety_score=safety_score
    )
    
    # Performance monitoring
    performance_monitor.record_latency(processing_time_ms)


def record_error(error_type: str):
    """Record an error across monitoring systems."""
    prometheus_metrics.record_error(error_type)
    metrics_collector.record_error()


def record_cache_operation(operation: str, hit: bool):
    """Record a cache operation."""
    result = "hit" if hit else "miss"
    prometheus_metrics.record_cache_operation(operation, result)
    
    if hit:
        metrics_collector.record_cache_hit()
    else:
        metrics_collector.record_cache_miss()


def record_detector_invocation(
    detector_name: str,
    detected: bool,
    duration_ms: float
):
    """Record a detector invocation."""
    result = "detected" if detected else "clean"
    duration_seconds = duration_ms / 1000.0
    
    prometheus_metrics.record_detector_invocation(
        detector_name=detector_name,
        result=result,
        duration=duration_seconds
    )
    
    metrics_collector.record_detector_metrics(
        detector_name=detector_name,
        detected=detected,
        duration_ms=duration_ms
    )


def get_monitoring_summary() -> Dict[str, Any]:
    """Get comprehensive monitoring summary."""
    return {
        'current_metrics': metrics_collector.get_current_summary(),
        'performance_stats': performance_monitor.get_performance_stats(),
        'recent_alerts': performance_monitor.get_recent_alerts(),
        'prometheus_available': True
    }