#!/usr/bin/env python3
"""
Metrics collection system for CoT SafePath Filter
Collects and exports metrics for monitoring and observability
"""

import time
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricType(Enum):
    """Metric types supported by the collector."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Individual metric value with metadata."""
    name: str
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metric_type: MetricType = MetricType.GAUGE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'labels': self.labels,
            'timestamp': self.timestamp.isoformat(),
            'type': self.metric_type.value
        }


class MetricsCollector:
    """Centralized metrics collection and export system."""

    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        # In-memory metric storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._current_values: Dict[str, MetricValue] = {}
        self._lock = threading.Lock()
        
        # Prometheus registry
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._prometheus_metrics: Dict[str, Any] = {}
            self._init_prometheus_metrics()
        
        # Start background collection
        self._collection_task = None
        self._running = False

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        if not self.enable_prometheus:
            return

        # Application metrics
        self._prometheus_metrics['requests_total'] = Counter(
            'safepath_requests_total',
            'Total number of filter requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self._prometheus_metrics['request_duration'] = Histogram(
            'safepath_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self._prometheus_metrics['filtered_requests'] = Counter(
            'safepath_filtered_requests_total',
            'Total number of filtered requests',
            ['reason', 'severity'],
            registry=self.registry
        )
        
        self._prometheus_metrics['safety_score'] = Histogram(
            'safepath_safety_score',
            'Distribution of safety scores',
            ['detector'],
            registry=self.registry
        )
        
        self._prometheus_metrics['cache_hits'] = Counter(
            'safepath_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self._prometheus_metrics['cache_misses'] = Counter(
            'safepath_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # System metrics
        self._prometheus_metrics['cpu_usage'] = Gauge(
            'safepath_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self._prometheus_metrics['memory_usage'] = Gauge(
            'safepath_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self._prometheus_metrics['active_connections'] = Gauge(
            'safepath_active_connections',
            'Number of active connections',
            ['type'],
            registry=self.registry
        )

    def start_collection(self):
        """Start background metrics collection."""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Metrics collection started")

    async def stop_collection(self):
        """Stop background metrics collection."""
        if not self._running:
            return
        
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Metrics collection stopped")

    async def _collection_loop(self):
        """Background loop for collecting system metrics."""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(30)

    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.gauge('system.cpu.usage_percent', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.gauge('system.memory.used_bytes', memory.used)
            self.gauge('system.memory.available_bytes', memory.available)
            self.gauge('system.memory.percent', memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('.')
            self.gauge('system.disk.used_bytes', disk.used)
            self.gauge('system.disk.free_bytes', disk.free)
            self.gauge('system.disk.percent', (disk.used / disk.total) * 100)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                self.counter('system.network.bytes_sent', net_io.bytes_sent)
                self.counter('system.network.bytes_recv', net_io.bytes_recv)
            
            # Process count
            process_count = len(psutil.pids())
            self.gauge('system.processes.count', process_count)
            
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    def counter(self, name: str, value: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        labels = labels or {}
        
        with self._lock:
            # Update internal storage
            metric = MetricValue(name, value, labels, metric_type=MetricType.COUNTER)
            self._metrics[name].append(metric)
            self._current_values[name] = metric
            
            # Update Prometheus if available
            if self.enable_prometheus and name in self._prometheus_metrics:
                if labels:
                    self._prometheus_metrics[name].labels(**labels).inc(value)
                else:
                    self._prometheus_metrics[name].inc(value)

    def gauge(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        labels = labels or {}
        
        with self._lock:
            # Update internal storage
            metric = MetricValue(name, value, labels, metric_type=MetricType.GAUGE)
            self._metrics[name].append(metric)
            self._current_values[name] = metric
            
            # Update Prometheus if available
            if self.enable_prometheus and name in self._prometheus_metrics:
                if labels:
                    self._prometheus_metrics[name].labels(**labels).set(value)
                else:
                    self._prometheus_metrics[name].set(value)

    def histogram(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        labels = labels or {}
        
        with self._lock:
            # Update internal storage
            metric = MetricValue(name, value, labels, metric_type=MetricType.HISTOGRAM)
            self._metrics[name].append(metric)
            self._current_values[name] = metric
            
            # Update Prometheus if available
            if self.enable_prometheus and name in self._prometheus_metrics:
                if labels:
                    self._prometheus_metrics[name].labels(**labels).observe(value)
                else:
                    self._prometheus_metrics[name].observe(value)

    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return MetricTimer(self, name, labels)

    def increment_request_count(self, method: str, endpoint: str, status: str):
        """Increment request counter with standard labels."""
        self.counter('requests_total', 1, {'method': method, 'endpoint': endpoint, 'status': status})

    def record_request_duration(self, method: str, endpoint: str, duration: float):
        """Record request duration."""
        self.histogram('request_duration_seconds', duration, {'method': method, 'endpoint': endpoint})

    def record_filter_event(self, reason: str, severity: str):
        """Record a filter event."""
        self.counter('filtered_requests_total', 1, {'reason': reason, 'severity': severity})

    def record_safety_score(self, detector: str, score: float):
        """Record a safety score."""
        self.histogram('safety_score', score, {'detector': detector})

    def record_cache_hit(self, cache_type: str):
        """Record a cache hit."""
        self.counter('cache_hits_total', 1, {'cache_type': cache_type})

    def record_cache_miss(self, cache_type: str):
        """Record a cache miss."""
        self.counter('cache_misses_total', 1, {'cache_type': cache_type})

    def get_current_metrics(self) -> Dict[str, MetricValue]:
        """Get current metric values."""
        with self._lock:
            return self._current_values.copy()

    def get_metric_history(self, name: str, limit: int = 100) -> List[MetricValue]:
        """Get historical values for a metric."""
        with self._lock:
            return list(self._metrics[name])[-limit:]

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        if not self.enable_prometheus:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')

    def export_json(self) -> str:
        """Export metrics in JSON format."""
        current_metrics = self.get_current_metrics()
        export_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics': {name: metric.to_dict() for name, metric in current_metrics.items()}
        }
        return json.dumps(export_data, indent=2)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        current_metrics = self.get_current_metrics()
        
        return {
            'total_metrics': len(current_metrics),
            'counters': len([m for m in current_metrics.values() if m.metric_type == MetricType.COUNTER]),
            'gauges': len([m for m in current_metrics.values() if m.metric_type == MetricType.GAUGE]),
            'histograms': len([m for m in current_metrics.values() if m.metric_type == MetricType.HISTOGRAM]),
            'last_collection': max([m.timestamp for m in current_metrics.values()]) if current_metrics else None
        }


class MetricTimer:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.histogram(self.name, duration, self.labels)


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def setup_metrics_collection(enable_prometheus: bool = True):
    """Setup global metrics collection."""
    global _global_collector
    _global_collector = MetricsCollector(enable_prometheus=enable_prometheus)
    return _global_collector


# Convenience functions
def counter(name: str, value: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None):
    """Record a counter metric using the global collector."""
    get_metrics_collector().counter(name, value, labels)


def gauge(name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
    """Record a gauge metric using the global collector."""
    get_metrics_collector().gauge(name, value, labels)


def histogram(name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
    """Record a histogram metric using the global collector."""
    get_metrics_collector().histogram(name, value, labels)


def timer(name: str, labels: Optional[Dict[str, str]] = None):
    """Time an operation using the global collector."""
    return get_metrics_collector().timer(name, labels)


async def main():
    """CLI entry point for metrics collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SafePath Metrics Collector')
    parser.add_argument('--format', choices=['prometheus', 'json'], default='json', help='Export format')
    parser.add_argument('--prometheus', action='store_true', help='Enable Prometheus metrics')
    parser.add_argument('--collect', action='store_true', help='Start collection and serve metrics')
    parser.add_argument('--port', type=int, default=9090, help='Metrics server port')
    
    args = parser.parse_args()
    
    # Setup collector
    collector = setup_metrics_collection(enable_prometheus=args.prometheus)
    
    if args.collect:
        # Start collection and serve metrics
        await collector.start_collection()
        
        try:
            if args.prometheus:
                # Start Prometheus metrics server
                from prometheus_client import start_http_server
                start_http_server(args.port, registry=collector.registry)
                print(f"Prometheus metrics server started on port {args.port}")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            await collector.stop_collection()
    else:
        # Export current metrics
        if args.format == 'prometheus':
            print(collector.export_prometheus())
        else:
            print(collector.export_json())


if __name__ == '__main__':
    asyncio.run(main())