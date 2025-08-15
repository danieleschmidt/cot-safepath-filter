"""
Performance Optimizer - Generation 3 Advanced Performance Features.

Adaptive performance optimization, resource scaling, and bottleneck detection.
"""

import asyncio
import time
import threading
import multiprocessing
import queue
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import logging
import psutil
import gc
import weakref
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .models import FilterRequest, FilterResult, ProcessingMetrics
from .intelligent_caching import IntelligentCache, FilterResultCache
from .advanced_monitoring import MetricCollector, MetricType


logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    CONSERVATIVE = "conservative"    # Safe, minimal changes
    BALANCED = "balanced"           # Moderate optimization
    AGGRESSIVE = "aggressive"       # Maximum performance
    ADAPTIVE = "adaptive"          # Self-adjusting based on load


class ResourceType(Enum):
    """Types of resources that can be optimized."""
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    CACHE = "cache"
    THREADS = "threads"
    PROCESSES = "processes"


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    CACHE_MISS = "cache_miss"
    THREAD_CONTENTION = "thread_contention"
    QUEUE_BACKLOG = "queue_backlog"
    DETECTOR_LATENCY = "detector_latency"


@dataclass
class PerformanceProfile:
    """Performance profile for a component or operation."""
    component: str
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    cpu_usage_percent: float
    memory_usage_mb: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_qps": self.throughput_qps,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "error_rate": self.error_rate,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Bottleneck:
    """Detected performance bottleneck."""
    bottleneck_id: str
    bottleneck_type: BottleneckType
    component: str
    severity: float  # 0.0 to 1.0
    description: str
    metrics: Dict[str, float]
    suggested_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bottleneck_id": self.bottleneck_id,
            "bottleneck_type": self.bottleneck_type.value,
            "component": self.component,
            "severity": self.severity,
            "description": self.description,
            "metrics": self.metrics,
            "suggested_actions": self.suggested_actions,
            "timestamp": self.timestamp.isoformat()
        }


class ResourceMonitor:
    """Monitors system and application resource usage."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Resource metrics history
        self.cpu_history: deque = deque(maxlen=3600)  # 1 hour at 1s intervals
        self.memory_history: deque = deque(maxlen=3600)
        self.io_history: deque = deque(maxlen=3600)
        self.network_history: deque = deque(maxlen=3600)
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("Resource monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._sample_resources()
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(5)
    
    def _sample_resources(self) -> None:
        """Sample current resource usage."""
        timestamp = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # IO metrics
        io_counters = psutil.disk_io_counters()
        
        # Network metrics
        net_counters = psutil.net_io_counters()
        
        with self._lock:
            self.cpu_history.append({
                "timestamp": timestamp,
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "load_avg": load_avg
            })
            
            self.memory_history.append({
                "timestamp": timestamp,
                "total_mb": memory.total / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "percent": memory.percent
            })
            
            if io_counters:
                self.io_history.append({
                    "timestamp": timestamp,
                    "read_bytes": io_counters.read_bytes,
                    "write_bytes": io_counters.write_bytes,
                    "read_count": io_counters.read_count,
                    "write_count": io_counters.write_count
                })
            
            if net_counters:
                self.network_history.append({
                    "timestamp": timestamp,
                    "bytes_sent": net_counters.bytes_sent,
                    "bytes_recv": net_counters.bytes_recv,
                    "packets_sent": net_counters.packets_sent,
                    "packets_recv": net_counters.packets_recv
                })
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        with self._lock:
            latest_cpu = self.cpu_history[-1] if self.cpu_history else {}
            latest_memory = self.memory_history[-1] if self.memory_history else {}
            latest_io = self.io_history[-1] if self.io_history else {}
            latest_network = self.network_history[-1] if self.network_history else {}
        
        return {
            "cpu": latest_cpu,
            "memory": latest_memory,
            "io": latest_io,
            "network": latest_network,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_usage_trends(self, minutes: int = 60) -> Dict[str, Any]:
        """Get resource usage trends over the specified period."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            # Filter recent data
            recent_cpu = [h for h in self.cpu_history if h["timestamp"] >= cutoff_time]
            recent_memory = [h for h in self.memory_history if h["timestamp"] >= cutoff_time]
            recent_io = [h for h in self.io_history if h["timestamp"] >= cutoff_time]
            recent_network = [h for h in self.network_history if h["timestamp"] >= cutoff_time]
        
        trends = {}
        
        # CPU trends
        if recent_cpu:
            cpu_values = [h["cpu_percent"] for h in recent_cpu]
            trends["cpu"] = {
                "avg": statistics.mean(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "trend": self._calculate_trend(cpu_values)
            }
        
        # Memory trends
        if recent_memory:
            memory_values = [h["percent"] for h in recent_memory]
            trends["memory"] = {
                "avg": statistics.mean(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "trend": self._calculate_trend(memory_values)
            }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        diff_percent = (second_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0
        
        if diff_percent > 5:
            return "increasing"
        elif diff_percent < -5:
            return "decreasing"
        else:
            return "stable"


class BottleneckDetector:
    """Detects performance bottlenecks in the system."""
    
    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self.detected_bottlenecks: List[Bottleneck] = []
        self.max_bottleneck_history = 1000
        
        # Detection thresholds
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 85.0  # %
        self.latency_threshold = 5000.0  # ms
        self.error_rate_threshold = 0.05  # 5%
        
        logger.info("Bottleneck detector initialized")
    
    def detect_bottlenecks(self, performance_profiles: Dict[str, PerformanceProfile]) -> List[Bottleneck]:
        """Detect current bottlenecks based on performance profiles."""
        bottlenecks = []
        current_time = datetime.utcnow()
        
        # Get current resource usage
        resource_usage = self.resource_monitor.get_current_usage()
        resource_trends = self.resource_monitor.get_usage_trends()
        
        # Check system-level bottlenecks
        system_bottlenecks = self._detect_system_bottlenecks(resource_usage, resource_trends)
        bottlenecks.extend(system_bottlenecks)
        
        # Check component-level bottlenecks
        for component, profile in performance_profiles.items():
            component_bottlenecks = self._detect_component_bottlenecks(component, profile)
            bottlenecks.extend(component_bottlenecks)
        
        # Store detected bottlenecks
        self.detected_bottlenecks.extend(bottlenecks)
        if len(self.detected_bottlenecks) > self.max_bottleneck_history:
            self.detected_bottlenecks = self.detected_bottlenecks[-self.max_bottleneck_history//2:]
        
        return bottlenecks
    
    def _detect_system_bottlenecks(self, usage: Dict[str, Any], trends: Dict[str, Any]) -> List[Bottleneck]:
        """Detect system-level bottlenecks."""
        bottlenecks = []
        
        # CPU bottleneck
        cpu_data = usage.get("cpu", {})
        if cpu_data.get("cpu_percent", 0) > self.cpu_threshold:
            severity = min(1.0, cpu_data["cpu_percent"] / 100.0)
            bottleneck = Bottleneck(
                bottleneck_id=f"cpu_bottleneck_{int(time.time())}",
                bottleneck_type=BottleneckType.CPU_BOUND,
                component="system",
                severity=severity,
                description=f"High CPU usage: {cpu_data['cpu_percent']:.1f}%",
                metrics={"cpu_percent": cpu_data["cpu_percent"]},
                suggested_actions=[
                    "Consider reducing concurrent processing",
                    "Optimize compute-intensive operations",
                    "Scale horizontally if possible"
                ]
            )
            bottlenecks.append(bottleneck)
        
        # Memory bottleneck
        memory_data = usage.get("memory", {})
        if memory_data.get("percent", 0) > self.memory_threshold:
            severity = min(1.0, memory_data["percent"] / 100.0)
            bottleneck = Bottleneck(
                bottleneck_id=f"memory_bottleneck_{int(time.time())}",
                bottleneck_type=BottleneckType.MEMORY_BOUND,
                component="system",
                severity=severity,
                description=f"High memory usage: {memory_data['percent']:.1f}%",
                metrics={"memory_percent": memory_data["percent"]},
                suggested_actions=[
                    "Reduce cache sizes",
                    "Optimize memory usage in components",
                    "Consider garbage collection tuning"
                ]
            )
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_component_bottlenecks(self, component: str, profile: PerformanceProfile) -> List[Bottleneck]:
        """Detect component-level bottlenecks."""
        bottlenecks = []
        
        # High latency bottleneck
        if profile.avg_latency_ms > self.latency_threshold:
            severity = min(1.0, profile.avg_latency_ms / (self.latency_threshold * 2))
            bottleneck = Bottleneck(
                bottleneck_id=f"{component}_latency_{int(time.time())}",
                bottleneck_type=BottleneckType.DETECTOR_LATENCY,
                component=component,
                severity=severity,
                description=f"High average latency: {profile.avg_latency_ms:.1f}ms",
                metrics={
                    "avg_latency_ms": profile.avg_latency_ms,
                    "p95_latency_ms": profile.p95_latency_ms
                },
                suggested_actions=[
                    "Optimize component processing logic",
                    "Add caching for expensive operations",
                    "Consider parallel processing"
                ]
            )
            bottlenecks.append(bottleneck)
        
        # High error rate bottleneck
        if profile.error_rate > self.error_rate_threshold:
            severity = min(1.0, profile.error_rate / (self.error_rate_threshold * 4))
            bottleneck = Bottleneck(
                bottleneck_id=f"{component}_errors_{int(time.time())}",
                bottleneck_type=BottleneckType.THREAD_CONTENTION,
                component=component,
                severity=severity,
                description=f"High error rate: {profile.error_rate:.1%}",
                metrics={"error_rate": profile.error_rate},
                suggested_actions=[
                    "Investigate error causes",
                    "Add error handling and retries",
                    "Check resource availability"
                ]
            )
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def get_recent_bottlenecks(self, hours: int = 24) -> List[Bottleneck]:
        """Get bottlenecks detected in the recent time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [b for b in self.detected_bottlenecks if b.timestamp >= cutoff_time]


class AdaptiveOptimizer:
    """Adaptive performance optimizer that adjusts system parameters."""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        self.optimization_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        # Current optimization parameters
        self.current_params = {
            "thread_pool_size": min(32, multiprocessing.cpu_count() * 2),
            "cache_size_mb": 100,
            "batch_size": 10,
            "timeout_seconds": 30,
            "gc_threshold": 1000
        }
        
        # Performance baselines
        self.baselines: Dict[str, float] = {}
        
        logger.info(f"Adaptive optimizer initialized with strategy: {strategy.value}")
    
    def optimize_based_on_bottlenecks(self, bottlenecks: List[Bottleneck], 
                                    current_performance: Dict[str, PerformanceProfile]) -> Dict[str, Any]:
        """Optimize system parameters based on detected bottlenecks."""
        optimizations = {}
        
        for bottleneck in bottlenecks:
            optimization = self._create_optimization_for_bottleneck(bottleneck)
            if optimization:
                optimizations.update(optimization)
        
        # Apply conservative limits based on strategy
        optimizations = self._apply_strategy_limits(optimizations)
        
        # Record optimization
        self._record_optimization(bottlenecks, optimizations, current_performance)
        
        return optimizations
    
    def _create_optimization_for_bottleneck(self, bottleneck: Bottleneck) -> Optional[Dict[str, Any]]:
        """Create specific optimization for a bottleneck type."""
        optimization = {}
        
        if bottleneck.bottleneck_type == BottleneckType.CPU_BOUND:
            # Reduce parallelism to decrease CPU load
            new_thread_count = max(1, int(self.current_params["thread_pool_size"] * 0.8))
            optimization["thread_pool_size"] = new_thread_count
            
        elif bottleneck.bottleneck_type == BottleneckType.MEMORY_BOUND:
            # Reduce cache size and batch size
            new_cache_size = max(10, int(self.current_params["cache_size_mb"] * 0.7))
            optimization["cache_size_mb"] = new_cache_size
            optimization["batch_size"] = max(1, int(self.current_params["batch_size"] * 0.5))
            
        elif bottleneck.bottleneck_type == BottleneckType.DETECTOR_LATENCY:
            # Increase timeout and potentially add more parallelism
            if bottleneck.severity > 0.7:
                optimization["timeout_seconds"] = min(60, self.current_params["timeout_seconds"] * 1.5)
                optimization["thread_pool_size"] = min(64, int(self.current_params["thread_pool_size"] * 1.2))
            
        elif bottleneck.bottleneck_type == BottleneckType.CACHE_MISS:
            # Increase cache size
            optimization["cache_size_mb"] = min(500, int(self.current_params["cache_size_mb"] * 1.5))
            
        return optimization if optimization else None
    
    def _apply_strategy_limits(self, optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strategy-specific limits to optimizations."""
        limited_optimizations = {}
        
        for param, value in optimizations.items():
            current_value = self.current_params.get(param, value)
            
            if self.strategy == OptimizationStrategy.CONSERVATIVE:
                # Limit changes to 20%
                max_change = abs(current_value * 0.2)
                if abs(value - current_value) <= max_change:
                    limited_optimizations[param] = value
                    
            elif self.strategy == OptimizationStrategy.BALANCED:
                # Limit changes to 50%
                max_change = abs(current_value * 0.5)
                if abs(value - current_value) <= max_change:
                    limited_optimizations[param] = value
                    
            elif self.strategy == OptimizationStrategy.AGGRESSIVE:
                # Allow larger changes but within reasonable bounds
                limited_optimizations[param] = value
                
            elif self.strategy == OptimizationStrategy.ADAPTIVE:
                # Use historical performance to determine limits
                if self._is_safe_optimization(param, value):
                    limited_optimizations[param] = value
        
        return limited_optimizations
    
    def _is_safe_optimization(self, param: str, value: Any) -> bool:
        """Check if an optimization is safe based on historical data."""
        # Simple heuristic - can be made more sophisticated
        current_value = self.current_params.get(param, value)
        
        # Don't allow extreme changes
        if isinstance(value, (int, float)) and isinstance(current_value, (int, float)):
            change_ratio = abs(value - current_value) / max(abs(current_value), 1)
            return change_ratio <= 1.0  # Max 100% change
        
        return True
    
    def _record_optimization(self, bottlenecks: List[Bottleneck], 
                           optimizations: Dict[str, Any],
                           performance: Dict[str, PerformanceProfile]) -> None:
        """Record optimization attempt for future learning."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "bottlenecks": [b.to_dict() for b in bottlenecks],
            "optimizations": optimizations,
            "performance_before": {k: v.to_dict() for k, v in performance.items()},
            "strategy": self.strategy.value
        }
        
        self.optimization_history.append(record)
        
        # Trim history
        if len(self.optimization_history) > self.max_history:
            self.optimization_history = self.optimization_history[-self.max_history//2:]
    
    def apply_optimizations(self, optimizations: Dict[str, Any]) -> None:
        """Apply optimizations to current parameters."""
        for param, value in optimizations.items():
            old_value = self.current_params.get(param)
            self.current_params[param] = value
            logger.info(f"Optimization applied: {param} {old_value} -> {value}")
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current optimization parameters."""
        return self.current_params.copy()


class PerformanceOptimizer:
    """Main performance optimization manager."""
    
    def __init__(self, 
                 metric_collector: Optional[MetricCollector] = None,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        
        self.metric_collector = metric_collector
        self.resource_monitor = ResourceMonitor()
        self.bottleneck_detector = BottleneckDetector(self.resource_monitor)
        self.adaptive_optimizer = AdaptiveOptimizer(optimization_strategy)
        
        # Caching systems
        self.filter_cache = FilterResultCache(max_size_bytes=100*1024*1024)  # 100MB
        self.general_cache = IntelligentCache(max_size_bytes=50*1024*1024)   # 50MB
        
        # Thread pools for different types of work
        self.io_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="io")
        self.cpu_executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count(), thread_name_prefix="cpu")
        
        # Performance tracking
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.optimization_active = False
        self.optimization_thread: Optional[threading.Thread] = None
        self.optimization_interval = 60  # seconds
        
        logger.info("Performance optimizer initialized")
    
    def start_optimization(self) -> None:
        """Start performance optimization."""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        # Start cache background tasks
        self.filter_cache.start_background_cleanup()
        self.general_cache.start_background_cleanup()
        
        # Start optimization loop
        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        logger.info("Performance optimization started")
    
    def stop_optimization(self) -> None:
        """Stop performance optimization."""
        self.optimization_active = False
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        
        # Stop cache background tasks
        self.filter_cache.stop_background_cleanup()
        self.general_cache.stop_background_cleanup()
        
        # Stop optimization loop
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
        logger.info("Performance optimization stopped")
    
    def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self.optimization_active:
            try:
                self._run_optimization_cycle()
                time.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _run_optimization_cycle(self) -> None:
        """Run one optimization cycle."""
        # Detect bottlenecks
        bottlenecks = self.bottleneck_detector.detect_bottlenecks(self.performance_profiles)
        
        if bottlenecks:
            logger.info(f"Detected {len(bottlenecks)} bottlenecks")
            
            # Generate optimizations
            optimizations = self.adaptive_optimizer.optimize_based_on_bottlenecks(
                bottlenecks, self.performance_profiles
            )
            
            if optimizations:
                logger.info(f"Applying optimizations: {optimizations}")
                
                # Apply optimizations
                self._apply_system_optimizations(optimizations)
                self.adaptive_optimizer.apply_optimizations(optimizations)
                
                # Record metrics
                if self.metric_collector:
                    self.metric_collector.record_counter("performance.optimizations_applied")
                    for param, value in optimizations.items():
                        self.metric_collector.record_gauge(f"performance.param.{param}", float(value))
    
    def _apply_system_optimizations(self, optimizations: Dict[str, Any]) -> None:
        """Apply optimizations to system components."""
        # Update thread pool sizes
        if "thread_pool_size" in optimizations:
            new_size = optimizations["thread_pool_size"]
            
            # Update CPU executor
            old_cpu_executor = self.cpu_executor
            self.cpu_executor = ThreadPoolExecutor(max_workers=new_size, thread_name_prefix="cpu")
            old_cpu_executor.shutdown(wait=False)
        
        # Update cache sizes
        if "cache_size_mb" in optimizations:
            new_size_bytes = optimizations["cache_size_mb"] * 1024 * 1024
            self.filter_cache.max_size_bytes = new_size_bytes
            self.general_cache.max_size_bytes = new_size_bytes // 2
        
        # Update batch sizes and timeouts would be applied to specific components
        # This would be implemented based on the specific architecture
    
    def optimize_filter_operation(self, request: FilterRequest, 
                                 filter_func: Callable[[FilterRequest], FilterResult]) -> FilterResult:
        """Optimize a filter operation with caching and resource management."""
        start_time = time.time()
        
        # Check cache first
        cached_result = self.filter_cache.get_cached_result(request)
        if cached_result is not None:
            # Record cache hit
            if self.metric_collector:
                self.metric_collector.record_counter("performance.cache_hits")
                self.metric_collector.record_timer("performance.cache_lookup_ms", (time.time() - start_time) * 1000)
            
            return cached_result
        
        # Execute filter operation
        try:
            # Use appropriate executor based on operation type
            future = self.cpu_executor.submit(filter_func, request)
            result = future.result(timeout=30)  # 30 second timeout
            
            # Cache the result
            self.filter_cache.cache_filter_result(request, result)
            
            # Record metrics
            processing_time = (time.time() - start_time) * 1000
            if self.metric_collector:
                self.metric_collector.record_counter("performance.cache_misses")
                self.metric_collector.record_timer("performance.filter_latency_ms", processing_time)
            
            # Update performance profile
            self._update_performance_profile("filter", processing_time, error=False)
            
            return result
            
        except Exception as e:
            # Record error
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_profile("filter", processing_time, error=True)
            
            if self.metric_collector:
                self.metric_collector.record_counter("performance.filter_errors")
            
            raise
    
    def _update_performance_profile(self, component: str, latency_ms: float, error: bool = False) -> None:
        """Update performance profile for a component."""
        if component not in self.performance_profiles:
            self.performance_profiles[component] = PerformanceProfile(
                component=component,
                avg_latency_ms=latency_ms,
                p95_latency_ms=latency_ms,
                p99_latency_ms=latency_ms,
                throughput_qps=0.0,
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                error_rate=0.0
            )
        
        profile = self.performance_profiles[component]
        
        # Update latency (simple moving average)
        profile.avg_latency_ms = (profile.avg_latency_ms * 0.9) + (latency_ms * 0.1)
        profile.p95_latency_ms = max(profile.p95_latency_ms * 0.95, latency_ms)
        profile.p99_latency_ms = max(profile.p99_latency_ms * 0.99, latency_ms)
        
        # Update error rate
        if error:
            profile.error_rate = (profile.error_rate * 0.9) + 0.1
        else:
            profile.error_rate = profile.error_rate * 0.9
        
        profile.timestamp = datetime.utcnow()
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            "optimization_active": self.optimization_active,
            "current_parameters": self.adaptive_optimizer.get_current_parameters(),
            "performance_profiles": {k: v.to_dict() for k, v in self.performance_profiles.items()},
            "recent_bottlenecks": [b.to_dict() for b in self.bottleneck_detector.get_recent_bottlenecks(hours=1)],
            "cache_stats": {
                "filter_cache": self.filter_cache.get_stats(),
                "general_cache": self.general_cache.get_stats()
            },
            "resource_usage": self.resource_monitor.get_current_usage(),
            "resource_trends": self.resource_monitor.get_usage_trends()
        }
    
    def cleanup(self) -> None:
        """Cleanup optimization resources."""
        self.stop_optimization()
        
        # Shutdown executors
        self.io_executor.shutdown(wait=True)
        self.cpu_executor.shutdown(wait=True)
        
        # Cleanup caches
        self.filter_cache.cleanup()
        self.general_cache.cleanup()
        
        logger.info("Performance optimizer cleaned up")