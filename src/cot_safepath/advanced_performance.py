"""
Advanced performance optimization features for CoT SafePath Filter.
"""

import asyncio
import time
import gc
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Mock psutil for basic functionality
    class MockPsutil:
        class Process:
            def memory_info(self):
                class MemInfo:
                    rss = 1024 * 1024 * 100  # 100MB mock
                return MemInfo()
            
            def cpu_percent(self):
                return 50.0
        
        @staticmethod
        def virtual_memory():
            class VirtMem:
                total = 1024 * 1024 * 1024 * 8  # 8GB mock
                percent = 50.0
            return VirtMem()
        
        @staticmethod
        def cpu_percent():
            return 50.0
    
    psutil = MockPsutil()
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from .models import FilterRequest, FilterResult
from .exceptions import TimeoutError, CapacityError
from .performance import PerformanceConfig

logger = logging.getLogger(__name__)


@dataclass
class AdvancedPerformanceConfig:
    """Advanced configuration for performance optimizations."""
    
    # AI/ML Optimizations
    enable_model_quantization: bool = True
    enable_gpu_acceleration: bool = False
    enable_dynamic_batching: bool = True
    
    # Adaptive settings
    adaptive_timeout: bool = True
    adaptive_batch_sizing: bool = True
    adaptive_cache_sizing: bool = True
    
    # Memory optimization
    aggressive_gc: bool = True
    memory_pressure_threshold: float = 0.8
    
    # Monitoring
    performance_monitoring_interval: int = 60
    metrics_retention_hours: int = 24


class AdaptivePerformanceOptimizer:
    """Adaptive performance optimization based on system load."""
    
    def __init__(self, config: AdvancedPerformanceConfig):
        self.config = config
        self.performance_history = []
        self.adaptive_settings = {
            'timeout_multiplier': 1.0,
            'batch_size_multiplier': 1.0,
            'cache_size_multiplier': 1.0
        }
        self.last_optimization = time.time()
        
    def record_performance(self, processing_time_ms: float, memory_usage_mb: float, cpu_percent: float):
        """Record performance metrics for adaptive optimization."""
        self.performance_history.append({
            'timestamp': time.time(),
            'processing_time_ms': processing_time_ms,
            'memory_usage_mb': memory_usage_mb,
            'cpu_percent': cpu_percent
        })
        
        # Keep only last 100 measurements
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Trigger optimization every 30 seconds
        if time.time() - self.last_optimization > 30:
            self._optimize_settings()
            self.last_optimization = time.time()
    
    def _optimize_settings(self):
        """Optimize settings based on performance history."""
        if len(self.performance_history) < 10:
            return
        
        recent_metrics = self.performance_history[-10:]
        avg_processing_time = sum(m['processing_time_ms'] for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m['memory_usage_mb'] for m in recent_metrics) / len(recent_metrics)
        avg_cpu_usage = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        
        # Adjust timeout based on processing time
        if avg_processing_time > 500:  # Slow processing
            self.adaptive_settings['timeout_multiplier'] = min(2.0, self.adaptive_settings['timeout_multiplier'] * 1.1)
        elif avg_processing_time < 100:  # Fast processing
            self.adaptive_settings['timeout_multiplier'] = max(0.5, self.adaptive_settings['timeout_multiplier'] * 0.9)
        
        # Adjust batch size based on CPU usage
        if avg_cpu_usage > 70:  # High CPU usage
            self.adaptive_settings['batch_size_multiplier'] = max(0.5, self.adaptive_settings['batch_size_multiplier'] * 0.9)
        elif avg_cpu_usage < 30:  # Low CPU usage
            self.adaptive_settings['batch_size_multiplier'] = min(2.0, self.adaptive_settings['batch_size_multiplier'] * 1.1)
        
        # Adjust cache size based on memory usage
        if PSUTIL_AVAILABLE:
            system_memory = psutil.virtual_memory().total / 1024 / 1024  # MB
        else:
            system_memory = 8192  # 8GB fallback
        
        if avg_memory_usage > system_memory * self.config.memory_pressure_threshold:
            self.adaptive_settings['cache_size_multiplier'] = max(0.5, self.adaptive_settings['cache_size_multiplier'] * 0.9)
            if self.config.aggressive_gc:
                gc.collect()
        elif avg_memory_usage < system_memory * 0.4:
            self.adaptive_settings['cache_size_multiplier'] = min(2.0, self.adaptive_settings['cache_size_multiplier'] * 1.1)
        
        logger.info(f"Adaptive optimization: timeout={self.adaptive_settings['timeout_multiplier']:.2f}, "
                   f"batch_size={self.adaptive_settings['batch_size_multiplier']:.2f}, "
                   f"cache_size={self.adaptive_settings['cache_size_multiplier']:.2f}")
    
    def get_adaptive_timeout(self, base_timeout_ms: int) -> int:
        """Get adaptively adjusted timeout."""
        return int(base_timeout_ms * self.adaptive_settings['timeout_multiplier'])
    
    def get_adaptive_batch_size(self, base_batch_size: int) -> int:
        """Get adaptively adjusted batch size."""
        return max(1, int(base_batch_size * self.adaptive_settings['batch_size_multiplier']))
    
    def get_adaptive_cache_size(self, base_cache_size: int) -> int:
        """Get adaptively adjusted cache size."""
        return max(10, int(base_cache_size * self.adaptive_settings['cache_size_multiplier']))


class AsyncFilterProcessor:
    """Asynchronous filter processing with advanced optimizations."""
    
    def __init__(self, filter_func: Callable, config: PerformanceConfig):
        self.filter_func = filter_func
        self.config = config
        self.advanced_config = AdvancedPerformanceConfig()
        self.performance_optimizer = AdaptivePerformanceOptimizer(self.advanced_config)
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._processing_stats = {
            'total_processed': 0,
            'total_errors': 0,
            'avg_processing_time_ms': 0,
            'memory_efficiency_score': 0.0
        }
        self._start_time = time.time()
    
    @asynccontextmanager
    async def _resource_context(self):
        """Context manager for resource tracking."""
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            # Memory efficiency tracking
            end_memory = self._get_memory_usage()
            memory_delta = end_memory - start_memory
            
            if memory_delta > 0:
                # Update memory efficiency score (lower is better)
                self._processing_stats['memory_efficiency_score'] = (
                    (self._processing_stats['memory_efficiency_score'] * 
                     self._processing_stats['total_processed'] + memory_delta) /
                    (self._processing_stats['total_processed'] + 1)
                )
            
            # Aggressive garbage collection if enabled
            if self.advanced_config.aggressive_gc and memory_delta > 50:  # 50MB threshold
                gc.collect()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            else:
                return 100.0  # Mock value
        except:
            return 0.0
    
    async def process_async(self, request: FilterRequest) -> FilterResult:
        """Process filter request asynchronously with advanced optimizations."""
        async with self.semaphore:
            async with self._resource_context():
                start_time = time.time()
                
                try:
                    # Get adaptive timeout
                    base_timeout = getattr(self.config, 'filter_timeout_ms', 1000)
                    timeout_ms = self.performance_optimizer.get_adaptive_timeout(base_timeout)
                    
                    # Process with timeout
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self.filter_func, request
                        ),
                        timeout=timeout_ms / 1000.0
                    )
                    
                    # Record performance metrics
                    processing_time_ms = (time.time() - start_time) * 1000
                    memory_usage_mb = self._get_memory_usage()
                    if PSUTIL_AVAILABLE:
                        cpu_percent = psutil.cpu_percent()
                    else:
                        cpu_percent = 50.0  # Mock value
                    
                    self.performance_optimizer.record_performance(
                        processing_time_ms, memory_usage_mb, cpu_percent
                    )
                    
                    # Update stats
                    self._processing_stats['total_processed'] += 1
                    self._processing_stats['avg_processing_time_ms'] = (
                        (self._processing_stats['avg_processing_time_ms'] * 
                         (self._processing_stats['total_processed'] - 1) + processing_time_ms) /
                        self._processing_stats['total_processed']
                    )
                    
                    return result
                    
                except asyncio.TimeoutError:
                    self._processing_stats['total_errors'] += 1
                    raise TimeoutError(f"Filter processing timed out after {timeout_ms}ms")
                
                except Exception as e:
                    self._processing_stats['total_errors'] += 1
                    logger.error(f"Async processing failed: {e}")
                    raise
    
    async def process_batch_async(self, requests: List[FilterRequest]) -> List[FilterResult]:
        """Process multiple requests in parallel with intelligent batching."""
        if not requests:
            return []
        
        # Get adaptive batch size
        optimal_batch_size = self.performance_optimizer.get_adaptive_batch_size(
            len(requests)
        )
        
        # Split into optimal-sized batches
        batches = [requests[i:i + optimal_batch_size] 
                  for i in range(0, len(requests), optimal_batch_size)]
        
        all_results = []
        
        for batch in batches:
            # Process batch in parallel
            tasks = [self.process_async(request) for request in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to None or handle appropriately
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            all_results.extend(processed_results)
        
        return all_results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        uptime_hours = (time.time() - self._start_time) / 3600
        
        return {
            **self._processing_stats,
            'uptime_hours': uptime_hours,
            'requests_per_hour': self._processing_stats['total_processed'] / max(uptime_hours, 0.001),
            'error_rate': (self._processing_stats['total_errors'] / 
                          max(self._processing_stats['total_processed'], 1)),
            'adaptive_settings': self.performance_optimizer.adaptive_settings,
            'system_memory_mb': psutil.virtual_memory().total / 1024 / 1024 if PSUTIL_AVAILABLE else 8192,
            'system_cpu_count': psutil.cpu_count() if PSUTIL_AVAILABLE else 4,
        }


class IntelligentCacheManager:
    """Intelligent caching with predictive eviction and adaptive sizing."""
    
    def __init__(self, config: AdvancedPerformanceConfig):
        self.config = config
        self.cache = {}
        self.access_frequency = {}
        self.access_history = {}
        self.last_cleanup = time.time()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with usage tracking."""
        if key in self.cache:
            self.access_frequency[key] = self.access_frequency.get(key, 0) + 1
            self.access_history[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Put item in cache with intelligent eviction."""
        current_time = time.time()
        
        # Clean expired items first
        self._cleanup_expired()
        
        # Check if we need to evict items
        if len(self.cache) >= self._get_adaptive_cache_size():
            self._intelligent_eviction()
        
        self.cache[key] = {
            'value': value,
            'created_at': current_time,
            'ttl_seconds': ttl_seconds,
            'expires_at': current_time + ttl_seconds
        }
        self.access_frequency[key] = 1
        self.access_history[key] = current_time
    
    def _get_adaptive_cache_size(self) -> int:
        """Get adaptive cache size based on system resources."""
        base_size = 1000
        memory_usage = psutil.virtual_memory().percent
        
        if memory_usage > 80:
            return int(base_size * 0.5)
        elif memory_usage < 40:
            return int(base_size * 1.5)
        else:
            return base_size
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, data in self.cache.items()
            if current_time > data['expires_at']
        ]
        
        for key in expired_keys:
            self._remove_key(key)
    
    def _intelligent_eviction(self):
        """Intelligently evict cache entries based on usage patterns."""
        if not self.cache:
            return
        
        # Calculate eviction scores (lower = more likely to evict)
        eviction_scores = {}
        current_time = time.time()
        
        for key in self.cache:
            frequency = self.access_frequency.get(key, 1)
            last_access = self.access_history.get(key, current_time)
            time_since_access = current_time - last_access
            
            # Score based on frequency and recency (LFU + LRU hybrid)
            score = frequency / (1 + time_since_access / 3600)  # Frequency per hour
            eviction_scores[key] = score
        
        # Evict 25% of items with lowest scores
        items_to_evict = len(self.cache) // 4
        sorted_items = sorted(eviction_scores.items(), key=lambda x: x[1])
        
        for key, _ in sorted_items[:items_to_evict]:
            self._remove_key(key)
    
    def _remove_key(self, key: str):
        """Remove key and its associated data."""
        self.cache.pop(key, None)
        self.access_frequency.pop(key, None)
        self.access_history.pop(key, None)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_accesses = sum(self.access_frequency.values())
        
        return {
            'cache_size': len(self.cache),
            'total_accesses': total_accesses,
            'unique_keys': len(self.access_frequency),
            'average_access_frequency': total_accesses / max(len(self.access_frequency), 1),
            'memory_usage_mb': self._estimate_memory_usage(),
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate cache memory usage in MB."""
        # Rough estimation - in practice would use sys.getsizeof
        return len(self.cache) * 0.01  # 10KB per entry estimate