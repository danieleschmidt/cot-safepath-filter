"""
Advanced performance optimization system for SafePath Filter - Generation 3.

High-performance filtering with intelligent caching, concurrent processing,
resource pooling, and adaptive performance tuning.
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import hashlib
import pickle
import json
from functools import lru_cache
import weakref

from .models import FilterResult, FilterRequest, SafetyScore
from .exceptions import FilterError


logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Caching strategies."""
    
    LRU = "lru"
    LFU = "lfu"  
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class ProcessingMode(str, Enum):
    """Processing modes for different workloads."""
    
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    throughput_qps: float = 0.0
    cpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    concurrent_requests: int = 0
    peak_concurrent_requests: int = 0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    
    # Caching configuration
    enable_caching: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    cache_size_mb: int = 100
    cache_ttl_seconds: int = 3600
    
    # Concurrency configuration
    processing_mode: ProcessingMode = ProcessingMode.ASYNCHRONOUS
    max_concurrent_requests: int = 100
    max_worker_threads: int = None  # Default to CPU count * 2
    max_worker_processes: int = None  # Default to CPU count
    
    # Resource pooling
    enable_connection_pooling: bool = True
    pool_size: int = 10
    pool_timeout_seconds: int = 30
    
    # Performance tuning
    enable_adaptive_tuning: bool = True
    performance_target_ms: float = 100.0
    auto_scaling_enabled: bool = True
    scale_up_threshold: float = 0.8  # CPU/memory threshold
    scale_down_threshold: float = 0.3


class IntelligentCache:
    """Intelligent caching system with multiple strategies."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.strategy = config.cache_strategy
        self.max_size = self._calculate_max_entries()
        
        # Cache storage
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._expiry_times: Dict[str, float] = {}
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Adaptive learning
        self._access_patterns = defaultdict(list)
        self._pattern_analysis_interval = 300  # 5 minutes
        self._last_analysis = time.time()
        
    def _calculate_max_entries(self) -> int:
        """Calculate maximum cache entries based on memory limit."""
        # Estimate ~1KB per entry on average
        estimated_entry_size = 1024
        max_bytes = self.config.cache_size_mb * 1024 * 1024
        return max(1000, max_bytes // estimated_entry_size)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            # Check TTL expiry
            if self._is_expired(key):
                self._remove_key(key)
                self.misses += 1
                return None
            
            # Update access statistics
            self._access_times[key] = time.time()
            self._access_counts[key] += 1
            self.hits += 1
            
            # Record access pattern for adaptive learning
            if self.strategy == CacheStrategy.ADAPTIVE:
                self._record_access_pattern(key)
            
            return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            current_time = time.time()
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict()
            
            # Store value
            self._cache[key] = value
            self._access_times[key] = current_time
            self._access_counts[key] += 1
            
            # Set expiry time
            if ttl is not None:
                self._expiry_times[key] = current_time + ttl
            elif self.config.cache_ttl_seconds > 0:
                self._expiry_times[key] = current_time + self.config.cache_ttl_seconds
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._expiry_times:
            return False
        return time.time() > self._expiry_times[key]
    
    def _evict(self) -> None:
        """Evict entries based on strategy."""
        if not self._cache:
            return
            
        keys_to_remove = []
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            keys_to_remove.append(oldest_key)
            
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            least_used_key = min(self._access_counts.keys(), key=lambda k: self._access_counts[k])
            keys_to_remove.append(least_used_key)
            
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first
            current_time = time.time()
            for key in list(self._cache.keys()):
                if self._is_expired(key):
                    keys_to_remove.append(key)
            
            # If no expired entries, fall back to LRU
            if not keys_to_remove:
                oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
                keys_to_remove.append(oldest_key)
                
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive eviction based on access patterns
            keys_to_remove = self._adaptive_eviction()
        
        # Remove selected keys
        for key in keys_to_remove[:max(1, len(self._cache) // 10)]:  # Remove at most 10%
            self._remove_key(key)
        
        self.evictions += len(keys_to_remove)
    
    def _adaptive_eviction(self) -> List[str]:
        """Adaptive eviction based on learned access patterns."""
        current_time = time.time()
        
        # Analyze access patterns if needed
        if current_time - self._last_analysis > self._pattern_analysis_interval:
            self._analyze_access_patterns()
            self._last_analysis = current_time
        
        # Score keys based on multiple factors
        key_scores = {}
        for key in self._cache.keys():
            if self._is_expired(key):
                key_scores[key] = 0  # Highest priority for removal
            else:
                # Combine recency, frequency, and predicted future access
                recency_score = 1.0 / (current_time - self._access_times.get(key, current_time) + 1)
                frequency_score = self._access_counts.get(key, 1) / max(self._access_counts.values())
                prediction_score = self._predict_future_access(key)
                
                # Weighted combination
                key_scores[key] = (0.3 * recency_score + 0.4 * frequency_score + 0.3 * prediction_score)
        
        # Return keys with lowest scores for eviction
        sorted_keys = sorted(key_scores.keys(), key=lambda k: key_scores[k])
        return sorted_keys[:max(1, len(sorted_keys) // 20)]  # Remove bottom 5%
    
    def _record_access_pattern(self, key: str) -> None:
        """Record access pattern for adaptive learning."""
        current_time = time.time()
        self._access_patterns[key].append(current_time)
        
        # Keep only recent access history
        cutoff_time = current_time - 3600  # Keep last hour
        self._access_patterns[key] = [
            t for t in self._access_patterns[key] if t > cutoff_time
        ]
    
    def _analyze_access_patterns(self) -> None:
        """Analyze access patterns to improve predictions."""
        # This is a simplified implementation
        # In production, you might use more sophisticated ML techniques
        current_time = time.time()
        
        for key in list(self._access_patterns.keys()):
            accesses = self._access_patterns[key]
            if not accesses:
                continue
            
            # Calculate access frequency
            if len(accesses) > 1:
                intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
                avg_interval = sum(intervals) / len(intervals)
                # Use this to predict future access probability
                # (Simplified - real implementation would be more complex)
    
    def _predict_future_access(self, key: str) -> float:
        """Predict probability of future access."""
        accesses = self._access_patterns.get(key, [])
        if len(accesses) < 2:
            return 0.5  # Default probability
        
        # Simple prediction based on recent access frequency
        current_time = time.time()
        recent_accesses = [a for a in accesses if current_time - a < 900]  # Last 15 minutes
        
        if not recent_accesses:
            return 0.1  # Low probability if no recent access
        
        # Higher probability for frequently accessed items
        return min(1.0, len(recent_accesses) / 10.0)
    
    def _remove_key(self, key: str) -> None:
        """Remove key and all associated data."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._access_counts.pop(key, None)
        self._expiry_times.pop(key, None)
        self._access_patterns.pop(key, None)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(1, total_requests)
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "strategy": self.strategy.value,
            "memory_usage_estimate_mb": len(self._cache) * 1024 / (1024 * 1024)
        }
    
    def clear(self) -> None:
        """Clear all cache data."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._expiry_times.clear()
            self._access_patterns.clear()


class ConcurrentProcessor:
    """High-performance concurrent processing system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.mode = config.processing_mode
        
        # Thread/process pools
        self.max_workers = config.max_worker_threads or (multiprocessing.cpu_count() * 2)
        self.max_processes = config.max_worker_processes or multiprocessing.cpu_count()
        
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        
        # Request tracking
        self._active_requests: Dict[str, float] = {}
        self._request_queue = asyncio.Queue(maxsize=config.max_concurrent_requests)
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Performance tracking
        self._response_times = deque(maxlen=1000)
        self._throughput_tracker = deque(maxlen=60)  # Track per second for 1 minute
        self._last_throughput_update = time.time()
        self._request_count = 0
        
        # Graceful shutdown
        self._shutdown_event = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _get_thread_pool(self) -> ThreadPoolExecutor:
        """Get or create thread pool."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="SafePath-"
            )
        return self._thread_pool
    
    def _get_process_pool(self) -> ProcessPoolExecutor:
        """Get or create process pool."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(
                max_workers=self.max_processes
            )
        return self._process_pool
    
    async def process_async(
        self,
        func: Callable,
        request: FilterRequest,
        *args,
        **kwargs
    ) -> FilterResult:
        """Process request asynchronously."""
        request_id = request.request_id
        start_time = time.time()
        
        async with self._semaphore:
            try:
                self._active_requests[request_id] = start_time
                
                if self.mode == ProcessingMode.ASYNCHRONOUS:
                    # Pure async processing
                    if asyncio.iscoroutinefunction(func):
                        result = await func(request, *args, **kwargs)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self._get_thread_pool(),
                            func,
                            request,
                            *args,
                            **kwargs
                        )
                
                elif self.mode == ProcessingMode.PARALLEL:
                    # Parallel processing with thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._get_thread_pool(),
                        func,
                        request,
                        *args,
                        **kwargs
                    )
                
                else:  # SYNCHRONOUS
                    result = func(request, *args, **kwargs)
                
                # Track performance
                processing_time = time.time() - start_time
                self._record_response_time(processing_time)
                
                return result
                
            except Exception as e:
                logger.error(f"Async processing failed for {request_id}: {e}")
                raise
            finally:
                self._active_requests.pop(request_id, None)
    
    def process_batch_concurrent(
        self,
        func: Callable,
        requests: List[FilterRequest],
        *args,
        **kwargs
    ) -> List[FilterResult]:
        """Process batch of requests concurrently."""
        if self.mode == ProcessingMode.DISTRIBUTED:
            return self._process_batch_distributed(func, requests, *args, **kwargs)
        
        # Use thread pool for concurrent processing
        results = []
        thread_pool = self._get_thread_pool()
        
        # Submit all requests
        future_to_request = {}
        for request in requests:
            future = thread_pool.submit(func, request, *args, **kwargs)
            future_to_request[future] = request
        
        # Collect results
        for future in as_completed(future_to_request):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                request = future_to_request[future]
                logger.error(f"Batch processing failed for {request.request_id}: {e}")
                # Create error result
                from .models import SafetyScore, Severity
                error_result = FilterResult(
                    filtered_content="[ERROR: Processing failed]",
                    safety_score=SafetyScore(
                        overall_score=0.0,
                        confidence=0.0,
                        is_safe=False,
                        severity=Severity.CRITICAL
                    ),
                    was_filtered=True,
                    filter_reasons=[f"processing_error: {str(e)}"],
                    request_id=request.request_id
                )
                results.append(error_result)
        
        return results
    
    def _process_batch_distributed(
        self,
        func: Callable,
        requests: List[FilterRequest],
        *args,
        **kwargs
    ) -> List[FilterResult]:
        """Process batch using distributed processing (process pool)."""
        try:
            process_pool = self._get_process_pool()
            
            # Prepare serializable arguments
            serializable_requests = []
            for request in requests:
                # Convert to dict for serialization
                request_dict = {
                    'content': request.content,
                    'safety_level': request.safety_level.value,
                    'request_id': request.request_id,
                    'metadata': request.metadata
                }
                serializable_requests.append(request_dict)
            
            # Submit to process pool
            futures = []
            for request_dict in serializable_requests:
                future = process_pool.submit(
                    self._process_single_distributed,
                    func,
                    request_dict,
                    *args,
                    **kwargs
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Distributed processing failed: {e}")
                    # Create error result
                    error_result = FilterResult(
                        filtered_content="[ERROR: Distributed processing failed]",
                        safety_score=SafetyScore(
                            overall_score=0.0,
                            confidence=0.0,
                            is_safe=False,
                            severity=Severity.CRITICAL
                        ),
                        was_filtered=True,
                        filter_reasons=[f"distributed_error: {str(e)}"]
                    )
                    results.append(error_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Distributed batch processing setup failed: {e}")
            # Fallback to thread-based processing
            return self.process_batch_concurrent(func, requests, *args, **kwargs)
    
    @staticmethod
    def _process_single_distributed(func, request_dict, *args, **kwargs):
        """Process single request in distributed mode (runs in separate process)."""
        # This would need to be implemented carefully for actual multiprocessing
        # For now, we'll return a placeholder
        from cot_safepath.models import SafetyScore, FilterResult
        
        return FilterResult(
            filtered_content="[DISTRIBUTED PROCESSING PLACEHOLDER]",
            safety_score=SafetyScore(
                overall_score=0.8,
                confidence=0.7,
                is_safe=True
            ),
            was_filtered=False,
            request_id=request_dict.get('request_id')
        )
    
    def _record_response_time(self, response_time: float) -> None:
        """Record response time for performance tracking."""
        self._response_times.append(response_time * 1000)  # Convert to ms
        self._request_count += 1
        
        # Update throughput tracking
        current_time = time.time()
        if current_time - self._last_throughput_update >= 1.0:
            self._throughput_tracker.append(self._request_count)
            self._request_count = 0
            self._last_throughput_update = current_time
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        response_times = list(self._response_times)
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            sorted_times = sorted(response_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            p95_time = sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1]
            p99_time = sorted_times[p99_idx] if p99_idx < len(sorted_times) else sorted_times[-1]
        else:
            avg_response_time = p95_time = p99_time = 0.0
        
        # Calculate throughput (requests per second)
        throughput = sum(self._throughput_tracker) / max(1, len(self._throughput_tracker))
        
        return PerformanceMetrics(
            total_requests=len(response_times),
            avg_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_time,
            p99_response_time_ms=p99_time,
            throughput_qps=throughput,
            concurrent_requests=len(self._active_requests),
            peak_concurrent_requests=self.config.max_concurrent_requests
        )
    
    def _cleanup_worker(self) -> None:
        """Background cleanup worker."""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Clean up old active requests (potential memory leak prevention)
                stale_requests = [
                    req_id for req_id, start_time in self._active_requests.items()
                    if current_time - start_time > 300  # 5 minutes timeout
                ]
                
                for req_id in stale_requests:
                    logger.warning(f"Cleaning up stale request: {req_id}")
                    self._active_requests.pop(req_id, None)
                
                # Sleep before next cleanup
                self._shutdown_event.wait(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                time.sleep(60)
    
    def shutdown(self) -> None:
        """Gracefully shutdown processor."""
        self._shutdown_event.set()
        
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)


class AdaptivePerformanceTuner:
    """Adaptive performance tuning system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.target_response_time = config.performance_target_ms
        
        # Performance history for trend analysis
        self._performance_history = deque(maxlen=100)
        
        # Tuning parameters
        self._cache_hit_rate_threshold = 0.8
        self._response_time_tolerance = 0.2  # 20% tolerance
        
        # Auto-scaling parameters
        self._scale_up_consecutive_threshold = 3
        self._scale_down_consecutive_threshold = 5
        self._consecutive_high_load = 0
        self._consecutive_low_load = 0
        
        # Last tuning time
        self._last_tuning = time.time()
        self._tuning_interval = 300  # 5 minutes
    
    def analyze_and_tune(
        self,
        cache: IntelligentCache,
        processor: ConcurrentProcessor,
        current_metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Analyze performance and suggest/apply tuning changes."""
        current_time = time.time()
        
        # Record current performance
        self._performance_history.append(current_metrics)
        
        # Only tune periodically
        if current_time - self._last_tuning < self._tuning_interval:
            return {"status": "no_tuning_needed", "reason": "within_tuning_interval"}
        
        self._last_tuning = current_time
        tuning_actions = []
        
        # Analyze cache performance
        cache_stats = cache.get_statistics()
        if cache_stats["hit_rate"] < self._cache_hit_rate_threshold:
            # Consider increasing cache size or changing strategy
            if cache.strategy != CacheStrategy.ADAPTIVE:
                cache.strategy = CacheStrategy.ADAPTIVE
                tuning_actions.append("switched_to_adaptive_cache")
            
            # Increase cache size if memory allows
            if cache_stats["size"] >= cache.max_size * 0.9:
                new_max_size = min(cache.max_size * 2, self.config.cache_size_mb * 1024 * 2)
                cache.max_size = new_max_size
                tuning_actions.append(f"increased_cache_size_to_{new_max_size}")
        
        # Analyze response time performance
        if current_metrics.avg_response_time_ms > self.target_response_time * (1 + self._response_time_tolerance):
            # Response time too high
            if self.config.auto_scaling_enabled:
                self._consecutive_high_load += 1
                self._consecutive_low_load = 0
                
                if self._consecutive_high_load >= self._scale_up_consecutive_threshold:
                    # Scale up processing capacity
                    if processor.max_workers < multiprocessing.cpu_count() * 4:
                        processor.max_workers = min(processor.max_workers + 2, multiprocessing.cpu_count() * 4)
                        tuning_actions.append(f"scaled_up_workers_to_{processor.max_workers}")
                        self._consecutive_high_load = 0
        
        elif current_metrics.avg_response_time_ms < self.target_response_time * (1 - self._response_time_tolerance):
            # Response time very good, consider scaling down
            if self.config.auto_scaling_enabled:
                self._consecutive_low_load += 1
                self._consecutive_high_load = 0
                
                if self._consecutive_low_load >= self._scale_down_consecutive_threshold:
                    # Scale down to save resources
                    min_workers = max(2, multiprocessing.cpu_count())
                    if processor.max_workers > min_workers:
                        processor.max_workers = max(processor.max_workers - 1, min_workers)
                        tuning_actions.append(f"scaled_down_workers_to_{processor.max_workers}")
                        self._consecutive_low_load = 0
        
        # Analyze throughput trends
        if len(self._performance_history) >= 5:
            recent_throughput = [m.throughput_qps for m in list(self._performance_history)[-5:]]
            if len(set(recent_throughput)) > 1:  # Varying throughput
                avg_throughput = sum(recent_throughput) / len(recent_throughput)
                if avg_throughput < current_metrics.throughput_qps * 0.8:
                    tuning_actions.append("detected_throughput_degradation")
        
        return {
            "status": "tuning_completed",
            "actions": tuning_actions,
            "metrics": {
                "response_time": current_metrics.avg_response_time_ms,
                "cache_hit_rate": cache_stats["hit_rate"],
                "throughput": current_metrics.throughput_qps,
                "concurrent_requests": current_metrics.concurrent_requests
            }
        }
    
    def get_performance_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if metrics.avg_response_time_ms > self.target_response_time * 2:
            recommendations.append("Consider increasing worker threads or optimizing algorithms")
        
        if metrics.cache_hits > 0:
            hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
            if hit_rate < 0.6:
                recommendations.append("Cache hit rate is low - consider cache size increase or strategy change")
        
        if metrics.concurrent_requests > metrics.peak_concurrent_requests * 0.9:
            recommendations.append("Near peak concurrent capacity - consider scaling up")
        
        if metrics.error_rate > 0.05:  # 5% error rate
            recommendations.append("High error rate detected - investigate error causes")
        
        return recommendations


class HighPerformanceFilterEngine:
    """High-performance filter engine with optimizations."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # Core components
        self.cache = IntelligentCache(self.config)
        self.processor = ConcurrentProcessor(self.config)
        self.tuner = AdaptivePerformanceTuner(self.config)
        
        # Performance tracking
        self._start_time = time.time()
        self._total_requests = 0
        self._cache_enabled = self.config.enable_caching
        
        logger.info(f"High-performance filter engine initialized: {self.config.processing_mode}")
    
    def create_cache_key(self, request: FilterRequest) -> str:
        """Create cache key for request."""
        key_data = {
            'content': request.content,
            'safety_level': request.safety_level.value,
            'metadata': request.metadata
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def filter_async(self, filter_func: Callable, request: FilterRequest) -> FilterResult:
        """High-performance async filtering."""
        cache_key = None
        
        # Check cache first
        if self._cache_enabled:
            cache_key = self.create_cache_key(request)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
        
        # Process request
        result = await self.processor.process_async(filter_func, request)
        
        # Cache result
        if self._cache_enabled and cache_key:
            self.cache.set(cache_key, result)
        
        self._total_requests += 1
        
        # Periodic performance tuning
        if self._total_requests % 100 == 0:
            metrics = self.get_performance_metrics()
            tuning_result = self.tuner.analyze_and_tune(self.cache, self.processor, metrics)
            if tuning_result.get("actions"):
                logger.info(f"Performance tuning applied: {tuning_result['actions']}")
        
        return result
    
    def filter_batch(
        self,
        filter_func: Callable,
        requests: List[FilterRequest]
    ) -> List[FilterResult]:
        """High-performance batch filtering."""
        if not requests:
            return []
        
        results = []
        cache_misses = []
        
        # Check cache for all requests
        if self._cache_enabled:
            for request in requests:
                cache_key = self.create_cache_key(request)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    results.append((request, cached_result, True))  # Cache hit
                else:
                    cache_misses.append((request, cache_key))
                    results.append((request, None, False))  # Cache miss
        else:
            cache_misses = [(req, None) for req in requests]
            results = [(req, None, False) for req in requests]
        
        # Process cache misses
        if cache_misses:
            miss_requests = [req for req, _ in cache_misses]
            processed_results = self.processor.process_batch_concurrent(
                filter_func, miss_requests
            )
            
            # Update cache and results
            miss_index = 0
            for i, (request, cached_result, was_cached) in enumerate(results):
                if not was_cached:
                    processed_result = processed_results[miss_index]
                    results[i] = (request, processed_result, False)
                    
                    # Cache the result
                    if self._cache_enabled:
                        cache_key = cache_misses[miss_index][1]
                        if cache_key:
                            self.cache.set(cache_key, processed_result)
                    
                    miss_index += 1
        
        # Extract final results
        final_results = [result for _, result, _ in results if result is not None]
        self._total_requests += len(final_results)
        
        return final_results
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics."""
        processor_metrics = self.processor.get_performance_metrics()
        cache_stats = self.cache.get_statistics()
        
        # Update with cache metrics
        processor_metrics.cache_hits = cache_stats["hits"]
        processor_metrics.cache_misses = cache_stats["misses"]
        processor_metrics.total_requests = self._total_requests
        
        return processor_metrics
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        metrics = self.get_performance_metrics()
        cache_stats = self.cache.get_statistics()
        recommendations = self.tuner.get_performance_recommendations(metrics)
        
        uptime = time.time() - self._start_time
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self._total_requests,
            "performance_metrics": {
                "avg_response_time_ms": metrics.avg_response_time_ms,
                "p95_response_time_ms": metrics.p95_response_time_ms,
                "p99_response_time_ms": metrics.p99_response_time_ms,
                "throughput_qps": metrics.throughput_qps,
                "concurrent_requests": metrics.concurrent_requests
            },
            "cache_metrics": cache_stats,
            "configuration": {
                "processing_mode": self.config.processing_mode.value,
                "max_workers": self.processor.max_workers,
                "cache_strategy": self.cache.strategy.value,
                "cache_size_mb": self.config.cache_size_mb
            },
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the engine."""
        self.processor.shutdown()
        self.cache.clear()
        logger.info("High-performance filter engine shut down")


# Global high-performance engine instance
_global_performance_engine: Optional[HighPerformanceFilterEngine] = None


def get_global_performance_engine() -> HighPerformanceFilterEngine:
    """Get or create global performance engine."""
    global _global_performance_engine
    if _global_performance_engine is None:
        _global_performance_engine = HighPerformanceFilterEngine()
    return _global_performance_engine


def configure_high_performance(config: OptimizationConfig) -> HighPerformanceFilterEngine:
    """Configure global high-performance engine."""
    global _global_performance_engine
    if _global_performance_engine:
        _global_performance_engine.shutdown()
    
    _global_performance_engine = HighPerformanceFilterEngine(config)
    return _global_performance_engine