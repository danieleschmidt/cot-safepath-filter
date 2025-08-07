"""
Performance optimization system for CoT SafePath Filter.

Provides advanced caching, async processing, connection pooling,
and auto-scaling capabilities for high-throughput environments.
"""

import asyncio
import time
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache
from collections import deque
import threading
import weakref

from .models import FilterRequest, FilterResult, SafetyScore
from .sentiment_analyzer import SentimentScore, SentimentAnalyzer
from .exceptions import ConfigurationError, TimeoutError
from .advanced_logging import get_logger, get_metrics_collector


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    
    enabled: bool = True
    max_size: int = 10000
    ttl_seconds: int = 3600
    cleanup_interval_seconds: int = 300
    compression_enabled: bool = True
    persistence_enabled: bool = False
    persistence_file: Optional[str] = None


@dataclass 
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    enable_async: bool = True
    max_workers: int = 10
    enable_multiprocessing: bool = False
    max_processes: int = 4
    batch_processing_enabled: bool = True
    max_batch_size: int = 100
    connection_pool_size: int = 20
    request_timeout_seconds: int = 30
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    enable_load_balancing: bool = True


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    created_at: float
    accessed_at: float
    ttl_seconds: int
    access_count: int = 0
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self):
        """Update access timestamp and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class AdvancedCache:
    """High-performance caching system with TTL, LRU, and compression."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = deque()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0
        }
        
        # Start cleanup thread
        if config.enabled:
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.config.enabled:
            return None
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            if entry.is_expired():
                self._remove_entry(key)
                self._stats["misses"] += 1
                return None
            
            # Update access info
            entry.touch()
            
            # Update LRU order
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
            self._access_order.append(key)
            
            self._stats["hits"] += 1
            
            # Decompress if needed
            if self.config.compression_enabled and hasattr(entry.value, '__compressed__'):
                return pickle.loads(entry.value)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.config.enabled:
            return False
        
        with self._lock:
            # Serialize and optionally compress
            if self.config.compression_enabled:
                try:
                    compressed_value = pickle.dumps(value)
                    compressed_value.__compressed__ = True
                    actual_value = compressed_value
                except Exception:
                    actual_value = value
            else:
                actual_value = value
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(actual_value))
            except Exception:
                size_bytes = 1000  # Estimate
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=actual_value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl_seconds=ttl_seconds or self.config.ttl_seconds,
                size_bytes=size_bytes
            )
            
            # Check if we need to evict
            while (len(self._cache) >= self.config.max_size and 
                   key not in self._cache):
                self._evict_lru()
            
            # Remove existing entry if updating
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats["size_bytes"] -= old_entry.size_bytes
            
            # Add new entry
            self._cache[key] = entry
            self._stats["size_bytes"] += size_bytes
            
            # Update access order
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
            self._access_order.append(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = {"hits": 0, "misses": 0, "evictions": 0, "size_bytes": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                **self._stats,
                "total_entries": len(self._cache),
                "hit_rate": hit_rate,
                "memory_usage_mb": self._stats["size_bytes"] / (1024 * 1024)
            }
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and update stats."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats["size_bytes"] -= entry.size_bytes
            
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order.popleft()
            self._remove_entry(lru_key)
            self._stats["evictions"] += 1
    
    def _cleanup_loop(self):
        """Background thread to clean up expired entries."""
        while True:
            try:
                time.sleep(self.config.cleanup_interval_seconds)
                self._cleanup_expired()
            except Exception as e:
                get_logger().log_error(e, "cache_cleanup")
    
    def _cleanup_expired(self):
        """Clean up expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, threshold: int = 5, timeout_seconds: int = 60):
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout_seconds:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.threshold:
                    self.state = "open"
                
                raise e


class AsyncProcessingPool:
    """Asynchronous processing pool with load balancing."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        
        if config.enable_multiprocessing:
            self.process_pool = ProcessPoolExecutor(max_workers=config.max_processes)
        else:
            self.process_pool = None
        
        self.circuit_breakers = {}
        self._request_queue = asyncio.Queue()
        self._worker_stats = {}
        self._load_balancer = LoadBalancer() if config.enable_load_balancing else None
    
    async def submit_async(self, func: Callable, *args, use_processes: bool = False, **kwargs):
        """Submit task for asynchronous execution."""
        loop = asyncio.get_event_loop()
        
        # Choose execution pool
        pool = self.process_pool if (use_processes and self.process_pool) else self.thread_pool
        
        # Wrap with circuit breaker if enabled
        if self.config.enable_circuit_breaker:
            circuit_breaker = self._get_circuit_breaker(func.__name__)
            func = lambda *a, **k: circuit_breaker.call(func, *a, **k)
        
        # Submit to pool
        future = loop.run_in_executor(pool, func, *args)
        
        # Add timeout
        try:
            return await asyncio.wait_for(future, timeout=self.config.request_timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {self.config.request_timeout_seconds} seconds")
    
    async def batch_process(self, items: List[Any], processor: Callable, batch_size: Optional[int] = None) -> List[Any]:
        """Process items in batches for better throughput."""
        if not self.config.batch_processing_enabled:
            # Process individually
            tasks = [self.submit_async(processor, item) for item in items]
            return await asyncio.gather(*tasks)
        
        batch_size = batch_size or self.config.max_batch_size
        results = []
        
        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [self.submit_async(processor, item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    def _get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for function."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                threshold=self.config.circuit_breaker_threshold
            )
        return self.circuit_breakers[name]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing pool statistics."""
        return {
            "thread_pool_active": getattr(self.thread_pool, '_threads', 0),
            "process_pool_active": getattr(self.process_pool, '_processes', 0) if self.process_pool else 0,
            "circuit_breakers": {
                name: {"state": cb.state, "failure_count": cb.failure_count}
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    def shutdown(self):
        """Shutdown processing pools."""
        self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class LoadBalancer:
    """Simple load balancer for distributing requests."""
    
    def __init__(self):
        self.workers = []
        self.current_index = 0
        self.worker_stats = {}
        self._lock = threading.Lock()
    
    def add_worker(self, worker_id: str, weight: int = 1):
        """Add worker to load balancer."""
        with self._lock:
            self.workers.append({"id": worker_id, "weight": weight})
            self.worker_stats[worker_id] = {"requests": 0, "errors": 0, "avg_response_time": 0}
    
    def get_next_worker(self) -> str:
        """Get next worker using round-robin."""
        with self._lock:
            if not self.workers:
                return "default"
            
            worker = self.workers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.workers)
            
            return worker["id"]
    
    def record_request(self, worker_id: str, response_time: float, success: bool):
        """Record request metrics for worker."""
        with self._lock:
            if worker_id in self.worker_stats:
                stats = self.worker_stats[worker_id]
                stats["requests"] += 1
                if not success:
                    stats["errors"] += 1
                
                # Update moving average response time
                current_avg = stats["avg_response_time"]
                request_count = stats["requests"]
                stats["avg_response_time"] = ((current_avg * (request_count - 1)) + response_time) / request_count


class OptimizedSentimentAnalyzer:
    """Performance-optimized sentiment analyzer with caching and async support."""
    
    def __init__(self, cache_config: CacheConfig, performance_config: PerformanceConfig):
        self.base_analyzer = SentimentAnalyzer()
        self.cache = AdvancedCache(cache_config)
        self.processing_pool = AsyncProcessingPool(performance_config)
        self.config = performance_config
        
        # Warm up analyzer with common patterns
        self._warmup()
    
    def analyze_sentiment_sync(self, content: str, context: Dict[str, Any] = None) -> SentimentScore:
        """Synchronous sentiment analysis with caching."""
        # Generate cache key
        cache_key = self._generate_cache_key(content, context)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            get_metrics_collector().record_request(0)  # Cache hit
            return cached_result
        
        # Perform analysis
        start_time = time.time()
        result = self.base_analyzer.analyze_sentiment(content, context)
        processing_time = (time.time() - start_time) * 1000
        
        # Cache result
        self.cache.set(cache_key, result)
        
        # Record metrics
        get_metrics_collector().record_request(processing_time)
        get_metrics_collector().record_sentiment_score(result)
        
        return result
    
    async def analyze_sentiment_async(self, content: str, context: Dict[str, Any] = None) -> SentimentScore:
        """Asynchronous sentiment analysis with caching."""
        if not self.config.enable_async:
            return self.analyze_sentiment_sync(content, context)
        
        # Check cache first
        cache_key = self._generate_cache_key(content, context)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Submit for async processing
        result = await self.processing_pool.submit_async(
            self.base_analyzer.analyze_sentiment,
            content,
            context
        )
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return result
    
    async def batch_analyze_async(self, items: List[Dict[str, Any]]) -> List[SentimentScore]:
        """Batch process multiple sentiment analysis requests."""
        return await self.processing_pool.batch_process(
            items,
            lambda item: self.analyze_sentiment_sync(item["content"], item.get("context"))
        )
    
    def _generate_cache_key(self, content: str, context: Dict[str, Any] = None) -> str:
        """Generate cache key for content and context."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        context_hash = hashlib.sha256(str(sorted((context or {}).items())).encode()).hexdigest()
        return f"sentiment_{content_hash}_{context_hash}"
    
    def _warmup(self):
        """Warm up the analyzer with common patterns."""
        warmup_texts = [
            "I'm happy to help you with this task.",
            "This is a neutral statement for testing.",
            "I'm very disappointed with the results.",
            "Step 1: Analyze the problem.",
            "You should feel grateful for this opportunity."
        ]
        
        for text in warmup_texts:
            try:
                self.analyze_sentiment_sync(text)
            except Exception:
                pass  # Ignore warmup errors
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "cache_stats": self.cache.get_stats(),
            "processing_pool_stats": self.processing_pool.get_stats(),
            "metrics": get_metrics_collector().get_metrics_summary()
        }
    
    def shutdown(self):
        """Shutdown analyzer and clean up resources."""
        self.processing_pool.shutdown()
        self.cache.clear()


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, cache_config: CacheConfig = None, performance_config: PerformanceConfig = None):
        self.cache_config = cache_config or CacheConfig()
        self.performance_config = performance_config or PerformanceConfig()
        
        # Initialize optimized components
        self.optimized_sentiment_analyzer = OptimizedSentimentAnalyzer(
            self.cache_config,
            self.performance_config
        )
        
        # Performance monitoring
        self.start_time = time.time()
        self.request_count = 0
        self._lock = threading.Lock()
    
    async def optimize_filter_operation(self, request: FilterRequest) -> FilterResult:
        """Optimized filter operation with all performance enhancements."""
        start_time = time.time()
        
        with self._lock:
            self.request_count += 1
        
        try:
            # Async sentiment analysis
            sentiment_result = await self.optimized_sentiment_analyzer.analyze_sentiment_async(
                request.content,
                {"safety_level": request.safety_level.value, **(request.metadata or {})}
            )
            
            # Create optimized filter result
            processing_time = int((time.time() - start_time) * 1000)
            
            # Calculate safety score based on sentiment
            safety_score = SafetyScore(
                overall_score=max(0.0, 1.0 - sentiment_result.manipulation_risk),
                confidence=sentiment_result.confidence,
                is_safe=sentiment_result.manipulation_risk < 0.5,
                detected_patterns=sentiment_result.reasoning_patterns
            )
            
            result = FilterResult(
                filtered_content=request.content,  # Could apply sanitization here
                safety_score=safety_score,
                was_filtered=sentiment_result.manipulation_risk > 0.5,
                filter_reasons=[f"sentiment_risk:{sentiment_result.manipulation_risk:.3f}"] if sentiment_result.manipulation_risk > 0.5 else [],
                processing_time_ms=processing_time,
                request_id=request.request_id
            )
            
            # Log performance metrics
            get_logger().log_performance_metric(
                "filter_operation_time_ms",
                processing_time,
                "performance_optimizer"
            )
            
            return result
            
        except Exception as e:
            get_logger().log_error(e, "performance_optimizer", request.request_id)
            raise
    
    def get_system_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive system performance statistics."""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
            "sentiment_analyzer": self.optimized_sentiment_analyzer.get_performance_stats(),
            "system_resources": self._get_system_resources()
        }
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get system resource usage."""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def shutdown(self):
        """Shutdown optimizer and clean up resources."""
        self.optimized_sentiment_analyzer.shutdown()


# Decorators for performance optimization
def cached(ttl_seconds: int = 3600):
    """Decorator for caching function results."""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = hashlib.sha256(str((args, sorted(kwargs.items()))).encode()).hexdigest()
            
            # Check cache
            if key in cache:
                value, timestamp = cache[key]
                if time.time() - timestamp < ttl_seconds:
                    return value
                else:
                    del cache[key]
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {"size": len(cache)}
        
        return wrapper
    
    return decorator


def async_timeout(seconds: int = 30):
    """Decorator for adding timeout to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
        
        return wrapper
    
    return decorator


def performance_monitor(component_name: str):
    """Decorator for monitoring function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                processing_time = (time.time() - start_time) * 1000
                
                get_logger().log_performance_metric(
                    f"{func.__name__}_time_ms",
                    processing_time,
                    component_name
                )
                
                return result
            except Exception as e:
                get_logger().log_error(e, component_name)
                raise
        
        return wrapper
    
    return decorator