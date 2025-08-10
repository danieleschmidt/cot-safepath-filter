"""
Performance optimization and scaling features for CoT SafePath Filter.
"""

import asyncio
import concurrent.futures
import threading
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue, Empty
import weakref

from .models import FilterRequest, FilterResult, SafetyScore, DetectionResult
from .exceptions import TimeoutError, CapacityError
from .utils import measure_performance


logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    
    # Concurrency settings
    max_concurrent_requests: int = 50
    thread_pool_size: int = 10
    process_pool_size: int = 4
    
    # Caching settings  
    enable_result_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Timeout settings
    filter_timeout_ms: int = 1000
    detector_timeout_ms: int = 200
    
    # Batching settings
    enable_batching: bool = True
    batch_size: int = 10
    batch_timeout_ms: int = 100
    
    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: int = 80
    
    # Optimization flags
    enable_async: bool = True
    enable_precomputation: bool = True
    enable_streaming: bool = False


class ResourceMonitor:
    """Monitor system resources and enforce limits."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.active_requests = 0
        self.peak_requests = 0
        self.total_requests = 0
        self.lock = threading.RLock()
        
        # Resource tracking
        self._start_time = time.time()
        self._memory_usage = 0
        self._cpu_usage = 0
        
    def acquire_request_slot(self) -> bool:
        """Acquire a slot for processing a request."""
        with self.lock:
            if self.active_requests >= self.config.max_concurrent_requests:
                return False
            
            self.active_requests += 1
            self.total_requests += 1
            self.peak_requests = max(self.peak_requests, self.active_requests)
            return True
    
    def release_request_slot(self):
        """Release a request processing slot."""
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)
    
    def check_resource_limits(self) -> bool:
        """Check if system is within resource limits."""
        try:
            import psutil
            
            # Check memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > self.config.max_memory_mb:
                logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.config.max_memory_mb}MB")
                return False
            
            # Check CPU usage
            cpu_percent = process.cpu_percent()
            if cpu_percent > self.config.max_cpu_percent:
                logger.warning(f"CPU usage {cpu_percent:.1f}% exceeds limit {self.config.max_cpu_percent}%")
                return False
            
            self._memory_usage = memory_mb
            self._cpu_usage = cpu_percent
            return True
            
        except ImportError:
            # psutil not available, assume OK
            return True
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return True  # Fail open
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource monitoring statistics."""
        uptime_seconds = time.time() - self._start_time
        
        with self.lock:
            return {
                "active_requests": self.active_requests,
                "peak_requests": self.peak_requests,
                "total_requests": self.total_requests,
                "memory_usage_mb": self._memory_usage,
                "cpu_usage_percent": self._cpu_usage,
                "uptime_seconds": uptime_seconds,
                "requests_per_second": self.total_requests / max(uptime_seconds, 1)
            }


class RequestBatcher:
    """Batch requests for more efficient processing."""
    
    def __init__(self, config: PerformanceConfig, process_func: Callable):
        self.config = config
        self.process_func = process_func
        self.batch_queue = Queue()
        self.batch_thread = None
        self.running = False
        
    def start(self):
        """Start the batching processor."""
        if self.running:
            return
        
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
        logger.info("Request batcher started")
    
    def stop(self):
        """Stop the batching processor."""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=1.0)
    
    def submit_request(self, request: FilterRequest, callback: Callable) -> None:
        """Submit a request for batched processing."""
        if not self.config.enable_batching:
            # Process immediately
            try:
                result = self.process_func(request)
                callback(result, None)
            except Exception as e:
                callback(None, e)
            return
        
        self.batch_queue.put((request, callback))
    
    def _batch_processor(self):
        """Process requests in batches."""
        while self.running:
            batch = []
            batch_callbacks = []
            
            # Collect batch
            end_time = time.time() + self.config.batch_timeout_ms / 1000.0
            
            while len(batch) < self.config.batch_size and time.time() < end_time:
                try:
                    timeout = max(0.001, end_time - time.time())
                    item = self.batch_queue.get(timeout=timeout)
                    batch.append(item[0])
                    batch_callbacks.append(item[1])
                except Empty:
                    break
            
            if not batch:
                continue
            
            # Process batch
            try:
                results = self._process_batch(batch)
                
                # Send results to callbacks
                for i, callback in enumerate(batch_callbacks):
                    if i < len(results):
                        callback(results[i], None)
                    else:
                        callback(None, Exception("Batch processing failed"))
                        
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Send error to all callbacks
                for callback in batch_callbacks:
                    callback(None, e)
    
    def _process_batch(self, requests: List[FilterRequest]) -> List[FilterResult]:
        """Process a batch of requests efficiently."""
        # For now, process individually
        # In a real implementation, this could do batch optimization
        results = []
        for request in requests:
            try:
                result = self.process_func(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch item processing error: {e}")
                # Create error result
                results.append(FilterResult(
                    filtered_content=request.content,
                    safety_score=SafetyScore(overall_score=0.0, confidence=0.0, is_safe=False),
                    was_filtered=False,
                    filter_reasons=[f"Processing error: {str(e)}"]
                ))
        
        return results


class AsyncFilterExecutor:
    """Asynchronous filter execution with concurrency control."""
    
    def __init__(self, config: PerformanceConfig, filter_func: Callable):
        self.config = config
        self.filter_func = filter_func
        self.resource_monitor = ResourceMonitor(config)
        
        # Thread pool for CPU-bound filtering
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.thread_pool_size,
            thread_name_prefix="SafePathFilter"
        )
        
        # Process pool for heavy computation (optional)
        self.process_pool = None
        if config.process_pool_size > 0:
            try:
                self.process_pool = ProcessPoolExecutor(
                    max_workers=config.process_pool_size
                )
            except Exception as e:
                logger.warning(f"Could not create process pool: {e}")
        
        # Request batcher
        if config.enable_batching:
            self.batcher = RequestBatcher(config, filter_func)
            self.batcher.start()
        else:
            self.batcher = None
        
        logger.info(f"Async executor initialized with {config.thread_pool_size} threads")
    
    async def filter_async(self, request: FilterRequest) -> FilterResult:
        """Filter request asynchronously."""
        # Check resource limits
        if not self.resource_monitor.check_resource_limits():
            raise CapacityError(
                "System resource limits exceeded",
                resource="memory/cpu",
                current=self.resource_monitor._memory_usage,
                limit=self.config.max_memory_mb
            )
        
        # Acquire request slot
        if not self.resource_monitor.acquire_request_slot():
            raise CapacityError(
                "Maximum concurrent requests exceeded", 
                resource="concurrent_requests",
                current=self.resource_monitor.active_requests,
                limit=self.config.max_concurrent_requests
            )
        
        try:
            # Execute with timeout
            loop = asyncio.get_event_loop()
            
            if self.batcher:
                # Use batched processing
                future = asyncio.Future()
                
                def callback(result, error):
                    if error:
                        future.set_exception(error)
                    else:
                        future.set_result(result)
                
                self.batcher.submit_request(request, callback)
                
                try:
                    result = await asyncio.wait_for(
                        future, 
                        timeout=self.config.filter_timeout_ms / 1000.0
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        "Filter request timed out",
                        timeout_ms=self.config.filter_timeout_ms,
                        elapsed_ms=self.config.filter_timeout_ms
                    )
            
            else:
                # Direct processing
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(self.thread_pool, self.filter_func, request),
                        timeout=self.config.filter_timeout_ms / 1000.0
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        "Filter request timed out",
                        timeout_ms=self.config.filter_timeout_ms,
                        elapsed_ms=self.config.filter_timeout_ms
                    )
            
            return result
            
        finally:
            self.resource_monitor.release_request_slot()
    
    def filter_batch_sync(self, requests: List[FilterRequest]) -> List[FilterResult]:
        """Process multiple requests efficiently in sync mode."""
        if not requests:
            return []
        
        # Check capacity
        if len(requests) > self.config.max_concurrent_requests:
            raise CapacityError(
                f"Batch size {len(requests)} exceeds maximum concurrent requests",
                resource="batch_size",
                current=len(requests),
                limit=self.config.max_concurrent_requests
            )
        
        # Process with thread pool
        futures = []
        for request in requests:
            future = self.thread_pool.submit(self.filter_func, request)
            futures.append(future)
        
        # Collect results with timeout
        results = []
        timeout_seconds = self.config.filter_timeout_ms / 1000.0
        
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=timeout_seconds)
                results.append(result)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Request {i} in batch timed out")
                results.append(FilterResult(
                    filtered_content=requests[i].content,
                    safety_score=SafetyScore(overall_score=0.0, confidence=0.0, is_safe=False),
                    was_filtered=False,
                    filter_reasons=["Request timed out"]
                ))
            except Exception as e:
                logger.error(f"Request {i} in batch failed: {e}")
                results.append(FilterResult(
                    filtered_content=requests[i].content,
                    safety_score=SafetyScore(overall_score=0.0, confidence=0.0, is_safe=False),
                    was_filtered=False,
                    filter_reasons=[f"Processing error: {str(e)}"]
                ))
        
        return results
    
    def shutdown(self):
        """Shutdown the executor and cleanup resources."""
        logger.info("Shutting down async filter executor...")
        
        if self.batcher:
            self.batcher.stop()
        
        self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("Async filter executor shut down")


class PerformanceOptimizer:
    """Optimize performance based on runtime metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.optimization_history = []
        
    @lru_cache(maxsize=1000)
    def cached_pattern_match(self, pattern: str, content: str) -> bool:
        """Cached pattern matching for common patterns."""
        import re
        return bool(re.search(pattern, content, re.IGNORECASE))
    
    def precompute_features(self, content: str) -> Dict[str, Any]:
        """Precompute features that can be reused across detectors."""
        # Cache frequently used computations
        features = {
            'length': len(content),
            'word_count': len(content.split()),
            'lowercase': content.lower(),
            'lines': content.split('\n'),
            'sentences': content.split('.'),
        }
        
        # Common patterns
        features.update({
            'has_steps': 'step' in features['lowercase'],
            'has_numbers': any(c.isdigit() for c in content),
            'has_special_chars': any(not c.isalnum() and not c.isspace() for c in content),
        })
        
        return features
    
    def optimize_detector_order(self, detectors: List[Any], performance_data: Dict[str, float]) -> List[Any]:
        """Optimize detector execution order based on performance data."""
        # Sort by speed/effectiveness ratio
        def score_detector(detector):
            name = getattr(detector, '__name__', str(detector))
            speed = performance_data.get(f"{name}_speed", 1.0)
            accuracy = performance_data.get(f"{name}_accuracy", 0.5)
            return accuracy / speed  # Higher is better
        
        return sorted(detectors, key=score_detector, reverse=True)
    
    def preprocess_content(self, content: str) -> str:
        """Preprocess content for optimal filtering performance."""
        # Basic content preprocessing
        processed = content.strip()
        processed = ' '.join(processed.split())  # Normalize whitespace
        return processed
    
    def batch_process(self, contents: List[str]) -> List[Dict[str, Any]]:
        """Process multiple contents in batch for better efficiency."""
        results = []
        for content in contents:
            processed = self.preprocess_content(content)
            features = self.precompute_features(processed)
            results.append({
                'content': processed,
                'features': features
            })
        return results


def performance_profile(func):
    """Decorator to profile function performance."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            
            # Log performance data
            logger.debug(f"{func.__name__} completed in {duration*1000:.2f}ms")
            
            # Store in global metrics if available
            if hasattr(wrapper, '_performance_data'):
                if func.__name__ not in wrapper._performance_data:
                    wrapper._performance_data[func.__name__] = []
                wrapper._performance_data[func.__name__].append(duration)
            
            return result
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {duration*1000:.2f}ms: {e}")
            raise
    
    wrapper._performance_data = {}
    return wrapper


class MemoryOptimizer:
    """Optimize memory usage and prevent leaks."""
    
    def __init__(self):
        self._weak_cache = weakref.WeakValueDictionary()
        self._cleanup_interval = 60  # seconds
        self._last_cleanup = time.time()
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result with automatic cleanup."""
        self._maybe_cleanup()
        return self._weak_cache.get(cache_key)
    
    def cache_result(self, cache_key: str, result: Any) -> None:
        """Cache result with weak references."""
        self._weak_cache[cache_key] = result
    
    def _maybe_cleanup(self):
        """Periodic cleanup of expired cache entries."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            # Weak references are cleaned automatically
            # Just update timestamp
            self._last_cleanup = now


# Global instances for easy access
_performance_config = PerformanceConfig()
_memory_optimizer = MemoryOptimizer()
_performance_optimizer = PerformanceOptimizer()


def configure_performance(config: PerformanceConfig) -> None:
    """Configure global performance settings."""
    global _performance_config
    _performance_config = config
    logger.info(f"Performance configured: {config.max_concurrent_requests} max requests")


def get_performance_config() -> PerformanceConfig:
    """Get current performance configuration."""
    return _performance_config


def get_memory_optimizer() -> MemoryOptimizer:
    """Get the global memory optimizer."""
    return _memory_optimizer


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer."""
    return _performance_optimizer