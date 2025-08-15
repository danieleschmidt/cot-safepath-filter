"""
Concurrent Processing Engine - Generation 3 Scalability Implementation.

Advanced concurrent processing with load balancing, work distribution, and auto-scaling.
"""

import asyncio
import threading
import multiprocessing
import queue
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple, Set, Union, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import weakref
import gc
import pickle

from .models import FilterRequest, FilterResult, ProcessingMetrics
from .advanced_monitoring import MetricCollector, MetricType
from .performance_optimizer import PerformanceOptimizer


logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing execution modes."""
    SYNCHRONOUS = "synchronous"      # Single-threaded processing
    THREADED = "threaded"           # Multi-threaded processing
    PROCESS_BASED = "process_based"  # Multi-process processing
    ASYNC_IO = "async_io"           # Async/await processing
    HYBRID = "hybrid"               # Combination of modes


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"


class WorkerStatus(Enum):
    """Worker status states."""
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class WorkerMetrics:
    """Metrics for a worker instance."""
    worker_id: str
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    current_load: int = 0
    avg_response_time_ms: float = 0.0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    status: WorkerStatus = WorkerStatus.IDLE
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.completed_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        return 1.0 - self.success_rate
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "current_load": self.current_load,
            "avg_response_time_ms": self.avg_response_time_ms,
            "last_activity": self.last_activity.isoformat(),
            "status": self.status.value,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate
        }


@dataclass
class WorkUnit:
    """A unit of work to be processed."""
    work_id: str
    request: FilterRequest
    callback: Optional[Callable[[FilterResult], None]] = None
    priority: int = 1
    timeout_seconds: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def processing_time_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


class WorkerPool:
    """Pool of workers for processing requests."""
    
    def __init__(self, 
                 worker_count: int,
                 processing_function: Callable[[FilterRequest], FilterResult],
                 mode: ProcessingMode = ProcessingMode.THREADED):
        
        self.worker_count = worker_count
        self.processing_function = processing_function
        self.mode = mode
        
        # Worker management
        self.workers: Dict[str, Any] = {}
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        self.work_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.result_queue: queue.Queue = queue.Queue()
        
        # Executors based on mode
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        
        # Pool state
        self.active = False
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.total_processed = 0
        self.total_errors = 0
        self.processing_times: deque = deque(maxlen=1000)
        
        self._initialize_workers()
        
        logger.info(f"Worker pool initialized: {worker_count} workers, mode: {mode.value}")
    
    def _initialize_workers(self) -> None:
        """Initialize workers based on processing mode."""
        if self.mode == ProcessingMode.THREADED:
            self.thread_executor = ThreadPoolExecutor(
                max_workers=self.worker_count,
                thread_name_prefix="filter_worker"
            )
            for i in range(self.worker_count):
                worker_id = f"thread_worker_{i}"
                self.worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
        
        elif self.mode == ProcessingMode.PROCESS_BASED:
            self.process_executor = ProcessPoolExecutor(
                max_workers=self.worker_count
            )
            for i in range(self.worker_count):
                worker_id = f"process_worker_{i}"
                self.worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
        
        elif self.mode == ProcessingMode.ASYNC_IO:
            # Async workers will be created dynamically
            for i in range(self.worker_count):
                worker_id = f"async_worker_{i}"
                self.worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
        
        elif self.mode == ProcessingMode.HYBRID:
            # Mix of thread and process workers
            thread_count = self.worker_count // 2
            process_count = self.worker_count - thread_count
            
            if thread_count > 0:
                self.thread_executor = ThreadPoolExecutor(
                    max_workers=thread_count,
                    thread_name_prefix="filter_thread"
                )
                for i in range(thread_count):
                    worker_id = f"thread_worker_{i}"
                    self.worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
            
            if process_count > 0:
                self.process_executor = ProcessPoolExecutor(
                    max_workers=process_count
                )
                for i in range(process_count):
                    worker_id = f"process_worker_{i}"
                    self.worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
    
    def start(self) -> None:
        """Start the worker pool."""
        if self.active:
            return
        
        self.active = True
        self.shutdown_event.clear()
        
        # Start result collection thread
        self.result_thread = threading.Thread(target=self._collect_results)
        self.result_thread.daemon = True
        self.result_thread.start()
        
        logger.info("Worker pool started")
    
    def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker pool."""
        if not self.active:
            return
        
        self.active = False
        self.shutdown_event.set()
        
        # Shutdown executors
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
        
        # Wait for result thread
        if hasattr(self, 'result_thread'):
            self.result_thread.join(timeout=5)
        
        logger.info("Worker pool stopped")
    
    def submit_work(self, work_unit: WorkUnit) -> bool:
        """Submit work to the pool."""
        if not self.active:
            return False
        
        try:
            # Priority queue uses negative priority for max-heap behavior
            self.work_queue.put((-work_unit.priority, work_unit.created_at, work_unit))
            return True
        except Exception as e:
            logger.error(f"Failed to submit work: {e}")
            return False
    
    def process_work_batch(self, batch_size: int = 10) -> List[concurrent.futures.Future]:
        """Process a batch of work units."""
        futures = []
        batch = []
        
        # Collect batch
        for _ in range(batch_size):
            try:
                priority, created_at, work_unit = self.work_queue.get_nowait()
                batch.append(work_unit)
            except queue.Empty:
                break
        
        if not batch:
            return futures
        
        # Submit batch to appropriate executor
        for work_unit in batch:
            future = self._submit_single_work(work_unit)
            if future:
                futures.append(future)
        
        return futures
    
    def _submit_single_work(self, work_unit: WorkUnit) -> Optional[concurrent.futures.Future]:
        """Submit a single work unit for processing."""
        work_unit.started_at = datetime.utcnow()
        
        # Select worker based on mode and load
        if self.mode == ProcessingMode.THREADED and self.thread_executor:
            return self.thread_executor.submit(self._process_work_unit, work_unit)
        
        elif self.mode == ProcessingMode.PROCESS_BASED and self.process_executor:
            return self.process_executor.submit(self._process_work_unit_process, work_unit)
        
        elif self.mode == ProcessingMode.HYBRID:
            # Route based on work characteristics
            if self._should_use_process(work_unit):
                if self.process_executor:
                    return self.process_executor.submit(self._process_work_unit_process, work_unit)
            else:
                if self.thread_executor:
                    return self.thread_executor.submit(self._process_work_unit, work_unit)
        
        return None
    
    def _process_work_unit(self, work_unit: WorkUnit) -> Tuple[WorkUnit, Optional[FilterResult], Optional[Exception]]:
        """Process a work unit in the current thread."""
        start_time = time.time()
        
        try:
            # Update worker metrics
            worker_id = f"thread_worker_{threading.current_thread().ident}"
            if worker_id in self.worker_metrics:
                metrics = self.worker_metrics[worker_id]
                metrics.current_load += 1
                metrics.status = WorkerStatus.BUSY
                metrics.total_requests += 1
            
            # Process the request
            result = self.processing_function(work_unit.request)
            
            # Update completion metrics
            work_unit.completed_at = datetime.utcnow()
            processing_time_ms = (time.time() - start_time) * 1000
            
            if worker_id in self.worker_metrics:
                metrics = self.worker_metrics[worker_id]
                metrics.completed_requests += 1
                metrics.current_load -= 1
                metrics.status = WorkerStatus.IDLE
                metrics.last_activity = datetime.utcnow()
                
                # Update average response time
                metrics.avg_response_time_ms = (
                    (metrics.avg_response_time_ms * (metrics.completed_requests - 1) + processing_time_ms) /
                    metrics.completed_requests
                )
            
            self.processing_times.append(processing_time_ms)
            self.total_processed += 1
            
            return work_unit, result, None
            
        except Exception as e:
            # Update error metrics
            work_unit.completed_at = datetime.utcnow()
            
            if worker_id in self.worker_metrics:
                metrics = self.worker_metrics[worker_id]
                metrics.failed_requests += 1
                metrics.current_load -= 1
                metrics.status = WorkerStatus.ERROR
                metrics.last_activity = datetime.utcnow()
            
            self.total_errors += 1
            
            return work_unit, None, e
    
    def _process_work_unit_process(self, work_unit: WorkUnit) -> Tuple[WorkUnit, Optional[FilterResult], Optional[Exception]]:
        """Process a work unit in a separate process."""
        # This is a simplified version - in practice, you'd need to handle
        # serialization of the processing function and other process-specific concerns
        try:
            # Since we can't easily serialize the processing function,
            # we'll use the thread-based processing for now
            return self._process_work_unit(work_unit)
        except Exception as e:
            return work_unit, None, e
    
    def _should_use_process(self, work_unit: WorkUnit) -> bool:
        """Determine if work should be processed in a separate process."""
        # Heuristics for process vs thread selection
        content_size = len(work_unit.request.content) if work_unit.request.content else 0
        
        # Use process for large content or CPU-intensive work
        return content_size > 10000 or work_unit.priority > 5
    
    def _collect_results(self) -> None:
        """Collect results from completed work."""
        while self.active and not self.shutdown_event.is_set():
            try:
                # This would collect results from the futures
                # Implementation depends on specific result handling needs
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error collecting results: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics."""
        queue_size = self.work_queue.qsize()
        avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0.0
        
        return {
            "worker_count": self.worker_count,
            "mode": self.mode.value,
            "active": self.active,
            "queue_size": queue_size,
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_processed, 1),
            "avg_processing_time_ms": avg_processing_time,
            "worker_metrics": {k: v.to_dict() for k, v in self.worker_metrics.items()}
        }


class LoadBalancer:
    """Load balancer for distributing work across multiple worker pools."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.worker_pools: List[WorkerPool] = []
        self.pool_weights: Dict[int, float] = {}  # Pool index -> weight
        self.current_index = 0  # For round-robin
        
        # Performance tracking
        self.routing_history: deque = deque(maxlen=1000)
        
        logger.info(f"Load balancer initialized with strategy: {strategy.value}")
    
    def add_worker_pool(self, pool: WorkerPool, weight: float = 1.0) -> None:
        """Add a worker pool to the load balancer."""
        pool_index = len(self.worker_pools)
        self.worker_pools.append(pool)
        self.pool_weights[pool_index] = weight
        
        logger.info(f"Added worker pool {pool_index} with weight {weight}")
    
    def route_request(self, work_unit: WorkUnit) -> Optional[WorkerPool]:
        """Route a request to the best worker pool."""
        if not self.worker_pools:
            return None
        
        selected_pool = None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected_pool = self._round_robin_selection()
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected_pool = self._least_connections_selection()
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            selected_pool = self._weighted_round_robin_selection()
        
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            selected_pool = self._least_response_time_selection()
        
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            selected_pool = self._resource_based_selection()
        
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            selected_pool = self._adaptive_selection(work_unit)
        
        # Record routing decision
        if selected_pool:
            pool_index = self.worker_pools.index(selected_pool)
            self.routing_history.append({
                "timestamp": datetime.utcnow(),
                "pool_index": pool_index,
                "strategy": self.strategy.value,
                "work_id": work_unit.work_id
            })
        
        return selected_pool
    
    def _round_robin_selection(self) -> WorkerPool:
        """Simple round-robin selection."""
        pool = self.worker_pools[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.worker_pools)
        return pool
    
    def _least_connections_selection(self) -> WorkerPool:
        """Select pool with least current connections."""
        min_load = float('inf')
        selected_pool = None
        
        for pool in self.worker_pools:
            current_load = sum(
                metrics.current_load 
                for metrics in pool.worker_metrics.values()
            )
            
            if current_load < min_load:
                min_load = current_load
                selected_pool = pool
        
        return selected_pool or self.worker_pools[0]
    
    def _weighted_round_robin_selection(self) -> WorkerPool:
        """Weighted round-robin selection."""
        # Simplified implementation - would need proper weight tracking
        return self._round_robin_selection()
    
    def _least_response_time_selection(self) -> WorkerPool:
        """Select pool with lowest average response time."""
        min_response_time = float('inf')
        selected_pool = None
        
        for pool in self.worker_pools:
            avg_response_time = statistics.mean([
                metrics.avg_response_time_ms 
                for metrics in pool.worker_metrics.values()
                if metrics.avg_response_time_ms > 0
            ]) if pool.worker_metrics else float('inf')
            
            if avg_response_time < min_response_time:
                min_response_time = avg_response_time
                selected_pool = pool
        
        return selected_pool or self.worker_pools[0]
    
    def _resource_based_selection(self) -> WorkerPool:
        """Select pool based on resource utilization."""
        # Would integrate with system resource monitoring
        # For now, use least connections as fallback
        return self._least_connections_selection()
    
    def _adaptive_selection(self, work_unit: WorkUnit) -> WorkerPool:
        """Adaptive selection based on work characteristics and pool performance."""
        # Score each pool based on multiple factors
        best_score = float('-inf')
        selected_pool = None
        
        for i, pool in enumerate(self.worker_pools):
            score = 0.0
            
            # Factor 1: Current load (lower is better)
            current_load = sum(m.current_load for m in pool.worker_metrics.values())
            load_score = 1.0 / (1.0 + current_load)
            
            # Factor 2: Success rate (higher is better)
            success_rates = [m.success_rate for m in pool.worker_metrics.values()]
            success_score = statistics.mean(success_rates) if success_rates else 0.5
            
            # Factor 3: Response time (lower is better)
            response_times = [m.avg_response_time_ms for m in pool.worker_metrics.values() if m.avg_response_time_ms > 0]
            if response_times:
                avg_response = statistics.mean(response_times)
                response_score = 1.0 / (1.0 + avg_response / 1000.0)  # Normalize by seconds
            else:
                response_score = 1.0
            
            # Factor 4: Pool weight
            weight = self.pool_weights.get(i, 1.0)
            
            # Combine scores
            score = (load_score * 0.4 + success_score * 0.3 + response_score * 0.2 + weight * 0.1)
            
            if score > best_score:
                best_score = score
                selected_pool = pool
        
        return selected_pool or self.worker_pools[0]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get load balancer routing statistics."""
        if not self.routing_history:
            return {"total_requests": 0}
        
        # Count requests per pool
        pool_counts = defaultdict(int)
        for entry in self.routing_history:
            pool_counts[entry["pool_index"]] += 1
        
        total_requests = len(self.routing_history)
        
        return {
            "strategy": self.strategy.value,
            "total_requests": total_requests,
            "pool_distribution": dict(pool_counts),
            "pool_percentages": {
                pool_idx: (count / total_requests) * 100
                for pool_idx, count in pool_counts.items()
            }
        }


class ConcurrentProcessingEngine:
    """Main concurrent processing engine."""
    
    def __init__(self, 
                 processing_function: Callable[[FilterRequest], FilterResult],
                 metric_collector: Optional[MetricCollector] = None):
        
        self.processing_function = processing_function
        self.metric_collector = metric_collector
        
        # Processing configuration
        self.default_pool_size = min(32, multiprocessing.cpu_count() * 2)
        self.auto_scaling_enabled = True
        self.min_pools = 1
        self.max_pools = 4
        
        # Load balancer and worker pools
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.ADAPTIVE)
        self.worker_pools: List[WorkerPool] = []
        
        # Auto-scaling
        self.scaling_thread: Optional[threading.Thread] = None
        self.scaling_active = False
        self.scaling_interval = 30  # seconds
        
        # Request tracking
        self.pending_requests: Dict[str, WorkUnit] = {}
        self.completed_requests: Dict[str, Tuple[WorkUnit, FilterResult]] = {}
        self.failed_requests: Dict[str, Tuple[WorkUnit, Exception]] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize default worker pool
        self._initialize_default_pools()
        
        logger.info("Concurrent processing engine initialized")
    
    def _initialize_default_pools(self) -> None:
        """Initialize default worker pools."""
        # Create initial worker pools
        thread_pool = WorkerPool(
            worker_count=self.default_pool_size,
            processing_function=self.processing_function,
            mode=ProcessingMode.THREADED
        )
        
        self.worker_pools.append(thread_pool)
        self.load_balancer.add_worker_pool(thread_pool, weight=1.0)
        
        # Add process-based pool for CPU-intensive work
        if multiprocessing.cpu_count() > 1:
            process_pool = WorkerPool(
                worker_count=max(1, multiprocessing.cpu_count() // 2),
                processing_function=self.processing_function,
                mode=ProcessingMode.PROCESS_BASED
            )
            
            self.worker_pools.append(process_pool)
            self.load_balancer.add_worker_pool(process_pool, weight=0.5)
    
    def start(self) -> None:
        """Start the concurrent processing engine."""
        # Start all worker pools
        for pool in self.worker_pools:
            pool.start()
        
        # Start auto-scaling if enabled
        if self.auto_scaling_enabled:
            self._start_auto_scaling()
        
        logger.info("Concurrent processing engine started")
    
    def stop(self) -> None:
        """Stop the concurrent processing engine."""
        # Stop auto-scaling
        if self.scaling_active:
            self.scaling_active = False
            if self.scaling_thread:
                self.scaling_thread.join(timeout=10)
        
        # Stop all worker pools
        for pool in self.worker_pools:
            pool.stop()
        
        logger.info("Concurrent processing engine stopped")
    
    def process_request_async(self, request: FilterRequest, 
                            priority: int = 1,
                            timeout_seconds: float = 30.0,
                            callback: Optional[Callable[[FilterResult], None]] = None) -> str:
        """Process a request asynchronously."""
        work_id = f"work_{int(time.time() * 1000)}_{id(request)}"
        
        work_unit = WorkUnit(
            work_id=work_id,
            request=request,
            callback=callback,
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        # Route to appropriate worker pool
        selected_pool = self.load_balancer.route_request(work_unit)
        
        if selected_pool and selected_pool.submit_work(work_unit):
            with self._lock:
                self.pending_requests[work_id] = work_unit
            
            # Record metrics
            if self.metric_collector:
                self.metric_collector.record_counter("concurrent.requests_submitted")
                self.metric_collector.record_gauge("concurrent.pending_requests", len(self.pending_requests))
            
            return work_id
        else:
            raise RuntimeError("Failed to submit work to any pool")
    
    def process_request_sync(self, request: FilterRequest, 
                           priority: int = 1,
                           timeout_seconds: float = 30.0) -> FilterResult:
        """Process a request synchronously."""
        result_event = threading.Event()
        result_container = {"result": None, "error": None}
        
        def callback(result: FilterResult):
            result_container["result"] = result
            result_event.set()
        
        def error_callback(error: Exception):
            result_container["error"] = error
            result_event.set()
        
        work_id = self.process_request_async(request, priority, timeout_seconds, callback)
        
        # Wait for completion
        if result_event.wait(timeout=timeout_seconds):
            if result_container["error"]:
                raise result_container["error"]
            return result_container["result"]
        else:
            raise TimeoutError(f"Request {work_id} timed out after {timeout_seconds} seconds")
    
    def get_request_status(self, work_id: str) -> Optional[str]:
        """Get the status of a request."""
        with self._lock:
            if work_id in self.pending_requests:
                return "pending"
            elif work_id in self.completed_requests:
                return "completed"
            elif work_id in self.failed_requests:
                return "failed"
        return None
    
    def get_request_result(self, work_id: str) -> Optional[FilterResult]:
        """Get the result of a completed request."""
        with self._lock:
            if work_id in self.completed_requests:
                return self.completed_requests[work_id][1]
        return None
    
    def _start_auto_scaling(self) -> None:
        """Start auto-scaling thread."""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.scaling_thread = threading.Thread(target=self._auto_scaling_loop)
        self.scaling_thread.daemon = True
        self.scaling_thread.start()
        
        logger.info("Auto-scaling started")
    
    def _auto_scaling_loop(self) -> None:
        """Auto-scaling monitoring loop."""
        while self.scaling_active:
            try:
                self._evaluate_scaling_needs()
                time.sleep(self.scaling_interval)
            except Exception as e:
                logger.error(f"Error in auto-scaling: {e}")
                time.sleep(60)
    
    def _evaluate_scaling_needs(self) -> None:
        """Evaluate if scaling is needed."""
        # Get current metrics
        total_queue_size = sum(pool.work_queue.qsize() for pool in self.worker_pools)
        total_workers = sum(pool.worker_count for pool in self.worker_pools)
        
        # Calculate average utilization
        total_load = 0
        total_capacity = 0
        
        for pool in self.worker_pools:
            pool_load = sum(m.current_load for m in pool.worker_metrics.values())
            total_load += pool_load
            total_capacity += pool.worker_count
        
        utilization = total_load / max(total_capacity, 1)
        
        # Scaling decisions
        if utilization > 0.8 and total_queue_size > 10:
            # Scale up
            if len(self.worker_pools) < self.max_pools:
                self._scale_up()
        elif utilization < 0.2 and total_queue_size < 5:
            # Scale down
            if len(self.worker_pools) > self.min_pools:
                self._scale_down()
        
        # Record metrics
        if self.metric_collector:
            self.metric_collector.record_gauge("concurrent.utilization", utilization)
            self.metric_collector.record_gauge("concurrent.total_queue_size", total_queue_size)
            self.metric_collector.record_gauge("concurrent.worker_pools", len(self.worker_pools))
    
    def _scale_up(self) -> None:
        """Add a new worker pool."""
        try:
            new_pool = WorkerPool(
                worker_count=self.default_pool_size,
                processing_function=self.processing_function,
                mode=ProcessingMode.THREADED
            )
            
            new_pool.start()
            self.worker_pools.append(new_pool)
            self.load_balancer.add_worker_pool(new_pool, weight=1.0)
            
            if self.metric_collector:
                self.metric_collector.record_counter("concurrent.scale_up_events")
            
            logger.info(f"Scaled up: added worker pool (total: {len(self.worker_pools)})")
            
        except Exception as e:
            logger.error(f"Failed to scale up: {e}")
    
    def _scale_down(self) -> None:
        """Remove a worker pool."""
        if len(self.worker_pools) <= self.min_pools:
            return
        
        try:
            # Find pool with lowest utilization
            min_utilization = float('inf')
            pool_to_remove = None
            
            for pool in self.worker_pools:
                pool_load = sum(m.current_load for m in pool.worker_metrics.values())
                utilization = pool_load / max(pool.worker_count, 1)
                
                if utilization < min_utilization:
                    min_utilization = utilization
                    pool_to_remove = pool
            
            if pool_to_remove and min_utilization < 0.1:
                pool_to_remove.stop()
                self.worker_pools.remove(pool_to_remove)
                
                if self.metric_collector:
                    self.metric_collector.record_counter("concurrent.scale_down_events")
                
                logger.info(f"Scaled down: removed worker pool (total: {len(self.worker_pools)})")
        
        except Exception as e:
            logger.error(f"Failed to scale down: {e}")
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        with self._lock:
            pending_count = len(self.pending_requests)
            completed_count = len(self.completed_requests)
            failed_count = len(self.failed_requests)
        
        pool_stats = [pool.get_metrics() for pool in self.worker_pools]
        routing_stats = self.load_balancer.get_routing_stats()
        
        return {
            "engine_active": any(pool.active for pool in self.worker_pools),
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "worker_pools_count": len(self.worker_pools),
            "requests": {
                "pending": pending_count,
                "completed": completed_count,
                "failed": failed_count,
                "total": pending_count + completed_count + failed_count
            },
            "worker_pools": pool_stats,
            "load_balancer": routing_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def cleanup(self) -> None:
        """Cleanup engine resources."""
        self.stop()
        
        # Clear request tracking
        with self._lock:
            self.pending_requests.clear()
            self.completed_requests.clear()
            self.failed_requests.clear()
        
        logger.info("Concurrent processing engine cleaned up")