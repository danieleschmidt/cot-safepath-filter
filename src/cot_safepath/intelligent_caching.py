"""
Intelligent Caching System - Generation 3 Performance Optimization.

Advanced caching with intelligent eviction, predictive loading, and adaptive strategies.
"""

import asyncio
import time
import hashlib
import json
import pickle
import threading
import weakref
from typing import Dict, List, Optional, Any, Callable, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import OrderedDict, defaultdict, deque
import logging
import statistics
import heapq
import gc
import sys

from .models import FilterResult, FilterRequest, SafetyScore
from .advanced_monitoring import MetricCollector, MetricType


logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    ADAPTIVE = "adaptive"     # Adaptive based on access patterns
    INTELLIGENT = "intelligent"  # ML-based intelligent caching


class CacheLevel(Enum):
    """Cache levels for hierarchical caching."""
    L1_MEMORY = "l1_memory"        # Fast in-memory cache
    L2_COMPRESSED = "l2_compressed"  # Compressed memory cache
    L3_DISK = "l3_disk"            # Disk-based cache
    L4_DISTRIBUTED = "l4_distributed"  # Distributed cache


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    priority: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def idle_seconds(self) -> float:
        """Get idle time since last access."""
        return (datetime.utcnow() - self.last_accessed).total_seconds()
    
    def access(self) -> None:
        """Record an access to this entry."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def calculate_size(self) -> int:
        """Calculate size of the cached value."""
        if self.size_bytes > 0:
            return self.size_bytes
        
        try:
            if isinstance(self.value, str):
                self.size_bytes = len(self.value.encode('utf-8'))
            elif isinstance(self.value, bytes):
                self.size_bytes = len(self.value)
            else:
                # Estimate size using pickle
                self.size_bytes = len(pickle.dumps(self.value))
        except Exception:
            self.size_bytes = sys.getsizeof(self.value)
        
        return self.size_bytes


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size_bytes": self.size_bytes,
            "entry_count": self.entry_count,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate
        }


class AccessPattern:
    """Tracks access patterns for intelligent caching."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.access_times: deque = deque(maxlen=max_history)
        self.access_intervals: deque = deque(maxlen=max_history)
        self.hourly_access_counts: Dict[int, int] = defaultdict(int)
        self.daily_access_counts: Dict[int, int] = defaultdict(int)
        self.last_access_time: Optional[datetime] = None
        
    def record_access(self, access_time: datetime = None) -> None:
        """Record an access at the specified time."""
        if access_time is None:
            access_time = datetime.utcnow()
        
        self.access_times.append(access_time)
        
        # Calculate interval if we have previous access
        if self.last_access_time:
            interval = (access_time - self.last_access_time).total_seconds()
            self.access_intervals.append(interval)
        
        # Update hourly and daily counters
        self.hourly_access_counts[access_time.hour] += 1
        self.daily_access_counts[access_time.weekday()] += 1
        
        self.last_access_time = access_time
    
    def predict_next_access(self) -> Optional[datetime]:
        """Predict when the next access might occur."""
        if len(self.access_intervals) < 3:
            return None
        
        # Simple prediction based on average interval
        avg_interval = statistics.mean(self.access_intervals)
        std_interval = statistics.stdev(self.access_intervals) if len(self.access_intervals) > 1 else 0
        
        # Use median for more robust prediction
        median_interval = statistics.median(self.access_intervals)
        
        # Predict next access time
        predicted_interval = min(avg_interval, median_interval + std_interval)
        return self.last_access_time + timedelta(seconds=predicted_interval)
    
    def get_access_frequency(self, window_hours: int = 24) -> float:
        """Get access frequency within the specified window."""
        cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
        recent_accesses = sum(1 for t in self.access_times if t >= cutoff_time)
        return recent_accesses / window_hours
    
    def is_hot_data(self, threshold_frequency: float = 1.0) -> bool:
        """Determine if this represents hot (frequently accessed) data."""
        return self.get_access_frequency() >= threshold_frequency


class IntelligentCache:
    """Advanced cache with intelligent eviction and prediction."""
    
    def __init__(self, 
                 max_size_bytes: int = 100 * 1024 * 1024,  # 100MB default
                 max_entries: int = 10000,
                 policy: CachePolicy = CachePolicy.ADAPTIVE,
                 metric_collector: Optional[MetricCollector] = None):
        
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.policy = policy
        self.metric_collector = metric_collector
        
        # Cache storage
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict = OrderedDict()  # For LRU
        self.frequency_heap: List[Tuple[int, str]] = []  # For LFU
        
        # Advanced features
        self.access_patterns: Dict[str, AccessPattern] = {}
        self.prediction_cache: Dict[str, Any] = {}  # Pre-loaded predicted items
        
        # Statistics
        self.stats = CacheStats()
        
        # Configuration
        self.enable_compression = True
        self.enable_prediction = True
        self.prediction_threshold = 0.7  # Confidence threshold for predictions
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self.cleanup_interval = 300  # 5 minutes
        self.cleanup_thread: Optional[threading.Thread] = None
        self.cleanup_active = False
        
        logger.info(f"Intelligent cache initialized: {policy.value}, max_size={max_size_bytes}, max_entries={max_entries}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()
        
        with self._lock:
            if key in self.entries:
                entry = self.entries[key]
                
                # Check if expired
                if entry.is_expired:
                    self._remove_entry(key)
                    self._record_miss(start_time)
                    return None
                
                # Update access information
                entry.access()
                self._update_access_order(key)
                self._record_access_pattern(key)
                
                self._record_hit(start_time)
                return entry.value
            
            # Check prediction cache
            if self.enable_prediction and key in self.prediction_cache:
                value = self.prediction_cache.pop(key)
                # Promote to main cache
                self.put(key, value)
                self._record_hit(start_time)
                return value
            
            self._record_miss(start_time)
            return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            priority: float = 1.0, metadata: Dict[str, Any] = None) -> bool:
        """Put value into cache."""
        start_time = time.time()
        
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl_seconds,
                priority=priority,
                metadata=metadata or {}
            )
            entry.calculate_size()
            
            # Check if we need to make space
            if not self._ensure_space(entry.size_bytes):
                logger.warning(f"Could not cache entry {key}: insufficient space")
                return False
            
            # Remove existing entry if present
            if key in self.entries:
                self._remove_entry(key)
            
            # Add new entry
            self.entries[key] = entry
            self._update_access_order(key)
            self._record_access_pattern(key)
            
            # Update statistics
            self.stats.entry_count = len(self.entries)
            self.stats.size_bytes += entry.size_bytes
            
            # Record metrics
            if self.metric_collector:
                self.metric_collector.record_timer("cache.put_latency_ms", (time.time() - start_time) * 1000)
                self.metric_collector.record_gauge("cache.size_bytes", self.stats.size_bytes)
                self.metric_collector.record_gauge("cache.entry_count", self.stats.entry_count)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self.entries:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.entries.clear()
            self.access_order.clear()
            self.frequency_heap.clear()
            self.access_patterns.clear()
            self.prediction_cache.clear()
            
            self.stats = CacheStats()
            
            if self.metric_collector:
                self.metric_collector.record_counter("cache.clears_total")
    
    def _ensure_space(self, required_bytes: int) -> bool:
        """Ensure there's enough space for a new entry."""
        # Check entry count limit
        while len(self.entries) >= self.max_entries:
            if not self._evict_one():
                return False
        
        # Check size limit
        while (self.stats.size_bytes + required_bytes) > self.max_size_bytes:
            if not self._evict_one():
                return False
        
        return True
    
    def _evict_one(self) -> bool:
        """Evict one entry based on the current policy."""
        if not self.entries:
            return False
        
        victim_key = None
        
        if self.policy == CachePolicy.LRU:
            victim_key = self._select_lru_victim()
        elif self.policy == CachePolicy.LFU:
            victim_key = self._select_lfu_victim()
        elif self.policy == CachePolicy.TTL:
            victim_key = self._select_ttl_victim()
        elif self.policy == CachePolicy.ADAPTIVE:
            victim_key = self._select_adaptive_victim()
        elif self.policy == CachePolicy.INTELLIGENT:
            victim_key = self._select_intelligent_victim()
        
        if victim_key:
            self._remove_entry(victim_key)
            self.stats.evictions += 1
            
            if self.metric_collector:
                self.metric_collector.record_counter("cache.evictions_total", labels={"policy": self.policy.value})
            
            return True
        
        return False
    
    def _select_lru_victim(self) -> Optional[str]:
        """Select victim using LRU policy."""
        if self.access_order:
            return next(iter(self.access_order))
        return None
    
    def _select_lfu_victim(self) -> Optional[str]:
        """Select victim using LFU policy."""
        if not self.entries:
            return None
        
        # Find entry with lowest access count
        min_key = min(self.entries.keys(), key=lambda k: self.entries[k].access_count)
        return min_key
    
    def _select_ttl_victim(self) -> Optional[str]:
        """Select victim based on TTL (expired first, then oldest)."""
        # First, try to find expired entries
        for key, entry in self.entries.items():
            if entry.is_expired:
                return key
        
        # If no expired entries, select oldest
        if self.entries:
            oldest_key = min(self.entries.keys(), key=lambda k: self.entries[k].created_at)
            return oldest_key
        
        return None
    
    def _select_adaptive_victim(self) -> Optional[str]:
        """Select victim using adaptive policy based on access patterns."""
        if not self.entries:
            return None
        
        # Score each entry based on multiple factors
        scores = {}
        
        for key, entry in self.entries.items():
            score = 0.0
            
            # Factor 1: Recency (higher score = more recent)
            age_score = 1.0 / (1.0 + entry.idle_seconds / 3600)  # Decay over hours
            
            # Factor 2: Frequency
            freq_score = min(entry.access_count / 10.0, 1.0)  # Cap at 10 accesses
            
            # Factor 3: Size penalty (larger items get lower scores)
            size_penalty = 1.0 / (1.0 + entry.size_bytes / (1024 * 1024))  # Penalty for MB
            
            # Factor 4: Priority
            priority_score = entry.priority
            
            # Factor 5: Access pattern prediction
            pattern_score = 0.0
            if key in self.access_patterns:
                pattern = self.access_patterns[key]
                if pattern.is_hot_data():
                    pattern_score = 0.5
            
            # Combine scores
            score = (age_score * 0.3 + 
                    freq_score * 0.3 + 
                    size_penalty * 0.2 + 
                    priority_score * 0.1 + 
                    pattern_score * 0.1)
            
            scores[key] = score
        
        # Select entry with lowest score
        victim_key = min(scores.keys(), key=lambda k: scores[k])
        return victim_key
    
    def _select_intelligent_victim(self) -> Optional[str]:
        """Select victim using intelligent prediction-based policy."""
        if not self.entries:
            return None
        
        # Use adaptive selection as base, but enhance with predictions
        current_time = datetime.utcnow()
        scores = {}
        
        for key, entry in self.entries.items():
            # Start with adaptive score
            base_score = self._calculate_adaptive_score(entry)
            
            # Enhance with prediction
            prediction_bonus = 0.0
            if key in self.access_patterns:
                pattern = self.access_patterns[key]
                next_access = pattern.predict_next_access()
                
                if next_access:
                    time_to_next = (next_access - current_time).total_seconds()
                    
                    # Bonus for items likely to be accessed soon
                    if time_to_next < 3600:  # Within an hour
                        prediction_bonus = 0.3
                    elif time_to_next < 7200:  # Within 2 hours
                        prediction_bonus = 0.1
            
            scores[key] = base_score + prediction_bonus
        
        victim_key = min(scores.keys(), key=lambda k: scores[k])
        return victim_key
    
    def _calculate_adaptive_score(self, entry: CacheEntry) -> float:
        """Calculate adaptive score for an entry."""
        age_score = 1.0 / (1.0 + entry.idle_seconds / 3600)
        freq_score = min(entry.access_count / 10.0, 1.0)
        size_penalty = 1.0 / (1.0 + entry.size_bytes / (1024 * 1024))
        priority_score = entry.priority
        
        return (age_score * 0.4 + freq_score * 0.3 + 
                size_penalty * 0.2 + priority_score * 0.1)
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry from the cache."""
        if key in self.entries:
            entry = self.entries[key]
            self.stats.size_bytes -= entry.size_bytes
            
            del self.entries[key]
            
            if key in self.access_order:
                del self.access_order[key]
        
        # Clean up prediction cache
        if key in self.prediction_cache:
            del self.prediction_cache[key]
        
        self.stats.entry_count = len(self.entries)
    
    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU tracking."""
        if key in self.access_order:
            self.access_order.move_to_end(key)
        else:
            self.access_order[key] = True
    
    def _record_access_pattern(self, key: str) -> None:
        """Record access pattern for intelligent caching."""
        if key not in self.access_patterns:
            self.access_patterns[key] = AccessPattern()
        
        self.access_patterns[key].record_access()
    
    def _record_hit(self, start_time: float) -> None:
        """Record cache hit."""
        self.stats.hits += 1
        
        if self.metric_collector:
            self.metric_collector.record_counter("cache.hits_total")
            self.metric_collector.record_timer("cache.get_latency_ms", (time.time() - start_time) * 1000)
    
    def _record_miss(self, start_time: float) -> None:
        """Record cache miss."""
        self.stats.misses += 1
        
        if self.metric_collector:
            self.metric_collector.record_counter("cache.misses_total")
            self.metric_collector.record_timer("cache.get_latency_ms", (time.time() - start_time) * 1000)
    
    def predict_and_preload(self) -> None:
        """Predict likely future accesses and preload them."""
        if not self.enable_prediction:
            return
        
        current_time = datetime.utcnow()
        predictions = []
        
        for key, pattern in self.access_patterns.items():
            if key not in self.entries:  # Only predict for items not in cache
                next_access = pattern.predict_next_access()
                
                if next_access and next_access > current_time:
                    time_to_access = (next_access - current_time).total_seconds()
                    
                    # Only predict for items likely to be accessed soon
                    if time_to_access < 3600:  # Within an hour
                        confidence = max(0.0, 1.0 - (time_to_access / 3600))
                        if confidence >= self.prediction_threshold:
                            predictions.append((key, confidence, time_to_access))
        
        # Sort by confidence and time
        predictions.sort(key=lambda x: (-x[1], x[2]))
        
        # This would trigger preloading in a real implementation
        logger.debug(f"Predicted {len(predictions)} cache entries for preloading")
    
    def start_background_cleanup(self) -> None:
        """Start background cleanup thread."""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return
        
        self.cleanup_active = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        
        logger.info("Cache background cleanup started")
    
    def stop_background_cleanup(self) -> None:
        """Stop background cleanup thread."""
        self.cleanup_active = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        logger.info("Cache background cleanup stopped")
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.cleanup_active:
            try:
                self._cleanup_expired()
                self._optimize_data_structures()
                self.predict_and_preload()
                
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _cleanup_expired(self) -> None:
        """Clean up expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self.entries.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
                self.stats.evictions += 1
            
            if expired_keys and self.metric_collector:
                self.metric_collector.record_counter("cache.expired_entries_cleaned", len(expired_keys))
    
    def _optimize_data_structures(self) -> None:
        """Optimize internal data structures."""
        with self._lock:
            # Clean up orphaned access patterns
            orphaned_patterns = set(self.access_patterns.keys()) - set(self.entries.keys())
            for key in orphaned_patterns:
                del self.access_patterns[key]
            
            # Trigger garbage collection occasionally
            if len(self.entries) > 1000:
                gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats_dict = self.stats.to_dict()
            
            # Add additional statistics
            stats_dict.update({
                "max_size_bytes": self.max_size_bytes,
                "max_entries": self.max_entries,
                "policy": self.policy.value,
                "utilization_ratio": self.stats.size_bytes / self.max_size_bytes,
                "fill_ratio": self.stats.entry_count / self.max_entries,
                "avg_entry_size": self.stats.size_bytes / max(self.stats.entry_count, 1),
                "prediction_cache_size": len(self.prediction_cache),
                "access_patterns_tracked": len(self.access_patterns),
                "background_cleanup_active": self.cleanup_active
            })
            
            return stats_dict
    
    def cleanup(self) -> None:
        """Cleanup cache resources."""
        self.stop_background_cleanup()
        self.clear()
        
        logger.info("Intelligent cache cleaned up")


class FilterResultCache(IntelligentCache):
    """Specialized cache for FilterResult objects."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enable_content_hashing = True
        
    def cache_filter_result(self, request: FilterRequest, result: FilterResult,
                          ttl_seconds: int = 3600) -> bool:
        """Cache a filter result."""
        cache_key = self._generate_cache_key(request)
        
        # Calculate priority based on result characteristics
        priority = self._calculate_result_priority(result)
        
        metadata = {
            "safety_score": result.safety_score.overall_score if result.safety_score else 0.0,
            "was_filtered": result.was_filtered,
            "processing_time_ms": result.processing_time_ms,
            "content_length": len(request.content) if request.content else 0
        }
        
        return self.put(cache_key, result, ttl_seconds, priority, metadata)
    
    def get_cached_result(self, request: FilterRequest) -> Optional[FilterResult]:
        """Get cached result for a request."""
        cache_key = self._generate_cache_key(request)
        return self.get(cache_key)
    
    def _generate_cache_key(self, request: FilterRequest) -> str:
        """Generate cache key for a filter request."""
        # Include content hash, safety level, and other relevant parameters
        content_hash = hashlib.sha256(request.content.encode('utf-8')).hexdigest()
        
        key_components = [
            content_hash,
            str(request.safety_level),
            str(sorted(request.metadata.items()) if request.metadata else "")
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def _calculate_result_priority(self, result: FilterResult) -> float:
        """Calculate priority for a filter result."""
        priority = 1.0
        
        # Higher priority for results that took longer to compute
        if result.processing_time_ms > 1000:  # > 1 second
            priority += 0.5
        
        # Higher priority for filtered content (more complex processing)
        if result.was_filtered:
            priority += 0.3
        
        # Lower priority for very low safety scores (likely to change)
        if result.safety_score and result.safety_score.overall_score < 0.3:
            priority -= 0.2
        
        return max(0.1, priority)  # Minimum priority of 0.1