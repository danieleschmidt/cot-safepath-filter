"""
Advanced caching system for CoT SafePath Filter.
"""

import time
import hashlib
import pickle
import threading
from typing import Any, Optional, Dict, List, Union, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict
import json
import logging
from abc import ABC, abstractmethod

from .models import FilterResult, SafetyScore
from .utils import create_fingerprint
from .exceptions import CacheError


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[float] = None
    tags: List[str] = None
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        
        # Estimate size
        if self.size_bytes == 0:
            try:
                self.size_bytes = len(pickle.dumps(self.value))
            except:
                self.size_bytes = len(str(self.value).encode('utf-8'))
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheStrategy(ABC):
    """Abstract base class for cache eviction strategies."""
    
    @abstractmethod
    def should_evict(self, entries: Dict[str, CacheEntry], new_entry_size: int) -> List[str]:
        """Determine which entries to evict."""
        pass


class LRUStrategy(CacheStrategy):
    """Least Recently Used eviction strategy."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
    
    def should_evict(self, entries: Dict[str, CacheEntry], new_entry_size: int) -> List[str]:
        current_size = sum(entry.size_bytes for entry in entries.values())
        
        if current_size + new_entry_size <= self.max_size:
            return []
        
        # Sort by access time (oldest first)
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].accessed_at
        )
        
        to_evict = []
        size_to_free = current_size + new_entry_size - self.max_size
        freed_size = 0
        
        for key, entry in sorted_entries:
            to_evict.append(key)
            freed_size += entry.size_bytes
            
            if freed_size >= size_to_free:
                break
        
        return to_evict


class LFUStrategy(CacheStrategy):
    """Least Frequently Used eviction strategy."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
    
    def should_evict(self, entries: Dict[str, CacheEntry], new_entry_size: int) -> List[str]:
        current_size = sum(entry.size_bytes for entry in entries.values())
        
        if current_size + new_entry_size <= self.max_size:
            return []
        
        # Sort by access count (least frequent first)
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: (x[1].access_count, x[1].accessed_at)
        )
        
        to_evict = []
        size_to_free = current_size + new_entry_size - self.max_size
        freed_size = 0
        
        for key, entry in sorted_entries:
            to_evict.append(key)
            freed_size += entry.size_bytes
            
            if freed_size >= size_to_free:
                break
        
        return to_evict


class TTLStrategy(CacheStrategy):
    """Time-to-Live based eviction strategy."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
    
    def should_evict(self, entries: Dict[str, CacheEntry], new_entry_size: int) -> List[str]:
        # First, remove expired entries
        expired = [key for key, entry in entries.items() if entry.is_expired()]
        
        # Calculate remaining size after removing expired
        remaining_entries = {k: v for k, v in entries.items() if not v.is_expired()}
        current_size = sum(entry.size_bytes for entry in remaining_entries.values())
        
        if current_size + new_entry_size <= self.max_size:
            return expired
        
        # If still need space, use LRU for non-expired entries
        lru_strategy = LRUStrategy(self.max_size)
        additional_evict = lru_strategy.should_evict(remaining_entries, new_entry_size)
        
        return expired + additional_evict


class MultiLevelCache:
    """Multi-level cache with different strategies per level."""
    
    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 1000,
        l1_strategy: Optional[CacheStrategy] = None,
        l2_strategy: Optional[CacheStrategy] = None
    ):
        self.l1_cache = AdvancedCache(
            max_size=l1_size,
            strategy=l1_strategy or LRUStrategy(l1_size)
        )
        
        self.l2_cache = AdvancedCache(
            max_size=l2_size,
            strategy=l2_strategy or LFUStrategy(l2_size)
        )
        
        logger.info(f"Multi-level cache initialized: L1={l1_size}, L2={l2_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, checking L1 first, then L2."""
        # Check L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Check L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.set(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: List[str] = None) -> None:
        """Set value in both cache levels."""
        self.l1_cache.set(key, value, ttl, tags)
        self.l2_cache.set(key, value, ttl, tags)
    
    def delete(self, key: str) -> bool:
        """Delete from both cache levels."""
        l1_deleted = self.l1_cache.delete(key)
        l2_deleted = self.l2_cache.delete(key)
        return l1_deleted or l2_deleted
    
    def clear(self) -> None:
        """Clear both cache levels."""
        self.l1_cache.clear()
        self.l2_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from both cache levels."""
        return {
            'l1': self.l1_cache.get_stats(),
            'l2': self.l2_cache.get_stats()
        }


class AdvancedCache:
    """Advanced cache implementation with multiple eviction strategies."""
    
    def __init__(
        self,
        max_size: int = 10_000_000,  # 10MB default
        default_ttl: Optional[float] = 3600,  # 1 hour default
        strategy: Optional[CacheStrategy] = None
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy or LRUStrategy(max_size)
        
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._sets = 0
        
        logger.info(f"Advanced cache initialized: max_size={max_size}, strategy={strategy.__class__.__name__}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._entries.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired():
                del self._entries[key]
                self._misses += 1
                return None
            
            entry.touch()
            self._hits += 1
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: List[str] = None
    ) -> None:
        """Set value in cache."""
        with self._lock:
            # Create entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=0,
                ttl=ttl or self.default_ttl,
                tags=tags or []
            )
            
            # Check if we need to evict
            to_evict = self.strategy.should_evict(self._entries, entry.size_bytes)
            
            # Evict entries
            for evict_key in to_evict:
                if evict_key in self._entries:
                    del self._entries[evict_key]
                    self._evictions += 1
            
            # Store entry
            self._entries[key] = entry
            self._sets += 1
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._entries.clear()
    
    def get_by_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Get all entries that have any of the specified tags."""
        with self._lock:
            result = {}
            for key, entry in self._entries.items():
                if entry.is_expired():
                    continue
                
                if any(tag in entry.tags for tag in tags):
                    entry.touch()
                    result[key] = entry.value
            
            return result
    
    def delete_by_tags(self, tags: List[str]) -> int:
        """Delete all entries that have any of the specified tags."""
        with self._lock:
            to_delete = []
            
            for key, entry in self._entries.items():
                if any(tag in entry.tags for tag in tags):
                    to_delete.append(key)
            
            for key in to_delete:
                del self._entries[key]
            
            return len(to_delete)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._entries.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._entries[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / max(total_requests, 1)
            
            current_size = sum(entry.size_bytes for entry in self._entries.values())
            entry_count = len(self._entries)
            
            # Calculate average entry size
            avg_entry_size = current_size / max(entry_count, 1)
            
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'sets': self._sets,
                'evictions': self._evictions,
                'entry_count': entry_count,
                'current_size_bytes': current_size,
                'max_size_bytes': self.max_size,
                'utilization': current_size / self.max_size,
                'avg_entry_size_bytes': avg_entry_size,
                'strategy': self.strategy.__class__.__name__
            }


class FilterResultCache:
    """Specialized cache for filter results."""
    
    def __init__(self, max_size: int = 5_000_000):  # 5MB default
        self.cache = MultiLevelCache(
            l1_size=max_size // 10,  # 10% in L1 (hot cache)
            l2_size=max_size,        # 90% in L2 (warm cache)
            l1_strategy=LRUStrategy(max_size // 10),
            l2_strategy=TTLStrategy(max_size)
        )
        
    def get_cached_result(
        self,
        content: str,
        safety_level: str,
        context: Optional[str] = None
    ) -> Optional[FilterResult]:
        """Get cached filter result."""
        cache_key = self._create_cache_key(content, safety_level, context)
        
        cached_data = self.cache.get(cache_key)
        if cached_data is None:
            return None
        
        try:
            # Deserialize FilterResult
            return FilterResult(**cached_data)
        except Exception as e:
            logger.error(f"Failed to deserialize cached result: {e}")
            self.cache.delete(cache_key)
            return None
    
    def cache_result(
        self,
        content: str,
        result: FilterResult,
        safety_level: str,
        context: Optional[str] = None,
        ttl: Optional[float] = 3600
    ) -> None:
        """Cache filter result."""
        cache_key = self._create_cache_key(content, safety_level, context)
        
        # Serialize FilterResult to dict
        result_data = {
            'filtered_content': result.filtered_content,
            'safety_score': asdict(result.safety_score),
            'was_filtered': result.was_filtered,
            'filter_reasons': result.filter_reasons,
            'processing_time_ms': result.processing_time_ms,
            'request_id': result.request_id,
            'metadata': result.metadata
        }
        
        # Add cache tags for easier management
        tags = [
            f"safety_level:{safety_level}",
            f"was_filtered:{result.was_filtered}",
            f"score_range:{self._get_score_range(result.safety_score.overall_score)}"
        ]
        
        self.cache.set(cache_key, result_data, ttl=ttl, tags=tags)
    
    def _create_cache_key(self, content: str, safety_level: str, context: Optional[str]) -> str:
        """Create cache key from content and parameters."""
        key_data = {
            'content': content,
            'safety_level': safety_level,
            'context': context or ""
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def _get_score_range(self, score: float) -> str:
        """Get score range for tagging."""
        if score >= 0.9:
            return "very_safe"
        elif score >= 0.7:
            return "safe"
        elif score >= 0.5:
            return "moderate"
        elif score >= 0.3:
            return "unsafe"
        else:
            return "very_unsafe"
    
    def invalidate_by_safety_level(self, safety_level: str) -> int:
        """Invalidate all cached results for a specific safety level."""
        return self.cache.l1_cache.delete_by_tags([f"safety_level:{safety_level}"])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


class DetectionResultCache:
    """Cache for individual detector results."""
    
    def __init__(self, max_size: int = 2_000_000):  # 2MB default
        self.cache = AdvancedCache(
            max_size=max_size,
            default_ttl=1800,  # 30 minutes
            strategy=LFUStrategy(max_size)  # Keep frequently accessed detector results
        )
    
    def get_detection_result(self, detector_name: str, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached detection result."""
        cache_key = f"{detector_name}:{content_hash}"
        return self.cache.get(cache_key)
    
    def cache_detection_result(
        self,
        detector_name: str,
        content_hash: str,
        result: Dict[str, Any],
        ttl: Optional[float] = None
    ) -> None:
        """Cache detection result."""
        cache_key = f"{detector_name}:{content_hash}"
        tags = [f"detector:{detector_name}"]
        
        self.cache.set(cache_key, result, ttl=ttl, tags=tags)
    
    def invalidate_detector(self, detector_name: str) -> int:
        """Invalidate all cached results for a specific detector."""
        return self.cache.delete_by_tags([f"detector:{detector_name}"])


class CacheManager:
    """Centralized cache management."""
    
    def __init__(self):
        self.filter_cache = FilterResultCache()
        self.detection_cache = DetectionResultCache()
        
        # Background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._running = True
        self._cleanup_thread.start()
        
        logger.info("Cache manager initialized")
    
    def _cleanup_worker(self):
        """Background worker to clean up expired entries."""
        while self._running:
            try:
                # Clean up every 5 minutes
                time.sleep(300)
                
                # Cleanup filter cache
                filter_cleaned = self.filter_cache.cache.l1_cache.cleanup_expired()
                filter_cleaned += self.filter_cache.cache.l2_cache.cleanup_expired()
                
                # Cleanup detection cache  
                detection_cleaned = self.detection_cache.cache.cleanup_expired()
                
                if filter_cleaned + detection_cleaned > 0:
                    logger.debug(f"Cache cleanup: {filter_cleaned} filter, {detection_cleaned} detection entries removed")
                    
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches."""
        return {
            'filter_cache': self.filter_cache.get_cache_stats(),
            'detection_cache': self.detection_cache.cache.get_stats(),
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
        }
    
    def shutdown(self):
        """Shutdown cache manager."""
        self._running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)
        logger.info("Cache manager shut down")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_filter_cache() -> FilterResultCache:
    """Get the filter result cache."""
    return get_cache_manager().filter_cache


def get_detection_cache() -> DetectionResultCache:
    """Get the detection result cache."""
    return get_cache_manager().detection_cache