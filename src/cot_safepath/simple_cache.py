"""
Simple caching system without external dependencies.
"""

import time
import threading
from typing import Any, Optional, Dict


class AdvancedCache:
    """Simple cache implementation for testing."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                # Check expiration
                if entry.get('expires', float('inf')) > time.time():
                    entry['accessed'] = time.time()
                    entry['access_count'] = entry.get('access_count', 0) + 1
                    self._hits += 1
                    return entry['value']
                else:
                    del self._cache[key]
            
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = 3600) -> None:
        """Set value in cache."""
        with self._lock:
            # Simple eviction if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Remove oldest entry
                oldest_key = min(
                    self._cache.keys(), 
                    key=lambda k: self._cache[k].get('accessed', 0)
                )
                del self._cache[oldest_key]
            
            expires = time.time() + ttl if ttl else float('inf')
            self._cache[key] = {
                'value': value,
                'created': time.time(),
                'accessed': time.time(),
                'expires': expires,
                'access_count': 0
            }
    
    def delete(self, key: str) -> bool:
        """Delete from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / max(total_requests, 1)
            
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'size': len(self._cache),
                'max_size': self.max_size
            }