"""
Caching strategies for SafePath filtering operations.
"""

import hashlib
import time
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import asdict

from .redis_manager import RedisManager, get_redis_manager
from ..models import FilterResult, SafetyScore
from ..exceptions import CacheError


logger = logging.getLogger(__name__)


class CacheStrategy:
    """Base caching strategy with common functionality."""
    
    def __init__(self, redis_manager: RedisManager = None, prefix: str = "safepath"):
        self.redis = redis_manager or get_redis_manager()
        self.prefix = prefix
        self.default_ttl = 3600  # 1 hour
    
    def _make_key(self, *parts: str) -> str:
        """Create a cache key with prefix."""
        return f"{self.prefix}:{':'.join(parts)}"
    
    def _hash_content(self, content: str) -> str:
        """Create hash of content for cache key."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace."""
        try:
            pattern = f"{self.prefix}:{namespace}:*"
            keys = self.redis.client.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to clear cache namespace '{namespace}': {e}")
            return 0


class FilterResultCache(CacheStrategy):
    """Caching strategy for filter operation results."""
    
    def __init__(self, redis_manager: RedisManager = None, ttl: int = 3600):
        super().__init__(redis_manager, "safepath:filter")
        self.ttl = ttl
    
    def get_cached_result(
        self, 
        content: str, 
        safety_level: str = None,
        context: str = None
    ) -> Optional[FilterResult]:
        """
        Get cached filter result.
        
        Args:
            content: Content to check cache for
            safety_level: Safety level used in filtering
            context: Additional context
            
        Returns:
            Cached FilterResult if found, None otherwise
        """
        try:
            cache_key = self._build_cache_key(content, safety_level, context)
            cached_data = self.redis.get(cache_key, deserialize=True)
            
            if cached_data:
                # Reconstruct FilterResult from cached data
                safety_score = SafetyScore(**cached_data['safety_score'])
                
                result = FilterResult(
                    filtered_content=cached_data['filtered_content'],
                    safety_score=safety_score,
                    was_filtered=cached_data['was_filtered'],
                    filter_reasons=cached_data['filter_reasons'],
                    original_content=cached_data.get('original_content'),
                    processing_time_ms=cached_data['processing_time_ms'],
                    request_id=cached_data.get('request_id'),
                    metadata=cached_data.get('metadata', {})
                )
                
                logger.debug(f"Cache hit for content hash: {self._hash_content(content)}")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached result: {e}")
            return None
    
    def cache_result(
        self,
        content: str,
        result: FilterResult,
        safety_level: str = None,
        context: str = None
    ) -> bool:
        """
        Cache a filter result.
        
        Args:
            content: Original content
            result: FilterResult to cache
            safety_level: Safety level used
            context: Additional context
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            cache_key = self._build_cache_key(content, safety_level, context)
            
            # Serialize FilterResult
            cache_data = {
                'filtered_content': result.filtered_content,
                'safety_score': asdict(result.safety_score),
                'was_filtered': result.was_filtered,
                'filter_reasons': result.filter_reasons,
                'original_content': result.original_content,
                'processing_time_ms': result.processing_time_ms,
                'request_id': result.request_id,
                'metadata': result.metadata,
                'cached_at': datetime.utcnow().isoformat()
            }
            
            success = self.redis.set(cache_key, cache_data, ttl=self.ttl)
            
            if success:
                logger.debug(f"Cached result for content hash: {self._hash_content(content)}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
            return False
    
    def invalidate_cache(self, content: str, safety_level: str = None, context: str = None) -> bool:
        """
        Invalidate cache for specific content.
        
        Args:
            content: Content to invalidate cache for
            safety_level: Safety level
            context: Additional context
            
        Returns:
            True if invalidated, False otherwise
        """
        try:
            cache_key = self._build_cache_key(content, safety_level, context)
            return self.redis.delete(cache_key) > 0
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            pattern = f"{self.prefix}:*"
            keys = self.redis.client.keys(pattern)
            
            total_keys = len(keys)
            total_memory = 0
            expired_keys = 0
            
            for key in keys[:100]:  # Sample first 100 keys
                ttl = self.redis.ttl(key)
                if ttl == -2:  # Key doesn't exist (expired)
                    expired_keys += 1
                
                # Estimate memory usage (rough approximation)
                try:
                    memory = self.redis.client.memory_usage(key)
                    if memory:
                        total_memory += memory
                except:
                    pass  # Not all Redis versions support MEMORY USAGE
            
            return {
                'total_keys': total_keys,
                'estimated_memory_bytes': total_memory,
                'expired_keys_sample': expired_keys,
                'hit_rate': self._calculate_hit_rate(),
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def _build_cache_key(self, content: str, safety_level: str = None, context: str = None) -> str:
        """Build cache key for content and parameters."""
        content_hash = self._hash_content(content)
        
        # Include safety level and context in key if provided
        key_parts = ["result", content_hash]
        
        if safety_level:
            key_parts.append(f"level:{safety_level}")
        
        if context:
            context_hash = self._hash_content(context)
            key_parts.append(f"ctx:{context_hash}")
        
        return self._make_key(*key_parts)
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate from Redis stats."""
        try:
            info = self.redis.info()
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            
            if hits + misses > 0:
                return hits / (hits + misses)
            
            return 0.0
            
        except Exception:
            return 0.0


class RateLimitCache(CacheStrategy):
    """Caching strategy for rate limiting."""
    
    def __init__(self, redis_manager: RedisManager = None):
        super().__init__(redis_manager, "safepath:ratelimit")
    
    def check_rate_limit(
        self, 
        identifier: str, 
        limit: int, 
        window_seconds: int
    ) -> Dict[str, Any]:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: Unique identifier (user ID, IP, etc.)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Dictionary with rate limit status
        """
        try:
            key = self._make_key("window", identifier)
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Use sliding window algorithm with sorted sets
            pipe = self.redis.pipeline()
            
            # Remove old entries outside the window
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, window_seconds + 60)  # Extra buffer
            
            results = pipe.execute()
            current_count = results[1] + 1  # +1 for the request we just added
            
            is_allowed = current_count <= limit
            remaining = max(0, limit - current_count)
            reset_time = current_time + window_seconds
            
            return {
                'allowed': is_allowed,
                'current_count': current_count,
                'limit': limit,
                'remaining': remaining,
                'reset_time': reset_time,
                'window_seconds': window_seconds
            }
            
        except Exception as e:
            logger.error(f"Rate limit check failed for '{identifier}': {e}")
            # On error, allow the request (fail open)
            return {
                'allowed': True,
                'current_count': 0,
                'limit': limit,
                'remaining': limit,
                'reset_time': int(time.time()) + window_seconds,
                'window_seconds': window_seconds,
                'error': str(e)
            }
    
    def get_rate_limit_status(self, identifier: str, window_seconds: int) -> Dict[str, Any]:
        """
        Get current rate limit status without incrementing.
        
        Args:
            identifier: Unique identifier
            window_seconds: Time window in seconds
            
        Returns:
            Current rate limit status
        """
        try:
            key = self._make_key("window", identifier)
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Clean up old entries and count current
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            results = pipe.execute()
            
            current_count = results[1]
            
            return {
                'current_count': current_count,
                'window_start': window_start,
                'window_end': current_time,
                'window_seconds': window_seconds
            }
            
        except Exception as e:
            logger.error(f"Failed to get rate limit status for '{identifier}': {e}")
            return {'current_count': 0, 'error': str(e)}
    
    def reset_rate_limit(self, identifier: str) -> bool:
        """
        Reset rate limit for identifier.
        
        Args:
            identifier: Unique identifier to reset
            
        Returns:
            True if reset successfully
        """
        try:
            key = self._make_key("window", identifier)
            return self.redis.delete(key) > 0
        except Exception as e:
            logger.error(f"Failed to reset rate limit for '{identifier}': {e}")
            return False


class SessionCache(CacheStrategy):
    """Caching strategy for user sessions."""
    
    def __init__(self, redis_manager: RedisManager = None, ttl: int = 86400):
        super().__init__(redis_manager, "safepath:session")
        self.ttl = ttl  # 24 hours default
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        try:
            key = self._make_key("data", session_id)
            return self.redis.get(key, deserialize=True)
        except Exception as e:
            logger.error(f"Failed to get session '{session_id}': {e}")
            return None
    
    def set_session(self, session_id: str, data: Dict[str, Any], ttl: int = None) -> bool:
        """Set session data."""
        try:
            key = self._make_key("data", session_id)
            return self.redis.set(key, data, ttl=ttl or self.ttl)
        except Exception as e:
            logger.error(f"Failed to set session '{session_id}': {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session data."""
        try:
            key = self._make_key("data", session_id)
            return self.redis.delete(key) > 0
        except Exception as e:
            logger.error(f"Failed to delete session '{session_id}': {e}")
            return False
    
    def extend_session(self, session_id: str, ttl: int = None) -> bool:
        """Extend session expiration."""
        try:
            key = self._make_key("data", session_id)
            return self.redis.expire(key, ttl or self.ttl)
        except Exception as e:
            logger.error(f"Failed to extend session '{session_id}': {e}")
            return False