"""
Redis connection and management for caching operations.
"""

import os
import json
import logging
from typing import Any, Optional, Dict, List, Union
from datetime import timedelta
import redis
from redis.connection import ConnectionPool
from redis.exceptions import RedisError, ConnectionError

from ..exceptions import ConfigurationError


logger = logging.getLogger(__name__)


class RedisManager:
    """Redis connection manager with connection pooling and error handling."""
    
    def __init__(
        self,
        redis_url: str = None,
        max_connections: int = None,
        timeout: int = None,
        **kwargs
    ):
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL", 
            "redis://localhost:6379/0"
        )
        self.max_connections = max_connections or int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
        self.timeout = timeout or int(os.getenv("REDIS_TIMEOUT", "5"))
        
        # Create connection pool
        try:
            self.connection_pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                socket_timeout=self.timeout,
                socket_connect_timeout=self.timeout,
                health_check_interval=30,
                **kwargs
            )
            
            # Create Redis client
            self.client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=True,
                socket_timeout=self.timeout
            )
            
            # Test connection
            self.client.ping()
            logger.info(f"Redis connection established: {self._mask_url(self.redis_url)}")
            
        except (RedisError, ConnectionError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConfigurationError(f"Redis connection failed: {e}")
    
    def get_client(self) -> redis.Redis:
        """Get Redis client instance."""
        return self.client
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = None, 
        serialize: bool = True
    ) -> bool:
        """
        Set a key-value pair in Redis.
        
        Args:
            key: Redis key
            value: Value to store
            ttl: Time to live in seconds
            serialize: Whether to JSON serialize the value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if serialize and not isinstance(value, str):
                value = json.dumps(value, default=str)
            
            if ttl:
                return self.client.setex(key, ttl, value)
            else:
                return self.client.set(key, value)
                
        except RedisError as e:
            logger.error(f"Redis SET failed for key '{key}': {e}")
            return False
    
    def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """
        Get a value from Redis.
        
        Args:
            key: Redis key
            deserialize: Whether to JSON deserialize the value
            
        Returns:
            The value if found, None otherwise
        """
        try:
            value = self.client.get(key)
            if value is None:
                return None
            
            if deserialize:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            
            return value
            
        except RedisError as e:
            logger.error(f"Redis GET failed for key '{key}': {e}")
            return None
    
    def delete(self, *keys: str) -> int:
        """
        Delete one or more keys from Redis.
        
        Args:
            keys: Redis keys to delete
            
        Returns:
            Number of keys deleted
        """
        try:
            return self.client.delete(*keys)
        except RedisError as e:
            logger.error(f"Redis DELETE failed for keys {keys}: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: Redis key to check
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            return bool(self.client.exists(key))
        except RedisError as e:
            logger.error(f"Redis EXISTS failed for key '{key}': {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for a key.
        
        Args:
            key: Redis key
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.client.expire(key, ttl)
        except RedisError as e:
            logger.error(f"Redis EXPIRE failed for key '{key}': {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """
        Get remaining time to live for a key.
        
        Args:
            key: Redis key
            
        Returns:
            TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        try:
            return self.client.ttl(key)
        except RedisError as e:
            logger.error(f"Redis TTL failed for key '{key}': {e}")
            return -2
    
    def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a key's value.
        
        Args:
            key: Redis key
            amount: Amount to increment by
            
        Returns:
            New value after increment, None if failed
        """
        try:
            return self.client.incr(key, amount)
        except RedisError as e:
            logger.error(f"Redis INCR failed for key '{key}': {e}")
            return None
    
    def sadd(self, key: str, *values: str) -> int:
        """
        Add members to a set.
        
        Args:
            key: Redis key
            values: Values to add to set
            
        Returns:
            Number of new members added
        """
        try:
            return self.client.sadd(key, *values)
        except RedisError as e:
            logger.error(f"Redis SADD failed for key '{key}': {e}")
            return 0
    
    def sismember(self, key: str, value: str) -> bool:
        """
        Check if value is member of set.
        
        Args:
            key: Redis key
            value: Value to check
            
        Returns:
            True if member of set, False otherwise
        """
        try:
            return self.client.sismember(key, value)
        except RedisError as e:
            logger.error(f"Redis SISMEMBER failed for key '{key}': {e}")
            return False
    
    def hset(self, key: str, mapping: Dict[str, Any]) -> int:
        """
        Set multiple hash fields.
        
        Args:
            key: Redis key
            mapping: Dictionary of field-value pairs
            
        Returns:
            Number of fields set
        """
        try:
            # Serialize values if needed
            serialized_mapping = {}
            for field, value in mapping.items():
                if isinstance(value, (dict, list)):
                    serialized_mapping[field] = json.dumps(value, default=str)
                else:
                    serialized_mapping[field] = str(value)
            
            return self.client.hset(key, mapping=serialized_mapping)
        except RedisError as e:
            logger.error(f"Redis HSET failed for key '{key}': {e}")
            return 0
    
    def hget(self, key: str, field: str) -> Optional[str]:
        """
        Get hash field value.
        
        Args:
            key: Redis key
            field: Hash field
            
        Returns:
            Field value if exists, None otherwise
        """
        try:
            return self.client.hget(key, field)
        except RedisError as e:
            logger.error(f"Redis HGET failed for key '{key}', field '{field}': {e}")
            return None
    
    def hgetall(self, key: str) -> Dict[str, str]:
        """
        Get all hash fields and values.
        
        Args:
            key: Redis key
            
        Returns:
            Dictionary of field-value pairs
        """
        try:
            return self.client.hgetall(key)
        except RedisError as e:
            logger.error(f"Redis HGETALL failed for key '{key}': {e}")
            return {}
    
    def pipeline(self):
        """Get Redis pipeline for batch operations."""
        return self.client.pipeline()
    
    def flushdb(self) -> bool:
        """
        Flush current database (use with caution!).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.flushdb()
            logger.warning("Redis database flushed")
            return True
        except RedisError as e:
            logger.error(f"Redis FLUSHDB failed: {e}")
            return False
    
    def info(self) -> Dict[str, Any]:
        """
        Get Redis server information.
        
        Returns:
            Dictionary with server info
        """
        try:
            return self.client.info()
        except RedisError as e:
            logger.error(f"Redis INFO failed: {e}")
            return {}
    
    def ping(self) -> bool:
        """
        Ping Redis server to check connectivity.
        
        Returns:
            True if ping successful, False otherwise
        """
        try:
            return self.client.ping()
        except RedisError as e:
            logger.error(f"Redis PING failed: {e}")
            return False
    
    def close(self):
        """Close Redis connections."""
        try:
            self.connection_pool.disconnect()
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")
    
    def _mask_url(self, url: str) -> str:
        """Mask sensitive information in Redis URL."""
        if "@" in url:
            parts = url.split("@")
            if "://" in parts[0]:
                protocol_user = parts[0].split("://")
                if ":" in protocol_user[1]:
                    user_pass = protocol_user[1].split(":")
                    masked = f"{protocol_user[0]}://{user_pass[0]}:***@{parts[1]}"
                    return masked
        return url


# Global Redis manager instance
_redis_manager: Optional[RedisManager] = None


def init_redis(redis_url: str = None, **kwargs) -> RedisManager:
    """Initialize the global Redis manager."""
    global _redis_manager
    _redis_manager = RedisManager(redis_url, **kwargs)
    return _redis_manager


def get_redis_manager() -> RedisManager:
    """Get the global Redis manager instance."""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = init_redis()
    return _redis_manager


def get_redis_client() -> redis.Redis:
    """Get Redis client from the global manager."""
    return get_redis_manager().get_client()