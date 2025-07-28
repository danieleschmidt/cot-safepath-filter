#!/usr/bin/env python3
"""
Health check utilities for CoT SafePath Filter
Provides comprehensive health monitoring for all system components
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
import redis
import psycopg2
from psycopg2.extras import RealDictCursor


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    response_time_ms: float
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


class HealthChecker:
    """Comprehensive health checker for all system components."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.default_timeout = self.config.get('timeout', 10.0)
        self.database_url = self.config.get('database_url', 'postgresql://localhost:5432/safepath')
        self.redis_url = self.config.get('redis_url', 'redis://localhost:6379/0')
        self.api_base_url = self.config.get('api_base_url', 'http://localhost:8080')

    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks and return results."""
        checks = [
            self.check_application(),
            self.check_database(),
            self.check_redis(),
            self.check_disk_space(),
            self.check_memory(),
            self.check_external_apis(),
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        health_results = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed with exception: {result}")
                health_results['unknown_error'] = HealthCheckResult(
                    name="unknown_error",
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    message=f"Health check failed: {str(result)}"
                )
            else:
                health_results[result.name] = result
        
        return health_results

    async def check_application(self) -> HealthCheckResult:
        """Check application API health."""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.default_timeout) as client:
                response = await client.get(f"{self.api_base_url}/health")
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    return HealthCheckResult(
                        name="application",
                        status=HealthStatus.HEALTHY,
                        response_time_ms=response_time,
                        message="Application is responding normally",
                        details=response.json() if response.headers.get('content-type', '').startswith('application/json') else None
                    )
                else:
                    return HealthCheckResult(
                        name="application",
                        status=HealthStatus.UNHEALTHY,
                        response_time_ms=response_time,
                        message=f"Application returned status {response.status_code}"
                    )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="application",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Application health check failed: {str(e)}"
            )

    async def check_database(self) -> HealthCheckResult:
        """Check PostgreSQL database connectivity and performance."""
        start_time = time.time()
        
        try:
            # Parse database URL for connection
            import urllib.parse
            parsed = urllib.parse.urlparse(self.database_url)
            
            conn_params = {
                'host': parsed.hostname,
                'port': parsed.port or 5432,
                'database': parsed.path.lstrip('/'),
                'user': parsed.username,
                'password': parsed.password,
            }
            
            # Connect and run basic query
            conn = psycopg2.connect(**conn_params, connect_timeout=int(self.default_timeout))
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Test basic query
            cursor.execute("SELECT version(), current_timestamp, pg_database_size(current_database())")
            result = cursor.fetchone()
            
            # Test application tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'safepath'
            """)
            tables = cursor.fetchall()
            
            response_time = (time.time() - start_time) * 1000
            
            conn.close()
            
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                message="Database is accessible and responsive",
                details={
                    'version': result['version'].split()[1] if result else 'unknown',
                    'database_size_bytes': result['pg_database_size'] if result else 0,
                    'table_count': len(tables),
                    'tables': [t['table_name'] for t in tables]
                }
            )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Database health check failed: {str(e)}"
            )

    async def check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity and performance."""
        start_time = time.time()
        
        try:
            # Connect to Redis
            redis_client = redis.from_url(self.redis_url, socket_timeout=self.default_timeout)
            
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            redis_client.set(test_key, "test_value", ex=60)
            retrieved_value = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            # Get Redis info
            info = redis_client.info()
            
            response_time = (time.time() - start_time) * 1000
            
            if retrieved_value == b"test_value":
                return HealthCheckResult(
                    name="redis",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time,
                    message="Redis is accessible and responsive",
                    details={
                        'version': info.get('redis_version', 'unknown'),
                        'used_memory_human': info.get('used_memory_human', 'unknown'),
                        'connected_clients': info.get('connected_clients', 0),
                        'total_commands_processed': info.get('total_commands_processed', 0),
                        'keyspace_hits': info.get('keyspace_hits', 0),
                        'keyspace_misses': info.get('keyspace_misses', 0)
                    }
                )
            else:
                return HealthCheckResult(
                    name="redis",
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time,
                    message="Redis connected but set/get test failed"
                )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Redis health check failed: {str(e)}"
            )

    async def check_disk_space(self) -> HealthCheckResult:
        """Check available disk space."""
        start_time = time.time()
        
        try:
            import shutil
            
            # Check disk space for current directory
            total, used, free = shutil.disk_usage('.')
            
            free_percentage = (free / total) * 100
            used_percentage = (used / total) * 100
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on free space
            if free_percentage > 20:
                status = HealthStatus.HEALTHY
                message = f"Disk space is adequate ({free_percentage:.1f}% free)"
            elif free_percentage > 10:
                status = HealthStatus.DEGRADED
                message = f"Disk space is getting low ({free_percentage:.1f}% free)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Disk space is critically low ({free_percentage:.1f}% free)"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                response_time_ms=response_time,
                message=message,
                details={
                    'total_bytes': total,
                    'used_bytes': used,
                    'free_bytes': free,
                    'used_percentage': used_percentage,
                    'free_percentage': free_percentage
                }
            )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                response_time_ms=response_time,
                message=f"Disk space check failed: {str(e)}"
            )

    async def check_memory(self) -> HealthCheckResult:
        """Check memory usage."""
        start_time = time.time()
        
        try:
            import psutil
            
            # Get memory information
            memory = psutil.virtual_memory()
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on memory usage
            if memory.percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Memory usage is normal ({memory.percent:.1f}%)"
            elif memory.percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Memory usage is high ({memory.percent:.1f}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage is critically high ({memory.percent:.1f}%)"
            
            return HealthCheckResult(
                name="memory",
                status=status,
                response_time_ms=response_time,
                message=message,
                details={
                    'total_bytes': memory.total,
                    'available_bytes': memory.available,
                    'used_bytes': memory.used,
                    'free_bytes': memory.free,
                    'percent_used': memory.percent
                }
            )
        
        except ImportError:
            # psutil not available, skip this check
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                response_time_ms=response_time,
                message="Memory check skipped (psutil not available)"
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                response_time_ms=response_time,
                message=f"Memory check failed: {str(e)}"
            )

    async def check_external_apis(self) -> HealthCheckResult:
        """Check external API dependencies."""
        start_time = time.time()
        
        external_apis = [
            {'name': 'OpenAI', 'url': 'https://api.openai.com/v1/models', 'timeout': 5},
            {'name': 'Hugging Face', 'url': 'https://huggingface.co/api/models', 'timeout': 5},
        ]
        
        results = []
        
        try:
            async with httpx.AsyncClient() as client:
                for api in external_apis:
                    try:
                        response = await client.get(
                            api['url'], 
                            timeout=api['timeout'],
                            headers={'User-Agent': 'SafePath-HealthCheck/1.0'}
                        )
                        
                        if response.status_code < 500:
                            results.append({'name': api['name'], 'status': 'accessible'})
                        else:
                            results.append({'name': api['name'], 'status': 'degraded'})
                    
                    except Exception as e:
                        results.append({'name': api['name'], 'status': 'unreachable', 'error': str(e)})
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine overall status
            accessible_count = sum(1 for r in results if r['status'] == 'accessible')
            total_count = len(results)
            
            if accessible_count == total_count:
                status = HealthStatus.HEALTHY
                message = "All external APIs are accessible"
            elif accessible_count > 0:
                status = HealthStatus.DEGRADED
                message = f"Some external APIs are accessible ({accessible_count}/{total_count})"
            else:
                status = HealthStatus.UNHEALTHY
                message = "No external APIs are accessible"
            
            return HealthCheckResult(
                name="external_apis",
                status=status,
                response_time_ms=response_time,
                message=message,
                details={'api_results': results}
            )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="external_apis",
                status=HealthStatus.UNKNOWN,
                response_time_ms=response_time,
                message=f"External API check failed: {str(e)}"
            )

    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determine overall system health status."""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results.values()]
        
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report."""
        results = await self.check_all()
        overall_status = self.get_overall_status(results)
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': overall_status.value,
            'checks': {name: result.to_dict() for name, result in results.items()},
            'summary': {
                'total_checks': len(results),
                'healthy': sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                'degraded': sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
                'unhealthy': sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
                'unknown': sum(1 for r in results.values() if r.status == HealthStatus.UNKNOWN),
            }
        }


async def main():
    """CLI entry point for health checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SafePath Health Checker')
    parser.add_argument('--format', choices=['json', 'text'], default='text', help='Output format')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--timeout', type=float, default=10.0, help='Check timeout in seconds')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {'timeout': args.timeout}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))
    
    # Run health checks
    checker = HealthChecker(config)
    report = await checker.generate_health_report()
    
    if args.format == 'json':
        print(json.dumps(report, indent=2))
    else:
        # Text format
        print(f"SafePath Health Report - {report['timestamp']}")
        print(f"Overall Status: {report['overall_status'].upper()}")
        print("\nComponent Status:")
        print("-" * 50)
        
        for name, check in report['checks'].items():
            status_emoji = {
                'healthy': '✅',
                'degraded': '⚠️',
                'unhealthy': '❌',
                'unknown': '❓'
            }.get(check['status'], '❓')
            
            print(f"{status_emoji} {name}: {check['status']} ({check['response_time_ms']:.1f}ms)")
            print(f"   {check['message']}")
        
        print(f"\nSummary: {report['summary']['healthy']} healthy, {report['summary']['degraded']} degraded, {report['summary']['unhealthy']} unhealthy")


if __name__ == '__main__':
    asyncio.run(main())