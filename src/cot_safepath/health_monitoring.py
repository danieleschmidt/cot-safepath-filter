"""
Health monitoring and system diagnostics for SafePath Filter - Generation 2.

Comprehensive health checks, system diagnostics, performance monitoring,
and automated health reporting for production deployments.
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

from .models import ProcessingMetrics, SafetyScore
from .exceptions import FilterError


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels."""
    
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of system components."""
    
    CORE_FILTER = "core_filter"
    DETECTOR = "detector"
    CACHE = "cache"
    DATABASE = "database"
    NETWORK = "network"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    component: str
    component_type: ComponentType
    status: HealthStatus
    timestamp: datetime
    response_time_ms: float
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    details: Optional[str] = None


@dataclass 
class SystemMetrics:
    """System-level performance metrics."""
    
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    available_memory_mb: float
    load_average: List[float]
    open_file_descriptors: int
    network_connections: int
    uptime_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HealthChecker:
    """Individual health checker for specific components."""
    
    def __init__(
        self,
        name: str,
        component_type: ComponentType,
        check_func: Callable[[], Union[bool, HealthCheckResult]],
        timeout_seconds: float = 5.0,
        interval_seconds: float = 60.0
    ):
        self.name = name
        self.component_type = component_type
        self.check_func = check_func
        self.timeout_seconds = timeout_seconds
        self.interval_seconds = interval_seconds
        self.last_check_time: Optional[datetime] = None
        self.last_result: Optional[HealthCheckResult] = None
        self.consecutive_failures = 0
        
    async def run_check(self) -> HealthCheckResult:
        """Execute health check with timeout and error handling."""
        start_time = time.time()
        
        try:
            # Run check with timeout
            if asyncio.iscoroutinefunction(self.check_func):
                result = await asyncio.wait_for(
                    self.check_func(), 
                    timeout=self.timeout_seconds
                )
            else:
                result = self.check_func()
            
            response_time = (time.time() - start_time) * 1000
            
            # Process result
            if isinstance(result, HealthCheckResult):
                result.response_time_ms = response_time
                status = result.status
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                result = HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=status,
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time,
                    message="Health check completed"
                )
            else:
                # Assume healthy if function returns without exception
                status = HealthStatus.HEALTHY
                result = HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=status,
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time,
                    message="Health check passed"
                )
            
            # Reset failure count on success
            if status == HealthStatus.HEALTHY:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            self.consecutive_failures += 1
            result = HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                message=f"Health check timed out after {self.timeout_seconds}s",
                details="Timeout exceeded"
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.consecutive_failures += 1
            result = HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                message=f"Health check failed: {str(e)[:100]}",
                details=str(e)
            )
        
        self.last_check_time = datetime.utcnow()
        self.last_result = result
        
        return result


class SystemDiagnostics:
    """System-level diagnostics and monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
        self._metrics_history: List[SystemMetrics] = []
        self.max_history_size = 1440  # 24 hours at 1 minute intervals
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_memory_mb = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_usage_percent = disk_usage.percent
            
            # Load average (Unix-like systems)
            try:
                load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                # Windows doesn't have load average
                load_average = [0.0, 0.0, 0.0]
            
            # File descriptors
            try:
                process = psutil.Process()
                open_fds = process.num_fds()
            except (AttributeError, psutil.NoSuchProcess):
                # Windows doesn't have num_fds
                open_fds = 0
            
            # Network connections
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, OSError):
                connections = 0
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage_percent=disk_usage_percent,
                available_memory_mb=available_memory_mb,
                load_average=load_average,
                open_file_descriptors=open_fds,
                network_connections=connections,
                uptime_seconds=uptime_seconds
            )
            
            # Store metrics history
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self.max_history_size:
                self._metrics_history = self._metrics_history[-self.max_history_size // 2:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics on error
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                available_memory_mb=0.0,
                load_average=[0.0, 0.0, 0.0],
                open_file_descriptors=0,
                network_connections=0,
                uptime_seconds=time.time() - self.start_time
            )
    
    def get_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """Get metrics history for specified time range."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            metrics for metrics in self._metrics_history 
            if metrics.timestamp >= cutoff
        ]
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from metrics history."""
        if len(self._metrics_history) < 2:
            return {}
        
        recent_metrics = self.get_metrics_history(hours=1)
        if not recent_metrics:
            return {}
        
        # Calculate trends
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        trends = {
            "cpu_trend": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "is_increasing": len(cpu_values) > 1 and cpu_values[-1] > cpu_values[-2]
            },
            "memory_trend": {
                "current": memory_values[-1] if memory_values else 0,
                "average": sum(memory_values) / len(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "is_increasing": len(memory_values) > 1 and memory_values[-1] > memory_values[-2]
            },
            "data_points": len(recent_metrics)
        }
        
        return trends


class HealthMonitor:
    """Main health monitoring coordinator."""
    
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.system_diagnostics = SystemDiagnostics()
        self.running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self.health_history: List[HealthCheckResult] = []
        self.max_history_size = 10000
        
        # Setup default health checks
        self._setup_default_checks()
        
    def _setup_default_checks(self):
        """Setup default health checks for SafePath components."""
        
        # Core filter health check
        def check_filter_health():
            """Basic filter functionality check."""
            try:
                from .core import SafePathFilter
                from .models import FilterRequest, SafetyLevel
                
                filter_instance = SafePathFilter()
                test_request = FilterRequest(
                    content="Test content for health check",
                    safety_level=SafetyLevel.BALANCED
                )
                result = filter_instance.filter(test_request)
                
                return HealthCheckResult(
                    component="core_filter",
                    component_type=ComponentType.CORE_FILTER,
                    status=HealthStatus.HEALTHY,
                    timestamp=datetime.utcnow(),
                    response_time_ms=0,  # Will be set by checker
                    message="Filter processing test successful"
                )
            except Exception as e:
                return HealthCheckResult(
                    component="core_filter",
                    component_type=ComponentType.CORE_FILTER,
                    status=HealthStatus.CRITICAL,
                    timestamp=datetime.utcnow(),
                    response_time_ms=0,
                    message=f"Filter test failed: {str(e)[:100]}"
                )
        
        self.add_health_check(
            "core_filter",
            ComponentType.CORE_FILTER,
            check_filter_health,
            timeout_seconds=10.0
        )
        
        # Memory health check
        def check_memory_health():
            """Check memory usage levels."""
            try:
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                if memory_percent > 90:
                    status = HealthStatus.CRITICAL
                    message = f"Critical memory usage: {memory_percent:.1f}%"
                elif memory_percent > 80:
                    status = HealthStatus.WARNING
                    message = f"High memory usage: {memory_percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Memory usage normal: {memory_percent:.1f}%"
                
                return HealthCheckResult(
                    component="memory",
                    component_type=ComponentType.MEMORY,
                    status=status,
                    timestamp=datetime.utcnow(),
                    response_time_ms=0,
                    message=message,
                    metadata={"memory_percent": memory_percent}
                )
            except Exception as e:
                return HealthCheckResult(
                    component="memory",
                    component_type=ComponentType.MEMORY,
                    status=HealthStatus.UNKNOWN,
                    timestamp=datetime.utcnow(),
                    response_time_ms=0,
                    message=f"Memory check failed: {str(e)}"
                )
        
        self.add_health_check(
            "memory",
            ComponentType.MEMORY,
            check_memory_health,
            timeout_seconds=5.0
        )
        
        # CPU health check
        def check_cpu_health():
            """Check CPU usage levels."""
            try:
                cpu_percent = psutil.cpu_percent(interval=1.0)
                
                if cpu_percent > 95:
                    status = HealthStatus.CRITICAL
                    message = f"Critical CPU usage: {cpu_percent:.1f}%"
                elif cpu_percent > 85:
                    status = HealthStatus.WARNING
                    message = f"High CPU usage: {cpu_percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"CPU usage normal: {cpu_percent:.1f}%"
                
                return HealthCheckResult(
                    component="cpu",
                    component_type=ComponentType.CPU,
                    status=status,
                    timestamp=datetime.utcnow(),
                    response_time_ms=0,
                    message=message,
                    metadata={"cpu_percent": cpu_percent}
                )
            except Exception as e:
                return HealthCheckResult(
                    component="cpu",
                    component_type=ComponentType.CPU,
                    status=HealthStatus.UNKNOWN,
                    timestamp=datetime.utcnow(),
                    response_time_ms=0,
                    message=f"CPU check failed: {str(e)}"
                )
        
        self.add_health_check(
            "cpu",
            ComponentType.CPU,
            check_cpu_health,
            timeout_seconds=5.0
        )
    
    def add_health_check(
        self,
        name: str,
        component_type: ComponentType,
        check_func: Callable,
        timeout_seconds: float = 5.0,
        interval_seconds: float = None
    ):
        """Add a health check."""
        if interval_seconds is None:
            interval_seconds = self.check_interval
            
        checker = HealthChecker(
            name=name,
            component_type=component_type,
            check_func=check_func,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds
        )
        
        self.health_checkers[name] = checker
        logger.info(f"Health check added: {name}")
    
    def remove_health_check(self, name: str):
        """Remove a health check."""
        if name in self.health_checkers:
            del self.health_checkers[name]
            logger.info(f"Health check removed: {name}")
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks and return results."""
        results = {}
        
        # Run health checks concurrently
        tasks = []
        for name, checker in self.health_checkers.items():
            task = asyncio.create_task(checker.run_check())
            tasks.append((name, task))
        
        # Wait for all checks to complete
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
                
                # Store in history
                self.health_history.append(result)
                
                # Log critical issues
                if result.status == HealthStatus.CRITICAL:
                    logger.error(f"Critical health issue: {name} - {result.message}")
                elif result.status == HealthStatus.WARNING:
                    logger.warning(f"Health warning: {name} - {result.message}")
                
            except Exception as e:
                logger.error(f"Health check {name} failed with exception: {e}")
                results[name] = HealthCheckResult(
                    component=name,
                    component_type=ComponentType.UNKNOWN,
                    status=HealthStatus.CRITICAL,
                    timestamp=datetime.utcnow(),
                    response_time_ms=0,
                    message=f"Health check failed: {str(e)}"
                )
        
        # Maintain history size
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size // 2:]
        
        return results
    
    async def start(self):
        """Start the health monitoring system."""
        if self.running:
            return
        
        self.running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def stop(self):
        """Stop the health monitoring system."""
        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Run health checks
                await self.run_all_checks()
                
                # Collect system metrics
                self.system_diagnostics.collect_system_metrics()
                
                # Wait for next interval
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.health_checkers:
            return {
                "overall_status": HealthStatus.UNKNOWN,
                "message": "No health checks configured",
                "component_count": 0
            }
        
        # Get latest results
        latest_results = {}
        for name, checker in self.health_checkers.items():
            if checker.last_result:
                latest_results[name] = checker.last_result
        
        if not latest_results:
            return {
                "overall_status": HealthStatus.UNKNOWN,
                "message": "No health check results available",
                "component_count": len(self.health_checkers)
            }
        
        # Determine overall status
        statuses = [result.status for result in latest_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
            message = "One or more critical issues detected"
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
            message = "One or more warnings detected"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All components healthy"
        
        # Component breakdown
        status_counts = {}
        for status in statuses:
            status_counts[status.value] = status_counts.get(status.value, 0) + 1
        
        return {
            "overall_status": overall_status,
            "message": message,
            "component_count": len(latest_results),
            "status_breakdown": status_counts,
            "components": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "last_check": result.timestamp.isoformat(),
                    "response_time_ms": result.response_time_ms
                }
                for name, result in latest_results.items()
            }
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        overall_health = self.get_overall_health()
        system_metrics = self.system_diagnostics.collect_system_metrics()
        performance_trends = self.system_diagnostics.analyze_performance_trends()
        
        # Recent health issues
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_issues = [
            {
                "component": result.component,
                "status": result.status.value,
                "message": result.message,
                "timestamp": result.timestamp.isoformat()
            }
            for result in self.health_history
            if result.timestamp >= one_hour_ago and result.status != HealthStatus.HEALTHY
        ]
        
        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "overall_health": overall_health,
            "system_metrics": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "disk_usage_percent": system_metrics.disk_usage_percent,
                "available_memory_mb": system_metrics.available_memory_mb,
                "uptime_seconds": system_metrics.uptime_seconds,
                "load_average": system_metrics.load_average,
                "open_file_descriptors": system_metrics.open_file_descriptors,
                "network_connections": system_metrics.network_connections
            },
            "performance_trends": performance_trends,
            "recent_issues_count": len(recent_issues),
            "recent_issues": recent_issues[:10]  # Limit to 10 most recent
        }


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None


def get_global_health_monitor() -> HealthMonitor:
    """Get or create global health monitor."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


async def start_health_monitoring(check_interval: float = 60.0):
    """Start global health monitoring."""
    monitor = get_global_health_monitor()
    monitor.check_interval = check_interval
    await monitor.start()
    return monitor


async def stop_health_monitoring():
    """Stop global health monitoring."""
    global _global_health_monitor
    if _global_health_monitor:
        await _global_health_monitor.stop()


def health_check_endpoint():
    """Simple health check endpoint for load balancers."""
    try:
        monitor = get_global_health_monitor()
        health = monitor.get_overall_health()
        
        if health["overall_status"] == HealthStatus.HEALTHY:
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        else:
            return {
                "status": health["overall_status"].value,
                "message": health["message"],
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }