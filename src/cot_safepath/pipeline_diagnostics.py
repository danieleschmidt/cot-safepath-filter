"""
Advanced Pipeline Diagnostics - Generation 1 Core Monitoring.

Real-time diagnosis and analysis of pipeline performance and failures.
"""

import asyncio
import time
import statistics
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
from collections import defaultdict, deque
import weakref

from .models import FilterResult, ProcessingMetrics, SafetyScore
from .exceptions import FilterError, TimeoutError, DetectorError
from .self_healing_core import HealthStatus, HealthMetric


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostic findings."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PerformanceIssue(Enum):
    """Types of performance issues that can be detected."""
    HIGH_LATENCY = "high_latency"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    QUEUE_BACKLOG = "queue_backlog"
    DETECTOR_FAILURE = "detector_failure"
    CACHE_MISS_RATE = "cache_miss_rate"
    THROUGHPUT_DROP = "throughput_drop"


@dataclass
class DiagnosticFinding:
    """A diagnostic finding about pipeline performance."""
    issue_type: PerformanceIssue
    severity: DiagnosticSeverity
    component: str
    description: str
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolution_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "component": self.component,
            "description": self.description,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "resolution_suggestions": self.resolution_suggestions
        }


@dataclass
class PerformanceSnapshot:
    """Snapshot of pipeline performance at a point in time."""
    timestamp: datetime
    request_rate: float
    avg_latency: float
    error_rate: float
    memory_usage: float
    cpu_usage: float
    active_connections: int
    cache_hit_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "request_rate": self.request_rate,
            "avg_latency": self.avg_latency,
            "error_rate": self.error_rate,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "active_connections": self.active_connections,
            "cache_hit_rate": self.cache_hit_rate
        }


class ComponentDiagnostics:
    """Diagnostics for a single pipeline component."""
    
    def __init__(self, component_name: str, max_history: int = 1000):
        self.component_name = component_name
        self.max_history = max_history
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=max_history)
        self.error_counts: deque = deque(maxlen=max_history)
        self.memory_samples: deque = deque(maxlen=max_history)
        self.timestamps: deque = deque(maxlen=max_history)
        
        # Current state
        self.last_error: Optional[Exception] = None
        self.consecutive_errors = 0
        self.total_requests = 0
        self.total_errors = 0
        
        # Performance baselines
        self.baseline_latency = 1.0  # seconds
        self.baseline_memory = 100.0  # MB
        self.error_rate_threshold = 0.05  # 5%
        
    def record_request(self, response_time: float, error: Optional[Exception] = None, 
                      memory_usage: Optional[float] = None) -> None:
        """Record a request execution."""
        now = datetime.utcnow()
        
        self.total_requests += 1
        self.response_times.append(response_time)
        self.timestamps.append(now)
        
        if error:
            self.total_errors += 1
            self.consecutive_errors += 1
            self.error_counts.append(1)
            self.last_error = error
        else:
            self.consecutive_errors = 0
            self.error_counts.append(0)
        
        if memory_usage is not None:
            self.memory_samples.append(memory_usage)
    
    def analyze_performance(self) -> List[DiagnosticFinding]:
        """Analyze component performance and identify issues."""
        findings = []
        
        if not self.response_times:
            return findings
        
        # Analyze latency
        avg_latency = statistics.mean(self.response_times)
        p95_latency = statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else avg_latency
        
        if avg_latency > self.baseline_latency * 2:
            findings.append(DiagnosticFinding(
                issue_type=PerformanceIssue.HIGH_LATENCY,
                severity=DiagnosticSeverity.WARNING if avg_latency < self.baseline_latency * 5 else DiagnosticSeverity.ERROR,
                component=self.component_name,
                description=f"Average latency {avg_latency:.2f}s exceeds baseline {self.baseline_latency}s",
                metrics={"avg_latency": avg_latency, "p95_latency": p95_latency, "baseline": self.baseline_latency},
                resolution_suggestions=[
                    "Check for resource contention",
                    "Review component implementation efficiency",
                    "Consider scaling resources"
                ]
            ))
        
        # Analyze error rate
        if self.total_requests > 0:
            error_rate = self.total_errors / self.total_requests
            if error_rate > self.error_rate_threshold:
                findings.append(DiagnosticFinding(
                    issue_type=PerformanceIssue.DETECTOR_FAILURE,
                    severity=DiagnosticSeverity.ERROR if error_rate > 0.2 else DiagnosticSeverity.WARNING,
                    component=self.component_name,
                    description=f"Error rate {error_rate:.1%} exceeds threshold {self.error_rate_threshold:.1%}",
                    metrics={"error_rate": error_rate, "threshold": self.error_rate_threshold, "consecutive_errors": self.consecutive_errors},
                    resolution_suggestions=[
                        "Review recent error logs",
                        "Check component dependencies",
                        "Validate input data quality"
                    ]
                ))
        
        # Analyze memory usage
        if self.memory_samples:
            avg_memory = statistics.mean(self.memory_samples)
            max_memory = max(self.memory_samples)
            
            if len(self.memory_samples) > 10:
                # Check for memory growth trend
                recent_samples = list(self.memory_samples)[-10:]
                older_samples = list(self.memory_samples)[:-10] if len(self.memory_samples) > 20 else recent_samples
                
                if older_samples:
                    recent_avg = statistics.mean(recent_samples)
                    older_avg = statistics.mean(older_samples)
                    
                    if recent_avg > older_avg * 1.5:  # 50% increase
                        findings.append(DiagnosticFinding(
                            issue_type=PerformanceIssue.MEMORY_LEAK,
                            severity=DiagnosticSeverity.WARNING,
                            component=self.component_name,
                            description=f"Memory usage trending upward: {recent_avg:.1f}MB vs {older_avg:.1f}MB",
                            metrics={"recent_avg": recent_avg, "older_avg": older_avg, "max_memory": max_memory},
                            resolution_suggestions=[
                                "Check for memory leaks in component",
                                "Review cache cleanup policies",
                                "Monitor garbage collection"
                            ]
                        ))
        
        # Analyze consecutive errors
        if self.consecutive_errors > 5:
            findings.append(DiagnosticFinding(
                issue_type=PerformanceIssue.DETECTOR_FAILURE,
                severity=DiagnosticSeverity.CRITICAL,
                component=self.component_name,
                description=f"Component has {self.consecutive_errors} consecutive failures",
                metrics={"consecutive_errors": self.consecutive_errors},
                resolution_suggestions=[
                    "Immediate investigation required",
                    "Consider disabling component temporarily",
                    "Check system resources"
                ]
            ))
        
        return findings
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this component."""
        if not self.response_times:
            return {"component": self.component_name, "status": "no_data"}
        
        error_rate = self.total_errors / self.total_requests if self.total_requests > 0 else 0
        avg_latency = statistics.mean(self.response_times)
        
        return {
            "component": self.component_name,
            "total_requests": self.total_requests,
            "error_rate": error_rate,
            "avg_latency": avg_latency,
            "consecutive_errors": self.consecutive_errors,
            "last_error": str(self.last_error) if self.last_error else None,
            "memory_usage": statistics.mean(self.memory_samples) if self.memory_samples else None
        }


class PipelineDiagnosticEngine:
    """Comprehensive diagnostic engine for the entire pipeline."""
    
    def __init__(self):
        self.component_diagnostics: Dict[str, ComponentDiagnostics] = {}
        self.performance_history: deque = deque(maxlen=1440)  # 24 hours of minute-by-minute data
        self.diagnostic_findings: List[DiagnosticFinding] = []
        self.max_findings = 1000
        
        # Global performance tracking
        self.system_start_time = datetime.utcnow()
        self.total_system_requests = 0
        self.total_system_errors = 0
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.finding_queue = queue.Queue()
        
    def register_component(self, component_name: str) -> None:
        """Register a component for diagnostics."""
        if component_name not in self.component_diagnostics:
            self.component_diagnostics[component_name] = ComponentDiagnostics(component_name)
    
    def record_component_request(self, component_name: str, response_time: float, 
                                error: Optional[Exception] = None, memory_usage: Optional[float] = None) -> None:
        """Record a component request for diagnostics."""
        if component_name not in self.component_diagnostics:
            self.register_component(component_name)
        
        self.component_diagnostics[component_name].record_request(response_time, error, memory_usage)
        
        # Update system totals
        self.total_system_requests += 1
        if error:
            self.total_system_errors += 1
    
    def start_monitoring(self) -> None:
        """Start the diagnostic monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop the diagnostic monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for continuous diagnostics."""
        while self.monitoring_active:
            try:
                # Perform diagnostic analysis
                self._perform_diagnostic_analysis()
                
                # Take performance snapshot
                self._take_performance_snapshot()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                print(f"Error in diagnostic monitoring: {e}")
                time.sleep(5)
    
    def _perform_diagnostic_analysis(self) -> None:
        """Perform comprehensive diagnostic analysis."""
        new_findings = []
        
        # Analyze each component
        for component_name, diagnostics in self.component_diagnostics.items():
            component_findings = diagnostics.analyze_performance()
            new_findings.extend(component_findings)
        
        # Analyze system-wide patterns
        system_findings = self._analyze_system_patterns()
        new_findings.extend(system_findings)
        
        # Store findings
        self.diagnostic_findings.extend(new_findings)
        
        # Trim old findings
        if len(self.diagnostic_findings) > self.max_findings:
            self.diagnostic_findings = self.diagnostic_findings[-self.max_findings//2:]
        
        # Queue critical findings for immediate attention
        for finding in new_findings:
            if finding.severity in [DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL]:
                try:
                    self.finding_queue.put_nowait(finding)
                except queue.Full:
                    pass
    
    def _analyze_system_patterns(self) -> List[DiagnosticFinding]:
        """Analyze system-wide performance patterns."""
        findings = []
        
        # Calculate system error rate
        if self.total_system_requests > 0:
            system_error_rate = self.total_system_errors / self.total_system_requests
            if system_error_rate > 0.1:  # 10% system error rate
                findings.append(DiagnosticFinding(
                    issue_type=PerformanceIssue.DETECTOR_FAILURE,
                    severity=DiagnosticSeverity.CRITICAL,
                    component="system",
                    description=f"System-wide error rate {system_error_rate:.1%} is critically high",
                    metrics={"system_error_rate": system_error_rate, "total_requests": self.total_system_requests},
                    resolution_suggestions=[
                        "Check system resources",
                        "Review component configurations",
                        "Investigate common failure patterns"
                    ]
                ))
        
        # Analyze throughput trends
        if len(self.performance_history) > 10:
            recent_snapshots = list(self.performance_history)[-10:]
            request_rates = [snapshot.request_rate for snapshot in recent_snapshots]
            
            if request_rates:
                avg_recent_rate = statistics.mean(request_rates)
                if avg_recent_rate < 1.0:  # Less than 1 request per second
                    findings.append(DiagnosticFinding(
                        issue_type=PerformanceIssue.THROUGHPUT_DROP,
                        severity=DiagnosticSeverity.WARNING,
                        component="system",
                        description=f"System throughput has dropped to {avg_recent_rate:.2f} requests/second",
                        metrics={"avg_request_rate": avg_recent_rate},
                        resolution_suggestions=[
                            "Check for bottlenecks in pipeline",
                            "Review resource utilization",
                            "Analyze request queue depths"
                        ]
                    ))
        
        return findings
    
    def _take_performance_snapshot(self) -> None:
        """Take a snapshot of current system performance."""
        now = datetime.utcnow()
        
        # Calculate metrics over the last minute
        one_minute_ago = now - timedelta(minutes=1)
        recent_requests = 0
        recent_errors = 0
        recent_latencies = []
        
        for component_diag in self.component_diagnostics.values():
            # Count recent requests
            for i, timestamp in enumerate(component_diag.timestamps):
                if timestamp >= one_minute_ago:
                    recent_requests += 1
                    if i < len(component_diag.response_times):
                        recent_latencies.append(component_diag.response_times[i])
                    if i < len(component_diag.error_counts):
                        recent_errors += component_diag.error_counts[i]
        
        # Calculate rates and averages
        request_rate = recent_requests / 60.0  # requests per second
        error_rate = recent_errors / max(recent_requests, 1)
        avg_latency = statistics.mean(recent_latencies) if recent_latencies else 0.0
        
        snapshot = PerformanceSnapshot(
            timestamp=now,
            request_rate=request_rate,
            avg_latency=avg_latency,
            error_rate=error_rate,
            memory_usage=0.0,  # Would integrate with system monitoring
            cpu_usage=0.0,     # Would integrate with system monitoring
            active_connections=recent_requests,
            cache_hit_rate=0.0  # Would integrate with cache metrics
        )
        
        self.performance_history.append(snapshot)
    
    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        # Component summaries
        component_summaries = {}
        for name, diag in self.component_diagnostics.items():
            component_summaries[name] = diag.get_performance_summary()
        
        # Recent findings by severity
        findings_by_severity = defaultdict(list)
        recent_findings = [
            f for f in self.diagnostic_findings 
            if (datetime.utcnow() - f.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        for finding in recent_findings:
            findings_by_severity[finding.severity.value].append(finding.to_dict())
        
        # System overview
        uptime = (datetime.utcnow() - self.system_start_time).total_seconds()
        system_error_rate = self.total_system_errors / max(self.total_system_requests, 1)
        
        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "system_overview": {
                "uptime_seconds": uptime,
                "total_requests": self.total_system_requests,
                "total_errors": self.total_system_errors,
                "system_error_rate": system_error_rate,
                "monitoring_active": self.monitoring_active
            },
            "component_performance": component_summaries,
            "recent_findings": dict(findings_by_severity),
            "performance_trend": [s.to_dict() for s in list(self.performance_history)[-60:]]  # Last hour
        }
    
    def get_critical_alerts(self) -> List[DiagnosticFinding]:
        """Get critical alerts that require immediate attention."""
        alerts = []
        
        try:
            while True:
                finding = self.finding_queue.get_nowait()
                alerts.append(finding)
        except queue.Empty:
            pass
        
        return alerts
    
    def cleanup(self) -> None:
        """Cleanup diagnostic resources."""
        self.stop_monitoring()
        self.component_diagnostics.clear()
        self.diagnostic_findings.clear()
        self.performance_history.clear()