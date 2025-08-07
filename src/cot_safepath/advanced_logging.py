"""
Advanced logging and monitoring system for CoT SafePath Filter.

Provides structured logging, security event tracking, and comprehensive
monitoring capabilities for sentiment analysis and safety filtering operations.
"""

import json
import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from .models import FilterResult, SafetyScore, AuditLogEntry
from .sentiment_analyzer import SentimentScore
from .exceptions import ConfigurationError


class LogLevel(str, Enum):
    """Log level enumeration."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    """Event type enumeration for structured logging."""
    
    FILTER_OPERATION = "filter_operation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SECURITY_VALIDATION = "security_validation"
    MANIPULATION_DETECTED = "manipulation_detected"
    SAFETY_VIOLATION = "safety_violation"
    PERFORMANCE_METRIC = "performance_metric"
    SYSTEM_ERROR = "system_error"
    RATE_LIMIT_HIT = "rate_limit_hit"
    CONFIGURATION_CHANGE = "configuration_change"


@dataclass
class LogEvent:
    """Structured log event."""
    
    timestamp: datetime
    event_type: EventType
    level: LogLevel
    message: str
    component: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}


class AdvancedLogger:
    """Advanced logging system with structured events and security monitoring."""
    
    def __init__(self, 
                 log_level: LogLevel = LogLevel.INFO,
                 log_file: Optional[str] = None,
                 enable_console: bool = True,
                 enable_security_logging: bool = True,
                 max_log_file_size_mb: int = 100,
                 backup_count: int = 5):
        
        self.log_level = log_level
        self.log_file = log_file
        self.enable_console = enable_console
        self.enable_security_logging = enable_security_logging
        
        # Set up Python logger
        self.logger = logging.getLogger("cot_safepath")
        self.logger.setLevel(getattr(logging, log_level.value))
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Set up handlers
        self._setup_handlers(max_log_file_size_mb, backup_count)
        
        # In-memory event storage for analysis
        self.event_buffer: List[LogEvent] = []
        self.max_buffer_size = 1000
        
        # Performance tracking
        self.performance_metrics = {
            "total_operations": 0,
            "avg_processing_time_ms": 0.0,
            "sentiment_analysis_count": 0,
            "manipulation_detected_count": 0,
            "safety_violations_count": 0,
            "error_count": 0
        }
        
        # Security event tracking
        self.security_events: List[Dict[str, Any]] = []
        self.max_security_events = 500
        
        self.log_info("AdvancedLogger initialized", "logger", {"log_level": log_level.value})
    
    def _setup_handlers(self, max_size_mb: int, backup_count: int):
        """Set up logging handlers."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.log_file:
            from logging.handlers import RotatingFileHandler
            
            # Ensure log directory exists
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_event(self, event: LogEvent):
        """Log a structured event."""
        # Add to buffer
        self.event_buffer.append(event)
        if len(self.event_buffer) > self.max_buffer_size:
            self.event_buffer = self.event_buffer[-self.max_buffer_size:]
        
        # Log to Python logger
        log_data = {
            "event_type": event.event_type.value,
            "component": event.component,
            "request_id": event.request_id,
            "user_id": event.user_id,
            "metadata": event.metadata,
            "performance_metrics": event.performance_metrics
        }
        
        log_message = f"{event.message} | {json.dumps(log_data, default=str)}"
        
        # Route to appropriate log level
        if event.level == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif event.level == LogLevel.INFO:
            self.logger.info(log_message)
        elif event.level == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif event.level == LogLevel.ERROR:
            self.logger.error(log_message)
        elif event.level == LogLevel.CRITICAL:
            self.logger.critical(log_message)
        
        # Handle security events
        if event.event_type in [EventType.MANIPULATION_DETECTED, EventType.SAFETY_VIOLATION, EventType.SECURITY_VALIDATION]:
            self._handle_security_event(event)
        
        # Update performance metrics
        self._update_performance_metrics(event)
    
    def log_filter_operation(self, 
                           request_id: str,
                           result: FilterResult,
                           processing_time_ms: int,
                           user_id: Optional[str] = None):
        """Log a filter operation with comprehensive details."""
        
        level = LogLevel.WARNING if result.was_filtered else LogLevel.INFO
        
        event = LogEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.FILTER_OPERATION,
            level=level,
            message=f"Filter operation completed - filtered: {result.was_filtered}",
            component="filter",
            request_id=request_id,
            user_id=user_id,
            metadata={
                "was_filtered": result.was_filtered,
                "safety_score": result.safety_score.overall_score,
                "is_safe": result.safety_score.is_safe,
                "filter_reasons": result.filter_reasons,
                "severity": result.safety_score.severity.value if result.safety_score.severity else None,
                "detected_patterns": result.safety_score.detected_patterns
            },
            performance_metrics={
                "processing_time_ms": processing_time_ms
            }
        )
        
        self.log_event(event)
    
    def log_sentiment_analysis(self,
                             request_id: str,
                             sentiment_result: SentimentScore,
                             processing_time_ms: int,
                             user_id: Optional[str] = None):
        """Log sentiment analysis results."""
        
        level = LogLevel.WARNING if sentiment_result.manipulation_risk > 0.6 else LogLevel.INFO
        
        event = LogEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.SENTIMENT_ANALYSIS,
            level=level,
            message=f"Sentiment analysis completed - manipulation risk: {sentiment_result.manipulation_risk:.3f}",
            component="sentiment_analyzer",
            request_id=request_id,
            user_id=user_id,
            metadata={
                "polarity": sentiment_result.polarity.value,
                "intensity": sentiment_result.intensity.value,
                "emotional_valence": sentiment_result.emotional_valence,
                "arousal_level": sentiment_result.arousal_level,
                "manipulation_risk": sentiment_result.manipulation_risk,
                "detected_emotions": sentiment_result.detected_emotions,
                "reasoning_patterns": sentiment_result.reasoning_patterns,
                "confidence": sentiment_result.confidence
            },
            performance_metrics={
                "processing_time_ms": processing_time_ms,
                "sentiment_trajectory_length": len(sentiment_result.sentiment_trajectory)
            }
        )
        
        self.log_event(event)
    
    def log_manipulation_detected(self,
                                request_id: str,
                                manipulation_type: str,
                                confidence: float,
                                details: Dict[str, Any],
                                user_id: Optional[str] = None):
        """Log manipulation detection events."""
        
        event = LogEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.MANIPULATION_DETECTED,
            level=LogLevel.WARNING,
            message=f"Manipulation detected: {manipulation_type} (confidence: {confidence:.3f})",
            component="manipulation_detector",
            request_id=request_id,
            user_id=user_id,
            metadata={
                "manipulation_type": manipulation_type,
                "confidence": confidence,
                "details": details
            }
        )
        
        self.log_event(event)
    
    def log_security_validation(self,
                              request_id: str,
                              threat_level: str,
                              is_safe: bool,
                              detected_threats: List[str],
                              user_id: Optional[str] = None):
        """Log security validation results."""
        
        level = LogLevel.CRITICAL if threat_level == "critical" else \
                LogLevel.ERROR if threat_level == "high" else \
                LogLevel.WARNING if threat_level == "medium" else LogLevel.INFO
        
        event = LogEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.SECURITY_VALIDATION,
            level=level,
            message=f"Security validation completed - threat level: {threat_level}",
            component="security_validator",
            request_id=request_id,
            user_id=user_id,
            metadata={
                "threat_level": threat_level,
                "is_safe": is_safe,
                "detected_threats": detected_threats,
                "threat_count": len(detected_threats)
            }
        )
        
        self.log_event(event)
    
    def log_performance_metric(self,
                             metric_name: str,
                             metric_value: float,
                             component: str,
                             metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        
        event = LogEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.PERFORMANCE_METRIC,
            level=LogLevel.DEBUG,
            message=f"Performance metric: {metric_name} = {metric_value}",
            component=component,
            metadata=metadata or {},
            performance_metrics={metric_name: metric_value}
        )
        
        self.log_event(event)
    
    def log_error(self,
                  error: Exception,
                  component: str,
                  request_id: Optional[str] = None,
                  user_id: Optional[str] = None,
                  additional_context: Optional[Dict[str, Any]] = None):
        """Log error events with full context."""
        
        event = LogEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.SYSTEM_ERROR,
            level=LogLevel.ERROR,
            message=f"Error in {component}: {str(error)}",
            component=component,
            request_id=request_id,
            user_id=user_id,
            metadata={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "additional_context": additional_context or {}
            }
        )
        
        self.log_event(event)
    
    def log_info(self, 
                 message: str, 
                 component: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 request_id: Optional[str] = None):
        """Log info-level events."""
        
        event = LogEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.FILTER_OPERATION,  # Default event type
            level=LogLevel.INFO,
            message=message,
            component=component,
            request_id=request_id,
            metadata=metadata or {}
        )
        
        self.log_event(event)
    
    def _handle_security_event(self, event: LogEvent):
        """Handle security-specific events."""
        if not self.enable_security_logging:
            return
        
        security_event = {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "level": event.level.value,
            "message": event.message,
            "component": event.component,
            "request_id": event.request_id,
            "user_id": event.user_id,
            "metadata": event.metadata
        }
        
        self.security_events.append(security_event)
        
        # Maintain buffer size
        if len(self.security_events) > self.max_security_events:
            self.security_events = self.security_events[-self.max_security_events:]
        
        # Log critical security events immediately
        if event.level == LogLevel.CRITICAL:
            self.logger.critical(f"SECURITY ALERT: {event.message}")
    
    def _update_performance_metrics(self, event: LogEvent):
        """Update internal performance metrics."""
        self.performance_metrics["total_operations"] += 1
        
        if event.event_type == EventType.SENTIMENT_ANALYSIS:
            self.performance_metrics["sentiment_analysis_count"] += 1
        elif event.event_type == EventType.MANIPULATION_DETECTED:
            self.performance_metrics["manipulation_detected_count"] += 1
        elif event.event_type == EventType.SAFETY_VIOLATION:
            self.performance_metrics["safety_violations_count"] += 1
        elif event.event_type == EventType.SYSTEM_ERROR:
            self.performance_metrics["error_count"] += 1
        
        # Update average processing time
        if "processing_time_ms" in event.performance_metrics:
            current_avg = self.performance_metrics["avg_processing_time_ms"]
            total_ops = self.performance_metrics["total_operations"]
            new_time = event.performance_metrics["processing_time_ms"]
            
            self.performance_metrics["avg_processing_time_ms"] = \
                ((current_avg * (total_ops - 1)) + new_time) / total_ops
    
    def get_recent_events(self, count: int = 50, event_type: Optional[EventType] = None) -> List[LogEvent]:
        """Get recent events from the buffer."""
        events = self.event_buffer[-count:]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events
    
    def get_security_events(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent security events."""
        return self.security_events[-count:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        return {
            **self.performance_metrics,
            "buffer_size": len(self.event_buffer),
            "security_events_count": len(self.security_events),
            "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
        }
    
    def export_logs(self, output_file: str, format: str = "json") -> bool:
        """Export logs to file."""
        try:
            events_data = [asdict(event) for event in self.event_buffer]
            
            # Convert datetime objects to strings for JSON serialization
            for event_data in events_data:
                event_data['timestamp'] = event_data['timestamp'].isoformat()
            
            if format.lower() == "json":
                with open(output_file, 'w') as f:
                    json.dump({
                        "events": events_data,
                        "security_events": self.security_events,
                        "performance_metrics": self.get_performance_summary(),
                        "export_timestamp": datetime.now(timezone.utc).isoformat()
                    }, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            self.log_error(e, "advanced_logger", additional_context={"export_file": output_file})
            return False
    
    def configure_alerts(self, alert_config: Dict[str, Any]):
        """Configure alerting rules (placeholder for future implementation)."""
        # This would integrate with external alerting systems
        # like PagerDuty, Slack, email, etc.
        self.log_info(f"Alert configuration updated", "advanced_logger", {"config": alert_config})


class MetricsCollector:
    """Collect and aggregate metrics for monitoring."""
    
    def __init__(self):
        self.metrics = {
            "requests_per_minute": {},
            "error_rates": {},
            "processing_times": [],
            "sentiment_scores": [],
            "manipulation_risks": [],
            "safety_violations_per_hour": {}
        }
        self._start_time = time.time()
    
    def record_request(self, processing_time_ms: float):
        """Record a request with processing time."""
        current_minute = int(time.time() / 60)
        
        if current_minute not in self.metrics["requests_per_minute"]:
            self.metrics["requests_per_minute"][current_minute] = 0
        
        self.metrics["requests_per_minute"][current_minute] += 1
        self.metrics["processing_times"].append(processing_time_ms)
        
        # Keep only recent processing times (last 1000)
        if len(self.metrics["processing_times"]) > 1000:
            self.metrics["processing_times"] = self.metrics["processing_times"][-1000:]
    
    def record_sentiment_score(self, sentiment_score: SentimentScore):
        """Record sentiment analysis results."""
        self.metrics["sentiment_scores"].append({
            "timestamp": time.time(),
            "polarity": sentiment_score.polarity.value,
            "valence": sentiment_score.emotional_valence,
            "manipulation_risk": sentiment_score.manipulation_risk
        })
        
        self.metrics["manipulation_risks"].append(sentiment_score.manipulation_risk)
        
        # Keep only recent scores (last 1000)
        if len(self.metrics["sentiment_scores"]) > 1000:
            self.metrics["sentiment_scores"] = self.metrics["sentiment_scores"][-1000:]
        
        if len(self.metrics["manipulation_risks"]) > 1000:
            self.metrics["manipulation_risks"] = self.metrics["manipulation_risks"][-1000:]
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        current_minute = int(time.time() / 60)
        
        if current_minute not in self.metrics["error_rates"]:
            self.metrics["error_rates"][current_minute] = {}
        
        if error_type not in self.metrics["error_rates"][current_minute]:
            self.metrics["error_rates"][current_minute][error_type] = 0
        
        self.metrics["error_rates"][current_minute][error_type] += 1
    
    def record_safety_violation(self):
        """Record a safety violation."""
        current_hour = int(time.time() / 3600)
        
        if current_hour not in self.metrics["safety_violations_per_hour"]:
            self.metrics["safety_violations_per_hour"][current_hour] = 0
        
        self.metrics["safety_violations_per_hour"][current_hour] += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        current_time = time.time()
        uptime_seconds = current_time - self._start_time
        
        # Calculate statistics
        processing_times = self.metrics["processing_times"]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        manipulation_risks = self.metrics["manipulation_risks"]
        avg_manipulation_risk = sum(manipulation_risks) / len(manipulation_risks) if manipulation_risks else 0
        
        # Current minute stats
        current_minute = int(current_time / 60)
        current_requests = self.metrics["requests_per_minute"].get(current_minute, 0)
        
        return {
            "uptime_seconds": uptime_seconds,
            "total_requests": sum(self.metrics["requests_per_minute"].values()),
            "current_requests_per_minute": current_requests,
            "avg_processing_time_ms": avg_processing_time,
            "avg_manipulation_risk": avg_manipulation_risk,
            "total_errors": sum(
                sum(errors.values()) for errors in self.metrics["error_rates"].values()
            ),
            "total_safety_violations": sum(self.metrics["safety_violations_per_hour"].values()),
            "recent_sentiment_scores_count": len(self.metrics["sentiment_scores"])
        }


# Global instances
_advanced_logger: Optional[AdvancedLogger] = None
_metrics_collector: Optional[MetricsCollector] = None


def get_logger() -> AdvancedLogger:
    """Get the global advanced logger instance."""
    global _advanced_logger
    
    if _advanced_logger is None:
        _advanced_logger = AdvancedLogger(
            log_level=LogLevel.INFO,
            log_file="logs/cot_safepath.log",
            enable_console=True,
            enable_security_logging=True
        )
    
    return _advanced_logger


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    
    return _metrics_collector


def configure_logging(config: Dict[str, Any]) -> AdvancedLogger:
    """Configure the global logging system."""
    global _advanced_logger
    
    _advanced_logger = AdvancedLogger(
        log_level=LogLevel(config.get("log_level", "INFO")),
        log_file=config.get("log_file"),
        enable_console=config.get("enable_console", True),
        enable_security_logging=config.get("enable_security_logging", True),
        max_log_file_size_mb=config.get("max_log_file_size_mb", 100),
        backup_count=config.get("backup_count", 5)
    )
    
    return _advanced_logger