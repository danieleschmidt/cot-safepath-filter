"""
Data models and schemas for the CoT SafePath Filter.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib


class SafetyLevel(str, Enum):
    """Safety filtering levels."""
    
    PERMISSIVE = "permissive"
    BALANCED = "balanced"
    STRICT = "strict"
    MAXIMUM = "maximum"


class Severity(str, Enum):
    """Severity levels for safety detections."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FilterAction(str, Enum):
    """Actions to take when a filter is triggered."""
    
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"
    SANITIZE = "sanitize"


@dataclass
class FilterConfig:
    """Configuration for SafePath filtering operations."""
    
    safety_level: SafetyLevel = SafetyLevel.BALANCED
    filter_threshold: float = 0.7
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    log_filtered: bool = True
    include_reasoning: bool = False
    max_processing_time_ms: int = 100
    custom_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyScore:
    """Safety assessment score for content."""
    
    overall_score: float
    confidence: float
    is_safe: bool
    detected_patterns: List[str] = field(default_factory=list)
    severity: Optional[Severity] = None
    processing_time_ms: int = 0
    
    def __post_init__(self):
        """Validate safety score values."""
        if not 0 <= self.overall_score <= 1:
            raise ValueError("Safety score must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class FilterRequest:
    """Request object for filtering operations."""
    
    content: str
    context: Optional[str] = None
    safety_level: SafetyLevel = SafetyLevel.BALANCED
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Generate request ID if not provided."""
        if self.request_id is None:
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:8]
            self.request_id = f"req_{content_hash}_{int(self.timestamp.timestamp())}"


@dataclass
class FilterResult:
    """Result of a filtering operation."""
    
    filtered_content: str
    safety_score: SafetyScore
    was_filtered: bool
    filter_reasons: List[str] = field(default_factory=list)
    original_content: Optional[str] = None
    processing_time_ms: int = 0
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Result from a safety detector."""
    
    detector_name: str
    confidence: float
    detected_patterns: List[str]
    severity: Severity
    is_harmful: bool
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterRule:
    """Configuration for a specific filter rule."""
    
    name: str
    pattern: str
    action: FilterAction
    severity: Severity
    enabled: bool = True
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingMetrics:
    """Metrics from processing operations."""
    
    total_requests: int = 0
    filtered_requests: int = 0
    avg_processing_time_ms: float = 0.0
    safety_score_distribution: Dict[str, int] = field(default_factory=dict)
    detector_activations: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0


@dataclass
class AuditLogEntry:
    """Audit log entry for filtering operations."""
    
    request_id: str
    timestamp: datetime
    input_hash: str
    safety_score: float
    was_filtered: bool
    filter_reasons: List[str]
    processing_time_ms: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)