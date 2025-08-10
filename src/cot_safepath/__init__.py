"""
CoT SafePath Filter - Real-time middleware for AI safety.

This package provides real-time filtering and sanitization of chain-of-thought
reasoning from AI systems to prevent harmful or deceptive reasoning patterns
from reaching end users.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "safety@terragonlabs.com"

from .core import SafePathFilter, FilterPipeline
from .detectors import (
    DeceptionDetector,
    HarmfulPlanningDetector,
    CapabilityConcealmentDetector,
    ManipulationDetector,
)
from .models import SafetyLevel, FilterConfig, SafetyScore, FilterResult, FilterRequest
from .exceptions import SafePathError, FilterError, DetectorError

__all__ = [
    "SafePathFilter",
    "FilterPipeline", 
    "FilterResult",
    "FilterRequest",
    "DeceptionDetector",
    "HarmfulPlanningDetector",
    "CapabilityConcealmentDetector",
    "ManipulationDetector",
    "SafetyLevel",
    "FilterConfig",
    "SafetyScore",
    "SafePathError",
    "FilterError", 
    "DetectorError",
]