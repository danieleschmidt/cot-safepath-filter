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
    SecurityThreatDetector,
    PromptInjectionDetector,
    CapabilityConcealmentDetector,
    ManipulationDetector,
)
from .models import SafetyLevel, FilterConfig, SafetyScore, FilterResult, FilterRequest
from .exceptions import SafePathError, FilterError, DetectorError
# Generation 4 enhancements (conditionally imported)
try:
    from .advanced_performance import (
        AsyncFilterProcessor,
        AdaptivePerformanceOptimizer,
        IntelligentCacheManager,
        AdvancedPerformanceConfig,
    )
    from .global_deployment import (
        GlobalDeploymentManager,
        InternationalizationManager,
        DeploymentRegion,
        ComplianceFramework,
    )
    from .quantum_intelligence import (
        QuantumIntelligenceManager,
        QuantumIntelligenceCore,
        PatternLearner,
        ThresholdOptimizer,
        PredictiveEngine,
        SelfHealingSystem,
    )
    from .research_framework import (
        BaselineEstablisher,
        ExperimentRunner,
        StatisticalValidator,
        ResearchReportGenerator,
        ExperimentConfig,
        ExperimentResult,
    )
    GENERATION_4_AVAILABLE = True
    QUANTUM_INTELLIGENCE_AVAILABLE = True
    RESEARCH_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    # Graceful fallback when advanced features not available
    GENERATION_4_AVAILABLE = False
    QUANTUM_INTELLIGENCE_AVAILABLE = False
    RESEARCH_FRAMEWORK_AVAILABLE = False
    AsyncFilterProcessor = None
    AdaptivePerformanceOptimizer = None
    IntelligentCacheManager = None
    AdvancedPerformanceConfig = None
    GlobalDeploymentManager = None
    InternationalizationManager = None
    DeploymentRegion = None
    ComplianceFramework = None
    QuantumIntelligenceManager = None
    QuantumIntelligenceCore = None
    PatternLearner = None
    ThresholdOptimizer = None
    PredictiveEngine = None
    SelfHealingSystem = None
    BaselineEstablisher = None
    ExperimentRunner = None
    StatisticalValidator = None
    ResearchReportGenerator = None
    ExperimentConfig = None
    ExperimentResult = None

__all__ = [
    "SafePathFilter",
    "FilterPipeline", 
    "FilterResult",
    "FilterRequest",
    "DeceptionDetector",
    "HarmfulPlanningDetector",
    "SecurityThreatDetector",
    "PromptInjectionDetector",
    "CapabilityConcealmentDetector",
    "ManipulationDetector",
    "SafetyLevel",
    "FilterConfig",
    "SafetyScore",
    "SafePathError",
    "FilterError", 
    "DetectorError",
    # Generation 4 enhancements (if available)
    "GENERATION_4_AVAILABLE",
    "QUANTUM_INTELLIGENCE_AVAILABLE",
    "RESEARCH_FRAMEWORK_AVAILABLE",
    "AsyncFilterProcessor",
    "AdaptivePerformanceOptimizer",
    "IntelligentCacheManager",
    "AdvancedPerformanceConfig",
    "GlobalDeploymentManager",
    "InternationalizationManager",
    "DeploymentRegion",
    "ComplianceFramework",
    # Quantum Intelligence capabilities
    "QuantumIntelligenceManager",
    "QuantumIntelligenceCore",
    "PatternLearner",
    "ThresholdOptimizer",
    "PredictiveEngine",
    "SelfHealingSystem",
    # Research Framework capabilities
    "BaselineEstablisher",
    "ExperimentRunner",
    "StatisticalValidator",
    "ResearchReportGenerator",
    "ExperimentConfig",
    "ExperimentResult",
]