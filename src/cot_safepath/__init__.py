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

# Generation 1 Enhanced Integrations
try:
    from .enhanced_integrations import (
        OpenAIIntegration,
        LangChainIntegration, 
        AutoGenIntegration,
        StreamingIntegration,
        IntegrationFactory,
        IntegrationConfig,
        BaseLLMIntegration,
        wrap_openai_client,
        wrap_langchain_llm,
        wrap_autogen_agent,
        create_safe_streaming_filter,
    )
    from .realtime_monitoring import (
        SafePathMonitor,
        MonitoringConfig,
        MetricsCollector,
        AlertManager,
        AlertRule,
        Alert,
        MetricPoint,
        get_global_monitor,
        configure_monitoring,
        monitor_filter_operation,
    )
    ENHANCED_INTEGRATIONS_AVAILABLE = True
    REALTIME_MONITORING_AVAILABLE = True
except ImportError as e:
    # Graceful fallback when enhanced features not available
    ENHANCED_INTEGRATIONS_AVAILABLE = False
    REALTIME_MONITORING_AVAILABLE = False
    OpenAIIntegration = None
    LangChainIntegration = None
    AutoGenIntegration = None
    StreamingIntegration = None
    IntegrationFactory = None
    IntegrationConfig = None
    BaseLLMIntegration = None
    wrap_openai_client = None
    wrap_langchain_llm = None
    wrap_autogen_agent = None
    create_safe_streaming_filter = None
    SafePathMonitor = None
    MonitoringConfig = None
    MetricsCollector = None
    AlertManager = None
    AlertRule = None
    Alert = None
    MetricPoint = None
    get_global_monitor = None
    configure_monitoring = None
    monitor_filter_operation = None

# Generation 2 Robust Features
try:
    from .robust_error_handling import (
        RobustErrorHandler,
        CircuitBreaker,
        ErrorClassifier,
        RetryManager,
        FallbackManager,
        CircuitBreakerConfig,
        RetryConfig,
        FallbackConfig,
    )
    from .health_monitoring import (
        HealthMonitor,
        SystemHealthChecker,
        HealthCheckResult,
        HealthStatus,
        HealthMetrics,
    )
    from .robust_security import (
        SecurityHardening,
        InputValidator,
        ThreatDetector,
        SecurityConfig,
        ThreatLevel,
        SecurityViolation,
    )
    GENERATION_2_AVAILABLE = True
    ROBUST_ERROR_HANDLING_AVAILABLE = True
    HEALTH_MONITORING_AVAILABLE = True
    SECURITY_HARDENING_AVAILABLE = True
except ImportError as e:
    GENERATION_2_AVAILABLE = False
    ROBUST_ERROR_HANDLING_AVAILABLE = False
    HEALTH_MONITORING_AVAILABLE = False
    SECURITY_HARDENING_AVAILABLE = False
    RobustErrorHandler = None
    CircuitBreaker = None
    ErrorClassifier = None
    RetryManager = None
    FallbackManager = None
    CircuitBreakerConfig = None
    RetryConfig = None
    FallbackConfig = None
    HealthMonitor = None
    SystemHealthChecker = None
    HealthCheckResult = None
    HealthStatus = None
    HealthMetrics = None
    SecurityHardening = None
    InputValidator = None
    ThreatDetector = None
    SecurityConfig = None
    ThreatLevel = None
    SecurityViolation = None

# Generation 3 Performance Optimization
try:
    from .performance_optimization import (
        IntelligentCache,
        CacheStrategy,
        CacheConfig,
        ConcurrentProcessor,
        ProcessingConfig,
        AdaptivePerformanceTuner,
        TuningConfig,
        HighPerformanceFilterEngine,
        OptimizationConfig,
    )
    GENERATION_3_AVAILABLE = True
    PERFORMANCE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    GENERATION_3_AVAILABLE = False
    PERFORMANCE_OPTIMIZATION_AVAILABLE = False
    IntelligentCache = None
    CacheStrategy = None
    CacheConfig = None
    ConcurrentProcessor = None
    ProcessingConfig = None
    AdaptivePerformanceTuner = None
    TuningConfig = None
    HighPerformanceFilterEngine = None
    OptimizationConfig = None

# Generation 5 enhancements (conditionally imported)
try:
    from .generation_5_lite import (
        MultimodalProcessorLite,
        FederatedLearningManagerLite,
        NeuralArchitectureSearchLite,
        Generation5ManagerLite,
        Generation5SafePathFilterLite,
        ModalityType as Generation5ModalityType,
        MultimodalInput as Generation5MultimodalInput,
        MultimodalAnalysis as Generation5MultimodalAnalysis,
        FederatedLearningUpdate as Generation5FederatedLearningUpdate,
    )
    from .threat_intelligence import (
        ThreatIntelligenceManager,
        ThreatIntelligenceFeed,
        ThreatPatternLearner,
        ThreatType,
        ThreatSeverity,
        ThreatIndicator,
        ThreatPattern,
        ThreatEvent,
    )
    GENERATION_5_AVAILABLE = True
    MULTIMODAL_PROCESSING_AVAILABLE = True
    THREAT_INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    # Graceful fallback when Generation 5 features not available
    GENERATION_5_AVAILABLE = False
    MULTIMODAL_PROCESSING_AVAILABLE = False
    THREAT_INTELLIGENCE_AVAILABLE = False
    MultimodalProcessorLite = None
    FederatedLearningManagerLite = None
    NeuralArchitectureSearchLite = None
    Generation5ManagerLite = None
    Generation5SafePathFilterLite = None
    Generation5ModalityType = None
    Generation5MultimodalInput = None
    Generation5MultimodalAnalysis = None
    Generation5FederatedLearningUpdate = None
    ThreatIntelligenceManager = None
    ThreatIntelligenceFeed = None
    ThreatPatternLearner = None
    ThreatType = None
    ThreatSeverity = None
    ThreatIndicator = None
    ThreatPattern = None
    ThreatEvent = None

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
    # Generation 1 Enhanced Integrations
    "ENHANCED_INTEGRATIONS_AVAILABLE",
    "REALTIME_MONITORING_AVAILABLE",
    "OpenAIIntegration",
    "LangChainIntegration", 
    "AutoGenIntegration",
    "StreamingIntegration",
    "IntegrationFactory",
    "IntegrationConfig",
    "BaseLLMIntegration",
    "wrap_openai_client",
    "wrap_langchain_llm",
    "wrap_autogen_agent",
    "create_safe_streaming_filter",
    "SafePathMonitor",
    "MonitoringConfig",
    "MetricsCollector",
    "AlertManager",
    "AlertRule",
    "Alert",
    "MetricPoint",
    "get_global_monitor",
    "configure_monitoring",
    "monitor_filter_operation",
    # Generation 2 Robust Features
    "GENERATION_2_AVAILABLE",
    "ROBUST_ERROR_HANDLING_AVAILABLE",
    "HEALTH_MONITORING_AVAILABLE",
    "SECURITY_HARDENING_AVAILABLE",
    "RobustErrorHandler",
    "CircuitBreaker",
    "ErrorClassifier",
    "RetryManager",
    "FallbackManager",
    "CircuitBreakerConfig",
    "RetryConfig",
    "FallbackConfig",
    "HealthMonitor",
    "SystemHealthChecker",
    "HealthCheckResult",
    "HealthStatus",
    "HealthMetrics",
    "SecurityHardening",
    "InputValidator",
    "ThreatDetector",
    "SecurityConfig",
    "ThreatLevel",
    "SecurityViolation",
    # Generation 3 Performance Optimization
    "GENERATION_3_AVAILABLE",
    "PERFORMANCE_OPTIMIZATION_AVAILABLE",
    "IntelligentCache",
    "CacheStrategy",
    "CacheConfig",
    "ConcurrentProcessor",
    "ProcessingConfig",
    "AdaptivePerformanceTuner",
    "TuningConfig",
    "HighPerformanceFilterEngine",
    "OptimizationConfig",
    # Generation 5 capabilities
    "GENERATION_5_AVAILABLE",
    "MULTIMODAL_PROCESSING_AVAILABLE", 
    "THREAT_INTELLIGENCE_AVAILABLE",
    "MultimodalProcessorLite",
    "FederatedLearningManagerLite",
    "NeuralArchitectureSearchLite", 
    "Generation5ManagerLite",
    "Generation5SafePathFilterLite",
    "Generation5ModalityType",
    "Generation5MultimodalInput",
    "Generation5MultimodalAnalysis",
    "Generation5FederatedLearningUpdate",
    "ThreatIntelligenceManager",
    "ThreatIntelligenceFeed",
    "ThreatPatternLearner",
    "ThreatType",
    "ThreatSeverity", 
    "ThreatIndicator",
    "ThreatPattern",
    "ThreatEvent",
]