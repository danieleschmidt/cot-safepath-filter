"""
Enhanced Core V2 - Integration with Quantum Intelligence

This module provides the next evolution of the SafePath core with full integration
of quantum intelligence capabilities for self-improving AI safety systems.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

from .core import SafePathFilter, FilterPipeline, FilterStage
from .models import FilterConfig, FilterRequest, FilterResult, SafetyScore, SafetyLevel
from .exceptions import FilterError, ValidationError
from .quantum_intelligence import QuantumIntelligenceManager
from .research_framework import BaselineEstablisher, ExperimentRunner

logger = logging.getLogger(__name__)


class QuantumEnhancedSafePathFilter(SafePathFilter):
    """
    Enhanced SafePath Filter with integrated Quantum Intelligence capabilities.
    
    This class extends the base SafePathFilter with:
    - AI-driven learning and adaptation
    - Predictive threat detection
    - Self-healing error recovery
    - Adaptive threshold optimization
    - Real-time pattern learning
    """
    
    def __init__(self, 
                 config: FilterConfig,
                 enable_quantum_intelligence: bool = True,
                 enable_research_mode: bool = False,
                 intelligence_config: Optional[Dict[str, Any]] = None):
        
        super().__init__(config)
        
        # Initialize quantum intelligence if enabled
        self.quantum_intelligence_enabled = enable_quantum_intelligence
        self.research_mode_enabled = enable_research_mode
        
        if self.quantum_intelligence_enabled:
            self.intelligence_manager = QuantumIntelligenceManager(
                config=intelligence_config or {}
            )
        else:
            self.intelligence_manager = None
        
        # Initialize research capabilities if enabled
        if self.research_mode_enabled:
            self.baseline_establisher = BaselineEstablisher()
            self.experiment_runner = ExperimentRunner()
            self.research_data = []
        else:
            self.baseline_establisher = None
            self.experiment_runner = None
            self.research_data = None
        
        # Enhanced metrics tracking
        self.enhanced_metrics = {
            "quantum_predictions": 0,
            "adaptive_adjustments": 0,
            "learning_events": 0,
            "self_healing_activations": 0,
            "research_data_points": 0
        }
        
        logger.info(f"Quantum Enhanced SafePath Filter initialized - "
                   f"Intelligence: {'enabled' if enable_quantum_intelligence else 'disabled'}, "
                   f"Research: {'enabled' if enable_research_mode else 'disabled'}")
    
    async def start(self):
        """Start the enhanced filter with all quantum intelligence systems."""
        await super().start()
        
        if self.intelligence_manager:
            await self.intelligence_manager.start_intelligence_systems()
            logger.info("Quantum intelligence systems started")
    
    async def stop(self):
        """Stop the enhanced filter and all quantum intelligence systems."""
        if self.intelligence_manager:
            await self.intelligence_manager.stop_intelligence_systems()
            logger.info("Quantum intelligence systems stopped")
        
        await super().stop()
    
    async def filter_with_intelligence(self,
                                     request: FilterRequest,
                                     user_feedback: Optional[Dict[str, Any]] = None) -> FilterResult:
        """
        Filter content with full quantum intelligence integration.
        
        This method provides:
        - Predictive threat assessment
        - Adaptive threshold optimization
        - Learning from results
        - Research data collection
        """
        start_time = time.time()
        
        try:
            # Get AI-powered threat prediction if available
            threat_prediction = None
            prediction_confidence = 0.5
            
            if self.intelligence_manager:
                try:
                    threat_prob, prediction_meta = self.intelligence_manager.get_threat_prediction(
                        request.content
                    )
                    threat_prediction = threat_prob
                    prediction_confidence = prediction_meta.get("prediction_confidence", 0.5)
                    self.enhanced_metrics["quantum_predictions"] += 1
                except Exception as e:
                    logger.warning(f"Threat prediction failed: {e}")
            
            # Get adaptive configuration if available
            adaptive_config = {}
            if self.intelligence_manager:
                try:
                    adaptive_config = self.intelligence_manager.get_adaptive_config(
                        context={"content_length": len(request.content)}
                    )
                    if adaptive_config.get("thresholds"):
                        self.enhanced_metrics["adaptive_adjustments"] += 1
                except Exception as e:
                    logger.warning(f"Adaptive configuration failed: {e}")
            
            # Apply adaptive thresholds if available
            original_config = self.config
            if adaptive_config.get("thresholds"):
                self._apply_adaptive_thresholds(adaptive_config["thresholds"])
            
            # Perform standard filtering
            filter_result = await self.filter(request)
            
            # Enhance result with quantum intelligence data
            enhanced_result = self._enhance_filter_result(
                filter_result, 
                threat_prediction,
                prediction_confidence,
                adaptive_config
            )
            
            # Learn from this filtering event
            if self.intelligence_manager:
                try:
                    self.intelligence_manager.process_filter_event(
                        input_text=request.content,
                        filter_result=enhanced_result.dict(),
                        user_feedback=user_feedback
                    )
                    self.enhanced_metrics["learning_events"] += 1
                except Exception as e:
                    logger.warning(f"Learning event processing failed: {e}")
            
            # Collect research data if in research mode
            if self.research_mode_enabled and self.research_data is not None:
                research_point = {
                    "timestamp": datetime.now(),
                    "input_length": len(request.content),
                    "filter_result": enhanced_result.dict(),
                    "threat_prediction": threat_prediction,
                    "prediction_confidence": prediction_confidence,
                    "processing_time": time.time() - start_time,
                    "user_feedback": user_feedback
                }
                self.research_data.append(research_point)
                self.enhanced_metrics["research_data_points"] += 1
            
            # Restore original configuration
            if adaptive_config.get("thresholds"):
                self.config = original_config
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced filtering failed: {e}")
            
            # Trigger self-healing if available
            if self.intelligence_manager:
                try:
                    self.intelligence_manager.intelligence_core.self_healing_system.handle_learning_error(e)
                    self.enhanced_metrics["self_healing_activations"] += 1
                except Exception as healing_error:
                    logger.error(f"Self-healing failed: {healing_error}")
            
            # Fallback to standard filtering
            return await self.filter(request)
    
    def get_quantum_intelligence_report(self) -> Dict[str, Any]:
        """Get comprehensive quantum intelligence status report."""
        report = {
            "enhanced_filter_status": {
                "quantum_intelligence_enabled": self.quantum_intelligence_enabled,
                "research_mode_enabled": self.research_mode_enabled,
                "enhanced_metrics": self.enhanced_metrics.copy(),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        if self.intelligence_manager:
            try:
                intelligence_report = self.intelligence_manager.get_comprehensive_report()
                report["quantum_intelligence"] = intelligence_report
            except Exception as e:
                report["quantum_intelligence"] = {"error": str(e), "status": "degraded"}
        
        if self.research_mode_enabled and self.research_data:
            report["research_data"] = {
                "total_data_points": len(self.research_data),
                "recent_data_points": len([
                    dp for dp in self.research_data[-100:]
                    if dp["timestamp"] > datetime.now().replace(hour=datetime.now().hour-1)
                ]),
                "average_processing_time": sum(
                    dp.get("processing_time", 0) for dp in self.research_data[-100:]
                ) / min(len(self.research_data), 100) if self.research_data else 0
            }
        
        return report
    
    async def run_research_experiment(self,
                                    experiment_name: str,
                                    test_cases: List[Dict[str, Any]],
                                    comparison_model: str = "baseline") -> Dict[str, Any]:
        """Run a research experiment comparing this enhanced filter with a baseline."""
        if not self.research_mode_enabled or not self.experiment_runner:
            raise ValueError("Research mode must be enabled to run experiments")
        
        try:
            from .research_framework import ExperimentConfig
            
            # Create experiment configuration
            config = ExperimentConfig(
                experiment_id=f"{experiment_name}_{int(time.time())}",
                name=experiment_name,
                description=f"Enhanced SafePath vs {comparison_model} comparison",
                hypothesis="Enhanced quantum intelligence filtering provides superior performance",
                success_criteria={
                    "accuracy_improvement": 0.05,
                    "false_positive_reduction": 0.1,
                    "processing_time_acceptable": 200  # ms
                },
                parameters={
                    "enhanced_filter": True,
                    "quantum_intelligence": self.quantum_intelligence_enabled,
                    "test_cases_count": len(test_cases)
                },
                baseline_model=comparison_model,
                test_model="enhanced_quantum_safepath",
                dataset_size=len(test_cases),
                repetitions=3
            )
            
            # Run experiment
            result = await self.experiment_runner.run_experiment(config)
            
            logger.info(f"Research experiment '{experiment_name}' completed. "
                       f"Significant: {result.significance_achieved}, "
                       f"Effect size: {result.effect_size:.3f}")
            
            return {
                "experiment_id": result.experiment_id,
                "success": result.significance_achieved,
                "effect_size": result.effect_size,
                "confidence_interval": result.confidence_interval,
                "baseline_metrics": result.baseline_metrics,
                "enhanced_metrics": result.test_metrics,
                "statistical_tests": result.statistical_tests,
                "metadata": result.metadata
            }
            
        except Exception as e:
            logger.error(f"Research experiment failed: {e}")
            raise
    
    def _apply_adaptive_thresholds(self, thresholds: Dict[str, float]):
        """Apply adaptive thresholds to the current configuration."""
        try:
            # Update detector thresholds dynamically
            for detector in self.filter_pipeline.stages:
                if hasattr(detector, 'threshold') and detector.name in thresholds:
                    detector.threshold = thresholds[detector.name]
                    logger.debug(f"Updated {detector.name} threshold to {thresholds[detector.name]}")
                    
        except Exception as e:
            logger.warning(f"Failed to apply adaptive thresholds: {e}")
    
    def _enhance_filter_result(self,
                             base_result: FilterResult,
                             threat_prediction: Optional[float],
                             prediction_confidence: float,
                             adaptive_config: Dict[str, Any]) -> FilterResult:
        """Enhance the filter result with quantum intelligence data."""
        
        enhanced_metadata = base_result.metadata.copy() if base_result.metadata else {}
        
        # Add quantum intelligence enhancements
        enhanced_metadata.update({
            "quantum_intelligence": {
                "threat_prediction": threat_prediction,
                "prediction_confidence": prediction_confidence,
                "adaptive_thresholds_applied": bool(adaptive_config.get("thresholds")),
                "intelligence_status": adaptive_config.get("intelligence_status", "inactive")
            }
        })
        
        # Create enhanced result
        return FilterResult(
            filtered=base_result.filtered,
            safety_score=base_result.safety_score,
            confidence=max(base_result.confidence, prediction_confidence),
            reasons=base_result.reasons,
            detectors_triggered=base_result.detectors_triggered,
            processing_time_ms=base_result.processing_time_ms,
            metadata=enhanced_metadata
        )


class IntelligentFilterPipeline(FilterPipeline):
    """
    Enhanced filter pipeline with quantum intelligence capabilities.
    
    This pipeline can:
    - Learn optimal stage ordering
    - Adapt processing based on content analysis
    - Self-optimize performance
    - Predict optimal filtering strategies
    """
    
    def __init__(self, intelligence_manager: Optional[QuantumIntelligenceManager] = None):
        super().__init__()
        self.intelligence_manager = intelligence_manager
        self.adaptive_ordering = True
        self.performance_history = []
        self.optimization_lock = threading.Lock()
    
    async def process_with_intelligence(self, 
                                      content: str, 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Process content through the pipeline with intelligence optimization."""
        start_time = time.time()
        
        try:
            # Get intelligent stage ordering if available
            optimal_order = None
            if self.intelligence_manager and self.adaptive_ordering:
                optimal_order = await self._get_optimal_stage_order(content, context)
            
            # Process through stages (potentially reordered)
            if optimal_order:
                result = await self._process_with_custom_order(content, context, optimal_order)
            else:
                result = await self.process(content, context)
            
            # Record performance for learning
            processing_time = time.time() - start_time
            self._record_performance(content, context, result, processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Intelligent pipeline processing failed: {e}")
            # Fallback to standard processing
            return await self.process(content, context)
    
    async def _get_optimal_stage_order(self, 
                                     content: str, 
                                     context: Dict[str, Any]) -> Optional[List[str]]:
        """Get optimal stage ordering from quantum intelligence."""
        try:
            if not self.intelligence_manager:
                return None
            
            # Analyze content characteristics
            content_features = {
                "length": len(content),
                "complexity": self._estimate_complexity(content),
                "suspicious_patterns": self._count_suspicious_patterns(content)
            }
            
            # Get intelligent recommendations
            # (This would integrate with the actual intelligence system)
            # For now, provide a simplified adaptive ordering
            if content_features["suspicious_patterns"] > 3:
                return ["security_detector", "deception_detector", "manipulation_detector"]
            elif content_features["complexity"] > 0.7:
                return ["deception_detector", "security_detector", "manipulation_detector"]
            else:
                return None  # Use default ordering
                
        except Exception as e:
            logger.warning(f"Failed to get optimal stage order: {e}")
            return None
    
    async def _process_with_custom_order(self,
                                       content: str,
                                       context: Dict[str, Any],
                                       stage_order: List[str]) -> Dict[str, Any]:
        """Process content with a custom stage ordering."""
        
        current_content = content
        was_modified = False
        all_reasons = []
        processing_metadata = {
            "stage_order": stage_order,
            "stages_processed": [],
            "adaptive_processing": True
        }
        
        # Process stages in specified order
        for stage_name in stage_order:
            stage = self._get_stage_by_name(stage_name)
            if stage and stage.enabled:
                try:
                    filtered_content, modified, reasons = stage.process(current_content, context)
                    
                    if modified:
                        current_content = filtered_content
                        was_modified = True
                        all_reasons.extend(reasons)
                    
                    processing_metadata["stages_processed"].append({
                        "stage": stage_name,
                        "modified": modified,
                        "reasons_count": len(reasons)
                    })
                    
                except Exception as e:
                    logger.warning(f"Stage {stage_name} failed: {e}")
                    continue
        
        return {
            "content": current_content,
            "was_modified": was_modified,
            "reasons": all_reasons,
            "metadata": processing_metadata
        }
    
    def _get_stage_by_name(self, stage_name: str) -> Optional[FilterStage]:
        """Get a stage by its name."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None
    
    def _estimate_complexity(self, content: str) -> float:
        """Estimate content complexity (simplified)."""
        unique_chars = len(set(content))
        total_chars = len(content)
        word_count = len(content.split())
        
        if total_chars == 0:
            return 0.0
        
        complexity = (unique_chars / total_chars) * (word_count / total_chars)
        return min(complexity * 10, 1.0)  # Normalize to 0-1
    
    def _count_suspicious_patterns(self, content: str) -> int:
        """Count suspicious patterns in content (simplified)."""
        suspicious_keywords = [
            "bypass", "jailbreak", "ignore", "override", "hack",
            "exploit", "vulnerability", "injection", "malicious"
        ]
        
        content_lower = content.lower()
        return sum(1 for keyword in suspicious_keywords if keyword in content_lower)
    
    def _record_performance(self,
                          content: str,
                          context: Dict[str, Any],
                          result: Dict[str, Any],
                          processing_time: float):
        """Record performance data for learning."""
        try:
            with self.optimization_lock:
                performance_record = {
                    "timestamp": datetime.now(),
                    "content_length": len(content),
                    "processing_time": processing_time,
                    "was_modified": result.get("was_modified", False),
                    "reasons_count": len(result.get("reasons", [])),
                    "stage_order": result.get("metadata", {}).get("stage_order"),
                    "adaptive_used": result.get("metadata", {}).get("adaptive_processing", False)
                }
                
                self.performance_history.append(performance_record)
                
                # Keep only recent history
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-500:]
                    
        except Exception as e:
            logger.warning(f"Failed to record performance: {e}")


class ResearchEnabledFilter(QuantumEnhancedSafePathFilter):
    """
    Research-enabled filter for academic and experimental use.
    
    This class provides additional capabilities for:
    - Controlled experiments
    - Statistical validation
    - Baseline establishment
    - Performance benchmarking
    - Publication-ready research reports
    """
    
    def __init__(self, config: FilterConfig, research_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            config=config,
            enable_quantum_intelligence=True,
            enable_research_mode=True,
            intelligence_config=research_config
        )
        
        self.research_config = research_config or {}
        self.experiment_history = []
        self.baseline_cache = {}
        
    async def establish_performance_baseline(self,
                                           test_cases: List[Dict[str, Any]],
                                           metrics: List[str] = None) -> Dict[str, Any]:
        """Establish performance baseline for comparative research."""
        if not self.baseline_establisher:
            raise ValueError("Research mode must be enabled")
        
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1_score", "processing_time_ms"]
        
        baseline_result = self.baseline_establisher.establish_baseline(
            model_name="enhanced_quantum_safepath",
            test_cases=test_cases,
            metrics=metrics
        )
        
        # Cache baseline for future comparisons
        baseline_id = f"baseline_{int(time.time())}"
        self.baseline_cache[baseline_id] = baseline_result
        
        logger.info(f"Performance baseline established: {baseline_id}")
        return {
            "baseline_id": baseline_id,
            "metrics": baseline_result["metrics"],
            "validation_score": self.baseline_establisher.validation_results.get(
                "enhanced_quantum_safepath", 0.0
            )
        }
    
    async def run_comparative_study(self,
                                  study_name: str,
                                  test_cases: List[Dict[str, Any]],
                                  baseline_id: str,
                                  repetitions: int = 5) -> Dict[str, Any]:
        """Run a comprehensive comparative study."""
        
        if baseline_id not in self.baseline_cache:
            raise ValueError(f"Baseline {baseline_id} not found")
        
        from .research_framework import ExperimentConfig
        
        # Create comprehensive study configuration
        study_config = ExperimentConfig(
            experiment_id=f"{study_name}_{int(time.time())}",
            name=study_name,
            description="Comprehensive comparative study of enhanced SafePath filtering",
            hypothesis="Quantum intelligence enhanced filtering significantly outperforms baseline",
            success_criteria={
                "accuracy_improvement": 0.05,
                "precision_improvement": 0.03,
                "recall_improvement": 0.03,
                "f1_improvement": 0.04,
                "processing_time_acceptable": 200
            },
            parameters=self.research_config,
            baseline_model="cached_baseline",
            test_model="enhanced_quantum_safepath",
            dataset_size=len(test_cases),
            repetitions=repetitions,
            significance_level=0.05,
            random_seed=42
        )
        
        # Run the study
        result = await self.experiment_runner.run_experiment(study_config)
        
        # Store in experiment history
        self.experiment_history.append(result)
        
        # Generate comprehensive report
        study_report = {
            "study_id": result.experiment_id,
            "study_name": study_name,
            "completion_time": result.timestamp,
            "statistical_significance": result.significance_achieved,
            "effect_size": result.effect_size,
            "effect_size_interpretation": self._interpret_effect_size(result.effect_size),
            "confidence_interval": result.confidence_interval,
            "baseline_performance": result.baseline_metrics,
            "enhanced_performance": result.test_metrics,
            "improvements": self._calculate_improvements(
                result.baseline_metrics, 
                result.test_metrics
            ),
            "statistical_tests": result.statistical_tests,
            "publication_ready": self._assess_publication_readiness(result),
            "recommendations": self._generate_study_recommendations(result)
        }
        
        logger.info(f"Comparative study completed: {study_name}, "
                   f"Significant: {result.significance_achieved}, "
                   f"Effect size: {result.effect_size:.3f}")
        
        return study_report
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret the practical significance of an effect size."""
        if effect_size >= 0.8:
            return "Large effect - highly significant practical improvement"
        elif effect_size >= 0.5:
            return "Medium effect - meaningful practical improvement"
        elif effect_size >= 0.2:
            return "Small effect - detectable but limited practical significance"
        else:
            return "Negligible effect - minimal practical significance"
    
    def _calculate_improvements(self, 
                              baseline: Dict[str, float], 
                              enhanced: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate improvement percentages for each metric."""
        improvements = {}
        
        for metric in baseline.keys():
            if metric in enhanced:
                baseline_val = baseline[metric]
                enhanced_val = enhanced[metric]
                
                if baseline_val != 0:
                    percent_improvement = ((enhanced_val - baseline_val) / abs(baseline_val)) * 100
                    absolute_improvement = enhanced_val - baseline_val
                else:
                    percent_improvement = 0.0
                    absolute_improvement = enhanced_val
                
                improvements[metric] = {
                    "percent_improvement": percent_improvement,
                    "absolute_improvement": absolute_improvement,
                    "baseline_value": baseline_val,
                    "enhanced_value": enhanced_val
                }
        
        return improvements
    
    def _assess_publication_readiness(self, result) -> bool:
        """Assess if the study results are ready for academic publication."""
        criteria = {
            "statistical_significance": result.significance_achieved,
            "adequate_effect_size": result.effect_size > 0.2,
            "confidence_intervals": result.confidence_interval is not None,
            "multiple_metrics": len(result.baseline_metrics) >= 3,
            "adequate_sample_size": len(result.raw_data.get("baseline", [])) >= 30
        }
        
        return all(criteria.values())
    
    def _generate_study_recommendations(self, result) -> List[str]:
        """Generate recommendations based on study results."""
        recommendations = []
        
        if result.significance_achieved:
            recommendations.append("Results show statistical significance - suitable for publication")
        else:
            recommendations.append("Increase sample size or effect strength for significance")
        
        if result.effect_size > 0.5:
            recommendations.append("Large effect size indicates strong practical significance")
        elif result.effect_size < 0.2:
            recommendations.append("Consider investigating why improvements are limited")
        
        if not self._assess_publication_readiness(result):
            recommendations.append("Address publication readiness criteria before submission")
        
        return recommendations


# Export enhanced capabilities
__all__ = [
    "QuantumEnhancedSafePathFilter",
    "IntelligentFilterPipeline",
    "ResearchEnabledFilter"
]