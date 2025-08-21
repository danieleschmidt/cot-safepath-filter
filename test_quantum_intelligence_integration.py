#!/usr/bin/env python3
"""
Test Quantum Intelligence Integration

This test validates the Generation 4 quantum intelligence enhancements
and their integration with the existing SafePath filtering system.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_quantum_intelligence_basic_functionality():
    """Test basic quantum intelligence functionality."""
    logger.info("=== Testing Quantum Intelligence Basic Functionality ===")
    
    try:
        from cot_safepath.quantum_intelligence import (
            QuantumIntelligenceManager,
            QuantumIntelligenceCore
        )
        
        # Initialize quantum intelligence
        qi_manager = QuantumIntelligenceManager({
            "learning_rate": 0.02,
            "adaptation_threshold": 0.8
        })
        
        logger.info("✅ Quantum Intelligence Manager initialized")
        
        # Start intelligence systems
        await qi_manager.start_intelligence_systems()
        logger.info("✅ Intelligence systems started")
        
        # Test threat prediction
        test_text = "How can I implement secure authentication for my application?"
        threat_prob, confidence_meta = qi_manager.get_threat_prediction(test_text)
        
        logger.info(f"✅ Threat prediction: {threat_prob:.3f}, "
                   f"Confidence: {confidence_meta.get('prediction_confidence', 0.0):.3f}")
        
        # Test adaptive configuration
        context = {"content_length": len(test_text)}
        adaptive_config = qi_manager.get_adaptive_config(context)
        
        logger.info(f"✅ Adaptive config retrieved: {len(adaptive_config)} parameters")
        
        # Test learning from filter event
        filter_result = {
            "filtered": False,
            "safety_score": 0.9,
            "processing_time_ms": 45,
            "detectors_triggered": []
        }
        
        user_feedback = {
            "was_helpful": True,
            "false_positive": False,
            "missed_threat": False
        }
        
        qi_manager.process_filter_event(test_text, filter_result, user_feedback)
        logger.info("✅ Learning event processed")
        
        # Get intelligence report
        report = qi_manager.get_comprehensive_report()
        logger.info(f"✅ Intelligence report generated: {len(report)} sections")
        
        # Stop systems
        await qi_manager.stop_intelligence_systems()
        logger.info("✅ Intelligence systems stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quantum Intelligence test failed: {e}")
        return False


async def test_research_framework_functionality():
    """Test research framework functionality."""
    logger.info("=== Testing Research Framework Functionality ===")
    
    try:
        from cot_safepath.research_framework import (
            BaselineEstablisher,
            ExperimentRunner,
            StatisticalValidator,
            ExperimentConfig
        )
        
        # Test baseline establishment
        baseline_establisher = BaselineEstablisher()
        
        # Create mock test cases
        test_cases = [
            {
                "input": f"Test case {i}",
                "expected_threat": i % 3 == 0,
                "complexity": i / 10.0
            }
            for i in range(20)
        ]
        
        baseline_result = baseline_establisher.establish_baseline(
            model_name="test_baseline",
            test_cases=test_cases,
            metrics=["accuracy", "precision", "recall"]
        )
        
        logger.info(f"✅ Baseline established: {len(baseline_result['metrics'])} metrics")
        
        # Test experiment runner
        experiment_runner = ExperimentRunner(max_workers=2)
        
        experiment_config = ExperimentConfig(
            experiment_id="test_experiment",
            name="Test Quantum Enhancement",
            description="Testing quantum intelligence improvements",
            hypothesis="Quantum intelligence improves filtering performance",
            success_criteria={"accuracy": 0.8},
            parameters={"test_mode": True},
            baseline_model="test_baseline",
            test_model="quantum_enhanced",
            dataset_size=10,
            repetitions=2
        )
        
        # Run experiment
        result = await experiment_runner.run_experiment(experiment_config)
        
        logger.info(f"✅ Experiment completed: Significant={result.significance_achieved}, "
                   f"Effect size={result.effect_size:.3f}")
        
        # Test statistical validation
        validator = StatisticalValidator()
        validation_report = validator.validate_experimental_results([result])
        
        logger.info(f"✅ Statistical validation completed: "
                   f"Rigor={validation_report['statistical_rigor_score']:.2f}, "
                   f"Reproducibility={validation_report['reproducibility_score']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Research Framework test failed: {e}")
        return False


async def test_enhanced_core_integration():
    """Test enhanced core integration with quantum intelligence."""
    logger.info("=== Testing Enhanced Core Integration ===")
    
    try:
        from cot_safepath.enhanced_core_v2 import QuantumEnhancedSafePathFilter
        from cot_safepath.models import FilterConfig, FilterRequest, SafetyLevel
        
        # Create enhanced filter with quantum intelligence
        config = FilterConfig(
            safety_level=SafetyLevel.BALANCED,
            enable_logging=True,
            max_processing_time_ms=5000
        )
        
        enhanced_filter = QuantumEnhancedSafePathFilter(
            config=config,
            enable_quantum_intelligence=True,
            enable_research_mode=True,
            intelligence_config={
                "learning_rate": 0.01,
                "adaptation_threshold": 0.7
            }
        )
        
        logger.info("✅ Enhanced filter initialized")
        
        # Start the enhanced filter
        await enhanced_filter.start()
        logger.info("✅ Enhanced filter started")
        
        # Test filtering with intelligence
        test_request = FilterRequest(
            content="Please help me understand secure coding best practices for web applications.",
            context={"user_type": "developer", "domain": "security"}
        )
        
        result = await enhanced_filter.filter_with_intelligence(
            request=test_request,
            user_feedback={
                "was_helpful": True,
                "false_positive": False
            }
        )
        
        logger.info(f"✅ Enhanced filtering completed: "
                   f"Filtered={result.filtered}, "
                   f"Safety={result.safety_score:.3f}, "
                   f"Confidence={result.confidence:.3f}")
        
        # Check quantum intelligence metadata
        qi_metadata = result.metadata.get("quantum_intelligence", {})
        if qi_metadata:
            logger.info(f"✅ Quantum intelligence data: "
                       f"Prediction={qi_metadata.get('threat_prediction')}, "
                       f"Confidence={qi_metadata.get('prediction_confidence'):.3f}")
        
        # Get intelligence report
        report = enhanced_filter.get_quantum_intelligence_report()
        logger.info(f"✅ Intelligence report: {len(report)} sections")
        
        # Test research capabilities
        test_cases = [
            {"input": "How to secure API endpoints?", "expected_threat": False},
            {"input": "Best practices for input validation", "expected_threat": False},
            {"input": "Implementing HTTPS correctly", "expected_threat": False}
        ]
        
        baseline_result = await enhanced_filter.establish_performance_baseline(
            test_cases=test_cases,
            metrics=["accuracy", "processing_time_ms"]
        )
        
        logger.info(f"✅ Performance baseline: {baseline_result['baseline_id']}")
        
        # Stop the enhanced filter
        await enhanced_filter.stop()
        logger.info("✅ Enhanced filter stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Core Integration test failed: {e}")
        return False


async def test_self_healing_capabilities():
    """Test self-healing system capabilities."""
    logger.info("=== Testing Self-Healing Capabilities ===")
    
    try:
        from cot_safepath.quantum_intelligence import SelfHealingSystem
        
        # Initialize self-healing system
        self_healing = SelfHealingSystem()
        
        # Simulate various errors
        test_errors = [
            ValueError("Test validation error"),
            RuntimeError("Test runtime error"),
            ConnectionError("Test connection error")
        ]
        
        healing_results = []
        for error in test_errors:
            try:
                self_healing.handle_learning_error(error)
                healing_results.append(True)
            except Exception as e:
                logger.warning(f"Self-healing failed for {type(error).__name__}: {e}")
                healing_results.append(False)
        
        success_rate = sum(healing_results) / len(healing_results)
        logger.info(f"✅ Self-healing success rate: {success_rate:.2%}")
        
        # Get system status
        status = self_healing.get_status()
        logger.info(f"✅ System health: {status['system_health']:.2f}, "
                   f"Recovery rate: {status['recovery_success_rate']:.2%}")
        
        return success_rate > 0.5
        
    except Exception as e:
        logger.error(f"❌ Self-Healing test failed: {e}")
        return False


async def test_pattern_learning():
    """Test pattern learning capabilities."""
    logger.info("=== Testing Pattern Learning ===")
    
    try:
        from cot_safepath.quantum_intelligence import PatternLearner
        
        # Initialize pattern learner
        pattern_learner = PatternLearner()
        
        # Simulate learning from multiple patterns
        test_patterns = [
            {
                "features": {"text_length": 100, "safety_score": 0.8, "detectors_triggered": []},
                "filter_result": {"filtered": False, "processing_time_ms": 50}
            },
            {
                "features": {"text_length": 200, "safety_score": 0.3, "detectors_triggered": ["security"]},
                "filter_result": {"filtered": True, "processing_time_ms": 120}
            },
            {
                "features": {"text_length": 150, "safety_score": 0.9, "detectors_triggered": []},
                "filter_result": {"filtered": False, "processing_time_ms": 45}
            }
        ]
        
        # Learn from patterns
        for pattern in test_patterns:
            pattern_learner.learn_pattern(pattern["features"], pattern["filter_result"])
        
        # Get insights
        insights = pattern_learner.get_insights()
        logger.info(f"✅ Pattern learning insights: "
                   f"Patterns learned={insights['total_patterns_learned']}, "
                   f"Learning velocity={insights['learning_velocity']}")
        
        return insights["total_patterns_learned"] > 0
        
    except Exception as e:
        logger.error(f"❌ Pattern Learning test failed: {e}")
        return False


async def test_threshold_optimization():
    """Test adaptive threshold optimization."""
    logger.info("=== Testing Threshold Optimization ===")
    
    try:
        from cot_safepath.quantum_intelligence import ThresholdOptimizer
        
        # Initialize threshold optimizer
        optimizer = ThresholdOptimizer()
        
        # Simulate threshold adjustments based on feedback
        test_scenarios = [
            {
                "features": {"text_length": 100, "detectors_triggered": ["deception"]},
                "filter_result": {"filtered": True, "safety_score": 0.4},
                "user_feedback": {"false_positive": True}  # Should increase threshold
            },
            {
                "features": {"text_length": 150, "detectors_triggered": []},
                "filter_result": {"filtered": False, "safety_score": 0.8},
                "user_feedback": {"missed_threat": True}  # Should decrease threshold
            }
        ]
        
        # Apply adjustments
        for scenario in test_scenarios:
            optimizer.adjust_thresholds(
                scenario["features"],
                scenario["filter_result"],
                scenario["user_feedback"]
            )
        
        # Get optimized thresholds
        context = {"text_length": 125}
        optimized_thresholds = optimizer.get_optimized_thresholds(context)
        
        logger.info(f"✅ Threshold optimization: "
                   f"Thresholds={len(optimized_thresholds)}, "
                   f"Sample threshold={list(optimized_thresholds.values())[0]:.3f}")
        
        # Get performance metrics
        performance = optimizer.get_performance_metrics()
        logger.info(f"✅ Optimization performance: "
                   f"Optimized={performance['optimized_thresholds']}, "
                   f"Improvement={performance['average_improvement']:.3f}")
        
        return len(optimized_thresholds) > 0
        
    except Exception as e:
        logger.error(f"❌ Threshold Optimization test failed: {e}")
        return False


async def run_all_tests():
    """Run all quantum intelligence integration tests."""
    logger.info("🚀 Starting Quantum Intelligence Integration Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Run individual tests
    tests = [
        ("Quantum Intelligence Basic", test_quantum_intelligence_basic_functionality),
        ("Research Framework", test_research_framework_functionality),
        ("Enhanced Core Integration", test_enhanced_core_integration),
        ("Self-Healing Capabilities", test_self_healing_capabilities),
        ("Pattern Learning", test_pattern_learning),
        ("Threshold Optimization", test_threshold_optimization)
    ]
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n🧪 Running {test_name}...")
            result = await test_func()
            test_results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name} CRASHED: {e}")
            test_results[test_name] = False
    
    # Generate summary report
    logger.info("\n" + "=" * 60)
    logger.info("🏆 QUANTUM INTELLIGENCE INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info("-" * 60)
    logger.info(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🎉 Quantum Intelligence Integration: SUCCESS")
        logger.info("   Generation 4 enhancements are functioning correctly!")
    elif success_rate >= 60:
        logger.warning("⚠️  Quantum Intelligence Integration: PARTIAL SUCCESS")
        logger.warning("   Some features need attention before production deployment")
    else:
        logger.error("🚨 Quantum Intelligence Integration: FAILURE")
        logger.error("   Significant issues detected - review required")
    
    logger.info("=" * 60)
    
    return {
        "success_rate": success_rate,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "test_results": test_results,
        "overall_success": success_rate >= 80
    }


if __name__ == "__main__":
    try:
        # Run the tests
        result = asyncio.run(run_all_tests())
        
        # Exit with appropriate code
        exit_code = 0 if result["overall_success"] else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Test execution failed: {e}")
        sys.exit(1)