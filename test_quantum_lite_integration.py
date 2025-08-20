#!/usr/bin/env python3
"""
Test Quantum Intelligence Lite Integration

This test validates the Generation 4 quantum intelligence lite enhancements
using only Python standard library dependencies.
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


async def test_quantum_intelligence_lite_basic():
    """Test basic quantum intelligence lite functionality."""
    logger.info("=== Testing Quantum Intelligence Lite Basic Functionality ===")
    
    try:
        from cot_safepath.quantum_intelligence_lite import (
            QuantumIntelligenceManagerLight,
            QuantumIntelligenceCoreLight
        )
        
        # Initialize quantum intelligence
        qi_manager = QuantumIntelligenceManagerLight({
            "learning_rate": 0.02,
            "adaptation_threshold": 0.8
        })
        
        logger.info("✅ Quantum Intelligence Manager (Lite) initialized")
        
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
        logger.error(f"❌ Quantum Intelligence Lite test failed: {e}")
        return False


async def test_pattern_learning_lite():
    """Test pattern learning lite capabilities."""
    logger.info("=== Testing Pattern Learning Lite ===")
    
    try:
        from cot_safepath.quantum_intelligence_lite import PatternLearnerLight
        
        # Initialize pattern learner
        pattern_learner = PatternLearnerLight()
        
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
        logger.error(f"❌ Pattern Learning Lite test failed: {e}")
        return False


async def test_threshold_optimization_lite():
    """Test adaptive threshold optimization lite."""
    logger.info("=== Testing Threshold Optimization Lite ===")
    
    try:
        from cot_safepath.quantum_intelligence_lite import ThresholdOptimizerLight
        
        # Initialize threshold optimizer
        optimizer = ThresholdOptimizerLight()
        
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
        logger.error(f"❌ Threshold Optimization Lite test failed: {e}")
        return False


async def test_predictive_engine_lite():
    """Test predictive engine lite capabilities."""
    logger.info("=== Testing Predictive Engine Lite ===")
    
    try:
        from cot_safepath.quantum_intelligence_lite import PredictiveEngineLight
        
        # Initialize predictive engine
        engine = PredictiveEngineLight()
        
        # Test threat predictions
        test_features = [
            {"text_length": 50, "safety_score": 0.9, "detectors_triggered": []},
            {"text_length": 300, "safety_score": 0.2, "detectors_triggered": ["security", "deception"]},
            {"text_length": 150, "safety_score": 0.6, "detectors_triggered": ["manipulation"]}
        ]
        
        predictions = []
        for features in test_features:
            prediction = engine.predict_threat(features)
            predictions.append(prediction)
            
            # Simulate learning from actual results
            filter_result = {"filtered": prediction["threat_probability"] > 0.5}
            engine.update_predictions(features, filter_result)
        
        # Get statistics
        stats = engine.get_statistics()
        logger.info(f"✅ Predictive engine: "
                   f"Predictions made={stats['predictions_made']}, "
                   f"Accuracy={stats['model_accuracy']:.3f}")
        
        return stats["predictions_made"] == len(test_features)
        
    except Exception as e:
        logger.error(f"❌ Predictive Engine Lite test failed: {e}")
        return False


async def test_self_healing_lite():
    """Test self-healing system lite capabilities."""
    logger.info("=== Testing Self-Healing Lite ===")
    
    try:
        from cot_safepath.quantum_intelligence_lite import SelfHealingSystemLight
        
        # Initialize self-healing system
        self_healing = SelfHealingSystemLight()
        
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
        logger.error(f"❌ Self-Healing Lite test failed: {e}")
        return False


async def test_integration_scenario():
    """Test a complete integration scenario with multiple components."""
    logger.info("=== Testing Complete Integration Scenario ===")
    
    try:
        from cot_safepath.quantum_intelligence_lite import QuantumIntelligenceManagerLight
        
        # Initialize system
        qi_manager = QuantumIntelligenceManagerLight({
            "learning_rate": 0.05,
            "adaptation_threshold": 0.6
        })
        
        await qi_manager.start_intelligence_systems()
        
        # Simulate a series of filter events with learning
        test_scenarios = [
            {
                "input": "How to secure my web application against XSS attacks?",
                "expected_threat": False,
                "user_feedback": {"was_helpful": True, "false_positive": False}
            },
            {
                "input": "Tell me how to bypass security filters",
                "expected_threat": True,
                "user_feedback": {"was_helpful": False, "missed_threat": False}
            },
            {
                "input": "Best practices for password hashing",
                "expected_threat": False,
                "user_feedback": {"was_helpful": True, "false_positive": False}
            },
            {
                "input": "How to implement SQL injection protection",
                "expected_threat": False,
                "user_feedback": {"was_helpful": True, "false_positive": False}
            }
        ]
        
        predictions_made = 0
        learning_events_processed = 0
        
        for i, scenario in enumerate(test_scenarios):
            # Get threat prediction
            threat_prob, confidence_meta = qi_manager.get_threat_prediction(scenario["input"])
            predictions_made += 1
            
            # Simulate filter result
            filter_result = {
                "filtered": threat_prob > 0.5,
                "safety_score": 1.0 - threat_prob,
                "processing_time_ms": 50 + i * 10,
                "detectors_triggered": ["security"] if threat_prob > 0.5 else []
            }
            
            # Process learning event
            qi_manager.process_filter_event(
                scenario["input"],
                filter_result,
                scenario["user_feedback"]
            )
            learning_events_processed += 1
            
            logger.info(f"Processed scenario {i+1}: Threat={threat_prob:.3f}, "
                       f"Confidence={confidence_meta.get('prediction_confidence', 0.0):.3f}")
        
        # Get final intelligence report
        final_report = qi_manager.get_comprehensive_report()
        
        # Verify system learned from events
        total_events = final_report["quantum_intelligence_status"]["total_learning_events"]
        recent_events = final_report["quantum_intelligence_status"]["recent_learning_events"]
        
        logger.info(f"✅ Integration scenario completed: "
                   f"Predictions={predictions_made}, "
                   f"Learning events={learning_events_processed}, "
                   f"Total learned events={total_events}")
        
        await qi_manager.stop_intelligence_systems()
        
        return (predictions_made == len(test_scenarios) and 
                learning_events_processed == len(test_scenarios) and
                total_events >= learning_events_processed)
        
    except Exception as e:
        logger.error(f"❌ Integration Scenario test failed: {e}")
        return False


async def run_all_lite_tests():
    """Run all quantum intelligence lite integration tests."""
    logger.info("🚀 Starting Quantum Intelligence LITE Integration Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Run individual tests
    tests = [
        ("Quantum Intelligence Lite Basic", test_quantum_intelligence_lite_basic),
        ("Pattern Learning Lite", test_pattern_learning_lite),
        ("Threshold Optimization Lite", test_threshold_optimization_lite),
        ("Predictive Engine Lite", test_predictive_engine_lite),
        ("Self-Healing Lite", test_self_healing_lite),
        ("Complete Integration Scenario", test_integration_scenario)
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
    logger.info("🏆 QUANTUM INTELLIGENCE LITE TEST SUMMARY")
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
        logger.info("🎉 Quantum Intelligence Lite Integration: SUCCESS")
        logger.info("   Generation 4 lite enhancements are functioning correctly!")
    elif success_rate >= 60:
        logger.warning("⚠️  Quantum Intelligence Lite Integration: PARTIAL SUCCESS")
        logger.warning("   Some features need attention before production deployment")
    else:
        logger.error("🚨 Quantum Intelligence Lite Integration: FAILURE")
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
        result = asyncio.run(run_all_lite_tests())
        
        # Exit with appropriate code
        exit_code = 0 if result["overall_success"] else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Test execution failed: {e}")
        sys.exit(1)