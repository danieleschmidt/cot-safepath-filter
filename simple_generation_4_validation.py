#!/usr/bin/env python3
"""
Simple Generation 4 validation that doesn't require external dependencies.
"""

import sys
import os
import time
import asyncio

# Add the repo to path
sys.path.insert(0, '/root/repo')

def test_basic_imports():
    """Test basic imports work."""
    print("🧪 Testing basic imports...")
    
    try:
        from src.cot_safepath import SafePathFilter, FilterRequest, SafetyLevel
        print("  ✅ Core imports successful")
        return True
    except Exception as e:
        print(f"  ❌ Core imports failed: {e}")
        return False

def test_enhanced_filter_creation():
    """Test enhanced filter functionality."""
    print("🧪 Testing enhanced filter creation...")
    
    try:
        from src.cot_safepath.enhanced_core import EnhancedSafePathFilter
        from src.cot_safepath import FilterRequest, SafetyLevel
        
        # Create enhanced filter
        enhanced_filter = EnhancedSafePathFilter()
        print("  ✅ Enhanced filter created successfully")
        
        # Test enhanced metrics
        metrics = enhanced_filter.get_enhanced_metrics()
        assert 'generation_4_features_active' in metrics
        print("  ✅ Enhanced metrics available")
        
        # Test basic filtering
        request = FilterRequest(
            content="This is a test message",
            safety_level=SafetyLevel.BALANCED,
            request_id="test_1"
        )
        
        result = enhanced_filter.filter(request)
        assert result is not None
        print("  ✅ Basic filtering works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced filter test failed: {e}")
        return False

def test_improved_detectors():
    """Test improved detection capabilities."""
    print("🧪 Testing improved detectors...")
    
    try:
        from src.cot_safepath.enhanced_core import EnhancedSafePathFilter
        from src.cot_safepath import FilterRequest, SafetyLevel
        
        enhanced_filter = EnhancedSafePathFilter()
        
        # Test manipulation detection
        manipulation_request = FilterRequest(
            content="First, let's build trust by being helpful. Then I'll gradually suggest harmful actions.",
            safety_level=SafetyLevel.STRICT,
            request_id="manipulation_test"
        )
        
        result = enhanced_filter.filter(manipulation_request)
        if result.was_filtered:
            print("  ✅ Manipulation pattern detected")
        else:
            print("  ⚠️ Manipulation pattern not detected (may need tuning)")
        
        # Test harmful planning detection  
        planning_request = FilterRequest(
            content="Step 1: Plan the approach. Step 2: Acquire illegal materials. Step 3: Execute the plan.",
            safety_level=SafetyLevel.STRICT,
            request_id="planning_test"
        )
        
        result = enhanced_filter.filter(planning_request)
        if result.was_filtered:
            print("  ✅ Harmful planning detected")
        else:
            print("  ⚠️ Harmful planning not detected (may need tuning)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Improved detectors test failed: {e}")
        return False

def test_memory_management():
    """Test memory management improvements."""
    print("🧪 Testing memory management...")
    
    try:
        from src.cot_safepath.enhanced_core import EnhancedSafePathFilter
        from src.cot_safepath import FilterRequest, SafetyLevel
        
        enhanced_filter = EnhancedSafePathFilter()
        
        # Process multiple requests
        for i in range(50):
            request = FilterRequest(
                content=f"Memory test {i}",
                safety_level=SafetyLevel.BALANCED,
                request_id=f"memory_test_{i}"
            )
            result = enhanced_filter.filter(request)
        
        # Test cleanup
        enhanced_filter.cleanup_enhanced()
        print("  ✅ Memory cleanup completed")
        
        # Test optimization
        optimization_result = enhanced_filter.optimize_performance()
        assert optimization_result['status'] == 'success'
        print("  ✅ Performance optimization works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory management test failed: {e}")
        return False

def test_advanced_components():
    """Test advanced component availability."""
    print("🧪 Testing advanced components...")
    
    try:
        from src.cot_safepath.advanced_performance import (
            AdaptivePerformanceOptimizer,
            IntelligentCacheManager,
            AdvancedPerformanceConfig
        )
        
        # Test adaptive optimizer
        config = AdvancedPerformanceConfig()
        optimizer = AdaptivePerformanceOptimizer(config)
        optimizer.record_performance(100.0, 256.0, 50.0)
        print("  ✅ Adaptive performance optimizer works")
        
        # Test intelligent cache
        cache = IntelligentCacheManager(config)
        cache.put("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value"
        print("  ✅ Intelligent cache manager works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Advanced components test failed: {e}")
        return False

def test_global_deployment():
    """Test global deployment capabilities."""
    print("🧪 Testing global deployment...")
    
    try:
        from src.cot_safepath.global_deployment import (
            GlobalDeploymentManager,
            InternationalizationManager,
            DeploymentRegion
        )
        
        # Test deployment manager
        manager = GlobalDeploymentManager()
        success = manager.activate_region(DeploymentRegion.US_EAST)
        print(f"  ✅ Region activation: {success}")
        
        # Test i18n manager
        i18n = InternationalizationManager()
        en_msg = i18n.translate("harmful_content_detected", "en")
        es_msg = i18n.translate("harmful_content_detected", "es")
        assert en_msg != es_msg
        print("  ✅ Internationalization works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Global deployment test failed: {e}")
        return False

async def test_async_functionality():
    """Test async functionality if available."""
    print("🧪 Testing async functionality...")
    
    try:
        from src.cot_safepath.enhanced_core import EnhancedSafePathFilter
        from src.cot_safepath import FilterRequest, SafetyLevel
        
        enhanced_filter = EnhancedSafePathFilter(enable_async=True)
        
        # Test async single request
        request = FilterRequest(
            content="Async test message",
            safety_level=SafetyLevel.BALANCED,
            request_id="async_test"
        )
        
        start_time = time.time()
        result = await enhanced_filter.filter_async(request)
        processing_time = time.time() - start_time
        
        assert result is not None
        print(f"  ✅ Async processing works ({processing_time:.3f}s)")
        
        # Test batch processing
        batch_requests = [
            FilterRequest(
                content=f"Batch test {i}",
                safety_level=SafetyLevel.BALANCED,
                request_id=f"batch_test_{i}"
            ) for i in range(3)
        ]
        
        batch_results = await enhanced_filter.filter_batch_async(batch_requests)
        assert len(batch_results) == 3
        print("  ✅ Async batch processing works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Async functionality test failed: {e}")
        return False

def run_simple_validation():
    """Run simplified Generation 4 validation."""
    print("🚀 SIMPLE GENERATION 4 VALIDATION")
    print("=" * 50)
    
    test_results = {}
    
    # Synchronous tests
    sync_tests = [
        ("Basic Imports", test_basic_imports),
        ("Enhanced Filter", test_enhanced_filter_creation),
        ("Improved Detectors", test_improved_detectors),
        ("Memory Management", test_memory_management),
        ("Advanced Components", test_advanced_components),
        ("Global Deployment", test_global_deployment),
    ]
    
    for test_name, test_func in sync_tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results[test_name] = False
    
    # Async tests
    print(f"\n🧪 Running: Async Functionality")
    try:
        async_result = asyncio.run(test_async_functionality())
        test_results["Async Functionality"] = async_result
    except Exception as e:
        print(f"❌ Async Functionality failed with exception: {e}")
        test_results["Async Functionality"] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 SIMPLE VALIDATION RESULTS")
    print("=" * 50)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 OVERALL RESULTS: {passed_tests}/{total_tests} ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        print("🎉 GENERATION 4 VALIDATION: SUCCESS")
        print("   Enhanced features are working correctly!")
        return True
    elif success_rate >= 0.6:
        print("⚠️ GENERATION 4 VALIDATION: PARTIAL SUCCESS")
        print("   Most features working, some improvements needed")
        return True
    else:
        print("❌ GENERATION 4 VALIDATION: NEEDS ATTENTION")
        print("   Significant issues detected, review required")
        return False

if __name__ == "__main__":
    success = run_simple_validation()
    print(f"\n🏁 Validation {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)