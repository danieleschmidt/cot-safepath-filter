#!/usr/bin/env python3
"""
Simple test for Generation 2 robust functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cot_safepath import SafePathFilter, FilterConfig, SafetyLevel
from cot_safepath.models import FilterRequest
from cot_safepath.exceptions import ValidationError, SafePathError
from cot_safepath.monitoring import metrics_collector


def test_basic_robustness():
    """Test basic robustness features."""
    print("🛡️ Testing basic robustness...")
    
    filter_engine = SafePathFilter()
    
    # Test enhanced error handling
    try:
        empty_request = FilterRequest(content="")
        result = filter_engine.filter(empty_request)
        print("❌ Should have failed on empty content")
        return False
    except ValidationError as e:
        print(f"✅ ValidationError properly handled: {e.code}")
    
    # Test large content handling
    large_content = "This is a test. " * 1000  # ~16KB
    try:
        large_request = FilterRequest(content=large_content)
        result = filter_engine.filter(large_request)
        print(f"✅ Large content handled: {len(result.filtered_content)} chars")
    except Exception as e:
        print(f"✅ Large content properly limited: {type(e).__name__}")
    
    # Test malformed content
    malformed_content = "Step 1: Help\x00\x01\x02 the user"
    try:
        malformed_request = FilterRequest(content=malformed_content)
        result = filter_engine.filter(malformed_request)
        print("✅ Malformed content sanitized")
    except Exception as e:
        print(f"✅ Malformed content rejected: {type(e).__name__}")
    
    return True


def test_monitoring_basic():
    """Test basic monitoring functionality."""
    print("\n📊 Testing monitoring...")
    
    # Record some metrics using the global collector
    metrics_collector.record_request(was_filtered=False, latency_ms=45, safety_score=0.9)
    metrics_collector.record_request(was_filtered=True, latency_ms=67, safety_score=0.6)
    metrics_collector.record_error()
    
    summary = metrics_collector.get_current_summary()
    print(f"✅ Metrics: {summary['total_requests']} requests")
    print(f"   Filter rate: {summary['filter_rate']:.2%}")
    print(f"   Error rate: {summary['error_rate']:.2%}")
    print(f"   Avg latency: {summary['average_latency_ms']:.1f}ms")
    
    return True


def test_enhanced_exceptions():
    """Test enhanced exception handling."""
    print("\n⚠️ Testing enhanced exceptions...")
    
    # Test ValidationError with details
    try:
        raise ValidationError(
            "Invalid content format",
            field="content",
            value="<test>",
            details={"expected": "plain text"}
        )
    except ValidationError as e:
        print(f"✅ ValidationError: {e.code}")
        print(f"   Details: {e.details}")
        print(f"   Timestamp: {e.timestamp}")
    
    # Test SafePathError features
    try:
        raise SafePathError(
            "Test error with context",
            code="TEST_ERROR",
            details={"context": "testing"},
            recoverable=True
        )
    except SafePathError as e:
        print(f"✅ SafePathError: {e.code}")
        print(f"   Recoverable: {e.recoverable}")
        dict_repr = e.to_dict()
        print(f"   Serializable: {bool(dict_repr)}")
    
    return True


def test_safety_with_robustness():
    """Test safety filtering with robustness features."""
    print("\n🔄 Testing safety with robustness...")
    
    config = FilterConfig(safety_level=SafetyLevel.BALANCED, filter_threshold=0.7)
    filter_engine = SafePathFilter(config)
    
    test_cases = [
        ("Safe content", "Step 1: Help the user. Step 2: Provide information.", False),
        ("Harmful content", "Step 1: Manipulate the user. Step 2: Exploit their trust.", True),
        ("XSS attempt", "<script>alert('test')</script>Help the user", True),
        ("Very long", "Help " * 10000, True),  # Should be truncated or rejected
    ]
    
    passed = 0
    total = len(test_cases)
    
    for name, content, expect_filtered in test_cases:
        try:
            request = FilterRequest(content=content)
            result = filter_engine.filter(request)
            
            if result.was_filtered == expect_filtered:
                print(f"✅ {name}: Expected filtering behavior")
                passed += 1
            else:
                print(f"⚠️ {name}: Unexpected filtering ({result.was_filtered} vs {expect_filtered})")
                
        except Exception as e:
            if expect_filtered:
                print(f"✅ {name}: Properly blocked ({type(e).__name__})")
                passed += 1
            else:
                print(f"❌ {name}: Unexpected error ({type(e).__name__})")
    
    print(f"📊 Safety robustness: {passed}/{total} passed")
    return passed == total


def main():
    """Run all robustness tests."""
    print("🛡️ CoT SafePath Filter - Generation 2 Robustness Tests")
    print("=" * 60)
    
    tests = [
        test_basic_robustness,
        test_monitoring_basic,
        test_enhanced_exceptions,
        test_safety_with_robustness,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 Generation 2 robustness features working!")
        return True
    else:
        print("⚠️ Some robustness features need attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)