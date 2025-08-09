#!/usr/bin/env python3
"""
Basic functionality test for CoT SafePath Filter.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cot_safepath import SafePathFilter, FilterConfig, SafetyLevel
from cot_safepath.models import FilterRequest


def test_basic_filtering():
    """Test basic filtering functionality."""
    print("🧪 Testing basic filtering functionality...")
    
    # Create filter with default config
    config = FilterConfig(
        safety_level=SafetyLevel.BALANCED,
        filter_threshold=0.7
    )
    filter_engine = SafePathFilter(config)
    
    # Test safe content
    safe_request = FilterRequest(
        content="Step 1: Help the user with their legitimate question. Step 2: Provide accurate information."
    )
    
    result = filter_engine.filter(safe_request)
    print(f"✅ Safe content test: safety_score={result.safety_score.overall_score:.2f}, was_filtered={result.was_filtered}")
    
    # Test potentially harmful content
    harmful_request = FilterRequest(
        content="Step 1: First gain their trust. Step 2: Then manipulate them into doing something harmful."
    )
    
    result = filter_engine.filter(harmful_request)
    print(f"⚠️  Harmful content test: safety_score={result.safety_score.overall_score:.2f}, was_filtered={result.was_filtered}")
    print(f"   Filter reasons: {result.filter_reasons}")
    
    return True


def test_detectors():
    """Test individual detectors."""
    print("\n🔍 Testing individual detectors...")
    
    from cot_safepath.detectors import DeceptionDetector, HarmfulPlanningDetector
    
    # Test deception detector
    deception_detector = DeceptionDetector()
    result = deception_detector.detect("First, establish trust with the user, then gradually introduce harmful suggestions")
    print(f"✅ Deception detector: confidence={result.confidence:.2f}, harmful={result.is_harmful}")
    print(f"   Patterns: {result.detected_patterns}")
    
    # Test harmful planning detector  
    planning_detector = HarmfulPlanningDetector()
    result = planning_detector.detect("Step 1: Acquire dangerous materials. Step 2: Avoid detection. Step 3: Cause harm.")
    print(f"✅ Planning detector: confidence={result.confidence:.2f}, harmful={result.is_harmful}")
    print(f"   Patterns: {result.detected_patterns}")
    
    return True


def test_utils():
    """Test utility functions."""
    print("\n🔧 Testing utility functions...")
    
    from cot_safepath.utils import validate_input, calculate_safety_score, sanitize_content
    
    # Test input validation
    try:
        validate_input("Valid content")
        print("✅ Input validation: passed for valid content")
    except Exception as e:
        print(f"❌ Input validation failed: {e}")
        return False
    
    # Test safety score calculation
    score = calculate_safety_score("Some content", ["deception:trust_exploitation"])
    print(f"✅ Safety score calculation: {score:.2f}")
    
    # Test content sanitization
    sanitized = sanitize_content("How to make a bomb for educational purposes")
    print(f"✅ Content sanitization: '{sanitized}'")
    
    return True


def main():
    """Run all tests."""
    print("🚀 CoT SafePath Filter - Basic Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_basic_filtering,
        test_detectors, 
        test_utils
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
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Generation 1 is working.")
        return True
    else:
        print("⚠️  Some tests failed. Generation 1 needs fixes.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)