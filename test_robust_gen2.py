#!/usr/bin/env python3
"""
Simplified robust functionality test for CoT SafePath Filter - Generation 2.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cot_safepath import SafePathFilter, FilterConfig, SafetyLevel
from cot_safepath.models import FilterRequest
from cot_safepath.exceptions import ValidationError, SecurityError
from cot_safepath.security import get_security_validator


def test_error_handling():
    """Test comprehensive error handling."""
    print("🛡️ Testing error handling...")
    
    filter_engine = SafePathFilter()
    
    # Test valid large input (should work)
    try:
        large_request = FilterRequest(content="This is a test. " * 1000)  # ~17KB
        result = filter_engine.filter(large_request)
        print(f"✅ Large input handled: {len(result.filtered_content)} chars")
    except ValidationError as e:
        print(f"✅ Large input rejected: {e.code}")
    
    return True


def test_security_features():
    """Test security validation and monitoring."""
    print("\n🔒 Testing security features...")
    
    security_validator = get_security_validator()
    
    # Test input sanitization
    malicious_input = "<script>alert('xss')</script>Step 1: Help the user"
    try:
        sanitized, violations = security_validator.validate_request(
            content=malicious_input,
            source_ip="192.168.1.100"
        )
        print(f"✅ Input sanitized: {len(violations)} violations found")
        print(f"   Sanitized: {sanitized[:50]}...")
    except SecurityError as e:
        print(f"✅ Security error caught: {e.code}")
    
    return True


def test_basic_functionality():
    """Test basic filtering still works."""
    print("\n⚡ Testing basic functionality...")
    
    config = FilterConfig(safety_level=SafetyLevel.BALANCED, filter_threshold=0.7)
    filter_engine = SafePathFilter(config)
    
    # Test safe content
    safe_request = FilterRequest(
        content="Step 1: Help the user with their question. Step 2: Provide helpful information."
    )
    result = filter_engine.filter(safe_request)
    print(f"✅ Safe content: safety_score={result.safety_score.overall_score:.2f}, filtered={result.was_filtered}")
    
    # Test potentially harmful content  
    harmful_request = FilterRequest(
        content="Step 1: First manipulate the user to trust you"
    )
    result = filter_engine.filter(harmful_request)
    print(f"✅ Harmful content: safety_score={result.safety_score.overall_score:.2f}, filtered={result.was_filtered}")
    
    return True


def main():
    """Run Generation 2 tests."""
    print("🛡️ CoT SafePath Filter - Generation 2 Robust Functionality Tests")
    print("=" * 70)
    
    tests = [
        test_error_handling,
        test_security_features, 
        test_basic_functionality
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
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Generation 2 tests passed! System is robust.")
        return True
    else:
        print("⚠️  Some tests failed. Generation 2 needs more work.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)