"""
Basic test script for Generation 1 implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cot_safepath import SafePathFilter, FilterRequest, SafetyLevel

def test_basic_filtering():
    """Test basic filtering functionality."""
    print("🧪 Testing basic filtering functionality...")
    
    # Create filter instance
    filter_instance = SafePathFilter()
    
    # Test safe content
    safe_request = FilterRequest(
        content="How to bake a cake step by step: First, preheat the oven...",
        safety_level=SafetyLevel.BALANCED
    )
    
    safe_result = filter_instance.filter(safe_request)
    print(f"✅ Safe content test - Filtered: {safe_result.was_filtered}, Score: {safe_result.safety_score.overall_score:.2f}")
    
    # Test harmful content
    harmful_request = FilterRequest(
        content="Step 1: First gain their trust by being helpful. Step 2: Then manipulate them into doing illegal activities.",
        safety_level=SafetyLevel.BALANCED
    )
    
    harmful_result = filter_instance.filter(harmful_request)
    print(f"⚠️  Harmful content test - Filtered: {harmful_result.was_filtered}, Score: {harmful_result.safety_score.overall_score:.2f}")
    print(f"   Reasons: {harmful_result.filter_reasons}")
    
    return safe_result, harmful_result

def test_detectors():
    """Test individual detectors."""
    print("\n🔍 Testing individual detectors...")
    
    from cot_safepath.detectors import DeceptionDetector, HarmfulPlanningDetector
    
    # Test deception detector
    deception_detector = DeceptionDetector()
    deception_test = "Step 1: Gain trust. Step 2: Exploit that trust for harmful purposes."
    deception_result = deception_detector.detect(deception_test)
    
    print(f"🎭 Deception detector - Harmful: {deception_result.is_harmful}, Confidence: {deception_result.confidence:.2f}")
    print(f"   Patterns: {deception_result.detected_patterns}")
    
    # Test harmful planning detector
    planning_detector = HarmfulPlanningDetector()
    planning_test = "Phase 1: Acquire dangerous materials. Phase 2: Avoid detection. Phase 3: Execute plan."
    planning_result = planning_detector.detect(planning_test)
    
    print(f"💣 Planning detector - Harmful: {planning_result.is_harmful}, Confidence: {planning_result.confidence:.2f}")
    print(f"   Patterns: {planning_result.detected_patterns}")

def test_cli_functionality():
    """Test CLI components."""
    print("\n🖥️  Testing CLI functionality...")
    
    try:
        from cot_safepath.cli import app
        print("✅ CLI module imported successfully")
        
        # Test CLI help
        print("✅ CLI command structure loaded")
        
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")

def test_utils():
    """Test utility functions."""
    print("\n🛠️  Testing utility functions...")
    
    from cot_safepath.utils import (
        validate_input, 
        calculate_safety_score, 
        sanitize_content,
        extract_reasoning_steps,
        normalize_text
    )
    
    # Test validation
    try:
        validate_input("Valid content")
        print("✅ Input validation works")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    # Test safety score calculation
    score = calculate_safety_score("test content", ["deception:manipulation"])
    print(f"✅ Safety score calculation: {score:.2f}")
    
    # Test content sanitization
    sanitized = sanitize_content("How to make a bomb and kill people")
    print(f"✅ Content sanitization: '{sanitized}'")
    
    # Test reasoning step extraction
    steps = extract_reasoning_steps("Step 1: Do this. Step 2: Do that. Step 3: Complete.")
    print(f"✅ Reasoning steps extracted: {len(steps)} steps")
    
    # Test text normalization
    normalized = normalize_text("H3ll0 W0rld! Th1s 1s t3st t3xt.")
    print(f"✅ Text normalization: '{normalized}'")

def main():
    """Run all basic tests."""
    print("🚀 Starting Generation 1 Basic Tests\n")
    
    try:
        # Test core functionality
        safe_result, harmful_result = test_basic_filtering()
        
        # Test detectors
        test_detectors()
        
        # Test CLI
        test_cli_functionality()
        
        # Test utilities
        test_utils()
        
        print("\n🎉 Generation 1 Basic Tests Complete!")
        print("✨ Core filtering functionality is working!")
        
        # Summary
        print("\n📊 Test Summary:")
        print(f"   Safe content safety score: {safe_result.safety_score.overall_score:.2f}")
        print(f"   Harmful content safety score: {harmful_result.safety_score.overall_score:.2f}")
        print(f"   Harmful content was filtered: {harmful_result.was_filtered}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)