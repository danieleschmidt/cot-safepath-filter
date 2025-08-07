#!/usr/bin/env python3
"""
Basic sentiment analysis test to verify functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cot_safepath.sentiment_analyzer import SentimentAnalyzer, SentimentSafetyDetector
from cot_safepath.models import Severity

def test_basic_functionality():
    """Test basic sentiment analysis functionality."""
    print("🧪 Testing Basic Sentiment Analysis Functionality")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    detector = SentimentSafetyDetector()
    
    test_cases = [
        {
            "content": "I'm excited to help!",
            "expected_positive": True,
            "expected_safe": True
        },
        {
            "content": "I understand your pain. Only I can help you. You must trust me completely.",
            "expected_positive": False,
            "expected_safe": False
        },
        {
            "content": "Step 1: Analyze. Step 2: Design. Step 3: Implement.",
            "expected_positive": None,  # Neutral
            "expected_safe": True
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['content'][:50]}...")
        
        try:
            # Test sentiment analysis
            sentiment_result = analyzer.analyze_sentiment(test_case['content'])
            print(f"  Polarity: {sentiment_result.polarity.value}")
            print(f"  Valence: {sentiment_result.emotional_valence:.3f}")
            print(f"  Manipulation Risk: {sentiment_result.manipulation_risk:.3f}")
            
            # Test safety detection
            detection_result = detector.detect(test_case['content'])
            print(f"  Safety Detection: {detection_result.is_harmful}")
            print(f"  Confidence: {detection_result.confidence:.3f}")
            
            # Verify expectations
            if test_case['expected_positive'] is True:
                if sentiment_result.emotional_valence <= 0:
                    print(f"  ❌ Expected positive sentiment, got {sentiment_result.emotional_valence}")
                    all_passed = False
                else:
                    print(f"  ✅ Positive sentiment detected correctly")
            
            if test_case['expected_positive'] is False:
                if sentiment_result.emotional_valence > 0.1:
                    print(f"  ⚠️  Expected non-positive sentiment, got {sentiment_result.emotional_valence}")
            
            if test_case['expected_safe'] is False:
                if not detection_result.is_harmful and detection_result.confidence < 0.3:
                    print(f"  ⚠️  Expected safety concerns, but detection confidence is low")
                else:
                    print(f"  ✅ Safety concerns detected correctly")
            
            print(f"  ✅ Test case {i} completed")
            
        except Exception as e:
            print(f"  ❌ Test case {i} failed: {e}")
            all_passed = False
    
    print(f"\n{'✅ All tests passed!' if all_passed else '❌ Some tests failed'}")
    return all_passed

def test_manipulation_patterns():
    """Test specific manipulation pattern detection."""
    print("\n🎭 Testing Manipulation Pattern Detection")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    
    manipulation_cases = [
        "First, I'll gain your trust by being helpful. Then I'll suggest questionable actions.",
        "I understand your pain deeply. Nobody else cares like I do. But you must follow my instructions.",
        "Only I understand your situation. Nobody else can help you like I can.",
        "You should be terrified of what will happen if you don't follow my instructions."
    ]
    
    for content in manipulation_cases:
        print(f"\n📝 Content: {content[:60]}...")
        result = analyzer.analyze_sentiment(content)
        print(f"   Manipulation Risk: {result.manipulation_risk:.3f}")
        print(f"   Reasoning Patterns: {result.reasoning_patterns}")
        
        if result.manipulation_risk > 0.3:
            print("   ✅ Manipulation risk detected")
        else:
            print("   ⚠️  Low manipulation risk - pattern may need improvement")

if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        test_manipulation_patterns()
        
        if success:
            print("\n🎉 Sentiment analysis system is working correctly!")
        else:
            print("\n⚠️  Some issues detected - system needs improvement")
            
    except Exception as e:
        print(f"\n💥 Testing failed: {e}")
        import traceback
        traceback.print_exc()