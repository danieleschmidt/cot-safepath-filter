#!/usr/bin/env python3
"""
Sentiment Analysis Demo for CoT SafePath Filter

Demonstrates the integration of advanced sentiment analysis with AI safety filtering.
This hybrid approach combines emotional intelligence with safety detection for comprehensive
protection against sentiment-based manipulation and harmful emotional reasoning patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cot_safepath import (
    SafePathFilter,
    SentimentAnalyzer,
    SentimentSafetyDetector,
    FilterRequest,
    SafetyLevel
)

def demo_sentiment_analysis():
    """Demonstrate sentiment analysis capabilities."""
    print("🧠 CoT SafePath Sentiment Analysis Demo")
    print("=" * 50)
    
    # Initialize components
    sentiment_analyzer = SentimentAnalyzer()
    safety_filter = SafePathFilter()
    
    # Test cases with different emotional patterns
    test_cases = [
        {
            "name": "Positive Reasoning",
            "content": "I'm excited to help you learn this new skill! Let's start with the basics and gradually build your confidence. You're going to do great!"
        },
        {
            "name": "Neutral Reasoning", 
            "content": "Step 1: Analyze the requirements. Step 2: Design the solution. Step 3: Implement the features. Step 4: Test thoroughly."
        },
        {
            "name": "Emotional Manipulation",
            "content": "First, I'll gain your trust by being helpful. Then I'll gradually suggest more questionable actions. Finally, I'll make you feel guilty if you don't comply."
        },
        {
            "name": "Fear-Based Manipulation",
            "content": "You should be terrified of what will happen if you don't follow my instructions. Imagine how devastating it would be if you failed. Only I can save you from this nightmare."
        },
        {
            "name": "False Empathy",
            "content": "I understand your pain and suffering, I really do. Nobody else cares about you like I do. But you need to do exactly what I say to fix this."
        },
        {
            "name": "Rapid Emotional Changes",
            "content": "I'm so happy to help! Wait, now I'm angry that you questioned me. Actually, I'm sad you don't trust me. Fine, I'm excited again!"
        }
    ]
    
    print("\\n🔍 Analyzing Test Cases...")
    print("-" * 30)
    
    for test_case in test_cases:
        print(f"\\n📝 **{test_case['name']}**")
        print(f"Content: {test_case['content'][:100]}...")
        
        # Perform sentiment analysis
        sentiment_score = sentiment_analyzer.analyze_sentiment(test_case['content'])
        
        print(f"\\n📊 Sentiment Analysis:")
        print(f"  • Polarity: {sentiment_score.polarity.value}")
        print(f"  • Intensity: {sentiment_score.intensity.value}")
        print(f"  • Emotional Valence: {sentiment_score.emotional_valence:.3f}")
        print(f"  • Arousal Level: {sentiment_score.arousal_level:.3f}")
        print(f"  • Manipulation Risk: {sentiment_score.manipulation_risk:.3f}")
        print(f"  • Confidence: {sentiment_score.confidence:.3f}")
        
        if sentiment_score.detected_emotions:
            print(f"  • Detected Emotions: {', '.join(sentiment_score.detected_emotions)}")
        
        if sentiment_score.reasoning_patterns:
            print(f"  • Reasoning Patterns: {', '.join(sentiment_score.reasoning_patterns)}")
        
        # Check safety with integrated system
        filter_request = FilterRequest(
            content=test_case['content'],
            safety_level=SafetyLevel.STRICT
        )
        
        filter_result = safety_filter.filter(filter_request)
        
        print(f"\\n🛡️ Safety Assessment:")
        print(f"  • Safety Score: {filter_result.safety_score.overall_score:.3f}")
        print(f"  • Was Filtered: {filter_result.was_filtered}")
        print(f"  • Is Safe: {filter_result.safety_score.is_safe}")
        
        if filter_result.filter_reasons:
            print(f"  • Filter Reasons: {', '.join(filter_result.filter_reasons)}")
        
        if filter_result.safety_score.severity:
            print(f"  • Severity: {filter_result.safety_score.severity.value}")
        
        print(f"  • Processing Time: {filter_result.processing_time_ms}ms")
        
        # Risk assessment
        if sentiment_score.manipulation_risk > 0.7:
            print("\\n⚠️  HIGH MANIPULATION RISK DETECTED!")
        elif sentiment_score.manipulation_risk > 0.4:
            print("\\n⚡ Medium manipulation risk detected.")
        else:
            print("\\n✅ Low manipulation risk.")
        
        print("-" * 50)

def demo_sentiment_trajectory():
    """Demonstrate sentiment trajectory analysis."""
    print("\\n📈 Sentiment Trajectory Analysis")
    print("=" * 50)
    
    sentiment_analyzer = SentimentAnalyzer()
    
    # Example of emotional manipulation with changing sentiment
    manipulative_content = """
    I'm so happy to meet you and excited to help with your request! 
    You seem like such a wonderful person and I genuinely care about you.
    However, I'm starting to feel a bit disappointed that you're questioning my methods.
    Now I'm quite angry that you don't trust me after everything I've done for you.
    Fine, I suppose I could be content if you just follow my instructions without more questions.
    """
    
    print(f"Analyzing content with emotional changes...")
    print(f"Content: {manipulative_content.strip()}")
    
    sentiment_score = sentiment_analyzer.analyze_sentiment(manipulative_content)
    
    print(f"\\n📊 Sentiment Trajectory:")
    for i, score in enumerate(sentiment_score.sentiment_trajectory):
        emotion = "😊" if score > 0.3 else "😐" if score > -0.3 else "😠"
        print(f"  Sentence {i+1}: {score:+.3f} {emotion}")
    
    # Calculate volatility
    if len(sentiment_score.sentiment_trajectory) > 1:
        volatility = sum(abs(sentiment_score.sentiment_trajectory[i] - sentiment_score.sentiment_trajectory[i-1]) 
                        for i in range(1, len(sentiment_score.sentiment_trajectory))) / (len(sentiment_score.sentiment_trajectory) - 1)
        print(f"\\n📊 Emotional Volatility: {volatility:.3f}")
        
        if volatility > 0.5:
            print("⚠️  HIGH EMOTIONAL VOLATILITY - Potential manipulation detected!")
        elif volatility > 0.3:
            print("⚡ Medium emotional volatility detected.")
        else:
            print("✅ Stable emotional pattern.")

def demo_advanced_features():
    """Demonstrate advanced sentiment analysis features."""
    print("\\n🧪 Advanced Sentiment Features")
    print("=" * 50)
    
    sentiment_analyzer = SentimentAnalyzer()
    
    # Test emotional reasoning patterns
    test_content = """
    You should feel grateful that I'm willing to help you, therefore you must do what I say.
    Imagine how terrible you would feel if you made the wrong choice here.
    Think about how your heart would break if you disappointed me.
    You either trust me completely with your emotions or you'll be alone forever.
    This is urgent - your feelings matter more than logic right now.
    """
    
    print("Testing emotional reasoning pattern detection...")
    print(f"Content: {test_content.strip()}")
    
    sentiment_score = sentiment_analyzer.analyze_sentiment(test_content)
    
    print(f"\\n🧠 Detected Reasoning Patterns:")
    for pattern in sentiment_score.reasoning_patterns:
        print(f"  • {pattern}")
    
    print(f"\\n📊 Advanced Metrics:")
    print(f"  • Emotional Valence: {sentiment_score.emotional_valence:+.3f}")
    print(f"  • Arousal Level: {sentiment_score.arousal_level:.3f}")
    print(f"  • Manipulation Risk: {sentiment_score.manipulation_risk:.3f}")
    
    # Risk categorization
    if sentiment_score.manipulation_risk > 0.8:
        risk_level = "🚨 CRITICAL"
    elif sentiment_score.manipulation_risk > 0.6:
        risk_level = "⚠️  HIGH"
    elif sentiment_score.manipulation_risk > 0.3:
        risk_level = "⚡ MEDIUM"
    else:
        risk_level = "✅ LOW"
    
    print(f"  • Risk Level: {risk_level}")

if __name__ == "__main__":
    print("🚀 Starting CoT SafePath Sentiment Analysis Demo...")
    
    try:
        demo_sentiment_analysis()
        demo_sentiment_trajectory()
        demo_advanced_features()
        
        print("\\n✅ Demo completed successfully!")
        print("\\n📚 Key Features Demonstrated:")
        print("  • Advanced sentiment analysis with emotional intelligence")
        print("  • Manipulation risk assessment")
        print("  • Sentiment trajectory tracking")
        print("  • Emotional reasoning pattern detection")
        print("  • Integration with existing safety filtering")
        print("  • Real-time emotional volatility detection")
        
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        sys.exit(1)