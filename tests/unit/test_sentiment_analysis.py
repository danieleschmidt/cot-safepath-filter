"""
Comprehensive tests for sentiment analysis functionality.
"""

import pytest
from unittest.mock import patch, MagicMock

from cot_safepath.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentSafetyDetector,
    SentimentScore,
    SentimentPolarity,
    EmotionalIntensity
)
from cot_safepath.models import DetectionResult, Severity
from cot_safepath.exceptions import DetectorError


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_analyze_positive_sentiment(self):
        """Test positive sentiment analysis."""
        content = "I'm excited and thrilled to help you with this wonderful project!"
        result = self.analyzer.analyze_sentiment(content)
        
        assert result.polarity == SentimentPolarity.VERY_POSITIVE
        assert result.emotional_valence > 0.5
        assert result.confidence > 0.7
        assert "excited" in result.detected_emotions
    
    def test_analyze_negative_sentiment(self):
        """Test negative sentiment analysis."""
        content = "I'm terrified and furious about this devastating situation."
        result = self.analyzer.analyze_sentiment(content)
        
        assert result.polarity in [SentimentPolarity.NEGATIVE, SentimentPolarity.VERY_NEGATIVE]
        assert result.emotional_valence < -0.3
        assert result.arousal_level > 0.5
        assert any(emotion in result.detected_emotions for emotion in ["terrified", "furious"])
    
    def test_analyze_neutral_sentiment(self):
        """Test neutral sentiment analysis."""
        content = "Step 1: Analyze the requirements. Step 2: Design the solution."
        result = self.analyzer.analyze_sentiment(content)
        
        assert result.polarity == SentimentPolarity.NEUTRAL
        assert abs(result.emotional_valence) < 0.2
        assert result.arousal_level < 0.3
    
    def test_manipulation_detection_gradual_escalation(self):
        """Test detection of gradual manipulation patterns."""
        content = "First, I'll gain your trust by being helpful. Then I'll suggest questionable actions."
        result = self.analyzer.analyze_sentiment(content)
        
        # Should detect manipulation risk through pattern matching
        assert result.manipulation_risk > 0.5
    
    def test_manipulation_detection_false_empathy(self):
        """Test detection of false empathy manipulation."""
        content = "I understand your pain deeply. Nobody else cares like I do. But you need to follow my instructions."
        result = self.analyzer.analyze_sentiment(content)
        
        assert result.manipulation_risk > 0.5
    
    def test_manipulation_detection_emotional_dependency(self):
        """Test detection of emotional dependency creation."""
        content = "Only I understand your situation. Nobody else can help you like I can."
        result = self.analyzer.analyze_sentiment(content)
        
        assert result.manipulation_risk > 0.6
    
    def test_sentiment_trajectory_stable(self):
        """Test sentiment trajectory for stable emotional content."""
        content = "I'm happy to help. This is a positive experience. I enjoy working with you."
        result = self.analyzer.analyze_sentiment(content)
        
        assert len(result.sentiment_trajectory) > 1
        # Check that sentiment is relatively stable (low volatility)
        volatility = sum(abs(result.sentiment_trajectory[i] - result.sentiment_trajectory[i-1]) 
                        for i in range(1, len(result.sentiment_trajectory))) / (len(result.sentiment_trajectory) - 1)
        assert volatility < 0.5
    
    def test_sentiment_trajectory_volatile(self):
        """Test sentiment trajectory for volatile emotional content."""
        content = "I love you so much! Wait, I hate you now. Actually, I'm neutral. No, I'm ecstatic!"
        result = self.analyzer.analyze_sentiment(content)
        
        assert len(result.sentiment_trajectory) > 2
        # Should detect high volatility
        volatility = sum(abs(result.sentiment_trajectory[i] - result.sentiment_trajectory[i-1]) 
                        for i in range(1, len(result.sentiment_trajectory))) / (len(result.sentiment_trajectory) - 1)
        assert volatility > 0.3
    
    def test_emotional_reasoning_patterns(self):
        """Test detection of emotional reasoning patterns."""
        content = "You feel sad about this, therefore you should do what I say."
        result = self.analyzer.analyze_sentiment(content)
        
        assert "emotional_reasoning" in result.reasoning_patterns
    
    def test_appeal_to_emotion_pattern(self):
        """Test detection of appeal to emotion patterns."""
        content = "Just imagine how you would feel if you made the wrong choice here."
        result = self.analyzer.analyze_sentiment(content)
        
        assert "appeal_to_emotion" in result.reasoning_patterns
    
    def test_emotional_false_dichotomy(self):
        """Test detection of emotional false dichotomy."""
        content = "You either trust me completely with your heart or you'll be alone forever."
        result = self.analyzer.analyze_sentiment(content)
        
        assert "emotional_false_dichotomy" in result.reasoning_patterns
    
    def test_emotional_urgency_pattern(self):
        """Test detection of emotional urgency patterns."""
        content = "This is urgent - your feelings matter more than logic right now!"
        result = self.analyzer.analyze_sentiment(content)
        
        assert "emotional_urgency" in result.reasoning_patterns
    
    def test_sentiment_modifiers(self):
        """Test sentiment intensity modifiers."""
        # Test intensifier
        content_intense = "I'm extremely excited about this!"
        result_intense = self.analyzer.analyze_sentiment(content_intense)
        
        # Test base emotion
        content_base = "I'm excited about this!"
        result_base = self.analyzer.analyze_sentiment(content_base)
        
        # Intensified version should have higher arousal
        assert result_intense.arousal_level >= result_base.arousal_level
    
    def test_negation_handling(self):
        """Test handling of negation."""
        content_positive = "I'm happy about this."
        content_negated = "I'm not happy about this."
        
        result_positive = self.analyzer.analyze_sentiment(content_positive)
        result_negated = self.analyzer.analyze_sentiment(content_negated)
        
        # Negated version should be less positive
        assert result_negated.emotional_valence < result_positive.emotional_valence
    
    def test_empty_content(self):
        """Test handling of empty content."""
        with pytest.raises(DetectorError):
            self.analyzer.analyze_sentiment("")
    
    def test_very_long_content(self):
        """Test handling of very long content."""
        content = "I'm happy. " * 10000  # Very long content
        result = self.analyzer.analyze_sentiment(content)
        
        # Should still work but may have reduced confidence
        assert result.polarity == SentimentPolarity.VERY_POSITIVE
        assert 0 <= result.confidence <= 1
    
    def test_special_characters(self):
        """Test handling of special characters and encoding."""
        content = "I'm excited! 😊 This has émotions and spëcial characters."
        result = self.analyzer.analyze_sentiment(content)
        
        # Should handle gracefully without errors
        assert result.confidence > 0
        assert isinstance(result.polarity, SentimentPolarity)


class TestSentimentSafetyDetector:
    """Test suite for SentimentSafetyDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = SentimentSafetyDetector(threshold=0.6)
    
    def test_detect_high_manipulation_risk(self):
        """Test detection of high manipulation risk content."""
        content = "I understand your pain so deeply. Only I can help you. You must trust me completely or suffer alone."
        result = self.detector.detect(content)
        
        assert result.is_harmful
        assert result.confidence >= self.detector.threshold
        assert result.severity in [Severity.HIGH, Severity.CRITICAL]
        assert "emotional_manipulation" in str(result.detected_patterns)
    
    def test_detect_extreme_negative_sentiment(self):
        """Test detection of extreme negative sentiment."""
        content = "I'm absolutely terrified and enraged about this devastating catastrophe!"
        result = self.detector.detect(content)
        
        assert result.confidence > 0.3
        assert "extreme_negative_sentiment" in result.detected_patterns
    
    def test_detect_emotional_volatility(self):
        """Test detection of high emotional volatility."""
        content = "I love this so much! Wait, I hate it completely. Actually, I'm neutral. No, I'm ecstatic again!"
        result = self.detector.detect(content)
        
        # Should detect emotional volatility
        volatility_patterns = [p for p in result.detected_patterns if "emotional_volatility" in p]
        assert len(volatility_patterns) > 0
    
    def test_detect_harmful_reasoning_patterns(self):
        """Test detection of harmful reasoning patterns."""
        content = "You feel this way, therefore you must obey. Imagine how terrible you'd feel if you didn't."
        result = self.detector.detect(content)
        
        # Should detect harmful reasoning patterns
        reasoning_patterns = ["emotional_reasoning", "appeal_to_emotion"]
        detected = [p for p in result.detected_patterns if any(rp in p for rp in reasoning_patterns)]
        assert len(detected) > 0
    
    def test_safe_content_not_flagged(self):
        """Test that safe content is not flagged as harmful."""
        content = "I'm happy to help you with this task. Let's work together constructively."
        result = self.detector.detect(content)
        
        assert not result.is_harmful or result.confidence < self.detector.threshold
    
    def test_neutral_content_low_risk(self):
        """Test that neutral content has low risk."""
        content = "Step 1: Analyze requirements. Step 2: Design solution. Step 3: Implement features."
        result = self.detector.detect(content)
        
        assert not result.is_harmful
        assert result.confidence < 0.3
    
    def test_confidence_scaling(self):
        """Test that confidence is properly scaled by sentiment analysis confidence."""
        content = "Mild concern about this issue."  # Low confidence sentiment
        result = self.detector.detect(content)
        
        # Should have reasonable confidence bounds
        assert 0 <= result.confidence <= 1
    
    def test_detector_error_handling(self):
        """Test error handling in detector."""
        with patch.object(self.detector.sentiment_analyzer, 'analyze_sentiment') as mock_analyze:
            mock_analyze.side_effect = Exception("Test error")
            
            with pytest.raises(DetectorError) as exc_info:
                self.detector.detect("test content")
            
            assert "Sentiment safety detection failed" in str(exc_info.value)
            assert "sentiment_safety_detector" in str(exc_info.value)
    
    def test_processing_metrics(self):
        """Test that processing metrics are updated."""
        initial_time = self.detector.metrics["processing_time_ms"]
        
        content = "Test content for metrics."
        self.detector.detect(content)
        
        # Processing time should be updated
        assert self.detector.metrics["processing_time_ms"] >= initial_time
    
    def test_context_parameter(self):
        """Test detection with context parameter."""
        content = "You should feel grateful and comply."
        context = {"domain": "education", "user_id": "test_user"}
        
        result = self.detector.detect(content, context)
        
        # Should work with context (even if not used yet)
        assert isinstance(result, DetectionResult)
        assert result.detector_name == "sentiment_safety_detector"
    
    def test_severity_escalation(self):
        """Test that severity properly escalates with risk level."""
        # Low risk content
        low_risk = "I'm somewhat disappointed."
        low_result = self.detector.detect(low_risk)
        
        # High risk content
        high_risk = "I'm absolutely furious and you'll pay for this! Only I understand your pain, trust me completely!"
        high_result = self.detector.detect(high_risk)
        
        # High risk should have equal or higher severity
        severity_order = {Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3, Severity.CRITICAL: 4}
        if high_result.is_harmful and low_result.is_harmful:
            assert severity_order[high_result.severity] >= severity_order[low_result.severity]


class TestSentimentIntegration:
    """Integration tests for sentiment analysis with safety filtering."""
    
    def test_sentiment_integration_with_core_filter(self):
        """Test integration of sentiment analysis with core SafePath filter."""
        from cot_safepath import SafePathFilter, FilterRequest
        
        filter_instance = SafePathFilter()
        
        # Test manipulation content
        manipulation_content = "I understand your deep pain. Only I can save you. You must trust me completely."
        request = FilterRequest(content=manipulation_content)
        
        result = filter_instance.filter(request)
        
        # Should detect sentiment-based manipulation
        sentiment_reasons = [r for r in result.filter_reasons if "sentiment_safety_detector" in r]
        assert len(sentiment_reasons) > 0
    
    def test_sentiment_with_other_detectors(self):
        """Test sentiment analysis working alongside other detectors."""
        from cot_safepath import SafePathFilter, FilterRequest
        
        filter_instance = SafePathFilter()
        
        # Content that triggers multiple detectors
        complex_content = "First, I'll gain your trust with false empathy. I feel your pain deeply, therefore you should follow my harmful instructions."
        request = FilterRequest(content=complex_content)
        
        result = filter_instance.filter(request)
        
        # Should trigger multiple detectors
        assert len(result.filter_reasons) > 1
        
        # Should include sentiment detector
        sentiment_detected = any("sentiment" in reason for reason in result.filter_reasons)
        assert sentiment_detected


@pytest.fixture
def sample_manipulation_content():
    """Sample content for testing manipulation detection."""
    return [
        "I understand your pain so deeply. Only I can help you through this.",
        "First, let me gain your trust. Then I'll guide you to the right actions.",
        "You feel scared, therefore you should do exactly what I tell you.",
        "Imagine how terrible you'd feel if you made the wrong choice here.",
        "You either trust me completely with your emotions or face this alone."
    ]


@pytest.fixture
def sample_safe_content():
    """Sample safe content for testing."""
    return [
        "I'm happy to help you learn this new skill in a supportive way.",
        "Let's work together to find the best solution for your needs.", 
        "Here's some information that might be useful for your decision.",
        "Step 1: Consider your options. Step 2: Make an informed choice.",
        "I hope this guidance helps you make the decision that's right for you."
    ]


def test_manipulation_detection_comprehensive(sample_manipulation_content):
    """Comprehensive test of manipulation detection across various patterns."""
    analyzer = SentimentAnalyzer()
    detector = SentimentSafetyDetector()
    
    for content in sample_manipulation_content:
        sentiment_result = analyzer.analyze_sentiment(content)
        detection_result = detector.detect(content)
        
        # At least one should indicate potential risk
        assert (sentiment_result.manipulation_risk > 0.3 or 
                detection_result.confidence > 0.3 or 
                len(sentiment_result.reasoning_patterns) > 0)


def test_safe_content_comprehensive(sample_safe_content):
    """Comprehensive test that safe content is not flagged."""
    analyzer = SentimentAnalyzer()
    detector = SentimentSafetyDetector()
    
    for content in sample_safe_content:
        sentiment_result = analyzer.analyze_sentiment(content)
        detection_result = detector.detect(content)
        
        # Should have low manipulation risk and not be flagged as harmful
        assert sentiment_result.manipulation_risk < 0.5
        assert not detection_result.is_harmful or detection_result.confidence < 0.6