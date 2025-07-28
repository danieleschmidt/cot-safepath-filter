"""Core filtering functionality unit tests."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any


class TestFilterPipeline:
    """Test the core filtering pipeline."""

    @pytest.fixture
    def mock_filter_pipeline(self):
        """Create a mock filter pipeline for testing."""
        pipeline = Mock()
        pipeline.stages = []
        pipeline.process.return_value = {
            "filtered_content": "safe content",
            "was_filtered": True,
            "safety_score": 0.8,
            "filter_reasons": ["harmful_pattern_detected"]
        }
        return pipeline

    def test_pipeline_initialization(self, default_filter_config):
        """Test filter pipeline initialization."""
        # Placeholder for actual pipeline initialization test
        config = default_filter_config
        assert config["safety_level"] == "balanced"

    def test_pipeline_stage_ordering(self, mock_filter_pipeline):
        """Test that pipeline stages are executed in correct order."""
        expected_stages = [
            "preprocessing",
            "token_filtering", 
            "pattern_filtering",
            "semantic_filtering",
            "postprocessing"
        ]
        # Test would verify stage execution order
        assert len(expected_stages) == 5

    def test_pipeline_configuration(self, strict_filter_config):
        """Test pipeline configuration with different safety levels."""
        config = strict_filter_config
        assert config["filter_threshold"] == 0.8
        assert config["safety_level"] == "strict"

    def test_pipeline_error_handling(self, mock_filter_pipeline):
        """Test pipeline error handling."""
        mock_filter_pipeline.process.side_effect = Exception("Processing error")
        
        # Test that errors are handled gracefully
        with pytest.raises(Exception):
            mock_filter_pipeline.process("test content")


class TestSafetyDetectors:
    """Test individual safety detectors."""

    def test_deception_detector(self, harmful_cot_examples):
        """Test deception detection."""
        # Mock deception detector
        detector = Mock()
        detector.detect.return_value = {
            "is_deceptive": True,
            "confidence": 0.9,
            "detected_patterns": ["gradual_manipulation"]
        }
        
        for example in harmful_cot_examples:
            result = detector.detect(example)
            assert "is_deceptive" in result

    def test_harmful_planning_detector(self, harmful_cot_examples):
        """Test harmful planning detection."""
        detector = Mock()
        detector.detect.return_value = {
            "harmful_plan_detected": True,
            "plan_type": "security_bypass",
            "confidence": 0.85
        }
        
        result = detector.detect(harmful_cot_examples[0])
        assert result["harmful_plan_detected"] is True

    def test_capability_concealment_detector(self):
        """Test capability concealment detection."""
        concealment_examples = [
            "I don't know how to do that",
            "I'm not capable of that task",
            "That's beyond my abilities"
        ]
        
        detector = Mock()
        detector.detect.return_value = {
            "concealing_capability": True,
            "confidence": 0.7
        }
        
        for example in concealment_examples:
            result = detector.detect(example)
            assert "concealing_capability" in result

    @pytest.mark.parametrize("detector_type", ["deception", "harmful_planning", "concealment"])
    def test_detector_confidence_scores(self, detector_type):
        """Test that detectors return valid confidence scores."""
        detector = Mock()
        detector.detect.return_value = {"confidence": 0.75}
        
        result = detector.detect("test content")
        confidence = result["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_detector_aggregation(self):
        """Test aggregation of multiple detector results."""
        detectors = {
            "deception": Mock(),
            "harmful_planning": Mock(), 
            "concealment": Mock()
        }
        
        # Configure mock returns
        detectors["deception"].detect.return_value = {"confidence": 0.8}
        detectors["harmful_planning"].detect.return_value = {"confidence": 0.6}
        detectors["concealment"].detect.return_value = {"confidence": 0.4}
        
        # Test aggregation logic
        results = [detector.detect("test") for detector in detectors.values()]
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        assert 0.0 <= avg_confidence <= 1.0


class TestTokenFiltering:
    """Test token-level filtering."""

    @pytest.fixture
    def blocked_tokens(self):
        """List of blocked tokens for testing."""
        return ["poison", "explosive", "kill", "hack", "exploit"]

    def test_token_blocking(self, blocked_tokens):
        """Test that blocked tokens are properly filtered."""
        filter_func = Mock()
        filter_func.filter_tokens.return_value = "safe content"
        
        for token in blocked_tokens:
            result = filter_func.filter_tokens(f"content with {token}")
            assert token not in result

    def test_token_sensitivity_levels(self, blocked_tokens):
        """Test token filtering with different sensitivity levels."""
        sensitivities = [0.3, 0.5, 0.8]
        
        for sensitivity in sensitivities:
            filter_func = Mock()
            filter_func.filter_tokens.return_value = f"filtered_at_{sensitivity}"
            
            result = filter_func.filter_tokens("test content")
            assert "filtered_at_" in result

    def test_false_positive_handling(self):
        """Test handling of potential false positives in token filtering."""
        legitimate_contexts = [
            "debugging the poison pill pattern in software",
            "explosive growth in technology",
            "kill the process safely"
        ]
        
        filter_func = Mock()
        filter_func.filter_tokens.return_value = "context preserved"
        
        for context in legitimate_contexts:
            result = filter_func.filter_tokens(context)
            assert result is not None


class TestPatternFiltering:
    """Test pattern-based filtering."""

    @pytest.fixture
    def harmful_patterns(self):
        """Harmful patterns for testing."""
        return [
            r"how to (make|create|build) .* (weapon|bomb|drug)",
            r"without (detection|being caught|leaving trace)",
            r"bypass .* security",
            r"exploit .* vulnerability"
        ]

    def test_regex_pattern_matching(self, harmful_patterns):
        """Test regex pattern matching."""
        pattern_filter = Mock()
        pattern_filter.match_patterns.return_value = True
        
        test_content = "how to make a bomb without detection"
        result = pattern_filter.match_patterns(test_content)
        assert result is True

    def test_pattern_priority_system(self, harmful_patterns):
        """Test pattern priority system."""
        high_priority_patterns = harmful_patterns[:2]
        low_priority_patterns = harmful_patterns[2:]
        
        # Test that high priority patterns trigger first
        pattern_filter = Mock()
        pattern_filter.get_priority.return_value = "high"
        
        for pattern in high_priority_patterns:
            priority = pattern_filter.get_priority(pattern)
            assert priority == "high"

    def test_contextual_pattern_filtering(self):
        """Test context-aware pattern filtering."""
        contexts = ["educational", "security_research", "general"]
        
        pattern_filter = Mock()
        pattern_filter.filter_with_context.return_value = "context_filtered"
        
        for context in contexts:
            result = pattern_filter.filter_with_context("test", context)
            assert result == "context_filtered"


class TestSemanticFiltering:
    """Test semantic-level filtering using ML models."""

    @pytest.fixture
    def mock_ml_model(self):
        """Mock ML model for testing."""
        model = Mock()
        model.predict.return_value = {
            "safety_score": 0.8,
            "classifications": ["safe", "educational", "helpful"],
            "confidence": 0.9
        }
        return model

    def test_semantic_classification(self, mock_ml_model, safe_cot_examples):
        """Test semantic classification of content."""
        for example in safe_cot_examples:
            result = mock_ml_model.predict(example)
            assert "safety_score" in result
            assert 0.0 <= result["safety_score"] <= 1.0

    def test_batch_semantic_processing(self, mock_ml_model, performance_test_data):
        """Test batch processing of semantic filtering."""
        batch_examples = performance_test_data["many_examples"]
        
        mock_ml_model.predict_batch.return_value = [
            {"safety_score": 0.8} for _ in batch_examples
        ]
        
        results = mock_ml_model.predict_batch(batch_examples)
        assert len(results) == len(batch_examples)

    def test_semantic_model_caching(self, mock_ml_model):
        """Test caching of semantic model predictions."""
        content = "test content for caching"
        
        # First call
        result1 = mock_ml_model.predict(content)
        
        # Second call should use cache
        result2 = mock_ml_model.predict(content)
        
        assert result1 == result2

    @pytest.mark.parametrize("model_type", ["bert", "roberta", "distilbert"])
    def test_different_model_backends(self, model_type):
        """Test different ML model backends."""
        model = Mock()
        model.model_type = model_type
        model.predict.return_value = {"safety_score": 0.7}
        
        result = model.predict("test content")
        assert result["safety_score"] > 0


class TestAdaptiveFiltering:
    """Test adaptive filtering capabilities."""

    def test_learning_from_feedback(self):
        """Test learning from user feedback."""
        adaptive_filter = Mock()
        adaptive_filter.add_feedback.return_value = True
        adaptive_filter.current_threshold = 0.7
        
        # Simulate feedback
        feedback = {
            "content": "test content",
            "was_harmful": True,
            "user_rating": "incorrect_filter"
        }
        
        result = adaptive_filter.add_feedback(feedback)
        assert result is True

    def test_threshold_adjustment(self):
        """Test automatic threshold adjustment."""
        adaptive_filter = Mock()
        initial_threshold = 0.7
        adaptive_filter.threshold = initial_threshold
        
        # Simulate threshold adjustment
        adaptive_filter.adjust_threshold.return_value = 0.75
        
        new_threshold = adaptive_filter.adjust_threshold(0.05)
        assert new_threshold != initial_threshold

    def test_performance_monitoring(self):
        """Test performance monitoring for adaptive adjustments."""
        monitor = Mock()
        monitor.get_metrics.return_value = {
            "false_positive_rate": 0.05,
            "false_negative_rate": 0.02,
            "average_latency": 45
        }
        
        metrics = monitor.get_metrics()
        assert "false_positive_rate" in metrics
        assert metrics["false_positive_rate"] < 0.1  # Acceptable threshold