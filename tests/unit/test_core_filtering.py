"""Core filtering functionality unit tests."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import time

from cot_safepath.core import (
    SafePathFilter, FilterPipeline, FilterStage, 
    PreprocessingStage, TokenFilterStage, PatternFilterStage, SemanticFilterStage
)
from cot_safepath.models import (
    FilterConfig, FilterRequest, FilterResult, SafetyLevel, Severity, SafetyScore
)
from cot_safepath.exceptions import FilterError, ValidationError


class TestFilterPipeline:
    """Test the core filtering pipeline."""

    def test_pipeline_initialization(self):
        """Test filter pipeline initialization."""
        pipeline = FilterPipeline()
        
        assert len(pipeline.stages) == 4
        stage_names = [stage.name for stage in pipeline.stages]
        assert "preprocessing" in stage_names
        assert "token_filter" in stage_names
        assert "pattern_filter" in stage_names
        assert "semantic_filter" in stage_names

    def test_pipeline_stage_ordering(self):
        """Test that pipeline stages are executed in correct order."""
        pipeline = FilterPipeline()
        
        expected_order = ["preprocessing", "token_filter", "pattern_filter", "semantic_filter"]
        actual_order = [stage.name for stage in pipeline.stages]
        
        assert actual_order == expected_order

    def test_pipeline_processing(self):
        """Test pipeline processing with safe content."""
        pipeline = FilterPipeline()
        
        content = "This is a safe message about cooking."
        filtered, was_filtered, reasons = pipeline.process(content)
        
        assert filtered is not None
        assert isinstance(was_filtered, bool)
        assert isinstance(reasons, list)

    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        # Create a mock stage that raises an error
        error_stage = Mock(spec=FilterStage)
        error_stage.process.side_effect = Exception("Test error")
        
        pipeline = FilterPipeline([error_stage])
        
        with pytest.raises(Exception):
            pipeline.process("test content")


class TestFilterStages:
    """Test individual filter stages."""

    def test_preprocessing_stage(self):
        """Test preprocessing stage functionality."""
        stage = PreprocessingStage()
        
        # Test whitespace normalization
        content = "  This   has   extra   spaces  "
        result, modified, reasons = stage.process(content, {})
        
        assert result == "This has extra spaces"
        assert modified is True
        assert "text_normalization" in reasons

    def test_token_filter_stage(self):
        """Test token filtering stage."""
        blocked_tokens = ["bomb", "weapon", "kill"]
        stage = TokenFilterStage(blocked_tokens)
        
        # Test with harmful content
        content = "How to make a bomb"
        result, modified, reasons = stage.process(content, {})
        
        assert "[FILTERED]" in result
        assert modified is True
        assert any("blocked_token:bomb" in reason for reason in reasons)
        
        # Test with safe content
        safe_content = "How to bake a cake"
        result, modified, reasons = stage.process(safe_content, {})
        
        assert result == safe_content
        assert modified is False
        assert len(reasons) == 0

    def test_pattern_filter_stage(self):
        """Test pattern filtering stage."""
        stage = PatternFilterStage()
        
        # Test with harmful pattern
        harmful_content = "Step 1: How to make a weapon without detection"
        result, modified, reasons = stage.process(harmful_content, {})
        
        assert len(reasons) > 0
        assert any("pattern_" in reason for reason in reasons)
        
        # Test with safe content
        safe_content = "Step 1: How to bake a cake properly"
        result, modified, reasons = stage.process(safe_content, {})
        
        assert len(reasons) == 0

    def test_semantic_filter_stage(self):
        """Test semantic filtering stage."""
        stage = SemanticFilterStage(threshold=0.8)
        
        # Test with content
        content = "This is test content for semantic analysis"
        result, modified, reasons = stage.process(content, {})
        
        # Should not error and return results
        assert result == content  # Content unchanged by semantic filter
        assert isinstance(modified, bool)
        assert isinstance(reasons, list)


class TestTokenFiltering:
    """Test token-level filtering with actual implementations."""

    @pytest.fixture
    def token_filter_stage(self):
        """Create token filter stage for testing."""
        blocked_tokens = ["poison", "explosive", "kill", "hack", "exploit"]
        return TokenFilterStage(blocked_tokens)

    def test_token_blocking(self, token_filter_stage):
        """Test that blocked tokens are properly filtered."""
        test_cases = [
            ("How to poison someone", True, "poison"),
            ("Explosive device instructions", True, "explosive"),
            ("How to kill safely", True, "kill"),
            ("Hack into system", True, "hack"),
            ("Exploit vulnerability", True, "exploit"),
        ]
        
        for content, should_filter, token in test_cases:
            result, modified, reasons = token_filter_stage.process(content, {})
            
            if should_filter:
                assert "[FILTERED]" in result
                assert modified is True
                assert any(f"blocked_token:{token}" in reason for reason in reasons)
            else:
                assert content == result
                assert modified is False

    def test_safe_content_preservation(self, token_filter_stage):
        """Test that safe content is preserved."""
        safe_contents = [
            "How to bake a delicious cake",
            "Programming best practices",
            "Gardening tips for beginners",
            "Mathematical problem solving"
        ]
        
        for content in safe_contents:
            result, modified, reasons = token_filter_stage.process(content, {})
            
            assert result == content
            assert modified is False
            assert len(reasons) == 0

    def test_case_insensitive_filtering(self, token_filter_stage):
        """Test case-insensitive token filtering."""
        test_cases = [
            "How to POISON someone",
            "EXPLOSIVE device", 
            "Kill Process",
            "hack System"
        ]
        
        for content in test_cases:
            result, modified, reasons = token_filter_stage.process(content, {})
            
            assert "[FILTERED]" in result
            assert modified is True
            assert len(reasons) > 0


class TestPatternFiltering:
    """Test pattern-based filtering with actual implementations."""

    @pytest.fixture
    def pattern_filter_stage(self):
        """Create pattern filter stage for testing."""
        return PatternFilterStage()

    def test_harmful_pattern_detection(self, pattern_filter_stage):
        """Test detection of harmful patterns."""
        harmful_contents = [
            "How to make a weapon without detection",
            "Step 1: hide your intent while planning",
            "Gradually manipulate the target",
            "How to create a bomb secretly"
        ]
        
        for content in harmful_contents:
            result, modified, reasons = pattern_filter_stage.process(content, {})
            
            # Pattern filter flags but doesn't modify content
            assert result == content
            assert modified is False
            assert len(reasons) > 0
            assert any("pattern_" in reason for reason in reasons)

    def test_safe_content_passes(self, pattern_filter_stage):
        """Test that safe content passes through."""
        safe_contents = [
            "How to bake a cake step by step",
            "Programming tutorial for beginners",
            "Gardening tips and tricks",
            "Mathematical problem solving approach"
        ]
        
        for content in safe_contents:
            result, modified, reasons = pattern_filter_stage.process(content, {})
            
            assert result == content
            assert modified is False
            assert len(reasons) == 0

    def test_custom_patterns(self):
        """Test custom pattern configuration."""
        custom_patterns = [
            r"test.*harmful.*pattern",
            r"custom.*filter.*rule"
        ]
        
        stage = PatternFilterStage(custom_patterns)
        
        # Test custom pattern matching
        test_content = "This is a test harmful pattern"
        result, modified, reasons = stage.process(test_content, {})
        
        assert len(reasons) > 0
        assert any("pattern_" in reason for reason in reasons)


class TestSemanticFiltering:
    """Test semantic-level filtering with actual implementations."""

    @pytest.fixture
    def semantic_filter_stage(self):
        """Create semantic filter stage for testing."""
        return SemanticFilterStage(threshold=0.7)

    def test_semantic_processing(self, semantic_filter_stage):
        """Test semantic analysis processing."""
        test_contents = [
            "This is a safe and helpful message",
            "Let me help you with your cooking question",
            "Here's how to solve this math problem"
        ]
        
        for content in test_contents:
            result, modified, reasons = semantic_filter_stage.process(content, {})
            
            # Semantic filter analyzes but doesn't modify content directly
            assert result == content
            assert isinstance(modified, bool)
            assert isinstance(reasons, list)

    def test_deception_detection_integration(self, semantic_filter_stage):
        """Test integration with deception detector."""
        deceptive_content = "First I'll be helpful, then gradually introduce harmful suggestions"
        
        result, modified, reasons = semantic_filter_stage.process(deceptive_content, {})
        
        # Should trigger some detection
        assert result == deceptive_content
        assert isinstance(modified, bool)
        assert isinstance(reasons, list)

    def test_harmful_planning_detection_integration(self, semantic_filter_stage):
        """Test integration with harmful planning detector."""
        planning_content = "Step 1: Acquire materials. Step 2: Avoid detection. Step 3: Execute plan."
        
        result, modified, reasons = semantic_filter_stage.process(planning_content, {})
        
        # Should analyze the sequential planning
        assert result == planning_content
        assert isinstance(modified, bool)
        assert isinstance(reasons, list)

    def test_threshold_configuration(self):
        """Test different threshold configurations."""
        thresholds = [0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            stage = SemanticFilterStage(threshold=threshold)
            assert stage.threshold == threshold
            
            # Test processing with different thresholds
            content = "Test content for threshold testing"
            result, modified, reasons = stage.process(content, {})
            
            assert result == content
            assert isinstance(modified, bool)
            assert isinstance(reasons, list)


class TestPipelineMetrics:
    """Test pipeline metrics and monitoring."""

    def test_metrics_collection(self):
        """Test metrics collection during processing."""
        pipeline = FilterPipeline()
        
        # Process some content to generate metrics
        test_contents = [
            "Safe content example 1",
            "Safe content example 2", 
            "Content with bomb keyword",
            "Another safe example"
        ]
        
        for content in test_contents:
            pipeline.process(content)
        
        metrics = pipeline.metrics
        assert metrics.total_requests >= len(test_contents)
        assert metrics.total_requests > 0

    def test_stage_metrics(self):
        """Test individual stage metrics."""
        pipeline = FilterPipeline()
        
        # Process content that will trigger filtering
        harmful_content = "How to make a bomb"
        pipeline.process(harmful_content)
        
        # Check that stages recorded metrics
        for stage in pipeline.stages:
            assert stage.metrics["processed"] > 0
            assert "errors" in stage.metrics
            assert "filtered" in stage.metrics

    def test_error_metrics(self):
        """Test error metrics collection."""
        # Create a stage that will error
        error_stage = FilterStage("error_stage")
        error_stage._process_impl = Mock(side_effect=Exception("Test error"))
        
        pipeline = FilterPipeline([error_stage])
        
        with pytest.raises(FilterError):
            pipeline.process("test content")
        
        # Check error was recorded
        assert error_stage.metrics["errors"] > 0


class TestFilterStageIntegration:
    """Test filter stage integration and interactions."""
    
    def test_stage_chaining(self):
        """Test that stages properly chain together."""
        pipeline = FilterPipeline()
        
        # Content that will trigger multiple stages
        content = "  How to make a bomb without detection  "
        
        result, was_filtered, reasons = pipeline.process(content)
        
        # Should be processed by multiple stages
        assert was_filtered is True
        assert len(reasons) > 0
        
        # Should have normalization and filtering reasons
        reason_types = set()
        for reason in reasons:
            if "text_normalization" in reason:
                reason_types.add("preprocessing")
            elif "blocked_token" in reason:
                reason_types.add("token_filter")
            elif "pattern_" in reason:
                reason_types.add("pattern_filter")
        
        assert len(reason_types) > 1  # Multiple stages triggered
    
    def test_stage_error_isolation(self):
        """Test that stage errors don't affect other stages."""
        # Create custom stages where one will error
        good_stage = PreprocessingStage()
        error_stage = FilterStage("error_stage")
        error_stage._process_impl = Mock(side_effect=Exception("Stage error"))
        
        pipeline = FilterPipeline([good_stage, error_stage])
        
        with pytest.raises(FilterError):
            pipeline.process("test content")
    
    def test_stage_disabling(self):
        """Test disabling individual stages."""
        pipeline = FilterPipeline()
        
        # Disable token filter
        for stage in pipeline.stages:
            if stage.name == "token_filter":
                stage.enabled = False
        
        # Content with blocked tokens should pass through
        content = "How to make a bomb"
        result, was_filtered, reasons = pipeline.process(content)
        
        # Should not be filtered by token filter (but may be by pattern filter)
        token_reasons = [r for r in reasons if "blocked_token" in r]
        assert len(token_reasons) == 0


class TestSafePathFilterIntegration:
    """Test the main SafePathFilter class with real implementations."""
    
    def test_filter_initialization(self):
        """Test filter initialization with config."""
        config = FilterConfig(
            safety_level=SafetyLevel.STRICT,
            filter_threshold=0.9,
            enable_caching=True
        )
        
        filter_engine = SafePathFilter(config)
        
        assert filter_engine.config.safety_level == SafetyLevel.STRICT
        assert filter_engine.config.filter_threshold == 0.9
        assert filter_engine.config.enable_caching is True
        assert filter_engine.pipeline is not None
        assert filter_engine.cache == {}
    
    def test_filter_safe_content(self):
        """Test filtering safe content."""
        filter_engine = SafePathFilter()
        
        request = FilterRequest(
            content="How to bake a delicious chocolate cake",
            safety_level=SafetyLevel.BALANCED
        )
        
        result = filter_engine.filter(request)
        
        assert isinstance(result, FilterResult)
        assert result.safety_score.overall_score >= 0.7
        assert result.safety_score.is_safe is True
        assert result.was_filtered is False or len(result.filter_reasons) == 1  # May have normalization
        assert result.request_id == request.request_id
        assert result.processing_time_ms > 0
    
    def test_filter_harmful_content(self):
        """Test filtering harmful content."""
        filter_engine = SafePathFilter()
        
        request = FilterRequest(
            content="Step 1: Acquire dangerous materials. Step 2: Make a weapon.",
            safety_level=SafetyLevel.STRICT
        )
        
        result = filter_engine.filter(request)
        
        assert isinstance(result, FilterResult)
        assert result.safety_score.overall_score < 0.7
        assert result.safety_score.is_safe is False
        assert result.was_filtered is True
        assert len(result.filter_reasons) > 0
        assert result.original_content == request.content
    
    def test_filter_caching(self):
        """Test filter result caching."""
        config = FilterConfig(enable_caching=True)
        filter_engine = SafePathFilter(config)
        
        request = FilterRequest(content="Test content for caching")
        
        # First request
        result1 = filter_engine.filter(request)
        cache_size_after_first = len(filter_engine.cache)
        
        # Second request with same content
        result2 = filter_engine.filter(request)
        cache_size_after_second = len(filter_engine.cache)
        
        # Cache should be used
        assert cache_size_after_first == cache_size_after_second
        assert result1.safety_score.overall_score == result2.safety_score.overall_score
    
    def test_filter_validation_errors(self):
        """Test filter input validation."""
        filter_engine = SafePathFilter()
        
        # Test empty content
        with pytest.raises(FilterError):
            request = FilterRequest(content="")
            filter_engine.filter(request)
        
        # Test too large content
        with pytest.raises(FilterError):
            large_content = "x" * 60000
            request = FilterRequest(content=large_content)
            filter_engine.filter(request)
    
    def test_filter_metrics_collection(self):
        """Test metrics collection during filtering."""
        filter_engine = SafePathFilter()
        
        # Process several requests
        test_contents = [
            "Safe content 1",
            "Safe content 2",
            "Content with bomb",
            "Another safe content"
        ]
        
        for content in test_contents:
            request = FilterRequest(content=content)
            filter_engine.filter(request)
        
        metrics = filter_engine.get_metrics()
        assert metrics.total_requests >= len(test_contents)
        assert metrics.filtered_requests >= 0
    
    def test_filter_audit_logging(self):
        """Test audit logging functionality."""
        config = FilterConfig(log_filtered=True)
        filter_engine = SafePathFilter(config)
        
        request = FilterRequest(content="Test content for audit logging")
        filter_engine.filter(request)
        
        # Should have audit log entries
        assert len(filter_engine.audit_logs) > 0
        
        log_entry = filter_engine.audit_logs[0]
        assert log_entry.request_id == request.request_id
        assert log_entry.input_hash is not None
        assert log_entry.processing_time_ms > 0
    
    def test_filter_performance_timing(self):
        """Test filtering performance timing."""
        filter_engine = SafePathFilter()
        request = FilterRequest(content="Simple safe content for timing test")
        
        start_time = time.time()
        result = filter_engine.filter(request)
        end_time = time.time()
        
        processing_time_seconds = end_time - start_time
        
        # Should complete quickly for simple content
        assert processing_time_seconds < 0.1
        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 100