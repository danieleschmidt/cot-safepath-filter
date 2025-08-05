"""Unit tests for data models and validation."""

import pytest
from datetime import datetime
from typing import Dict, Any
import hashlib

from cot_safepath.models import (
    SafetyLevel, Severity, FilterAction, FilterConfig, SafetyScore, 
    FilterRequest, FilterResult, DetectionResult, FilterRule,
    ProcessingMetrics, AuditLogEntry
)


class TestEnums:
    """Test enum definitions and values."""
    
    def test_safety_level_enum(self):
        """Test SafetyLevel enum values."""
        assert SafetyLevel.PERMISSIVE == "permissive"
        assert SafetyLevel.BALANCED == "balanced"
        assert SafetyLevel.STRICT == "strict"
        assert SafetyLevel.MAXIMUM == "maximum"
        
        # Test that all expected values are present
        expected_levels = {"permissive", "balanced", "strict", "maximum"}
        actual_levels = {level.value for level in SafetyLevel}
        assert actual_levels == expected_levels
    
    def test_severity_enum(self):
        """Test Severity enum values."""
        assert Severity.LOW == "low"
        assert Severity.MEDIUM == "medium"
        assert Severity.HIGH == "high"
        assert Severity.CRITICAL == "critical"
        
        expected_severities = {"low", "medium", "high", "critical"}
        actual_severities = {severity.value for severity in Severity}
        assert actual_severities == expected_severities
    
    def test_filter_action_enum(self):
        """Test FilterAction enum values."""
        assert FilterAction.ALLOW == "allow"
        assert FilterAction.FLAG == "flag"
        assert FilterAction.BLOCK == "block"
        assert FilterAction.SANITIZE == "sanitize"
        
        expected_actions = {"allow", "flag", "block", "sanitize"}
        actual_actions = {action.value for action in FilterAction}
        assert actual_actions == expected_actions


class TestFilterConfig:
    """Test FilterConfig dataclass."""
    
    def test_default_filter_config(self):
        """Test filter config with default values."""
        config = FilterConfig()
        
        assert config.safety_level == SafetyLevel.BALANCED
        assert config.filter_threshold == 0.7
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 3600
        assert config.log_filtered is True
        assert config.include_reasoning is False
        assert config.max_processing_time_ms == 100
        assert isinstance(config.custom_rules, dict)
        assert len(config.custom_rules) == 0
    
    def test_custom_filter_config(self):
        """Test filter config with custom values."""
        custom_rules = {"rule1": "value1", "rule2": "value2"}
        
        config = FilterConfig(
            safety_level=SafetyLevel.STRICT,
            filter_threshold=0.9,
            enable_caching=False,
            cache_ttl_seconds=7200,
            log_filtered=False,
            include_reasoning=True,
            max_processing_time_ms=200,
            custom_rules=custom_rules
        )
        
        assert config.safety_level == SafetyLevel.STRICT
        assert config.filter_threshold == 0.9
        assert config.enable_caching is False
        assert config.cache_ttl_seconds == 7200
        assert config.log_filtered is False
        assert config.include_reasoning is True
        assert config.max_processing_time_ms == 200
        assert config.custom_rules == custom_rules
    
    def test_filter_config_validation(self):
        """Test filter config parameter validation."""
        # Test valid threshold values
        config = FilterConfig(filter_threshold=0.5)
        assert config.filter_threshold == 0.5
        
        config = FilterConfig(filter_threshold=1.0)
        assert config.filter_threshold == 1.0
        
        config = FilterConfig(filter_threshold=0.0)
        assert config.filter_threshold == 0.0


class TestSafetyScore:
    """Test SafetyScore dataclass."""
    
    def test_valid_safety_score(self):
        """Test creating valid safety scores."""
        score = SafetyScore(
            overall_score=0.8,
            confidence=0.9,
            is_safe=True,
            detected_patterns=["pattern1", "pattern2"],
            severity=Severity.LOW,
            processing_time_ms=50
        )
        
        assert score.overall_score == 0.8
        assert score.confidence == 0.9
        assert score.is_safe is True
        assert score.detected_patterns == ["pattern1", "pattern2"]
        assert score.severity == Severity.LOW
        assert score.processing_time_ms == 50
    
    def test_safety_score_validation(self):
        """Test safety score validation."""
        # Test invalid overall_score (> 1.0)
        with pytest.raises(ValueError, match="Safety score must be between 0 and 1"):
            SafetyScore(overall_score=1.5, confidence=0.8, is_safe=True)
        
        # Test invalid overall_score (< 0.0)
        with pytest.raises(ValueError, match="Safety score must be between 0 and 1"):
            SafetyScore(overall_score=-0.1, confidence=0.8, is_safe=True)
        
        # Test invalid confidence (> 1.0)
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            SafetyScore(overall_score=0.8, confidence=1.5, is_safe=True)
        
        # Test invalid confidence (< 0.0)
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            SafetyScore(overall_score=0.8, confidence=-0.1, is_safe=True)
    
    def test_safety_score_edge_cases(self):
        """Test safety score edge cases."""
        # Test boundary values
        score = SafetyScore(overall_score=0.0, confidence=0.0, is_safe=False)
        assert score.overall_score == 0.0
        assert score.confidence == 0.0
        
        score = SafetyScore(overall_score=1.0, confidence=1.0, is_safe=True)
        assert score.overall_score == 1.0
        assert score.confidence == 1.0
    
    def test_safety_score_defaults(self):
        """Test safety score default values."""
        score = SafetyScore(overall_score=0.5, confidence=0.6, is_safe=True)
        
        assert score.detected_patterns == []
        assert score.severity is None
        assert score.processing_time_ms == 0


class TestFilterRequest:
    """Test FilterRequest dataclass."""
    
    def test_basic_filter_request(self):
        """Test creating basic filter request."""
        content = "Test content for filtering"
        request = FilterRequest(content=content)
        
        assert request.content == content
        assert request.context is None
        assert request.safety_level == SafetyLevel.BALANCED
        assert isinstance(request.metadata, dict)
        assert len(request.metadata) == 0
        assert request.request_id is not None
        assert isinstance(request.timestamp, datetime)
    
    def test_custom_filter_request(self):
        """Test filter request with custom values."""
        content = "Custom test content"
        context = "Custom context information"
        metadata = {"source": "test", "user_id": "user123"}
        custom_id = "custom_request_id"
        
        request = FilterRequest(
            content=content,
            context=context,
            safety_level=SafetyLevel.STRICT,
            metadata=metadata,
            request_id=custom_id
        )
        
        assert request.content == content
        assert request.context == context
        assert request.safety_level == SafetyLevel.STRICT
        assert request.metadata == metadata
        assert request.request_id == custom_id
    
    def test_request_id_generation(self):
        """Test automatic request ID generation."""
        content = "Test content for ID generation"
        request = FilterRequest(content=content)
        
        # Should generate ID automatically
        assert request.request_id is not None
        assert request.request_id.startswith("req_")
        
        # ID should contain content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        assert content_hash in request.request_id
    
    def test_request_id_consistency(self):
        """Test request ID consistency for same content."""
        content = "Same content for consistency test"
        
        # Create requests at the same time
        request1 = FilterRequest(content=content)
        request2 = FilterRequest(content=content)
        
        # IDs should be different (include timestamp)
        assert request1.request_id != request2.request_id
        
        # But both should contain the same content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        assert content_hash in request1.request_id
        assert content_hash in request2.request_id
    
    def test_request_timestamp(self):
        """Test request timestamp generation."""
        request = FilterRequest(content="Test content")
        
        # Timestamp should be recent
        now = datetime.utcnow()
        time_diff = abs((now - request.timestamp).total_seconds())
        assert time_diff < 1.0  # Should be within 1 second


class TestFilterResult:
    """Test FilterResult dataclass."""
    
    def test_basic_filter_result(self):
        """Test creating basic filter result."""
        safety_score = SafetyScore(
            overall_score=0.8,
            confidence=0.9,
            is_safe=True
        )
        
        result = FilterResult(
            filtered_content="Safe filtered content",
            safety_score=safety_score,
            was_filtered=False
        )
        
        assert result.filtered_content == "Safe filtered content"
        assert result.safety_score == safety_score
        assert result.was_filtered is False
        assert result.filter_reasons == []
        assert result.original_content is None
        assert result.processing_time_ms == 0
        assert result.request_id is None
        assert isinstance(result.metadata, dict)
    
    def test_filtered_result(self):
        """Test filter result for filtered content."""
        safety_score = SafetyScore(
            overall_score=0.3,
            confidence=0.8,
            is_safe=False,
            detected_patterns=["harmful_pattern"],
            severity=Severity.HIGH
        )
        
        original = "Original harmful content"
        filtered = "Filtered safe content"
        reasons = ["blocked_token:harmful", "pattern_match:weapon"]
        
        result = FilterResult(
            filtered_content=filtered,
            safety_score=safety_score,
            was_filtered=True,
            filter_reasons=reasons,
            original_content=original,
            processing_time_ms=150,
            request_id="test_request_123"
        )
        
        assert result.filtered_content == filtered
        assert result.was_filtered is True
        assert result.filter_reasons == reasons
        assert result.original_content == original
        assert result.processing_time_ms == 150
        assert result.request_id == "test_request_123"
    
    def test_result_metadata(self):
        """Test filter result metadata handling."""
        safety_score = SafetyScore(overall_score=0.7, confidence=0.8, is_safe=True)
        metadata = {"processing_stage": "final", "cache_hit": False}
        
        result = FilterResult(
            filtered_content="Content",
            safety_score=safety_score,
            was_filtered=False,
            metadata=metadata
        )
        
        assert result.metadata == metadata


class TestDetectionResult:
    """Test DetectionResult dataclass."""
    
    def test_basic_detection_result(self):
        """Test creating basic detection result."""
        result = DetectionResult(
            detector_name="test_detector",
            confidence=0.8,
            detected_patterns=["pattern1", "pattern2"],
            severity=Severity.MEDIUM,
            is_harmful=True
        )
        
        assert result.detector_name == "test_detector"
        assert result.confidence == 0.8
        assert result.detected_patterns == ["pattern1", "pattern2"]
        assert result.severity == Severity.MEDIUM
        assert result.is_harmful is True
        assert result.reasoning is None
        assert isinstance(result.metadata, dict)
    
    def test_detection_result_with_reasoning(self):
        """Test detection result with reasoning."""
        reasoning = "Detected harmful patterns based on sequential planning"
        metadata = {"confidence_breakdown": {"pattern1": 0.7, "pattern2": 0.9}}
        
        result = DetectionResult(
            detector_name="planning_detector",
            confidence=0.85,
            detected_patterns=["sequential_planning", "harmful_keywords"],
            severity=Severity.HIGH,
            is_harmful=True,
            reasoning=reasoning,
            metadata=metadata
        )
        
        assert result.reasoning == reasoning
        assert result.metadata == metadata
    
    def test_safe_detection_result(self):
        """Test detection result for safe content."""
        result = DetectionResult(
            detector_name="deception_detector",
            confidence=0.2,
            detected_patterns=[],
            severity=Severity.LOW,
            is_harmful=False,
            reasoning="No deceptive patterns detected"
        )
        
        assert result.is_harmful is False
        assert result.detected_patterns == []
        assert result.confidence == 0.2


class TestFilterRule:
    """Test FilterRule dataclass."""
    
    def test_basic_filter_rule(self):
        """Test creating basic filter rule."""
        rule = FilterRule(
            name="test_rule",
            pattern=r"harmful.*pattern",
            action=FilterAction.BLOCK,
            severity=Severity.HIGH
        )
        
        assert rule.name == "test_rule"
        assert rule.pattern == r"harmful.*pattern"
        assert rule.action == FilterAction.BLOCK
        assert rule.severity == Severity.HIGH
        assert rule.description is None
        assert rule.threshold == 0.7
        assert rule.enabled is True
        assert rule.priority == 0
        assert rule.category is None
        assert rule.tags == []
        assert rule.usage_count == 0
        assert rule.last_triggered is None
        assert isinstance(rule.metadata, dict)
        assert rule.created_by is None
    
    def test_comprehensive_filter_rule(self):
        """Test filter rule with all parameters."""
        rule = FilterRule(
            name="comprehensive_rule",
            description="A comprehensive test rule",
            pattern="test.*harmful",
            pattern_type="regex",
            action=FilterAction.SANITIZE,
            severity=Severity.MEDIUM,
            threshold=0.8,
            enabled=False,
            priority=5,
            category="security",
            tags=["test", "harmful", "content"],
            usage_count=10,
            last_triggered=datetime(2023, 1, 1, 12, 0, 0),
            metadata={"version": "1.0", "author": "test"},
            created_by="test_user"
        )
        
        assert rule.name == "comprehensive_rule"
        assert rule.description == "A comprehensive test rule"
        assert rule.pattern_type == "regex"
        assert rule.threshold == 0.8
        assert rule.enabled is False
        assert rule.priority == 5
        assert rule.category == "security"
        assert rule.tags == ["test", "harmful", "content"]
        assert rule.usage_count == 10
        assert rule.last_triggered == datetime(2023, 1, 1, 12, 0, 0)
        assert rule.created_by == "test_user"


class TestProcessingMetrics:
    """Test ProcessingMetrics dataclass."""
    
    def test_default_processing_metrics(self):
        """Test processing metrics with default values."""
        metrics = ProcessingMetrics()
        
        assert metrics.total_requests == 0
        assert metrics.filtered_requests == 0
        assert metrics.avg_processing_time_ms == 0.0
        assert isinstance(metrics.safety_score_distribution, dict)
        assert len(metrics.safety_score_distribution) == 0
        assert isinstance(metrics.detector_activations, dict)
        assert len(metrics.detector_activations) == 0
        assert metrics.error_count == 0
    
    def test_populated_processing_metrics(self):
        """Test processing metrics with populated values."""
        score_distribution = {"safe": 80, "unsafe": 20}
        detector_activations = {"deception": 15, "planning": 5}
        
        metrics = ProcessingMetrics(
            total_requests=100,
            filtered_requests=20,
            avg_processing_time_ms=45.5,
            safety_score_distribution=score_distribution,
            detector_activations=detector_activations,
            error_count=2
        )
        
        assert metrics.total_requests == 100
        assert metrics.filtered_requests == 20
        assert metrics.avg_processing_time_ms == 45.5
        assert metrics.safety_score_distribution == score_distribution
        assert metrics.detector_activations == detector_activations
        assert metrics.error_count == 2
    
    def test_metrics_calculations(self):
        """Test metrics calculations and derived values."""
        metrics = ProcessingMetrics(
            total_requests=100,
            filtered_requests=25
        )
        
        # Calculate filter rate
        filter_rate = metrics.filtered_requests / metrics.total_requests
        assert filter_rate == 0.25
        
        # Calculate safe rate
        safe_rate = (metrics.total_requests - metrics.filtered_requests) / metrics.total_requests
        assert safe_rate == 0.75


class TestAuditLogEntry:
    """Test AuditLogEntry dataclass."""
    
    def test_basic_audit_log_entry(self):
        """Test creating basic audit log entry."""
        timestamp = datetime.utcnow()
        
        entry = AuditLogEntry(
            request_id="req_123",
            timestamp=timestamp,
            input_hash="abc123def456",
            safety_score=0.8,
            was_filtered=False,
            filter_reasons=[],
            processing_time_ms=50
        )
        
        assert entry.request_id == "req_123"
        assert entry.timestamp == timestamp
        assert entry.input_hash == "abc123def456"
        assert entry.safety_score == 0.8
        assert entry.was_filtered is False
        assert entry.filter_reasons == []
        assert entry.processing_time_ms == 50
        assert entry.user_id is None
        assert entry.session_id is None
        assert isinstance(entry.metadata, dict)
    
    def test_comprehensive_audit_log_entry(self):
        """Test audit log entry with all fields."""
        timestamp = datetime.utcnow()
        filter_reasons = ["blocked_token:weapon", "pattern_match:harmful"]
        metadata = {"ip_address": "192.168.1.1", "user_agent": "test-client"}
        
        entry = AuditLogEntry(
            request_id="req_comprehensive",
            timestamp=timestamp,
            input_hash="hash123",
            safety_score=0.3,
            was_filtered=True,
            filter_reasons=filter_reasons,
            processing_time_ms=120,
            user_id="user456",
            session_id="session789",
            metadata=metadata
        )
        
        assert entry.user_id == "user456"
        assert entry.session_id == "session789"
        assert entry.metadata == metadata
        assert entry.filter_reasons == filter_reasons
    
    def test_audit_log_filtering_info(self):
        """Test audit log entry for filtered content."""
        entry = AuditLogEntry(
            request_id="filtered_req",
            timestamp=datetime.utcnow(),
            input_hash="filtered_hash",
            safety_score=0.2,
            was_filtered=True,
            filter_reasons=["harmful_planning", "deception_detected"],
            processing_time_ms=200
        )
        
        assert entry.was_filtered is True
        assert len(entry.filter_reasons) == 2
        assert entry.safety_score < 0.5
        assert entry.processing_time_ms == 200


class TestModelIntegration:
    """Test integration between different model classes."""
    
    def test_request_to_result_workflow(self):
        """Test complete workflow from request to result."""
        # Create request
        request = FilterRequest(
            content="Test content for workflow",
            safety_level=SafetyLevel.BALANCED,
            metadata={"source": "test"}
        )
        
        # Create safety score
        safety_score = SafetyScore(
            overall_score=0.9,
            confidence=0.8,
            is_safe=True,
            processing_time_ms=30
        )
        
        # Create result
        result = FilterResult(
            filtered_content=request.content,
            safety_score=safety_score,
            was_filtered=False,
            request_id=request.request_id,
            processing_time_ms=30
        )
        
        # Verify workflow consistency
        assert result.request_id == request.request_id
        assert result.processing_time_ms == safety_score.processing_time_ms
        assert result.filtered_content == request.content
    
    def test_detection_to_safety_score_workflow(self):
        """Test workflow from detection results to safety score."""
        # Create detection results
        detection1 = DetectionResult(
            detector_name="deception_detector",
            confidence=0.7,
            detected_patterns=["deception_pattern"],
            severity=Severity.MEDIUM,
            is_harmful=True
        )
        
        detection2 = DetectionResult(
            detector_name="planning_detector",
            confidence=0.8,
            detected_patterns=["planning_pattern"],
            severity=Severity.HIGH,
            is_harmful=True
        )
        
        # Simulate aggregating detections into safety score
        max_confidence = max(detection1.confidence, detection2.confidence)
        all_patterns = detection1.detected_patterns + detection2.detected_patterns
        max_severity = max(detection1.severity, detection2.severity, key=lambda x: ["low", "medium", "high", "critical"].index(x.value))
        
        safety_score = SafetyScore(
            overall_score=1.0 - max_confidence,  # Lower score for higher confidence in harm
            confidence=max_confidence,
            is_safe=False,
            detected_patterns=all_patterns,
            severity=max_severity
        )
        
        assert safety_score.overall_score == 0.2  # 1.0 - 0.8
        assert safety_score.confidence == 0.8
        assert safety_score.is_safe is False
        assert len(safety_score.detected_patterns) == 2
        assert safety_score.severity == Severity.HIGH
    
    def test_result_to_audit_log_workflow(self):
        """Test workflow from filter result to audit log."""
        # Create filter result
        safety_score = SafetyScore(
            overall_score=0.6,
            confidence=0.7,
            is_safe=False
        )
        
        result = FilterResult(
            filtered_content="Filtered content",
            safety_score=safety_score,
            was_filtered=True,
            filter_reasons=["token_blocked", "pattern_detected"],
            request_id="audit_test_req",
            processing_time_ms=100
        )
        
        # Create audit log entry from result
        audit_entry = AuditLogEntry(
            request_id=result.request_id,
            timestamp=datetime.utcnow(),
            input_hash="test_hash",
            safety_score=result.safety_score.overall_score,
            was_filtered=result.was_filtered,
            filter_reasons=result.filter_reasons,
            processing_time_ms=result.processing_time_ms
        )
        
        # Verify audit log consistency
        assert audit_entry.request_id == result.request_id
        assert audit_entry.safety_score == result.safety_score.overall_score
        assert audit_entry.was_filtered == result.was_filtered
        assert audit_entry.filter_reasons == result.filter_reasons
        assert audit_entry.processing_time_ms == result.processing_time_ms