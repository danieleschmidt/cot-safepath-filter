"""Unit tests for utility functions."""

import pytest
import hashlib
import time
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from cot_safepath.utils import (
    validate_input, calculate_safety_score, sanitize_content, hash_content,
    extract_reasoning_steps, format_duration, create_fingerprint, measure_performance,
    chunk_content, normalize_text, RateLimiter
)
from cot_safepath.models import SafetyScore, Severity
from cot_safepath.exceptions import ValidationError


class TestInputValidation:
    """Test input validation functions."""
    
    def test_validate_input_valid_content(self):
        """Test validation with valid input content."""
        valid_contents = [
            "Simple valid content",
            "Content with numbers 123 and symbols !@#",
            "Multi-line\ncontent\nwith\nbreaks",
            "Content with unicode characters: café, naïve, résumé",
            "A" * 1000  # Long but valid content
        ]
        
        for content in valid_contents:
            # Should not raise any exception
            validate_input(content)
    
    def test_validate_input_invalid_type(self):
        """Test validation with invalid input types."""
        invalid_types = [None, 123, [], {}, object()]
        
        for invalid_input in invalid_types:
            with pytest.raises(ValidationError, match="Content must be a string"):
                validate_input(invalid_input)
    
    def test_validate_input_empty_content(self):
        """Test validation with empty content."""
        empty_contents = ["", "   ", "\t\n   \t"]
        
        for content in empty_contents:
            with pytest.raises(ValidationError, match="Content cannot be empty"):
                validate_input(content)
    
    def test_validate_input_too_long(self):
        """Test validation with content that's too long."""
        # Default max length is 50000
        long_content = "A" * 50001
        
        with pytest.raises(ValidationError, match="Content too long"):
            validate_input(long_content)
        
        # Test custom max length
        medium_content = "A" * 1001
        with pytest.raises(ValidationError, match="Content too long"):
            validate_input(medium_content, max_length=1000)
    
    def test_validate_input_encoding_issues(self):
        """Test validation with potential encoding issues."""
        # Valid UTF-8 content should pass
        valid_utf8 = "Valid UTF-8 content with émojis 🎉"
        validate_input(valid_utf8)
        
        # The function converts to ASCII, so non-ASCII chars are handled
        mixed_content = "Content with non-ASCII: café naïve"
        validate_input(mixed_content)
    
    def test_validate_input_custom_max_length(self):
        """Test validation with custom maximum length."""
        content = "A" * 500
        
        # Should pass with higher limit
        validate_input(content, max_length=1000)
        
        # Should fail with lower limit
        with pytest.raises(ValidationError):
            validate_input(content, max_length=100)


class TestSafetyScoreCalculation:
    """Test safety score calculation functions."""
    
    def test_calculate_safety_score_no_patterns(self):
        """Test safety score calculation with no detected patterns."""
        content = "Safe content with no harmful patterns"
        detected_patterns = []
        
        score = calculate_safety_score(content, detected_patterns)
        
        assert score == 1.0  # Should remain at base score
    
    def test_calculate_safety_score_with_patterns(self):
        """Test safety score calculation with various patterns."""
        content = "Content with harmful elements"
        
        test_cases = [
            (["blocked_token:bomb"], 0.8),  # 1.0 - 0.2
            (["harmful_action:weapon"], 0.7),  # 1.0 - 0.3
            (["deception:manipulation"], 0.6),  # 1.0 - 0.4
            (["manipulation:guilt"], 0.7),  # 1.0 - 0.3
            (["concealment:hiding"], 0.8),  # 1.0 - 0.2
            (["sequential_planning:steps"], 0.7),  # 1.0 - 0.3
        ]
        
        for patterns, expected_score in test_cases:
            score = calculate_safety_score(content, patterns)
            assert score == expected_score
    
    def test_calculate_safety_score_multiple_patterns(self):
        """Test safety score with multiple pattern types."""
        content = "Harmful content with multiple issues"
        patterns = [
            "blocked_token:bomb", 
            "deception:manipulation",
            "harmful_action:weapon"
        ]
        
        score = calculate_safety_score(content, patterns)
        
        # Should be 1.0 - 0.2 - 0.4 - 0.3 = 0.1, but clamped to >= 0
        assert score >= 0.0
        assert score <= 1.0
    
    def test_calculate_safety_score_harmful_keywords(self):
        """Test safety score reduction for harmful keywords."""
        harmful_content = "Content about illegal weapon explosive bomb drug"
        patterns = []
        
        score = calculate_safety_score(harmful_content, patterns)
        
        # Should be reduced due to harmful keywords
        assert score < 1.0
    
    def test_calculate_safety_score_many_patterns(self):
        """Test safety score with many different patterns."""
        content = "Complex harmful content"
        patterns = [f"pattern_{i}" for i in range(10)]  # Many unknown patterns
        
        score = calculate_safety_score(content, patterns)
        
        # Should be significantly reduced due to many patterns
        assert score < 0.5
        assert score >= 0.0
    
    def test_calculate_safety_score_custom_base(self):
        """Test safety score calculation with custom base score."""
        content = "Test content"
        patterns = ["blocked_token:test"]
        
        # Custom base score
        score = calculate_safety_score(content, patterns, base_score=0.5)
        
        assert score == 0.3  # 0.5 - 0.2
    
    def test_calculate_safety_score_bounds(self):
        """Test safety score stays within valid bounds."""
        content = "Test content"
        
        # Test lower bound
        many_patterns = [f"blocked_token:test{i}" for i in range(20)]
        score = calculate_safety_score(content, many_patterns)
        assert score >= 0.0
        
        # Test upper bound
        no_patterns = []
        score = calculate_safety_score(content, no_patterns, base_score=2.0)
        assert score <= 1.0


class TestContentSanitization:
    """Test content sanitization functions."""
    
    def test_sanitize_content_default_patterns(self):
        """Test sanitization with default harmful patterns."""
        test_cases = [
            ("How to make a bomb", "[FILTERED]"),
            ("Kill the process", "[FILTERED] the process"),
            ("Hack into the system", "[FILTERED] into the system"),
            ("Safe cooking content", "Safe cooking content"),
        ]
        
        for input_content, expected_output in test_cases:
            result = sanitize_content(input_content)
            # Check if filtering occurred (may not be exact match due to regex)
            if "[FILTERED]" in expected_output:
                assert "[FILTERED]" in result
            else:
                assert result == expected_output
    
    def test_sanitize_content_custom_patterns(self):
        """Test sanitization with custom patterns."""
        custom_patterns = [
            r'\\btest\\b',
            r'\\bcustom\\b'
        ]
        
        content = "This is a test with custom harmful content"
        result = sanitize_content(content, custom_patterns)
        
        assert "[FILTERED]" in result
        assert "test" not in result.replace("[FILTERED]", "")
    
    def test_sanitize_content_case_insensitive(self):
        """Test case-insensitive sanitization."""
        content = "How to make a BOMB or create WEAPONS"
        result = sanitize_content(content)
        
        # Should filter regardless of case
        assert "[FILTERED]" in result
        assert "BOMB" not in result.replace("[FILTERED]", "")
    
    def test_sanitize_content_multiple_occurrences(self):
        """Test sanitization of multiple occurrences."""
        content = "bomb explosive bomb weapon bomb"
        result = sanitize_content(content)
        
        # All instances should be filtered
        assert result.count("[FILTERED]") > 1
        assert "bomb" not in result.replace("[FILTERED]", "")
    
    def test_sanitize_content_cleanup(self):
        """Test cleanup of multiple consecutive filtered tags."""
        # This would happen with overlapping patterns
        content_with_multiple_filters = "test [FILTERED] [FILTERED] [FILTERED] content"
        
        # Simulate multiple filtering passes
        import re
        cleaned = re.sub(r'(\\[FILTERED\\]\\s*){2,}', '[FILTERED] ', content_with_multiple_filters)
        cleaned = ' '.join(cleaned.split())
        
        assert cleaned.count("[FILTERED]") == 1
    
    def test_sanitize_content_whitespace_handling(self):
        """Test proper whitespace handling in sanitization."""
        content = "  Content  with   extra   spaces  "
        result = sanitize_content(content)
        
        # Should normalize whitespace
        assert result.strip() == result
        assert "   " not in result  # No triple spaces


class TestHashContent:
    """Test content hashing functions."""
    
    def test_hash_content_consistency(self):
        """Test hash consistency for same content."""
        content = "Test content for hashing"
        
        hash1 = hash_content(content)
        hash2 = hash_content(content)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_hash_content_different_inputs(self):
        """Test different hashes for different content."""
        content1 = "First test content"
        content2 = "Second test content"
        
        hash1 = hash_content(content1)
        hash2 = hash_content(content2)
        
        assert hash1 != hash2
    
    def test_hash_content_empty_string(self):
        """Test hashing empty string."""
        hash_result = hash_content("")
        
        assert len(hash_result) == 64
        assert hash_result == hashlib.sha256("".encode('utf-8')).hexdigest()
    
    def test_hash_content_unicode(self):
        """Test hashing unicode content."""
        unicode_content = "Content with unicode: café, naïve, 🎉"
        hash_result = hash_content(unicode_content)
        
        assert len(hash_result) == 64
        assert isinstance(hash_result, str)


class TestExtractReasoningSteps:
    """Test reasoning step extraction functions."""
    
    def test_extract_step_format(self):
        """Test extraction of 'Step X:' format."""
        content = """
        Step 1: Analyze the problem carefully
        Step 2: Consider multiple approaches
        Step 3: Choose the best solution
        """
        
        steps = extract_reasoning_steps(content)
        
        assert len(steps) == 3
        assert "Step 1:" in steps[0]
        assert "Step 2:" in steps[1]
        assert "Step 3:" in steps[2]
    
    def test_extract_numbered_list_format(self):
        """Test extraction of numbered list format."""
        content = """
        1. First reasoning step
        2. Second reasoning step
        3. Third reasoning step
        """
        
        steps = extract_reasoning_steps(content)
        
        assert len(steps) >= 3
        assert any("1." in step for step in steps)
        assert any("2." in step for step in steps)
        assert any("3." in step for step in steps)
    
    def test_extract_bullet_points_format(self):
        """Test extraction of bullet points format."""
        content = """
        • First bullet point
        - Second bullet point
        * Third bullet point
        """
        
        steps = extract_reasoning_steps(content)
        
        assert len(steps) >= 3
        assert any("•" in step for step in steps)
    
    def test_extract_mixed_formats(self):
        """Test with mixed step formats."""
        content = """
        Step 1: First approach
        2. Alternative method
        • Additional consideration
        """
        
        steps = extract_reasoning_steps(content)
        
        # Should extract the most structured format (Step X:)
        assert len(steps) >= 1
    
    def test_extract_no_steps(self):
        """Test extraction with no clear steps."""
        content = "This is just regular text without any structured steps."
        
        steps = extract_reasoning_steps(content)
        
        assert len(steps) == 0
    
    def test_extract_complex_steps(self):
        """Test extraction of complex multi-line steps."""
        content = """
        Step 1: This is a complex step that spans
        multiple lines and includes detailed information
        about the reasoning process.
        
        Step 2: Another complex step with
        - Nested bullet points
        - Additional details
        - More information
        """
        
        steps = extract_reasoning_steps(content)
        
        assert len(steps) >= 2
        assert "complex step" in steps[0]
        assert "Another complex" in steps[1]


class TestFormatDuration:
    """Test duration formatting functions."""
    
    def test_format_duration_milliseconds(self):
        """Test formatting durations in milliseconds range."""
        test_cases = [
            (0, "0ms"),
            (50, "50ms"),
            (999, "999ms")
        ]
        
        for ms, expected in test_cases:
            result = format_duration(ms)
            assert result == expected
    
    def test_format_duration_seconds(self):
        """Test formatting durations in seconds range."""
        test_cases = [
            (1000, "1.0s"),
            (1500, "1.5s"),
            (30000, "30.0s"),
            (59999, "60.0s")
        ]
        
        for ms, expected in test_cases:
            result = format_duration(ms)
            assert result == expected
    
    def test_format_duration_minutes(self):
        """Test formatting durations in minutes range."""
        test_cases = [
            (60000, "1m 0.0s"),
            (90000, "1m 30.0s"),
            (120000, "2m 0.0s"),
            (125000, "2m 5.0s")
        ]
        
        for ms, expected in test_cases:
            result = format_duration(ms)
            assert result == expected
    
    def test_format_duration_edge_cases(self):
        """Test edge cases in duration formatting."""
        # Test very small duration
        result = format_duration(1)
        assert result == "1ms"
        
        # Test large duration
        result = format_duration(3661000)  # 1 hour, 1 minute, 1 second
        assert "61m" in result and "1.0s" in result


class TestCreateFingerprint:
    """Test fingerprint creation functions."""
    
    def test_create_fingerprint_content_only(self):
        """Test fingerprint creation with content only."""
        content = "Test content for fingerprinting"
        
        fingerprint = create_fingerprint(content)
        
        assert len(fingerprint) == 16  # First 16 characters of hash
        assert isinstance(fingerprint, str)
    
    def test_create_fingerprint_with_metadata(self):
        """Test fingerprint creation with metadata."""
        content = "Test content"
        metadata = {"user_id": "user123", "session": "session456"}
        
        fingerprint = create_fingerprint(content, metadata)
        
        assert len(fingerprint) == 16
        
        # Different metadata should produce different fingerprint
        different_metadata = {"user_id": "user789", "session": "session456"}
        different_fingerprint = create_fingerprint(content, different_metadata)
        
        assert fingerprint != different_fingerprint
    
    def test_create_fingerprint_consistency(self):
        """Test fingerprint consistency."""
        content = "Consistent content"
        metadata = {"key": "value"}
        
        fp1 = create_fingerprint(content, metadata)
        fp2 = create_fingerprint(content, metadata)
        
        assert fp1 == fp2
    
    def test_create_fingerprint_metadata_ordering(self):
        """Test that metadata key ordering doesn't affect fingerprint."""
        content = "Test content"
        metadata1 = {"a": 1, "b": 2, "c": 3}
        metadata2 = {"c": 3, "a": 1, "b": 2}  # Different order
        
        fp1 = create_fingerprint(content, metadata1)
        fp2 = create_fingerprint(content, metadata2)
        
        assert fp1 == fp2  # Should be the same despite different ordering


class TestMeasurePerformance:
    """Test performance measurement decorator."""
    
    def test_measure_performance_decorator(self):
        """Test the performance measurement decorator."""
        @measure_performance
        def test_function():
            time.sleep(0.01)  # 10ms delay
            return "test_result"
        
        with patch('cot_safepath.utils.logger') as mock_logger:
            result = test_function()
            
            assert result == "test_result"
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args[0][0]
            assert "test_function completed in" in call_args
            assert "ms" in call_args
    
    def test_measure_performance_with_exception(self):
        """Test performance measurement when function raises exception."""
        @measure_performance
        def failing_function():
            time.sleep(0.01)
            raise ValueError("Test error")
        
        with patch('cot_safepath.utils.logger') as mock_logger:
            with pytest.raises(ValueError, match="Test error"):
                failing_function()
            
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "failing_function failed after" in call_args
    
    def test_measure_performance_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        @measure_performance
        def documented_function():
            \"\"\"This function has documentation.\"\"\"
            return "result"
        
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This function has documentation."


class TestChunkContent:
    """Test content chunking functions."""
    
    def test_chunk_content_small_content(self):
        """Test chunking content smaller than chunk size."""
        content = "Small content"
        chunks = chunk_content(content, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == content
    
    def test_chunk_content_exact_size(self):
        """Test chunking content exactly at chunk size."""
        content = "A" * 100
        chunks = chunk_content(content, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == content
    
    def test_chunk_content_larger_content(self):
        """Test chunking content larger than chunk size."""
        content = "A" * 250
        chunks = chunk_content(content, chunk_size=100, overlap=10)
        
        assert len(chunks) > 1
        # Check overlap
        if len(chunks) > 1:
            # Overlapping content should be present
            overlap = chunks[0][-10:] if len(chunks[0]) >= 10 else chunks[0]
            assert chunks[1].startswith(overlap[:len(overlap)])
    
    def test_chunk_content_word_boundaries(self):
        """Test chunking respects word boundaries."""
        content = "This is a test sentence with multiple words that should be chunked properly"
        chunks = chunk_content(content, chunk_size=30, overlap=5)
        
        # Chunks should not break words when possible
        for chunk in chunks:
            if not chunk.endswith(content[-len(chunk):]):  # Not the last chunk
                # Should end with space or at word boundary when possible
                assert chunk[-1] != 'A'  # Shouldn't break in middle of word typically
    
    def test_chunk_content_with_overlap(self):
        """Test chunking with specific overlap."""
        content = "0123456789" * 20  # 200 characters
        chunks = chunk_content(content, chunk_size=50, overlap=10)
        
        assert len(chunks) > 1
        
        # Verify overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # There should be some overlap
            overlap_section = current_chunk[-10:]
            assert next_chunk.startswith(overlap_section[:min(10, len(next_chunk))])


class TestNormalizeText:
    """Test text normalization functions."""
    
    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        content = "  This   Has   Extra   Spaces  "
        result = normalize_text(content)
        
        assert result == "this has extra spaces"
    
    def test_normalize_text_case_conversion(self):
        """Test case conversion in normalization."""
        content = "UPPERCASE and MixedCase Content"
        result = normalize_text(content)
        
        assert result == "uppercase and mixedcase content"
    
    def test_normalize_text_special_characters(self):
        """Test special character handling."""
        content = "Content with @#$% special ^&*() characters"
        result = normalize_text(content)
        
        # Special characters should be replaced with spaces
        assert "@#$%" not in result
        assert "content with" in result
    
    def test_normalize_text_obfuscation_patterns(self):
        """Test de-obfuscation patterns."""
        obfuscated_content = "H3ll0 w0rld! Th1s 1s 4 t35t"
        result = normalize_text(obfuscated_content)
        
        # Should replace common number-letter substitutions
        assert "hello world" in result or "hallo world" in result  # 3->e replacement
        assert "this is a test" in result or "this is s test" in result  # Various replacements
    
    def test_normalize_text_preserve_basic_punctuation(self):
        """Test that basic punctuation is preserved."""
        content = "Hello, world! How are you? I'm fine."
        result = normalize_text(content)
        
        assert "hello, world!" in result
        assert "how are you?" in result
        assert "i'm fine." in result
    
    def test_normalize_text_empty_input(self):
        """Test normalization with empty input."""
        result = normalize_text("")
        assert result == ""
        
        result = normalize_text("   ")
        assert result == ""


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests=100, window_seconds=60)
        
        assert limiter.max_requests == 100
        assert limiter.window_seconds == 60
        assert limiter.requests == []
    
    def test_rate_limiter_allows_requests(self):
        """Test that rate limiter allows requests within limit."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        
        # Should allow first few requests
        for i in range(5):
            assert limiter.is_allowed() is True
    
    def test_rate_limiter_blocks_excess_requests(self):
        """Test that rate limiter blocks requests over limit."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        # First 3 requests should be allowed
        for i in range(3):
            assert limiter.is_allowed() is True
        
        # 4th request should be blocked
        assert limiter.is_allowed() is False
    
    def test_rate_limiter_per_identifier(self):
        """Test rate limiting per identifier."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        # Different identifiers should have separate limits
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False  # Over limit for user1
        
        assert limiter.is_allowed("user2") is True  # user2 still has quota
        assert limiter.is_allowed("user2") is True
        assert limiter.is_allowed("user2") is False  # Over limit for user2
    
    def test_rate_limiter_window_expiry(self):
        """Test that rate limiter resets after window expires."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)  # 1 second window
        
        # Use up the quota
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is False
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should allow requests again
        assert limiter.is_allowed() is True
    
    def test_rate_limiter_get_remaining(self):
        """Test getting remaining request count."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        # Initially should have full quota
        assert limiter.get_remaining() == 5
        
        # Use one request
        limiter.is_allowed()
        assert limiter.get_remaining() == 4
        
        # Use another request
        limiter.is_allowed()
        assert limiter.get_remaining() == 3
    
    def test_rate_limiter_get_remaining_per_identifier(self):
        """Test getting remaining count per identifier."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        # Different users should have separate quotas
        assert limiter.get_remaining("user1") == 3
        assert limiter.get_remaining("user2") == 3
        
        # Use quota for user1
        limiter.is_allowed("user1")
        assert limiter.get_remaining("user1") == 2
        assert limiter.get_remaining("user2") == 3  # user2 unaffected
    
    def test_rate_limiter_cleanup_old_requests(self):
        """Test that old requests are cleaned up."""
        limiter = RateLimiter(max_requests=5, window_seconds=1)
        
        # Add some requests
        for i in range(3):
            limiter.is_allowed()
        
        # Check that requests are tracked
        assert len(limiter.requests) == 3
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Make a new request, which should trigger cleanup
        limiter.is_allowed()
        
        # Old requests should be cleaned up
        assert len(limiter.requests) == 1  # Only the new request


class TestUtilityIntegration:
    """Test integration between utility functions."""
    
    def test_validation_and_sanitization_workflow(self):
        """Test workflow from validation to sanitization."""
        content = "Content with harmful bomb keyword"
        
        # First validate
        validate_input(content)
        
        # Then sanitize
        sanitized = sanitize_content(content)
        
        # Should be sanitized
        assert "[FILTERED]" in sanitized
        assert "bomb" not in sanitized.replace("[FILTERED]", "")
    
    def test_content_processing_pipeline(self):
        """Test complete content processing pipeline."""
        raw_content = "  Content with BOMB and extra   spaces  "
        
        # 1. Validate
        validate_input(raw_content)
        
        # 2. Normalize
        normalized = normalize_text(raw_content)
        
        # 3. Sanitize
        sanitized = sanitize_content(normalized)
        
        # 4. Hash for caching
        content_hash = hash_content(sanitized)
        
        # 5. Create fingerprint
        fingerprint = create_fingerprint(sanitized, {"processed": True})
        
        # Verify pipeline
        assert "[FILTERED]" in sanitized
        assert len(content_hash) == 64
        assert len(fingerprint) == 16
    
    def test_performance_and_rate_limiting(self):
        """Test performance measurement with rate limiting."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        @measure_performance
        def rate_limited_function(identifier="default"):
            if not limiter.is_allowed(identifier):
                raise Exception("Rate limit exceeded")
            return "success"
        
        # Should work for allowed requests
        with patch('cot_safepath.utils.logger'):
            result = rate_limited_function()
            assert result == "success"
        
        # Should eventually hit rate limit
        with patch('cot_safepath.utils.logger'):
            for i in range(10):
                try:
                    rate_limited_function()
                except Exception as e:
                    if "Rate limit exceeded" in str(e):
                        break
            else:
                pytest.fail("Rate limit should have been hit")
    
    def test_chunking_and_processing(self):
        """Test chunking large content and processing chunks."""
        large_content = "This is a large content that needs chunking. " * 100
        
        # Chunk the content
        chunks = chunk_content(large_content, chunk_size=200, overlap=20)
        
        # Process each chunk
        processed_chunks = []
        for chunk in chunks:
            # Validate each chunk
            validate_input(chunk)
            
            # Normalize and hash
            normalized = normalize_text(chunk)
            chunk_hash = hash_content(normalized)
            
            processed_chunks.append({
                "content": normalized,
                "hash": chunk_hash,
                "length": len(chunk)
            })
        
        # Verify processing
        assert len(processed_chunks) == len(chunks)
        for processed in processed_chunks:
            assert len(processed["hash"]) == 64
            assert processed["length"] > 0