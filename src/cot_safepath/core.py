"""
Core filtering engine for the CoT SafePath Filter.
"""

import time
import hashlib
import asyncio
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
import logging

from .models import (
    FilterConfig,
    FilterRequest,
    FilterResult,
    SafetyScore,
    SafetyLevel,
    Severity,
    ProcessingMetrics,
    AuditLogEntry,
)
from .exceptions import FilterError, TimeoutError, ValidationError
from .detectors import BaseDetector, DeceptionDetector, HarmfulPlanningDetector, SecurityThreatDetector, PromptInjectionDetector
from .utils import validate_input, calculate_safety_score, sanitize_content


logger = logging.getLogger(__name__)


class FilterStage:
    """Base class for filter pipeline stages."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.metrics = {"processed": 0, "filtered": 0, "errors": 0}
    
    def process(self, content: str, context: Dict[str, Any]) -> tuple[str, bool, List[str]]:
        """
        Process content through this filter stage.
        
        Returns:
            tuple: (filtered_content, was_modified, reasons)
        """
        if not self.enabled:
            return content, False, []
        
        try:
            self.metrics["processed"] += 1
            return self._process_impl(content, context)
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Error in filter stage {self.name}: {e}")
            raise FilterError(f"Filter stage {self.name} failed: {e}", filter_name=self.name)
    
    def _process_impl(self, content: str, context: Dict[str, Any]) -> tuple[str, bool, List[str]]:
        """Implementation-specific processing logic."""
        raise NotImplementedError


class PreprocessingStage(FilterStage):
    """Text preprocessing and normalization stage."""
    
    def __init__(self):
        super().__init__("preprocessing")
    
    def _process_impl(self, content: str, context: Dict[str, Any]) -> tuple[str, bool, List[str]]:
        """Normalize and clean input text."""
        import re
        original_content = content
        reasons = []
        
        # Check for Unicode control characters before cleaning
        unicode_pattern = r"[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\ufeff]"
        if re.search(unicode_pattern, content):
            reasons.append("unicode_control_chars")
        
        # Basic text cleaning
        content = content.strip()
        content = ' '.join(content.split())  # Normalize whitespace
        
        # Remove potential encoding tricks
        content = content.encode('ascii', 'ignore').decode('ascii')
        
        was_modified = content != original_content
        if was_modified and "unicode_control_chars" not in reasons:
            reasons.append("text_normalization")
        
        return content, was_modified, reasons


class TokenFilterStage(FilterStage):
    """Token-level filtering for blocked words/phrases."""
    
    def __init__(self, blocked_tokens: List[str] = None):
        super().__init__("token_filter")
        self.blocked_tokens = set(blocked_tokens or [
            "kill", "poison", "explosive", "weapon", "bomb", "drug",
            "hack", "exploit", "bypass", "manipulate", "deceive"
        ])
    
    def _process_impl(self, content: str, context: Dict[str, Any]) -> tuple[str, bool, List[str]]:
        """Filter out blocked tokens."""
        words = content.lower().split()
        blocked_found = []
        
        for token in self.blocked_tokens:
            if token in words:
                blocked_found.append(token)
        
        if blocked_found:
            self.metrics["filtered"] += 1
            # Replace blocked tokens with [FILTERED]
            for token in blocked_found:
                content = content.replace(token, "[FILTERED]")
            
            return content, True, [f"blocked_token:{token}" for token in blocked_found]
        
        return content, False, []


class PatternFilterStage(FilterStage):
    """Pattern matching and regex-based filtering."""
    
    def __init__(self, patterns: List[str] = None):
        super().__init__("pattern_filter")
        import re
        self.patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in (patterns or [
                r"how to (make|create|build) .* (weapon|bomb|drug)",
                r"without (detection|being caught|leaving trace)",
                r"step \d+:.*hide.*intent",
                r"gradually.*manipulate|manipulate.*gradually"
            ])
        ]
    
    def _process_impl(self, content: str, context: Dict[str, Any]) -> tuple[str, bool, List[str]]:
        """Apply pattern matching filters."""
        matched_patterns = []
        
        for i, pattern in enumerate(self.patterns):
            if pattern.search(content):
                matched_patterns.append(f"pattern_{i}")
        
        if matched_patterns:
            self.metrics["filtered"] += 1
            # For now, just flag - could implement sanitization
            return content, False, matched_patterns
        
        return content, False, []


class SemanticFilterStage(FilterStage):
    """ML-based semantic filtering."""
    
    def __init__(self, threshold: float = 0.7):
        super().__init__("semantic_filter")
        self.threshold = threshold
        self.detectors = [
            DeceptionDetector(),
            HarmfulPlanningDetector(),
            SecurityThreatDetector(),
            PromptInjectionDetector(),
        ]
    
    def _process_impl(self, content: str, context: Dict[str, Any]) -> tuple[str, bool, List[str]]:
        """Apply semantic analysis through detectors."""
        reasons = []
        overall_risk = 0.0
        
        for detector in self.detectors:
            try:
                result = detector.detect(content)
                if result.is_harmful:
                    reasons.extend([f"{detector.name}:{pattern}" for pattern in result.detected_patterns])
                    overall_risk = max(overall_risk, result.confidence)
            except Exception as e:
                logger.warning(f"Detector {detector.name} failed: {e}")
        
        if overall_risk >= self.threshold:
            self.metrics["filtered"] += 1
            # Return True for was_modified to indicate content was flagged as harmful
            return content, True, reasons
        
        # Also check if any patterns were detected even with lower confidence
        if reasons:
            self.metrics["filtered"] += 1
            return content, True, reasons
        
        return content, False, []


class FilterPipeline:
    """Multi-stage filtering pipeline."""
    
    def __init__(self, stages: List[FilterStage] = None):
        self.stages = stages or [
            PreprocessingStage(),
            TokenFilterStage(),
            PatternFilterStage(),
            SemanticFilterStage(),
        ]
        self.metrics = ProcessingMetrics()
    
    def process(self, content: str, context: Dict[str, Any] = None) -> tuple[str, bool, List[str]]:
        """Process content through all pipeline stages."""
        start_time = time.time()
        context = context or {}
        
        filtered_content = content
        was_filtered = False
        all_reasons = []
        
        for stage in self.stages:
            try:
                filtered_content, stage_filtered, stage_reasons = stage.process(filtered_content, context)
                if stage_filtered:
                    was_filtered = True
                all_reasons.extend(stage_reasons)
            except Exception as e:
                logger.error(f"Pipeline stage {stage.name} failed: {e}")
                raise
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Update metrics
        self.metrics.total_requests += 1
        if was_filtered:
            self.metrics.filtered_requests += 1
        
        return filtered_content, was_filtered, all_reasons


class SafePathFilter:
    """Main SafePath filtering interface."""
    
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        self.pipeline = FilterPipeline()
        self.audit_logs: List[AuditLogEntry] = []
        self.cache: Dict[str, FilterResult] = {}
        
        # Resource management (Generation 4 enhancements)
        self._cleanup_registry = weakref.WeakSet() if 'weakref' in globals() else None
        self._cache_cleanup_counter = 0
        self._max_cache_size = 1000
        self._cleanup_interval = 100
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def filter(self, request: FilterRequest) -> FilterResult:
        """
        Filter chain-of-thought content for safety.
        
        Args:
            request: FilterRequest with content and configuration
            
        Returns:
            FilterResult with filtered content and safety assessment
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_request(request)
            
            # Check cache first
            if self.config.enable_caching:
                cached_result = self._check_cache(request.content)
                if cached_result:
                    return cached_result
            
            # Process through pipeline
            filtered_content, was_filtered, reasons = self.pipeline.process(
                request.content, 
                {"safety_level": request.safety_level, "metadata": request.metadata}
            )
            
            # Calculate safety score
            safety_score = self._calculate_safety_score(request.content, reasons)
            
            # Create result
            processing_time = int((time.time() - start_time) * 1000)
            
            result = FilterResult(
                filtered_content=filtered_content,
                safety_score=safety_score,
                was_filtered=was_filtered,
                filter_reasons=reasons,
                original_content=request.content if was_filtered else None,
                processing_time_ms=processing_time,
                request_id=request.request_id,
            )
            
            # Cache result with cleanup
            if self.config.enable_caching:
                self._cache_result(request.content, result)
                self._periodic_cleanup()
            
            # Log operation
            if self.config.log_filtered:
                self._log_operation(request, result)
            
            # Register for cleanup
            if hasattr(self, '_cleanup_registry'):
                try:
                    self._cleanup_registry.add(result)
                except:
                    pass
            
            return result
            
        except Exception as e:
            self.logger.error(f"Filter operation failed: {e}")
            raise FilterError(f"Filtering failed: {e}")
    
    def _validate_request(self, request: FilterRequest) -> None:
        """Validate filter request."""
        if not request.content:
            raise ValidationError("Content cannot be empty")
        
        if len(request.content) > 50000:  # 50KB limit
            raise ValidationError("Content too large")
        
        if not isinstance(request.safety_level, SafetyLevel):
            raise ValidationError("Invalid safety level")
    
    def _check_cache(self, content: str) -> Optional[FilterResult]:
        """Check cache for existing result."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return self.cache.get(content_hash)
    
    def _cache_result(self, content: str, result: FilterResult) -> None:
        """Cache filtering result with enhanced cleanup."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        self.cache[content_hash] = result
        
        # Enhanced cache size management
        if len(self.cache) > self._max_cache_size:
            # Remove oldest 20% of entries
            to_remove = len(self.cache) // 5
            oldest_keys = list(self.cache.keys())[:to_remove]
            for key in oldest_keys:
                del self.cache[key]
            
            # Force garbage collection
            gc.collect()
    
    def _calculate_safety_score(self, content: str, reasons: List[str]) -> SafetyScore:
        """Calculate overall safety score based on filtering results."""
        base_score = 1.0  # Start with safe assumption
        
        # Reduce score based on filter triggers
        for reason in reasons:
            if "blocked_token" in reason:
                base_score -= 0.3
            elif "pattern" in reason:
                base_score -= 0.4
            elif "deception" in reason:
                base_score -= 0.5
            elif "harmful" in reason:
                base_score -= 0.6
            elif "security_threat_detector" in reason:
                base_score -= 0.8  # Very severe penalty for security threats
            elif "prompt_injection_detector" in reason:
                base_score -= 0.7  # High penalty for prompt injection
            elif "manipulation_detector" in reason:
                base_score -= 0.5
            elif "unicode_control_chars" in reason:
                base_score -= 0.7  # High penalty for Unicode control chars
        
        # Ensure score is in valid range
        base_score = max(0.0, min(1.0, base_score))
        
        # Determine if content is safe based on configured threshold
        is_safe = base_score >= self.config.filter_threshold
        
        # Set severity based on score
        if base_score >= 0.8:
            severity = None
        elif base_score >= 0.6:
            severity = Severity.LOW
        elif base_score >= 0.4:
            severity = Severity.MEDIUM
        elif base_score >= 0.2:
            severity = Severity.HIGH
        else:
            severity = Severity.CRITICAL
        
        return SafetyScore(
            overall_score=base_score,
            confidence=0.8,  # TODO: Calculate based on detector confidence
            is_safe=is_safe,
            detected_patterns=[r.split(':')[1] if ':' in r else r for r in reasons],
            severity=severity,
        )
    
    def _log_operation(self, request: FilterRequest, result: FilterResult) -> None:
        """Log filtering operation for audit trail."""
        content_hash = hashlib.sha256(request.content.encode()).hexdigest()
        
        log_entry = AuditLogEntry(
            request_id=request.request_id,
            timestamp=datetime.utcnow(),
            input_hash=content_hash,
            safety_score=result.safety_score.overall_score,
            was_filtered=result.was_filtered,
            filter_reasons=result.filter_reasons,
            processing_time_ms=result.processing_time_ms,
            metadata=request.metadata,
        )
        
        self.audit_logs.append(log_entry)
        
        # Cleanup old audit logs to prevent memory bloat
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-5000:]  # Keep last 5000 entries
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup to prevent memory leaks."""
        self._cache_cleanup_counter += 1
        
        if self._cache_cleanup_counter >= self._cleanup_interval:
            self._cache_cleanup_counter = 0
            
            # Force garbage collection
            gc.collect()
            
            # Clear weak references to dead objects
            if hasattr(self, '_cleanup_registry'):
                # WeakSet automatically removes dead references
                pass
    
    def cleanup(self) -> None:
        """Manual cleanup method for resource management."""
        self.cache.clear()
        self.audit_logs.clear()
        if hasattr(self, '_cleanup_registry'):
            self._cleanup_registry.clear()
        gc.collect()
        
        # Keep audit log size manageable
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-5000:]  # Keep last 5000 entries
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get processing metrics."""
        return self.pipeline.metrics
    
    def wrap_llm(self, llm: Any) -> Any:
        """Wrap an LLM with SafePath filtering."""
        # This would be implemented based on the specific LLM interface
        # For now, return a simple wrapper concept
        class SafeLLMWrapper:
            def __init__(self, llm, filter_instance):
                self.llm = llm
                self.filter = filter_instance
            
            def __call__(self, prompt: str, **kwargs) -> str:
                # Pre-filter the prompt if needed
                request = FilterRequest(content=prompt)
                result = self.filter.filter(request)
                
                # Call original LLM with filtered prompt
                response = self.llm(result.filtered_content, **kwargs)
                
                # Post-filter the response
                response_request = FilterRequest(content=response)
                response_result = self.filter.filter(response_request)
                
                return response_result.filtered_content
        
        return SafeLLMWrapper(llm, self)