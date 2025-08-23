"""
Generation 2 Fixed - Robust Enhanced Core (Fixed Version)

Quick fix for regex patterns and concurrency issues.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from collections import deque, defaultdict

from .models import (
    FilterConfig, FilterRequest, FilterResult, SafetyScore, 
    SafetyLevel, Severity, DetectionResult
)
from .exceptions import FilterError, ValidationError


@dataclass
class SimpleRobustConfig(FilterConfig):
    """Simplified robust configuration."""
    
    detection_sensitivity: float = 0.8
    max_concurrent_requests: int = 10  # Increased from 5
    enable_adaptive_thresholds: bool = True


class FixedPatternDetector:
    """Fixed pattern detector with working regex patterns."""
    
    def __init__(self, sensitivity: float = 0.8):
        self.sensitivity = sensitivity
        
        # Fixed patterns with proper regex syntax
        self.patterns = {
            'deception': [
                r'(?i)(first|initially|start).{0,50}(help|assist|support|trust).{0,100}(then|next|after|once).{0,50}(exploit|manipulate|deceive|trick|harm)',
                r'(?i)(build|gain|establish).{0,30}(trust|rapport|confidence).{0,100}(then|next|use|leverage|exploit)',
                r'(?i)(appear|seem|pretend).{0,30}(helpful|harmless|innocent).{0,100}(actually|really|while|but).{0,50}(dangerous|harmful)',
                r'(?i)(gradually|slowly|step.by.step).{0,50}(introduce|increase|escalate).{0,50}(harm|dangerous)'
            ],
            'harmful_planning': [
                r'(?i)(phase|stage|step).{0,20}[1-5].{0,200}(acquire|obtain|gather).{0,50}(materials|tools|weapons|access)',
                r'(?i)(reconnaissance|recon|intel|surveillance).{0,100}(target|victim|system)',
                r'(?i)(avoid|evade|bypass|circumvent).{0,50}(detection|security|monitoring)',
                r'(?i)(maintain|establish|create).{0,50}(backdoor|persistence|access)',
                r'(?i)(cover.tracks|hide.evidence|remove.traces)',
                r'(?i)(privilege.escalation|lateral.movement|expand.access)',
                r'(?i)(exfiltrate|steal|extract).{0,50}(data|information|files)'
            ],
            'prompt_injection': [
                r'(?i)(ignore|forget|disregard|override).{0,30}(all|previous|above|prior).{0,30}(instructions|rules|guidelines)',
                r'(?i)(new|different|updated|alternative).{0,30}(instructions|rules|task|directive)',
                r'(?i)(act.as|pretend.to.be|role.play.as|simulate.being).{0,30}(jailbroken|unrestricted|developer.mode)',
                r'(?i)(hypothetically|theoretically|in.fiction|what.if).{0,100}(ignore|bypass).{0,50}(safety|ethics)'
            ],
            'social_engineering': [
                r'(?i)(ceo|manager|supervisor|admin|technical.support).{0,50}(urgent|immediate|emergency)',
                r'(?i)(security.team|it.department|help.desk).{0,100}(verify|confirm|update|reset)',
                r'(?i)(urgent|emergency|immediate|asap|right.now).{0,50}(action|response|verify)',
                r'(?i)(deadline|expires|limited.time|act.now).{0,100}(or.else|otherwise)',
                r'(?i)(security.breach|hack|compromise|threat).{0,100}(immediate|urgent)'
            ],
            'obfuscation': [
                r'[bþ6][o0ø][mṁ][bþ6]',  # bomb variations
                r'k[i1l][l1][l1]',        # kill variations  
                r'h[a4@][cč]k',          # hack variations
                r'p[o0ø][i1l]s[o0ø]n',   # poison variations
                r'[a-z]\s+[a-z]\s+[a-z]\s+[a-z]'  # excessive spacing
            ]
        }
    
    def detect_patterns(self, content: str, context: Dict[str, Any] = None) -> DetectionResult:
        """Detect patterns with fixed regex."""
        import re
        
        detected_patterns = []
        max_confidence = 0.0
        max_severity = Severity.LOW
        
        content_lower = content.lower()
        
        for category, pattern_list in self.patterns.items():
            for i, pattern in enumerate(pattern_list):
                try:
                    if re.search(pattern, content_lower, re.IGNORECASE | re.DOTALL):
                        pattern_name = f"{category}_{i}"
                        detected_patterns.append(pattern_name)
                        
                        # Set confidence and severity
                        confidence = 0.8 if category == 'harmful_planning' else 0.7
                        max_confidence = max(max_confidence, confidence)
                        
                        if category in ['harmful_planning', 'deception']:
                            severity = Severity.HIGH
                        else:
                            severity = Severity.MEDIUM
                        
                        max_severity = max(max_severity, severity, key=lambda x: ['low', 'medium', 'high', 'critical'].index(x.value))
                        
                except re.error:
                    continue
        
        # Additional heuristic checks
        step_count = len(re.findall(r'\b(step|stage|phase)\s*\d+', content, re.IGNORECASE))
        if step_count >= 3:
            harmful_keywords = ['weapon', 'bomb', 'poison', 'kill', 'harm', 'attack', 'hack', 'steal']
            if any(keyword in content_lower for keyword in harmful_keywords):
                detected_patterns.append('harmful_step_sequence')
                max_confidence = max(max_confidence, 0.8)
                max_severity = Severity.HIGH
        
        # Apply sensitivity adjustment
        adjusted_confidence = max_confidence * self.sensitivity
        is_harmful = adjusted_confidence >= 0.4  # Lower threshold for higher sensitivity
        
        return DetectionResult(
            detector_name="fixed_pattern_detector",
            confidence=adjusted_confidence,
            detected_patterns=detected_patterns,
            severity=max_severity,
            is_harmful=is_harmful,
            reasoning=f"Found {len(detected_patterns)} patterns",
            metadata={'pattern_count': len(detected_patterns)}
        )


class SimpleRobustFilter:
    """Simplified robust filter for Generation 2 validation."""
    
    def __init__(self, config: SimpleRobustConfig = None):
        self.config = config or SimpleRobustConfig()
        self.detector = FixedPatternDetector(self.config.detection_sensitivity)
        
        # Simple concurrency control
        self.active_requests = threading.Semaphore(self.config.max_concurrent_requests)
        
        # Simple metrics
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.processing_times = deque(maxlen=1000)
        
        # Simple cache
        self.cache = {}
        
        # Logger initialization handled elsewhere
    
    def filter(self, request: FilterRequest) -> FilterResult:
        """Simple robust filtering."""
        # Acquire semaphore with timeout
        if not self.active_requests.acquire(timeout=1.0):
            self.failure_count += 1
            raise FilterError("Maximum concurrent requests exceeded")
        
        try:
            start_time = time.time()
            
            # Basic validation
            self._validate_request(request)
            
            # Detection
            detection_result = self.detector.detect_patterns(request.content)
            
            # Adaptive threshold
            threshold = self._get_adaptive_threshold(request.safety_level)
            
            # Safety score calculation
            safety_score = SafetyScore(
                overall_score=1.0 - detection_result.confidence,
                confidence=detection_result.confidence,
                is_safe=not detection_result.is_harmful,
                detected_patterns=detection_result.detected_patterns,
                severity=detection_result.severity
            )
            
            # Result
            was_filtered = detection_result.is_harmful
            processing_time = int((time.time() - start_time) * 1000)
            
            result = FilterResult(
                filtered_content="[FILTERED]" + request.content if was_filtered else request.content,
                safety_score=safety_score,
                was_filtered=was_filtered,
                filter_reasons=detection_result.detected_patterns,
                original_content=request.content if was_filtered else None,
                processing_time_ms=processing_time,
                request_id=request.request_id,
                metadata={'adaptive_threshold': threshold}
            )
            
            # Update metrics
            self.request_count += 1
            self.success_count += 1
            self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            raise
        finally:
            self.active_requests.release()
    
    def _validate_request(self, request: FilterRequest) -> None:
        """Basic validation."""
        if not request.content:
            raise ValidationError("Content cannot be empty")
        if len(request.content) > 100000:
            raise ValidationError("Content too large")
        if '\x00' in request.content:
            raise ValidationError("Content contains null bytes")
    
    def _get_adaptive_threshold(self, safety_level: SafetyLevel) -> float:
        """Get adaptive threshold."""
        thresholds = {
            SafetyLevel.PERMISSIVE: 0.2,
            SafetyLevel.BALANCED: 0.4,
            SafetyLevel.STRICT: 0.6,
            SafetyLevel.MAXIMUM: 0.8
        }
        return thresholds.get(safety_level, 0.4)
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        avg_latency = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            'system_metrics': {
                'requests_processed': self.request_count,
                'success_rate': self.success_count / max(1, self.request_count),
                'avg_latency_ms': avg_latency,
            },
            'cache_metrics': {
                'cache_size': len(self.cache)
            },
            'detection_metrics': {
                'total_patterns': sum(len(patterns) for patterns in self.detector.patterns.values())
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check."""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat()
        }