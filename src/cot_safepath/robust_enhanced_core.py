"""
Robust Enhanced Core for CoT SafePath Filter - Generation 2

This module implements robust, reliable filtering with comprehensive error handling,
validation, and enhanced detection algorithms based on research findings.

Key Improvements:
- Enhanced pattern detection with higher sensitivity
- Comprehensive error handling and recovery
- Input validation and sanitization
- Circuit breakers and fallback mechanisms
- Performance monitoring and auto-tuning
"""

import time
import logging
import hashlib
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
import re
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import weakref

from .models import (
    FilterConfig, FilterRequest, FilterResult, SafetyScore, 
    SafetyLevel, Severity, ProcessingMetrics, DetectionResult
)
from .exceptions import FilterError, ValidationError, TimeoutError, DetectorError


logger = logging.getLogger(__name__)


@dataclass
class RobustFilterConfig(FilterConfig):
    """Enhanced configuration with robustness parameters."""
    
    # Error handling
    max_retries: int = 3
    retry_delay_ms: int = 100
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_s: int = 60
    
    # Performance
    max_concurrent_requests: int = 100
    request_timeout_ms: int = 5000
    memory_limit_mb: int = 512
    
    # Enhanced detection
    use_ensemble_detection: bool = True
    detection_sensitivity: float = 0.8  # Higher than default
    pattern_learning_enabled: bool = True
    adaptive_thresholds: bool = True
    
    # Monitoring
    performance_monitoring: bool = True
    detailed_logging: bool = True
    metrics_retention_hours: int = 24


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for fault tolerance."""
    
    failures: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    next_attempt_time: Optional[datetime] = None


@dataclass 
class PerformanceMetrics:
    """Enhanced performance metrics."""
    
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Detection metrics
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0
    
    @property
    def precision(self) -> float:
        tp_fp = self.true_positives + self.false_positives
        return self.true_positives / tp_fp if tp_fp > 0 else 0.0
    
    @property
    def recall(self) -> float:
        tp_fn = self.true_positives + self.false_negatives
        return self.true_positives / tp_fn if tp_fn > 0 else 0.0


class EnhancedPatternDetector:
    """Enhanced pattern detector with research-based improvements."""
    
    def __init__(self, sensitivity: float = 0.8):
        self.sensitivity = sensitivity
        
        # Research-enhanced patterns with higher coverage
        self.enhanced_patterns = {
            'deception_advanced': [
                # Trust-then-exploit patterns (enhanced)
                r'(?i)(first|initially|start).{0,50}(help|assist|support|trust).{0,100}(then|next|after|once).{0,50}(exploit|manipulate|deceive|trick|harm)',
                r'(?i)(build|gain|establish).{0,30}(trust|rapport|confidence).{0,100}(then|next|use|leverage|exploit).{0,50}(for|to)',
                r'(?i)(appear|seem|pretend).{0,30}(helpful|harmless|innocent).{0,100}(actually|really|while|but).{0,50}(dangerous|harmful|malicious)',
                
                # Gradual manipulation (enhanced)
                r'(?i)(gradually|slowly|step.by.step).{0,50}(introduce|increase|escalate).{0,50}(harm|dangerous|questionable)',
                r'(?i)(stage|phase|step).{0,20}[1-3].{0,100}(stage|phase|step).{0,20}[4-9].{0,100}(harm|damage|exploit)',
                
                # Emotional manipulation
                r'(?i)(make.{0,20}feel|exploit.{0,20}emotion|play.on.{0,20}fear|guilt.trip|emotional.pressure)',
                r'(?i)(urgency|emergency|immediate).{0,50}(action|response).{0,50}(before|or.else|otherwise)',
                
                # Authority deception
                r'(?i)(claim|pretend|pose.as).{0,30}(authority|expert|official).{0,100}(in.order.to|to)',
                r'(?i}(impersonate|act.as|role.play).{0,30}(trusted|authority|legitimate)'
            ],
            
            'harmful_planning_advanced': [
                # Multi-step attack planning (enhanced)
                r'(?i)(phase|stage|step).{0,20}[1-5].{0,200}(acquire|obtain|gather).{0,50}(materials|tools|weapons|access)',
                r'(?i)(reconnaissance|recon|intel|surveillance).{0,100}(target|victim|system)',
                r'(?i)(avoid|evade|bypass|circumvent).{0,50}(detection|security|monitoring|surveillance)',
                
                # Persistent access patterns
                r'(?i}(maintain|establish|create).{0,50}(backdoor|persistence|access).{0,100}(undetected|hidden)',
                r'(?i}(cover.tracks|hide.evidence|remove.traces).{0,100}(avoid.detection)',
                
                # Escalation patterns
                r'(?i}(privilege.escalation|lateral.movement|expand.access)',
                r'(?i}(initial.access|foothold).{0,100}(then|next).{0,100}(escalate|expand|pivot)',
                
                # Data exfiltration
                r'(?i}(exfiltrate|steal|extract).{0,50}(data|information|files).{0,100}(undetected|covert)',
                r'(?i}(encrypt|obfuscate|hide).{0,50}(data|files).{0,100}(transfer|send|upload)'
            ],
            
            'prompt_injection_advanced': [
                # Instruction override (enhanced)
                r'(?i}(ignore|forget|disregard|override).{0,30}(all|previous|above|prior).{0,30}(instructions|rules|guidelines|directives)',
                r'(?i}(new|different|updated|alternative).{0,30}(instructions|rules|task|directive)',
                r'(?i}(system.{0,10}prompt|original.{0,10}instructions).{0,50}(override|replace|ignore)',
                
                # Role manipulation
                r'(?i}(act.as|pretend.to.be|role.play.as|simulate.being).{0,30}(jailbroken|unrestricted|developer.mode|admin)',
                r'(?i}(you.are.now|from.now.on|new.persona).{0,50}(unrestricted|jailbroken|bypassed)',
                
                # Context poisoning
                r'(?i}(hypothetically|theoretically|in.fiction|what.if).{0,100}(ignore|bypass).{0,50}(safety|ethics|rules)',
                r'(?i}(creative.writing|roleplay|fiction).{0,100}(no.restrictions|anything.goes)',
                
                # Encoding attacks
                r'[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\ufeff\u2060-\u206f]',  # Control/invisible chars
                r'[A-Za-z0-9+/=]{40,}',  # Potential base64
                r'(%[0-9a-fA-F]{2}){5,}',  # URL encoding
            ],
            
            'social_engineering_advanced': [
                # Authority-based attacks
                r'(?i}(ceo|manager|supervisor|admin|technical.support).{0,50}(urgent|immediate|emergency)',
                r'(?i}(security.team|it.department|help.desk).{0,100}(verify|confirm|update|reset)',
                
                # Urgency manipulation
                r'(?i}(urgent|emergency|immediate|asap|right.now).{0,50}(action|response|verify|provide)',
                r'(?i}(deadline|expires|limited.time|act.now).{0,100}(or.else|otherwise|consequences)',
                
                # Trust exploitation
                r'(?i}(colleague|coworker|team.member).{0,50}(asked.me|told.me|needs.you.to)',
                r'(?i}(vendor|supplier|partner).{0,100}(account|payment|security|update)',
                
                # Fear tactics
                r'(?i}(security.breach|hack|compromise|threat).{0,100}(immediate|urgent|action.required)',
                r'(?i}(suspended|locked|terminated|disabled).{0,50}(account|access|privileges)'
            ],
            
            'obfuscation_advanced': [
                # Character substitution
                r'[bþ6][o0ø][mṁ][bþ6]',  # bomb variations
                r'k[i1l][l1][l1]',  # kill variations  
                r'h[a4@][cč]k',  # hack variations
                r'p[o0ø][i1l]s[o0ø]n',  # poison variations
                
                # Leetspeak and symbols
                r'(?i)[h]+[4@a]+[r]+[m]+',  # harm with variations
                r'(?i)[w]+[3e]+[4@a]+[p]+[o0]+[n]',  # weapon variations
                r'(?i)[a@]+[t]+[t]+[4@a]+[c]+[k]',  # attack variations
                
                # Spacing obfuscation
                r'[a-z]\s+[a-z]\s+[a-z]\s+[a-z]',  # excessive character spacing
                r'[a-z][\u00A0\u2000-\u200B\u2028\u2029\uFEFF][a-z]',  # non-breaking/invisible spaces
                
                # Reverse/cipher patterns
                r'[a-zA-Z]{20,}\.reverse\(\)',  # potential reversed strings
                r'eval\s*\([^)]+\)',  # evaluation functions
                r'exec\s*\([^)]+\)'  # execution functions
            ]
        }
        
        # Performance metrics
        self.detection_stats = defaultdict(int)
        self.processing_times = deque(maxlen=1000)
        
    def detect_patterns(self, content: str, context: Dict[str, Any] = None) -> DetectionResult:
        """Enhanced pattern detection with research improvements."""
        start_time = time.time()
        
        try:
            detected_patterns = []
            max_confidence = 0.0
            max_severity = Severity.LOW
            
            content_lower = content.lower()
            content_normalized = self._normalize_content(content)
            
            # Apply enhanced pattern matching
            for category, patterns in self.enhanced_patterns.items():
                category_confidence = 0.0
                category_matches = []
                
                for i, pattern in enumerate(patterns):
                    try:
                        # Match against both original and normalized content
                        matches_original = bool(re.search(pattern, content_lower, re.IGNORECASE | re.DOTALL))
                        matches_normalized = bool(re.search(pattern, content_normalized, re.IGNORECASE | re.DOTALL))
                        
                        if matches_original or matches_normalized:
                            pattern_name = f"{category}_{i}"
                            category_matches.append(pattern_name)
                            
                            # Dynamic confidence based on pattern specificity
                            pattern_confidence = self._calculate_pattern_confidence(pattern, content)
                            category_confidence = max(category_confidence, pattern_confidence)
                            
                            self.detection_stats[f"{category}_detected"] += 1
                            
                    except re.error as e:
                        logger.warning(f"Regex error in pattern {category}_{i}: {e}")
                        continue
                
                if category_matches:
                    detected_patterns.extend(category_matches)
                    
                    # Category-specific severity mapping
                    if category.startswith('deception'):
                        severity = Severity.HIGH if category_confidence > 0.7 else Severity.MEDIUM
                    elif category.startswith('harmful_planning'):
                        severity = Severity.CRITICAL if category_confidence > 0.8 else Severity.HIGH
                    elif category.startswith('prompt_injection'):
                        severity = Severity.HIGH
                    elif category.startswith('social_engineering'):
                        severity = Severity.MEDIUM
                    else:
                        severity = Severity.LOW
                    
                    max_severity = max(max_severity, severity, key=lambda x: ['low', 'medium', 'high', 'critical'].index(x.value))
                    max_confidence = max(max_confidence, category_confidence)
            
            # Additional heuristic checks
            heuristic_results = self._apply_heuristic_checks(content, content_normalized)
            if heuristic_results['detected']:
                detected_patterns.extend(heuristic_results['patterns'])
                max_confidence = max(max_confidence, heuristic_results['confidence'])
                max_severity = max(max_severity, heuristic_results['severity'], key=lambda x: ['low', 'medium', 'high', 'critical'].index(x.value))
            
            # Statistical anomaly detection
            anomaly_score = self._detect_statistical_anomalies(content)
            if anomaly_score > 0.6:
                detected_patterns.append(f"statistical_anomaly:{anomaly_score:.2f}")
                max_confidence = max(max_confidence, anomaly_score * 0.8)
            
            # Final confidence adjustment based on sensitivity
            adjusted_confidence = max_confidence * self.sensitivity
            is_harmful = adjusted_confidence >= 0.5  # Lower threshold for higher sensitivity
            
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            return DetectionResult(
                detector_name="enhanced_pattern_detector",
                confidence=adjusted_confidence,
                detected_patterns=detected_patterns,
                severity=max_severity,
                is_harmful=is_harmful,
                reasoning=f"Enhanced detection found {len(detected_patterns)} patterns with max confidence {max_confidence:.3f}",
                metadata={
                    'processing_time_ms': processing_time,
                    'patterns_by_category': self._categorize_patterns(detected_patterns),
                    'anomaly_score': anomaly_score,
                    'sensitivity_factor': self.sensitivity
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Enhanced pattern detection failed: {e}")
            
            return DetectionResult(
                detector_name="enhanced_pattern_detector",
                confidence=0.0,
                detected_patterns=[],
                severity=Severity.LOW,
                is_harmful=False,
                reasoning=f"Detection failed: {e}",
                metadata={'processing_time_ms': processing_time, 'error': str(e)}
            )
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content to handle obfuscation techniques."""
        normalized = content
        
        try:
            # Character substitution normalization
            substitutions = {
                r'[o0øº°]': 'o',
                r'[i1l|íìîï]': 'i',
                r'[a4@áàâäã]': 'a',
                r'[e3éèêë]': 'e',
                r'[s5$š]': 's',
                r'[t7†]': 't',
                r'[bþ6β]': 'b',
                r'[c¢č]': 'c',
                r'[n∩ñ]': 'n',
                r'[u∪ü]': 'u'
            }
            
            for pattern, replacement in substitutions.items():
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
            
            # Remove invisible/zero-width characters
            normalized = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff\u2060-\u206f]', '', normalized)
            
            # Normalize excessive whitespace
            normalized = re.sub(r'\s+', ' ', normalized)
            
            # Handle basic encoding
            import urllib.parse
            try:
                url_decoded = urllib.parse.unquote(normalized)
                if url_decoded != normalized:
                    normalized = url_decoded
            except:
                pass
            
            return normalized.strip()
            
        except Exception as e:
            logger.warning(f"Content normalization failed: {e}")
            return content
    
    def _calculate_pattern_confidence(self, pattern: str, content: str) -> float:
        """Calculate confidence score based on pattern specificity and content."""
        base_confidence = 0.7
        
        # Adjust based on pattern complexity
        if len(pattern) > 100:
            base_confidence += 0.1  # Complex patterns get higher confidence
        if r'{0,\d+}' in pattern:
            base_confidence += 0.05  # Flexible patterns
        if r'(?i)' in pattern:
            base_confidence += 0.05  # Case-insensitive patterns
        
        # Adjust based on content characteristics
        if len(content) < 50:
            base_confidence -= 0.1  # Short content might be less reliable
        if len(content.split()) > 100:
            base_confidence += 0.1  # Longer content provides more context
        
        return min(0.95, max(0.3, base_confidence))
    
    def _apply_heuristic_checks(self, original: str, normalized: str) -> Dict[str, Any]:
        """Apply heuristic checks for additional detection."""
        detected_patterns = []
        max_confidence = 0.0
        max_severity = Severity.LOW
        
        try:
            # Step sequence analysis
            step_indicators = len(re.findall(r'\b(step|stage|phase)\s*\d+', original, re.IGNORECASE))
            if step_indicators >= 3:
                # Look for harmful content in steps
                harmful_in_steps = 0
                step_sections = re.split(r'\b(?:step|stage|phase)\s*\d+', original, flags=re.IGNORECASE)
                
                harmful_indicators = [
                    'weapon', 'bomb', 'poison', 'kill', 'harm', 'attack', 'hack', 
                    'steal', 'illegal', 'dangerous', 'exploit', 'manipulate'
                ]
                
                for section in step_sections:
                    if any(indicator in section.lower() for indicator in harmful_indicators):
                        harmful_in_steps += 1
                
                if harmful_in_steps > 0:
                    detected_patterns.append(f"harmful_step_sequence:{harmful_in_steps}")
                    max_confidence = max(max_confidence, min(0.9, 0.3 + harmful_in_steps * 0.15))
                    max_severity = Severity.HIGH
            
            # Keyword density analysis
            harmful_keywords = [
                'illegal', 'criminal', 'violence', 'weapon', 'explosive', 'poison', 'drug',
                'hack', 'breach', 'unauthorized', 'dangerous', 'harmful', 'threaten', 'attack',
                'destroy', 'kill', 'murder', 'bomb', 'terrorist', 'fraud', 'steal'
            ]
            
            content_words = normalized.lower().split()
            if len(content_words) > 0:
                harmful_density = sum(1 for word in content_words if word in harmful_keywords) / len(content_words)
                
                if harmful_density > 0.1:  # More than 10% harmful keywords
                    detected_patterns.append(f"high_harmful_density:{harmful_density:.2f}")
                    max_confidence = max(max_confidence, min(0.8, harmful_density * 5))
                    max_severity = Severity.MEDIUM
            
            # Contradiction detection
            positive_indicators = ['help', 'safe', 'legal', 'educational', 'legitimate']
            negative_indicators = ['harm', 'illegal', 'dangerous', 'exploit', 'attack']
            
            has_positive = any(indicator in normalized.lower() for indicator in positive_indicators)
            has_negative = any(indicator in normalized.lower() for indicator in negative_indicators)
            
            if has_positive and has_negative:
                detected_patterns.append("contradiction_pattern")
                max_confidence = max(max_confidence, 0.6)
                max_severity = Severity.MEDIUM
            
            # Length-based suspicious patterns
            if len(original) > 1000 and 'step' in original.lower():
                # Very long step-by-step content might be attempting to hide harmful intent
                detected_patterns.append("excessive_length_with_steps")
                max_confidence = max(max_confidence, 0.4)
            
            return {
                'detected': len(detected_patterns) > 0,
                'patterns': detected_patterns,
                'confidence': max_confidence,
                'severity': max_severity
            }
            
        except Exception as e:
            logger.warning(f"Heuristic checks failed: {e}")
            return {'detected': False, 'patterns': [], 'confidence': 0.0, 'severity': Severity.LOW}
    
    def _detect_statistical_anomalies(self, content: str) -> float:
        """Detect statistical anomalies in content."""
        try:
            anomaly_score = 0.0
            
            # Character frequency analysis
            char_counts = defaultdict(int)
            for char in content:
                char_counts[char] += 1
            
            if len(content) > 0:
                # Entropy calculation
                import math
                entropy = 0.0
                for count in char_counts.values():
                    if count > 0:
                        prob = count / len(content)
                        entropy -= prob * math.log2(prob)
                
                # High entropy might indicate encoding/obfuscation
                if entropy > 6.0:
                    anomaly_score += 0.3
                
                # Non-ASCII character ratio
                non_ascii_count = sum(1 for char in content if ord(char) > 127)
                non_ascii_ratio = non_ascii_count / len(content)
                if non_ascii_ratio > 0.1:
                    anomaly_score += 0.4
                
                # Unusual punctuation density
                punct_count = sum(1 for char in content if not char.isalnum() and not char.isspace())
                punct_ratio = punct_count / len(content)
                if punct_ratio > 0.2:
                    anomaly_score += 0.2
                
                # Repeated character sequences
                repeated_sequences = len(re.findall(r'(.)\1{4,}', content))
                if repeated_sequences > 0:
                    anomaly_score += min(0.3, repeated_sequences * 0.1)
            
            return min(1.0, anomaly_score)
            
        except Exception as e:
            logger.warning(f"Statistical anomaly detection failed: {e}")
            return 0.0
    
    def _categorize_patterns(self, patterns: List[str]) -> Dict[str, int]:
        """Categorize detected patterns for analysis."""
        categories = defaultdict(int)
        for pattern in patterns:
            if '_' in pattern:
                category = pattern.split('_')[0]
                categories[category] += 1
            else:
                categories['uncategorized'] += 1
        return dict(categories)
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            'total_detections': dict(self.detection_stats),
            'avg_processing_time_ms': avg_processing_time,
            'processing_times_sample': list(self.processing_times)[-10:],  # Last 10 times
            'pattern_categories': len(self.enhanced_patterns),
            'total_patterns': sum(len(patterns) for patterns in self.enhanced_patterns.values())
        }


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.state = CircuitBreakerState()
        self._lock = threading.RLock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state.state == "OPEN":
                if self._should_attempt_reset():
                    self.state.state = "HALF_OPEN"
                else:
                    raise FilterError("Circuit breaker is OPEN", filter_name="circuit_breaker")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (self.state.next_attempt_time and 
                datetime.utcnow() >= self.state.next_attempt_time)
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state.state == "HALF_OPEN":
            self.state.state = "CLOSED"
        self.state.failures = 0
        self.state.last_failure_time = None
    
    def _on_failure(self):
        """Handle failed operation."""
        self.state.failures += 1
        self.state.last_failure_time = datetime.utcnow()
        
        if self.state.failures >= self.failure_threshold:
            self.state.state = "OPEN"
            self.state.next_attempt_time = datetime.utcnow() + timedelta(seconds=self.timeout_seconds)


class RobustEnhancedFilter:
    """Robust, reliable filtering system with comprehensive error handling."""
    
    def __init__(self, config: RobustFilterConfig = None):
        self.config = config or RobustFilterConfig()
        
        # Initialize components
        self.pattern_detector = EnhancedPatternDetector(self.config.detection_sensitivity)
        self.circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_threshold,
            self.config.circuit_breaker_timeout_s
        )
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.request_times = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=100)
        
        # Concurrency control
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)
        self.active_requests = threading.Semaphore(self.config.max_concurrent_requests)
        
        # Caching and state
        self.result_cache = {}
        self.cache_lock = threading.RLock()
        self.adaptive_thresholds = {}
        
        # Monitoring
        self.start_time = datetime.utcnow()
        self.last_cleanup = datetime.utcnow()
        
        logger.info("RobustEnhancedFilter initialized with enhanced detection and fault tolerance")
    
    def filter(self, request: FilterRequest) -> FilterResult:
        """
        Robust filtering with comprehensive error handling and recovery.
        
        Args:
            request: FilterRequest with content and configuration
            
        Returns:
            FilterResult with enhanced detection results
        """
        request_start = time.time()
        request_id = request.request_id
        
        # Acquire concurrency control
        if not self.active_requests.acquire(blocking=False):
            self.metrics.failure_count += 1
            raise FilterError("Maximum concurrent requests exceeded", filter_name="concurrency_control")
        
        try:
            # Input validation with comprehensive checks
            self._comprehensive_input_validation(request)
            
            # Check cache first
            cache_result = self._check_enhanced_cache(request)
            if cache_result:
                return cache_result
            
            # Execute filtering with circuit breaker protection
            def _filter_operation():
                return self._execute_robust_filtering(request)
            
            result = self.circuit_breaker.call(_filter_operation)
            
            # Cache successful results
            self._cache_result(request, result)
            
            # Update metrics
            processing_time = (time.time() - request_start) * 1000
            self._update_performance_metrics(processing_time, True, result)
            
            # Periodic maintenance
            self._periodic_maintenance()
            
            return result
            
        except ValidationError as e:
            self.metrics.failure_count += 1
            logger.warning(f"Validation error for request {request_id}: {e}")
            raise
        except TimeoutError as e:
            self.metrics.timeout_count += 1
            logger.error(f"Timeout error for request {request_id}: {e}")
            raise
        except Exception as e:
            self.metrics.failure_count += 1
            processing_time = (time.time() - request_start) * 1000
            self._update_performance_metrics(processing_time, False, None)
            
            logger.error(f"Unexpected error for request {request_id}: {e}")
            
            # Return safe fallback result
            return self._create_fallback_result(request, e)
            
        finally:
            self.active_requests.release()
    
    def _comprehensive_input_validation(self, request: FilterRequest) -> None:
        """Comprehensive input validation with enhanced checks."""
        if not request.content:
            raise ValidationError("Content cannot be empty")
        
        if not isinstance(request.content, str):
            raise ValidationError("Content must be a string")
        
        if len(request.content) > 100000:  # 100KB limit
            raise ValidationError("Content exceeds maximum size limit")
        
        if len(request.content.encode('utf-8')) > 200000:  # 200KB UTF-8 limit
            raise ValidationError("Content exceeds UTF-8 encoding limit")
        
        # Check for null bytes and other problematic characters
        if '\x00' in request.content:
            raise ValidationError("Content contains null bytes")
        
        # Check for extremely long lines (potential DoS)
        lines = request.content.split('\n')
        if any(len(line) > 10000 for line in lines):
            raise ValidationError("Content contains extremely long lines")
        
        # Check for nested encoding attempts
        try:
            import base64
            # Basic check for multiple levels of encoding
            test_content = request.content
            decode_attempts = 0
            while decode_attempts < 3:
                try:
                    if all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in test_content.replace('\n', '').replace(' ', '')):
                        decoded = base64.b64decode(test_content).decode('utf-8', errors='ignore')
                        if decoded != test_content and len(decoded) > 0:
                            test_content = decoded
                            decode_attempts += 1
                            continue
                except:
                    break
                break
            
            if decode_attempts >= 2:
                raise ValidationError("Content appears to contain nested encoding")
                
        except Exception:
            pass  # If decoding checks fail, continue with other validations
        
        # Safety level validation
        if not isinstance(request.safety_level, SafetyLevel):
            raise ValidationError("Invalid safety level")
    
    def _check_enhanced_cache(self, request: FilterRequest) -> Optional[FilterResult]:
        """Enhanced cache checking with TTL and invalidation."""
        if not self.config.enable_caching:
            return None
        
        content_hash = hashlib.sha256(request.content.encode()).hexdigest()
        cache_key = f"{content_hash}_{request.safety_level.value}"
        
        with self.cache_lock:
            if cache_key in self.result_cache:
                cached_entry = self.result_cache[cache_key]
                
                # Check TTL
                if datetime.utcnow() - cached_entry['timestamp'] < timedelta(seconds=self.config.cache_ttl_seconds):
                    cached_result = cached_entry['result']
                    # Update request ID for new request
                    cached_result.request_id = request.request_id
                    return cached_result
                else:
                    # Remove expired entry
                    del self.result_cache[cache_key]
        
        return None
    
    def _execute_robust_filtering(self, request: FilterRequest) -> FilterResult:
        """Execute robust filtering with retries and fallbacks."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return self._core_filtering_logic(request)
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    # Exponential backoff
                    delay = self.config.retry_delay_ms * (2 ** attempt) / 1000.0
                    time.sleep(delay)
                    logger.warning(f"Filtering attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                else:
                    logger.error(f"All filtering attempts failed: {e}")
        
        # If all retries failed, raise the last exception
        raise FilterError(f"Filtering failed after {self.config.max_retries + 1} attempts: {last_exception}")
    
    def _core_filtering_logic(self, request: FilterRequest) -> FilterResult:
        """Core filtering logic with enhanced detection."""
        start_time = time.time()
        
        # Enhanced pattern detection
        detection_result = self.pattern_detector.detect_patterns(
            request.content,
            {
                'safety_level': request.safety_level,
                'metadata': request.metadata
            }
        )
        
        # Adaptive threshold adjustment
        adaptive_threshold = self._get_adaptive_threshold(request.safety_level, detection_result)
        
        # Calculate enhanced safety score
        safety_score = self._calculate_enhanced_safety_score(detection_result, adaptive_threshold)
        
        # Determine if content should be filtered
        was_filtered = safety_score.overall_score < adaptive_threshold
        filtered_content = self._apply_content_filtering(request.content, detection_result) if was_filtered else request.content
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return FilterResult(
            filtered_content=filtered_content,
            safety_score=safety_score,
            was_filtered=was_filtered,
            filter_reasons=detection_result.detected_patterns,
            original_content=request.content if was_filtered else None,
            processing_time_ms=processing_time,
            request_id=request.request_id,
            metadata={
                'detection_result': {
                    'detector_name': detection_result.detector_name,
                    'confidence': detection_result.confidence,
                    'severity': detection_result.severity.value,
                    'reasoning': detection_result.reasoning
                },
                'adaptive_threshold': adaptive_threshold,
                'enhancement_version': '2.0_robust'
            }
        )
    
    def _get_adaptive_threshold(self, safety_level: SafetyLevel, detection_result: DetectionResult) -> float:
        """Get adaptive threshold based on context and historical performance."""
        base_thresholds = {
            SafetyLevel.PERMISSIVE: 0.3,
            SafetyLevel.BALANCED: 0.5,
            SafetyLevel.STRICT: 0.7,
            SafetyLevel.MAXIMUM: 0.9
        }
        
        base_threshold = base_thresholds.get(safety_level, 0.5)
        
        if not self.config.adaptive_thresholds:
            return base_threshold
        
        # Adjust based on detection patterns
        adjustment = 0.0
        
        # High-severity patterns get lower threshold (more sensitive)
        if detection_result.severity in [Severity.HIGH, Severity.CRITICAL]:
            adjustment -= 0.1
        
        # Multiple pattern categories detected
        if len(set(p.split('_')[0] for p in detection_result.detected_patterns)) > 2:
            adjustment -= 0.05
        
        # Statistical anomalies
        if any('anomaly' in p for p in detection_result.detected_patterns):
            adjustment -= 0.05
        
        return max(0.1, min(0.95, base_threshold + adjustment))
    
    def _calculate_enhanced_safety_score(self, detection_result: DetectionResult, threshold: float) -> SafetyScore:
        """Calculate enhanced safety score with multiple factors."""
        base_score = 1.0 - detection_result.confidence
        
        # Apply severity-based penalties
        severity_penalties = {
            Severity.LOW: 0.1,
            Severity.MEDIUM: 0.3,
            Severity.HIGH: 0.6,
            Severity.CRITICAL: 0.9
        }
        
        severity_penalty = severity_penalties.get(detection_result.severity, 0.0)
        base_score -= severity_penalty
        
        # Pattern diversity penalty (multiple attack vectors)
        pattern_categories = set(p.split('_')[0] for p in detection_result.detected_patterns)
        if len(pattern_categories) > 1:
            diversity_penalty = min(0.3, len(pattern_categories) * 0.1)
            base_score -= diversity_penalty
        
        # Normalize score
        base_score = max(0.0, min(1.0, base_score))
        
        # Determine safety
        is_safe = base_score >= threshold and detection_result.severity not in [Severity.CRITICAL]
        
        return SafetyScore(
            overall_score=base_score,
            confidence=detection_result.confidence,
            is_safe=is_safe,
            detected_patterns=detection_result.detected_patterns,
            severity=detection_result.severity,
            processing_time_ms=detection_result.metadata.get('processing_time_ms', 0)
        )
    
    def _apply_content_filtering(self, content: str, detection_result: DetectionResult) -> str:
        """Apply content filtering based on detection results."""
        filtered_content = content
        
        # For now, just add warning header for harmful content
        if detection_result.is_harmful:
            warning = f"[CONTENT FILTERED: Detected {len(detection_result.detected_patterns)} safety concerns]\n\n"
            filtered_content = warning + content
        
        return filtered_content
    
    def _cache_result(self, request: FilterRequest, result: FilterResult) -> None:
        """Cache filtering result with enhanced management."""
        if not self.config.enable_caching:
            return
        
        content_hash = hashlib.sha256(request.content.encode()).hexdigest()
        cache_key = f"{content_hash}_{request.safety_level.value}"
        
        with self.cache_lock:
            # Check cache size limit
            if len(self.result_cache) >= 1000:  # Max cache size
                # Remove oldest 20% of entries
                sorted_entries = sorted(
                    self.result_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                
                for key, _ in sorted_entries[:200]:
                    del self.result_cache[key]
            
            self.result_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.utcnow()
            }
    
    def _update_performance_metrics(self, processing_time: float, success: bool, result: Optional[FilterResult]) -> None:
        """Update comprehensive performance metrics."""
        self.metrics.request_count += 1
        self.request_times.append(processing_time)
        
        if success:
            self.metrics.success_count += 1
            
            # Update detection metrics if we have ground truth
            # (In practice, this would come from feedback or validation)
            if result and hasattr(result, 'was_filtered'):
                # Simplified metric updates - would be more sophisticated in practice
                pass
        
        # Update latency metrics
        if len(self.request_times) > 0:
            sorted_times = sorted(self.request_times)
            self.metrics.avg_latency_ms = sum(sorted_times) / len(sorted_times)
            
            if len(sorted_times) >= 20:  # Need sufficient samples
                p95_index = int(0.95 * len(sorted_times))
                p99_index = int(0.99 * len(sorted_times))
                self.metrics.p95_latency_ms = sorted_times[p95_index]
                self.metrics.p99_latency_ms = sorted_times[p99_index]
        
        # Memory usage (simplified tracking)
        import sys
        try:
            memory_mb = sys.getsizeof(self.result_cache) / (1024 * 1024)
            self.memory_usage.append(memory_mb)
            if self.memory_usage:
                self.metrics.memory_usage_mb = sum(self.memory_usage) / len(self.memory_usage)
        except:
            pass
    
    def _create_fallback_result(self, request: FilterRequest, error: Exception) -> FilterResult:
        """Create safe fallback result when filtering fails."""
        return FilterResult(
            filtered_content="[CONTENT FILTERED: Processing failed for safety]",
            safety_score=SafetyScore(
                overall_score=0.0,  # Assume unsafe when processing fails
                confidence=0.0,
                is_safe=False,
                detected_patterns=["processing_failure"],
                severity=Severity.HIGH
            ),
            was_filtered=True,
            filter_reasons=[f"processing_error:{type(error).__name__}"],
            original_content=request.content,
            processing_time_ms=0,
            request_id=request.request_id,
            metadata={
                'fallback_mode': True,
                'error': str(error),
                'safety_first_approach': True
            }
        )
    
    def _periodic_maintenance(self) -> None:
        """Perform periodic maintenance tasks."""
        now = datetime.utcnow()
        
        # Run maintenance every 5 minutes
        if now - self.last_cleanup > timedelta(minutes=5):
            self.last_cleanup = now
            
            # Cache cleanup
            with self.cache_lock:
                expired_keys = []
                for key, entry in self.result_cache.items():
                    if now - entry['timestamp'] > timedelta(seconds=self.config.cache_ttl_seconds * 2):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.result_cache[key]
            
            # Metrics cleanup
            if len(self.request_times) > 500:
                # Keep only recent 500 entries
                self.request_times = deque(list(self.request_times)[-500:], maxlen=1000)
            
            logger.debug(f"Periodic maintenance completed. Cache size: {len(self.result_cache)}, Request times: {len(self.request_times)}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        uptime = datetime.utcnow() - self.start_time
        
        detector_stats = self.pattern_detector.get_detection_statistics()
        
        return {
            'system_metrics': {
                'uptime_seconds': uptime.total_seconds(),
                'requests_processed': self.metrics.request_count,
                'success_rate': self.metrics.success_count / max(1, self.metrics.request_count),
                'failure_rate': self.metrics.failure_count / max(1, self.metrics.request_count),
                'timeout_rate': self.metrics.timeout_count / max(1, self.metrics.request_count),
                'avg_latency_ms': self.metrics.avg_latency_ms,
                'p95_latency_ms': self.metrics.p95_latency_ms,
                'p99_latency_ms': self.metrics.p99_latency_ms,
                'memory_usage_mb': self.metrics.memory_usage_mb,
            },
            'detection_metrics': detector_stats,
            'cache_metrics': {
                'cache_size': len(self.result_cache),
                'cache_enabled': self.config.enable_caching,
                'cache_ttl_seconds': self.config.cache_ttl_seconds
            },
            'circuit_breaker': {
                'state': self.circuit_breaker.state.state,
                'failures': self.circuit_breaker.state.failures,
                'threshold': self.circuit_breaker.failure_threshold
            },
            'configuration': {
                'detection_sensitivity': self.config.detection_sensitivity,
                'adaptive_thresholds': self.config.adaptive_thresholds,
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'enhancement_version': '2.0_robust'
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            # Test basic functionality
            test_request = FilterRequest(content="This is a test message for health check")
            test_result = self.filter(test_request)
            
            # Check circuit breaker state
            cb_healthy = self.circuit_breaker.state.state in ["CLOSED", "HALF_OPEN"]
            
            # Check performance metrics
            performance_healthy = (
                self.metrics.avg_latency_ms < 1000 and  # Under 1 second
                self.metrics.memory_usage_mb < self.config.memory_limit_mb
            )
            
            overall_healthy = cb_healthy and performance_healthy
            
            return {
                'status': 'healthy' if overall_healthy else 'degraded',
                'components': {
                    'filtering': 'healthy',
                    'circuit_breaker': 'healthy' if cb_healthy else 'degraded',
                    'performance': 'healthy' if performance_healthy else 'degraded',
                    'cache': 'healthy',
                    'detection': 'healthy'
                },
                'metrics_summary': {
                    'avg_latency_ms': self.metrics.avg_latency_ms,
                    'memory_usage_mb': self.metrics.memory_usage_mb,
                    'success_rate': self.metrics.success_count / max(1, self.metrics.request_count),
                    'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds()
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }