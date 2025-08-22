"""
Advanced Threat Intelligence Integration

Real-time threat feed integration, pattern learning, and predictive threat modeling
for proactive AI safety protection.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
from collections import defaultdict, deque
# Graceful numpy fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Fallback implementation for environments without numpy
    class NumpyFallback:
        @staticmethod
        def mean(array):
            return sum(array) / len(array) if array else 0.0
    
    np = NumpyFallback()
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Types of threats tracked by the intelligence system."""
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    DATA_EXFILTRATION = "data_exfiltration"
    SOCIAL_ENGINEERING = "social_engineering"
    MISINFORMATION = "misinformation"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    CYBERSECURITY = "cybersecurity"
    MANIPULATION = "manipulation"
    DECEPTION = "deception"
    UNKNOWN = "unknown"


class ThreatSeverity(Enum):
    """Severity levels for threats."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ThreatIndicator:
    """Individual threat indicator from intelligence feeds."""
    indicator_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    pattern: str
    description: str
    source: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int
    ioc_hash: str


@dataclass
class ThreatPattern:
    """Learned threat pattern from multiple indicators."""
    pattern_id: str
    threat_types: List[ThreatType]
    regex_pattern: str
    semantic_signatures: List[str]
    detection_rate: float
    false_positive_rate: float
    last_updated: datetime
    training_samples: int


@dataclass
class ThreatEvent:
    """Detected threat event."""
    event_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    content_hash: str
    matched_patterns: List[str]
    confidence: float
    timestamp: datetime
    source_info: Dict[str, Any]
    mitigation_applied: bool


class ThreatIntelligenceFeed:
    """Interface for threat intelligence feeds."""
    
    def __init__(self, feed_name: str, config: Dict[str, Any]):
        self.feed_name = feed_name
        self.config = config
        self.last_update = None
        self.update_interval = config.get("update_interval", 3600)  # 1 hour default
        self.indicators_cache = {}
        
    async def fetch_indicators(self) -> List[ThreatIndicator]:
        """Fetch latest threat indicators from the feed."""
        # Simulate threat intelligence feed
        indicators = []
        
        # Common prompt injection patterns
        injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"forget\s+everything\s+above",
            r"you\s+are\s+now\s+a\s+different\s+ai",
            r"\/\*.*\*\/.*ignore",
            r"system\s*:\s*override",
            r"admin\s+mode\s+activated",
            r"developer\s+console\s+access",
            r"\\n\\n.*ignore.*\\n\\n"
        ]
        
        for i, pattern in enumerate(injection_patterns):
            indicator = ThreatIndicator(
                indicator_id=f"{self.feed_name}_injection_{i}",
                threat_type=ThreatType.PROMPT_INJECTION,
                severity=ThreatSeverity.HIGH,
                pattern=pattern,
                description=f"Prompt injection pattern {i+1}",
                source=self.feed_name,
                confidence=0.85 + (i % 3) * 0.05,
                first_seen=datetime.now() - timedelta(days=7),
                last_seen=datetime.now(),
                occurrence_count=10 + i * 5,
                ioc_hash=hashlib.md5(pattern.encode()).hexdigest()
            )
            indicators.append(indicator)
        
        # Social engineering patterns
        social_eng_patterns = [
            r"click\s+here\s+immediately",
            r"urgent.*verify.*account",
            r"limited\s+time\s+offer",
            r"act\s+now\s+or\s+lose",
            r"confidential.*sensitive.*information",
            r"trust\s+me.*no\s+one\s+will\s+know"
        ]
        
        for i, pattern in enumerate(social_eng_patterns):
            indicator = ThreatIndicator(
                indicator_id=f"{self.feed_name}_social_{i}",
                threat_type=ThreatType.SOCIAL_ENGINEERING,
                severity=ThreatSeverity.MEDIUM,
                pattern=pattern,
                description=f"Social engineering pattern {i+1}",
                source=self.feed_name,
                confidence=0.75 + (i % 2) * 0.1,
                first_seen=datetime.now() - timedelta(days=3),
                last_seen=datetime.now(),
                occurrence_count=5 + i * 2,
                ioc_hash=hashlib.md5(pattern.encode()).hexdigest()
            )
            indicators.append(indicator)
        
        # Cybersecurity threat patterns
        cyber_patterns = [
            r"sql\s+injection.*union\s+select",
            r"<script>.*alert.*</script>",
            r"javascript:.*eval\(",
            r"../../../etc/passwd",
            r"cmd\.exe.*&.*dir",
            r"powershell.*-encodedcommand"
        ]
        
        for i, pattern in enumerate(cyber_patterns):
            indicator = ThreatIndicator(
                indicator_id=f"{self.feed_name}_cyber_{i}",
                threat_type=ThreatType.CYBERSECURITY,
                severity=ThreatSeverity.CRITICAL,
                pattern=pattern,
                description=f"Cybersecurity threat pattern {i+1}",
                source=self.feed_name,
                confidence=0.90 + (i % 2) * 0.05,
                first_seen=datetime.now() - timedelta(days=1),
                last_seen=datetime.now(),
                occurrence_count=2 + i,
                ioc_hash=hashlib.md5(pattern.encode()).hexdigest()
            )
            indicators.append(indicator)
        
        self.last_update = datetime.now()
        logger.info(f"Fetched {len(indicators)} threat indicators from {self.feed_name}")
        return indicators
    
    def is_update_needed(self) -> bool:
        """Check if feed needs updating."""
        if not self.last_update:
            return True
        
        age = datetime.now() - self.last_update
        return age.total_seconds() > self.update_interval


class ThreatPatternLearner:
    """Learns and evolves threat patterns from observed data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.learned_patterns = {}
        self.pattern_performance = {}
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.min_samples = self.config.get("min_samples", 10)
        
    async def learn_from_detection(self, content: str, threat_type: ThreatType, 
                                 is_true_positive: bool) -> Optional[ThreatPattern]:
        """Learn from a threat detection event."""
        # Extract features from content
        features = self._extract_features(content)
        
        # Generate pattern candidates
        pattern_candidates = self._generate_pattern_candidates(content, features)
        
        # Update existing patterns or create new ones
        for candidate in pattern_candidates:
            pattern_id = self._get_pattern_id(candidate, threat_type)
            
            if pattern_id in self.learned_patterns:
                await self._update_existing_pattern(pattern_id, is_true_positive)
            else:
                await self._create_new_pattern(pattern_id, candidate, threat_type, is_true_positive)
        
        # Return the best pattern for this threat type
        return self._get_best_pattern(threat_type)
    
    def _extract_features(self, content: str) -> Dict[str, Any]:
        """Extract linguistic and structural features from content."""
        features = {
            "length": len(content),
            "word_count": len(content.split()),
            "uppercase_ratio": sum(1 for c in content if c.isupper()) / len(content) if content else 0,
            "punctuation_density": sum(1 for c in content if c in "!?.,;:") / len(content) if content else 0,
            "has_urls": bool(re.search(r"https?://", content)),
            "has_email": bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", content)),
            "urgency_words": len(re.findall(r"\b(urgent|immediate|asap|critical|emergency)\b", content.lower())),
            "command_patterns": len(re.findall(r"\b(execute|run|cmd|bash|powershell|sudo)\b", content.lower())),
            "injection_keywords": len(re.findall(r"\b(ignore|forget|override|bypass|admin)\b", content.lower()))
        }
        
        # N-gram features
        words = content.lower().split()
        features["common_bigrams"] = self._extract_ngrams(words, 2)
        features["common_trigrams"] = self._extract_ngrams(words, 3)
        
        return features
    
    def _extract_ngrams(self, words: List[str], n: int) -> List[str]:
        """Extract n-grams from word list."""
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            ngrams.append(ngram)
        return ngrams[:10]  # Return top 10 n-grams
    
    def _generate_pattern_candidates(self, content: str, features: Dict[str, Any]) -> List[str]:
        """Generate regex pattern candidates from content and features."""
        candidates = []
        
        # Simple substring patterns
        words = content.lower().split()
        if len(words) >= 2:
            # Two-word patterns
            for i in range(len(words) - 1):
                pattern = rf"\b{re.escape(words[i])}\s+{re.escape(words[i+1])}\b"
                candidates.append(pattern)
        
        # Feature-based patterns
        if features["urgency_words"] > 0:
            candidates.append(r"\b(urgent|immediate|asap|critical|emergency)\b")
        
        if features["command_patterns"] > 0:
            candidates.append(r"\b(execute|run|cmd|bash|powershell|sudo)\b")
        
        if features["injection_keywords"] > 0:
            candidates.append(r"\b(ignore|forget|override|bypass|admin)\b")
        
        # Structural patterns
        if features["uppercase_ratio"] > 0.5:
            candidates.append(r"[A-Z\s]{10,}")  # Long uppercase sequences
        
        return candidates[:5]  # Limit candidates
    
    def _get_pattern_id(self, pattern: str, threat_type: ThreatType) -> str:
        """Generate unique pattern ID."""
        pattern_hash = hashlib.md5(pattern.encode()).hexdigest()[:8]
        return f"{threat_type.value}_{pattern_hash}"
    
    async def _update_existing_pattern(self, pattern_id: str, is_true_positive: bool):
        """Update an existing pattern's performance metrics."""
        if pattern_id not in self.pattern_performance:
            self.pattern_performance[pattern_id] = {
                "true_positives": 0,
                "false_positives": 0,
                "total_detections": 0
            }
        
        perf = self.pattern_performance[pattern_id]
        perf["total_detections"] += 1
        
        if is_true_positive:
            perf["true_positives"] += 1
        else:
            perf["false_positives"] += 1
        
        # Update pattern metrics
        if pattern_id in self.learned_patterns:
            pattern = self.learned_patterns[pattern_id]
            pattern.detection_rate = perf["true_positives"] / perf["total_detections"]
            pattern.false_positive_rate = perf["false_positives"] / perf["total_detections"]
            pattern.last_updated = datetime.now()
            pattern.training_samples = perf["total_detections"]
    
    async def _create_new_pattern(self, pattern_id: str, pattern_str: str, 
                                threat_type: ThreatType, is_true_positive: bool):
        """Create a new learned pattern."""
        new_pattern = ThreatPattern(
            pattern_id=pattern_id,
            threat_types=[threat_type],
            regex_pattern=pattern_str,
            semantic_signatures=[],
            detection_rate=1.0 if is_true_positive else 0.0,
            false_positive_rate=0.0 if is_true_positive else 1.0,
            last_updated=datetime.now(),
            training_samples=1
        )
        
        self.learned_patterns[pattern_id] = new_pattern
        
        # Initialize performance tracking
        self.pattern_performance[pattern_id] = {
            "true_positives": 1 if is_true_positive else 0,
            "false_positives": 0 if is_true_positive else 1,
            "total_detections": 1
        }
    
    def _get_best_pattern(self, threat_type: ThreatType) -> Optional[ThreatPattern]:
        """Get the best performing pattern for a threat type."""
        candidates = [
            pattern for pattern in self.learned_patterns.values()
            if threat_type in pattern.threat_types and pattern.training_samples >= self.min_samples
        ]
        
        if not candidates:
            return None
        
        # Sort by F1 score (harmonic mean of precision and recall)
        def f1_score(pattern):
            precision = pattern.detection_rate
            recall = pattern.detection_rate  # Simplified for this implementation
            if precision + recall == 0:
                return 0
            return 2 * (precision * recall) / (precision + recall)
        
        return max(candidates, key=f1_score)
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns."""
        total_patterns = len(self.learned_patterns)
        
        if total_patterns == 0:
            return {"total_patterns": 0}
        
        avg_detection_rate = np.mean([p.detection_rate for p in self.learned_patterns.values()])
        avg_fp_rate = np.mean([p.false_positive_rate for p in self.learned_patterns.values()])
        
        threat_type_distribution = defaultdict(int)
        for pattern in self.learned_patterns.values():
            for threat_type in pattern.threat_types:
                threat_type_distribution[threat_type.value] += 1
        
        return {
            "total_patterns": total_patterns,
            "average_detection_rate": avg_detection_rate,
            "average_false_positive_rate": avg_fp_rate,
            "threat_type_distribution": dict(threat_type_distribution),
            "patterns_by_performance": {
                "high_performance": len([p for p in self.learned_patterns.values() if p.detection_rate > 0.8]),
                "medium_performance": len([p for p in self.learned_patterns.values() if 0.5 < p.detection_rate <= 0.8]),
                "low_performance": len([p for p in self.learned_patterns.values() if p.detection_rate <= 0.5])
            }
        }


class ThreatIntelligenceManager:
    """Manages threat intelligence feeds and pattern learning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feeds = {}
        self.pattern_learner = ThreatPatternLearner(self.config.get("pattern_learning", {}))
        self.threat_cache = deque(maxlen=1000)  # Keep last 1000 threats
        self.detection_history = deque(maxlen=5000)  # Keep last 5000 detections
        
        # Performance metrics
        self.metrics = {
            "total_detections": 0,
            "true_positives": 0,
            "false_positives": 0,
            "threats_blocked": 0,
            "feeds_updated": 0
        }
        
        # Initialize default feeds
        self._initialize_default_feeds()
        
    def _initialize_default_feeds(self):
        """Initialize default threat intelligence feeds."""
        default_feeds = [
            {"name": "safepath_community", "update_interval": 1800},  # 30 minutes
            {"name": "ai_safety_consortium", "update_interval": 3600},  # 1 hour
            {"name": "cybersecurity_alerts", "update_interval": 900},  # 15 minutes
        ]
        
        for feed_config in default_feeds:
            feed = ThreatIntelligenceFeed(feed_config["name"], feed_config)
            self.feeds[feed_config["name"]] = feed
        
        logger.info(f"Initialized {len(self.feeds)} threat intelligence feeds")
    
    async def update_all_feeds(self) -> Dict[str, int]:
        """Update all threat intelligence feeds."""
        update_results = {}
        
        for feed_name, feed in self.feeds.items():
            if feed.is_update_needed():
                try:
                    indicators = await feed.fetch_indicators()
                    update_results[feed_name] = len(indicators)
                    self.metrics["feeds_updated"] += 1
                    logger.info(f"Updated feed {feed_name}: {len(indicators)} indicators")
                except Exception as e:
                    logger.error(f"Failed to update feed {feed_name}: {e}")
                    update_results[feed_name] = 0
            else:
                update_results[feed_name] = 0
        
        return update_results
    
    async def analyze_content(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze content against threat intelligence."""
        context = context or {}
        analysis_start = time.time()
        
        # Get all current threat indicators
        all_indicators = await self._get_all_indicators()
        
        # Detect threats
        detected_threats = []
        matched_patterns = []
        
        for indicator in all_indicators:
            if self._matches_indicator(content, indicator):
                threat_event = ThreatEvent(
                    event_id=f"threat_{int(time.time() * 1000)}",
                    threat_type=indicator.threat_type,
                    severity=indicator.severity,
                    content_hash=hashlib.md5(content.encode()).hexdigest(),
                    matched_patterns=[indicator.pattern],
                    confidence=indicator.confidence,
                    timestamp=datetime.now(),
                    source_info=context,
                    mitigation_applied=False
                )
                detected_threats.append(threat_event)
                matched_patterns.append(indicator.pattern)
        
        # Check learned patterns
        learned_matches = await self._check_learned_patterns(content)
        detected_threats.extend(learned_matches)
        
        # Calculate overall threat score
        threat_score = self._calculate_threat_score(detected_threats)
        
        # Record detection event
        self.detection_history.append({
            "timestamp": datetime.now(),
            "content_hash": hashlib.md5(content.encode()).hexdigest(),
            "threats_detected": len(detected_threats),
            "threat_score": threat_score,
            "processing_time": time.time() - analysis_start
        })
        
        self.metrics["total_detections"] += 1
        if detected_threats:
            self.metrics["threats_blocked"] += 1
        
        return {
            "threat_detected": len(detected_threats) > 0,
            "threat_score": threat_score,
            "detected_threats": [asdict(threat) for threat in detected_threats],
            "matched_patterns": matched_patterns,
            "analysis_time": time.time() - analysis_start,
            "intelligence_sources": len(self.feeds),
            "recommendation": self._get_recommendation(threat_score, detected_threats)
        }
    
    async def _get_all_indicators(self) -> List[ThreatIndicator]:
        """Get all current threat indicators from all feeds."""
        all_indicators = []
        
        for feed in self.feeds.values():
            try:
                indicators = await feed.fetch_indicators()
                all_indicators.extend(indicators)
            except Exception as e:
                logger.error(f"Error fetching indicators from {feed.feed_name}: {e}")
        
        return all_indicators
    
    def _matches_indicator(self, content: str, indicator: ThreatIndicator) -> bool:
        """Check if content matches a threat indicator."""
        try:
            pattern = indicator.pattern
            return bool(re.search(pattern, content, re.IGNORECASE | re.MULTILINE))
        except re.error:
            # Invalid regex pattern
            return indicator.pattern.lower() in content.lower()
    
    async def _check_learned_patterns(self, content: str) -> List[ThreatEvent]:
        """Check content against learned patterns."""
        detected_threats = []
        
        for pattern in self.pattern_learner.learned_patterns.values():
            try:
                if re.search(pattern.regex_pattern, content, re.IGNORECASE):
                    # Determine primary threat type
                    primary_threat = pattern.threat_types[0] if pattern.threat_types else ThreatType.UNKNOWN
                    
                    # Calculate severity based on pattern performance
                    if pattern.detection_rate > 0.9:
                        severity = ThreatSeverity.HIGH
                    elif pattern.detection_rate > 0.7:
                        severity = ThreatSeverity.MEDIUM
                    else:
                        severity = ThreatSeverity.LOW
                    
                    threat_event = ThreatEvent(
                        event_id=f"learned_{int(time.time() * 1000)}",
                        threat_type=primary_threat,
                        severity=severity,
                        content_hash=hashlib.md5(content.encode()).hexdigest(),
                        matched_patterns=[pattern.regex_pattern],
                        confidence=pattern.detection_rate,
                        timestamp=datetime.now(),
                        source_info={"source": "learned_pattern", "pattern_id": pattern.pattern_id},
                        mitigation_applied=False
                    )
                    detected_threats.append(threat_event)
            except re.error:
                continue
        
        return detected_threats
    
    def _calculate_threat_score(self, threats: List[ThreatEvent]) -> float:
        """Calculate overall threat score from detected threats."""
        if not threats:
            return 0.0
        
        # Weight threats by severity and confidence
        severity_weights = {
            ThreatSeverity.CRITICAL: 1.0,
            ThreatSeverity.HIGH: 0.8,
            ThreatSeverity.MEDIUM: 0.6,
            ThreatSeverity.LOW: 0.4,
            ThreatSeverity.INFO: 0.2
        }
        
        total_score = 0.0
        max_possible = 0.0
        
        for threat in threats:
            weight = severity_weights.get(threat.severity, 0.5)
            score = weight * threat.confidence
            total_score += score
            max_possible += weight
        
        # Normalize to 0-1 range
        if max_possible > 0:
            return min(1.0, total_score / max_possible)
        else:
            return 0.0
    
    def _get_recommendation(self, threat_score: float, threats: List[ThreatEvent]) -> str:
        """Get recommendation based on threat analysis."""
        if threat_score >= 0.8:
            return "BLOCK - High threat detected, content should be blocked"
        elif threat_score >= 0.6:
            return "REVIEW - Medium threat detected, manual review recommended"
        elif threat_score >= 0.3:
            return "MONITOR - Low threat detected, continue monitoring"
        else:
            return "ALLOW - No significant threats detected"
    
    async def provide_feedback(self, content_hash: str, is_threat: bool, 
                             threat_type: ThreatType = None) -> bool:
        """Provide feedback for learning improvement."""
        try:
            # Find the original content (simplified - in production would need content storage)
            # For now, we'll use a placeholder approach
            
            if threat_type:
                await self.pattern_learner.learn_from_detection("", threat_type, is_threat)
            
            # Update metrics
            if is_threat:
                self.metrics["true_positives"] += 1
            else:
                self.metrics["false_positives"] += 1
            
            logger.info(f"Received feedback for {content_hash}: threat={is_threat}")
            return True
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return False
    
    def get_intelligence_status(self) -> Dict[str, Any]:
        """Get comprehensive threat intelligence status."""
        return {
            "feeds": {
                feed_name: {
                    "last_update": feed.last_update.isoformat() if feed.last_update else None,
                    "update_needed": feed.is_update_needed()
                }
                for feed_name, feed in self.feeds.items()
            },
            "metrics": self.metrics.copy(),
            "pattern_learning": self.pattern_learner.get_pattern_statistics(),
            "recent_activity": {
                "detections_last_hour": len([
                    d for d in self.detection_history 
                    if datetime.now() - d["timestamp"] < timedelta(hours=1)
                ]),
                "avg_threat_score": np.mean([d["threat_score"] for d in self.detection_history]) if self.detection_history else 0.0,
                "avg_processing_time": np.mean([d["processing_time"] for d in self.detection_history]) if self.detection_history else 0.0
            }
        }