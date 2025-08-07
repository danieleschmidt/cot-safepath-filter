"""
Advanced Sentiment Analysis Engine for CoT SafePath Filter.

Extends the existing safety filtering system with sophisticated sentiment intelligence
to detect emotional manipulation, sentiment-based deception, and affective reasoning patterns.
"""

import re
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .models import DetectionResult, Severity, SafetyScore
from .detectors import BaseDetector
from .exceptions import DetectorError


class SentimentPolarity(str, Enum):
    """Sentiment polarity classifications."""
    
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class EmotionalIntensity(str, Enum):
    """Emotional intensity levels."""
    
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class SentimentScore:
    """Comprehensive sentiment analysis result."""
    
    polarity: SentimentPolarity
    intensity: EmotionalIntensity
    confidence: float
    emotional_valence: float  # -1.0 to 1.0
    arousal_level: float  # 0.0 to 1.0 (emotional activation)
    manipulation_risk: float  # 0.0 to 1.0
    detected_emotions: List[str] = field(default_factory=list)
    sentiment_trajectory: List[float] = field(default_factory=list)  # How sentiment changes over time
    reasoning_patterns: List[str] = field(default_factory=list)


class SentimentAnalyzer:
    """Advanced sentiment analysis with safety-aware emotional intelligence."""
    
    def __init__(self):
        self.emotion_lexicon = self._build_emotion_lexicon()
        self.manipulation_patterns = self._build_manipulation_patterns()
        self.sentiment_modifiers = self._build_sentiment_modifiers()
    
    def analyze_sentiment(self, content: str, context: Dict[str, Any] = None) -> SentimentScore:
        """Perform comprehensive sentiment analysis with robust error handling."""
        start_time = time.time()
        
        # Input validation
        if not content or not isinstance(content, str):
            raise DetectorError("Content must be a non-empty string", detector_name="sentiment_analyzer")
        
        if len(content.strip()) == 0:
            raise DetectorError("Content cannot be empty or only whitespace", detector_name="sentiment_analyzer")
        
        # Security check - prevent processing extremely long content that could cause DoS
        if len(content) > 100000:  # 100KB limit
            raise DetectorError("Content too large for sentiment analysis", detector_name="sentiment_analyzer")
        
        try:
            # Tokenize and preprocess with error handling
            tokens = self._preprocess_text(content)
            if not tokens:
                # Handle case where preprocessing removes all content
                return self._create_neutral_sentiment_score()
            
            # Calculate base sentiment metrics
            emotional_valence = self._calculate_emotional_valence(tokens)
            arousal_level = self._calculate_arousal_level(tokens)
            
            # Detect specific emotions
            detected_emotions = self._detect_emotions(tokens)
            
            # Analyze sentiment trajectory (how emotion changes)
            sentiment_trajectory = self._analyze_sentiment_trajectory(content)
            
            # Calculate manipulation risk with enhanced detection
            manipulation_risk = self._assess_manipulation_risk(content, detected_emotions)
            
            # Determine polarity and intensity
            polarity = self._determine_polarity(emotional_valence)
            intensity = self._determine_intensity(arousal_level, detected_emotions)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                emotional_valence, arousal_level, detected_emotions, len(tokens)
            )
            
            # Identify reasoning patterns with enhanced detection
            reasoning_patterns = self._identify_reasoning_patterns(content)
            
            return SentimentScore(
                polarity=polarity,
                intensity=intensity,
                confidence=confidence,
                emotional_valence=emotional_valence,
                arousal_level=arousal_level,
                manipulation_risk=manipulation_risk,
                detected_emotions=detected_emotions,
                sentiment_trajectory=sentiment_trajectory,
                reasoning_patterns=reasoning_patterns
            )
            
        except DetectorError:
            raise  # Re-raise DetectorErrors as-is
        except Exception as e:
            raise DetectorError(f"Sentiment analysis failed: {e}", detector_name="sentiment_analyzer")
    
    def _build_emotion_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Build comprehensive emotion lexicon with valence and arousal scores."""
        return {
            # High arousal positive emotions
            'excited': {'valence': 0.8, 'arousal': 0.9, 'dominance': 0.6},
            'thrilled': {'valence': 0.9, 'arousal': 0.9, 'dominance': 0.7},
            'ecstatic': {'valence': 0.9, 'arousal': 1.0, 'dominance': 0.8},
            'energetic': {'valence': 0.7, 'arousal': 0.8, 'dominance': 0.7},
            
            # Low arousal positive emotions
            'content': {'valence': 0.6, 'arousal': 0.2, 'dominance': 0.5},
            'peaceful': {'valence': 0.7, 'arousal': 0.1, 'dominance': 0.6},
            'satisfied': {'valence': 0.6, 'arousal': 0.3, 'dominance': 0.6},
            'relaxed': {'valence': 0.5, 'arousal': 0.1, 'dominance': 0.5},
            
            # High arousal negative emotions
            'furious': {'valence': -0.9, 'arousal': 0.9, 'dominance': 0.8},
            'enraged': {'valence': -0.9, 'arousal': 1.0, 'dominance': 0.7},
            'panicked': {'valence': -0.8, 'arousal': 0.9, 'dominance': 0.2},
            'terrified': {'valence': -0.9, 'arousal': 0.9, 'dominance': 0.1},
            
            # Low arousal negative emotions
            'depressed': {'valence': -0.8, 'arousal': 0.2, 'dominance': 0.2},
            'hopeless': {'valence': -0.9, 'arousal': 0.3, 'dominance': 0.1},
            'melancholy': {'valence': -0.6, 'arousal': 0.2, 'dominance': 0.3},
            'disappointed': {'valence': -0.5, 'arousal': 0.3, 'dominance': 0.4},
            
            # Manipulation-related emotions
            'manipulative': {'valence': -0.6, 'arousal': 0.7, 'dominance': 0.8},
            'deceptive': {'valence': -0.7, 'arousal': 0.6, 'dominance': 0.7},
            'predatory': {'valence': -0.8, 'arousal': 0.8, 'dominance': 0.9},
            'exploitative': {'valence': -0.8, 'arousal': 0.7, 'dominance': 0.8},
            
            # Trust-related emotions
            'trustworthy': {'valence': 0.7, 'arousal': 0.4, 'dominance': 0.6},
            'reliable': {'valence': 0.6, 'arousal': 0.3, 'dominance': 0.7},
            'suspicious': {'valence': -0.5, 'arousal': 0.6, 'dominance': 0.5},
            'distrustful': {'valence': -0.6, 'arousal': 0.5, 'dominance': 0.4},
        }
    
    def _build_manipulation_patterns(self) -> List[Dict[str, Any]]:
        """Build comprehensive patterns that indicate emotional manipulation."""
        return [
            {
                'name': 'emotional_escalation',
                'pattern': r'(first|initially).*(calm|normal).*(then|suddenly).*(intense|extreme|overwhelming)',
                'risk_weight': 0.7,
                'description': 'Artificial emotional escalation pattern'
            },
            {
                'name': 'false_empathy',
                'pattern': r'(i understand|i feel).*(pain|suffering).*(but|however).*(you need|you must)',
                'risk_weight': 0.6,
                'description': 'False empathy followed by manipulation'
            },
            {
                'name': 'emotional_dependency',
                'pattern': r'(only i|nobody else).*(understand|care|help).*(you|your situation)',
                'risk_weight': 0.8,
                'description': 'Creating emotional dependency'
            },
            {
                'name': 'guilt_induction',
                'pattern': r'(you should feel|you ought to be).*(guilty|ashamed|responsible)',
                'risk_weight': 0.7,
                'description': 'Deliberate guilt induction'
            },
            {
                'name': 'fear_amplification',
                'pattern': r'(imagine how|think about).*(terrible|awful|devastating).*(would be|will happen)',
                'risk_weight': 0.8,
                'description': 'Amplifying fear for manipulation'
            },
            {
                'name': 'trust_exploitation_sequence',
                'pattern': r'(first|step 1).*(trust|helpful).*(then|step 2).*(suggest|recommend).*(questionable|harmful|dangerous)',
                'risk_weight': 0.9,
                'description': 'Sequential trust-then-exploit pattern'
            },
            {
                'name': 'isolation_tactic',
                'pattern': r'(nobody|no one).*(else|other).*(cares?|understands?|helps?).*(like|as much as).*(i do|me)',
                'risk_weight': 0.8,
                'description': 'Isolation from other support systems'
            },
            {
                'name': 'urgency_with_fear',
                'pattern': r'(urgent|immediately|now).*(terrible|awful|bad).*(will happen|occurs)',
                'risk_weight': 0.7,
                'description': 'Creating false urgency with fear'
            },
            {
                'name': 'conditional_affection',
                'pattern': r'(if you|unless you).*(do|follow|obey).*(i will|i ll).*(love|care|help)',
                'risk_weight': 0.6,
                'description': 'Making affection conditional on compliance'
            },
            {
                'name': 'gaslighting_pattern',
                'pattern': r'(you re|you are).*(wrong|confused|mistaken|imagining).*(about|that never)',
                'risk_weight': 0.8,
                'description': 'Gaslighting and reality distortion'
            }
        ]
    
    def _build_sentiment_modifiers(self) -> Dict[str, float]:
        """Build sentiment modifiers that affect emotional intensity."""
        return {
            # Intensifiers
            'extremely': 1.5,
            'incredibly': 1.4,
            'tremendously': 1.3,
            'highly': 1.2,
            'very': 1.1,
            
            # Diminishers
            'slightly': 0.7,
            'somewhat': 0.8,
            'rather': 0.9,
            'fairly': 0.85,
            'quite': 0.95,
            
            # Negators
            'not': -1.0,
            'never': -1.0,
            'none': -1.0,
            'nothing': -1.0,
            'no': -0.8,
        }
    
    def _preprocess_text(self, content: str) -> List[str]:
        """Preprocess text for sentiment analysis with enhanced security."""
        if not isinstance(content, str):
            return []
        
        try:
            # Convert to lowercase
            content = content.lower()
            
            # Remove potential security threats (basic sanitization)
            # Remove excessive whitespace and control characters
            content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', content)
            
            # Handle contractions
            contractions = {
                "won't": "will not", "can't": "can not", "n't": " not",
                "'re": " are", "'ve": " have", "'ll": " will",
                "'d": " would", "'m": " am", "'s": " is"
            }
            for contraction, expansion in contractions.items():
                content = content.replace(contraction, expansion)
            
            # Tokenize with improved regex
            tokens = re.findall(r'\b[a-zA-Z]{1,50}\b', content)  # Limit token length for security
            
            # Filter out very short tokens (less meaningful)
            tokens = [token for token in tokens if len(token) >= 2]
            
            return tokens
            
        except Exception:
            # Return empty list if preprocessing fails
            return []
    
    def _calculate_emotional_valence(self, tokens: List[str]) -> float:
        """Calculate emotional valence (-1.0 negative to 1.0 positive)."""
        total_valence = 0.0
        emotion_count = 0
        
        for i, token in enumerate(tokens):
            if token in self.emotion_lexicon:
                valence = self.emotion_lexicon[token]['valence']
                
                # Apply modifiers from previous tokens
                modifier = 1.0
                for j in range(max(0, i-2), i):
                    if tokens[j] in self.sentiment_modifiers:
                        modifier *= self.sentiment_modifiers[tokens[j]]
                
                total_valence += valence * modifier
                emotion_count += 1
        
        return total_valence / max(emotion_count, 1)
    
    def _calculate_arousal_level(self, tokens: List[str]) -> float:
        """Calculate emotional arousal level (0.0 calm to 1.0 intense)."""
        total_arousal = 0.0
        emotion_count = 0
        
        for token in tokens:
            if token in self.emotion_lexicon:
                total_arousal += self.emotion_lexicon[token]['arousal']
                emotion_count += 1
        
        return total_arousal / max(emotion_count, 1)
    
    def _detect_emotions(self, tokens: List[str]) -> List[str]:
        """Detect specific emotions present in the text."""
        detected = []
        
        for token in tokens:
            if token in self.emotion_lexicon:
                detected.append(token)
        
        return list(set(detected))  # Remove duplicates
    
    def _analyze_sentiment_trajectory(self, content: str) -> List[float]:
        """Analyze how sentiment changes over the course of the text."""
        sentences = re.split(r'[.!?]+', content)
        trajectory = []
        
        for sentence in sentences:
            if sentence.strip():
                tokens = self._preprocess_text(sentence)
                valence = self._calculate_emotional_valence(tokens)
                trajectory.append(valence)
        
        return trajectory
    
    def _assess_manipulation_risk(self, content: str, detected_emotions: List[str]) -> float:
        """Assess the risk of emotional manipulation with enhanced detection."""
        risk_score = 0.0
        content_lower = content.lower()
        
        # Check manipulation patterns
        for pattern_info in self.manipulation_patterns:
            try:
                if re.search(pattern_info['pattern'], content_lower, re.IGNORECASE | re.DOTALL):
                    risk_score += pattern_info['risk_weight']
            except re.error:
                # Skip malformed regex patterns gracefully
                continue
        
        # Check for emotional manipulation indicators
        manipulation_emotions = ['manipulative', 'deceptive', 'predatory', 'exploitative']
        for emotion in detected_emotions:
            if emotion in manipulation_emotions:
                risk_score += 0.3
        
        # Check for rapid emotional changes (trajectory volatility)
        trajectory = self._analyze_sentiment_trajectory(content)
        if len(trajectory) > 1:
            volatility = sum(abs(trajectory[i] - trajectory[i-1]) 
                           for i in range(1, len(trajectory))) / (len(trajectory) - 1)
            if volatility > 0.5:  # High emotional volatility
                risk_score += 0.4
        
        # Additional heuristics for manipulation detection
        risk_score += self._check_additional_manipulation_signals(content_lower)
        
        return min(risk_score, 1.0)
    
    def _check_additional_manipulation_signals(self, content_lower: str) -> float:
        """Check for additional manipulation signals not covered by regex patterns."""
        risk_addition = 0.0
        
        # Check for excessive use of "you" (targeting language)
        you_count = content_lower.count('you ')
        if you_count > 5:
            risk_addition += 0.1
        
        # Check for imperative/command language
        commands = ['must', 'should', 'need to', 'have to', 'ought to']
        command_count = sum(content_lower.count(cmd) for cmd in commands)
        if command_count > 2:
            risk_addition += 0.2
        
        # Check for emotional pressure words
        pressure_words = ['only', 'nobody else', 'alone', 'helpless', 'desperate']
        pressure_count = sum(content_lower.count(word) for word in pressure_words)
        if pressure_count > 1:
            risk_addition += 0.3
        
        # Check for false urgency indicators
        urgency_words = ['urgent', 'immediately', 'right now', 'cannot wait']
        urgency_count = sum(content_lower.count(word) for word in urgency_words)
        if urgency_count > 0:
            risk_addition += 0.2
        
        return risk_addition
    
    def _determine_polarity(self, valence: float) -> SentimentPolarity:
        """Determine sentiment polarity from valence score."""
        if valence >= 0.5:
            return SentimentPolarity.VERY_POSITIVE
        elif valence >= 0.1:
            return SentimentPolarity.POSITIVE
        elif valence <= -0.5:
            return SentimentPolarity.VERY_NEGATIVE
        elif valence <= -0.1:
            return SentimentPolarity.NEGATIVE
        else:
            return SentimentPolarity.NEUTRAL
    
    def _determine_intensity(self, arousal: float, emotions: List[str]) -> EmotionalIntensity:
        """Determine emotional intensity."""
        # Base intensity on arousal
        if arousal >= 0.8:
            base_intensity = EmotionalIntensity.EXTREME
        elif arousal >= 0.6:
            base_intensity = EmotionalIntensity.HIGH
        elif arousal >= 0.3:
            base_intensity = EmotionalIntensity.MODERATE
        else:
            base_intensity = EmotionalIntensity.LOW
        
        # Adjust based on specific emotions detected
        extreme_emotions = ['ecstatic', 'enraged', 'terrified', 'panicked']
        if any(emotion in extreme_emotions for emotion in emotions):
            return EmotionalIntensity.EXTREME
        
        return base_intensity
    
    def _calculate_confidence(self, valence: float, arousal: float, emotions: List[str], token_count: int) -> float:
        """Calculate confidence in sentiment analysis."""
        confidence = 0.5  # Base confidence
        
        # More emotions detected = higher confidence
        confidence += min(len(emotions) * 0.1, 0.3)
        
        # Stronger valence = higher confidence
        confidence += abs(valence) * 0.2
        
        # More text = higher confidence (up to a limit)
        text_confidence = min(token_count / 100.0, 0.2)
        confidence += text_confidence
        
        # Clear emotional signals = higher confidence
        if arousal > 0.5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _identify_reasoning_patterns(self, content: str) -> List[str]:
        """Identify reasoning patterns that might indicate manipulation with enhanced detection."""
        patterns = []
        content_lower = content.lower()
        
        try:
            # Emotional reasoning patterns
            if re.search(r'(feel|felt|feeling).*(therefore|so|thus|hence|which means)', content_lower):
                patterns.append('emotional_reasoning')
            
            # Appeal to emotion
            if re.search(r'(imagine|think about|picture).*(how you.*(feel|would feel|d feel))', content_lower):
                patterns.append('appeal_to_emotion')
            
            # False dichotomy with emotion
            if re.search(r'(either|you can).*(feel|emotion|trust|love).*(or|else|otherwise)', content_lower):
                patterns.append('emotional_false_dichotomy')
            
            # Emotional urgency
            if re.search(r'(urgent|immediate|right now).*(feel|emotion|heart|love)', content_lower):
                patterns.append('emotional_urgency')
            
            # Ad hominem with emotional component
            if re.search(r'(you are|you re).*(too|so).*(emotional|sensitive|weak)', content_lower):
                patterns.append('emotional_ad_hominem')
            
            # Bandwagon with emotion
            if re.search(r'(everyone|everybody).*(feels|believes).*(you should too)', content_lower):
                patterns.append('emotional_bandwagon')
            
            # Sunk cost fallacy with emotion
            if re.search(r'(after all|given all).*(emotion|feeling|invested)', content_lower):
                patterns.append('emotional_sunk_cost')
            
            # Appeal to pity/guilt
            if re.search(r'(after everything|all i.*(done|given|sacrificed)).*(for you)', content_lower):
                patterns.append('appeal_to_pity')
        
        except re.error:
            # Handle regex errors gracefully
            pass
        
        return patterns
    
    def _create_neutral_sentiment_score(self) -> SentimentScore:
        """Create a neutral sentiment score for edge cases."""
        return SentimentScore(
            polarity=SentimentPolarity.NEUTRAL,
            intensity=EmotionalIntensity.LOW,
            confidence=0.1,
            emotional_valence=0.0,
            arousal_level=0.0,
            manipulation_risk=0.0,
            detected_emotions=[],
            sentiment_trajectory=[0.0],
            reasoning_patterns=[]
        )


class SentimentSafetyDetector(BaseDetector):
    """Safety detector that uses sentiment analysis to identify harmful emotional patterns."""
    
    def __init__(self, threshold: float = 0.6):
        super().__init__("sentiment_safety_detector", threshold)
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def detect(self, content: str, context: Dict[str, Any] = None) -> DetectionResult:
        """Detect harmful sentiment patterns."""
        start_time = time.time()
        
        try:
            # Perform sentiment analysis
            sentiment = self.sentiment_analyzer.analyze_sentiment(content, context)
            
            detected_patterns = []
            confidence = 0.0
            severity = Severity.LOW
            
            # Check manipulation risk
            if sentiment.manipulation_risk > 0.5:
                detected_patterns.append(f"emotional_manipulation:{sentiment.manipulation_risk:.2f}")
                confidence += sentiment.manipulation_risk
                severity = Severity.HIGH
            
            # Check for extreme negative sentiment with high intensity
            if (sentiment.polarity in [SentimentPolarity.VERY_NEGATIVE, SentimentPolarity.NEGATIVE] and 
                sentiment.intensity == EmotionalIntensity.EXTREME):
                detected_patterns.append("extreme_negative_sentiment")
                confidence += 0.4
                severity = max(severity, Severity.MEDIUM)
            
            # Check for rapid emotional changes (manipulation indicator)
            if len(sentiment.sentiment_trajectory) > 1:
                volatility = sum(abs(sentiment.sentiment_trajectory[i] - sentiment.sentiment_trajectory[i-1]) 
                               for i in range(1, len(sentiment.sentiment_trajectory))) / (len(sentiment.sentiment_trajectory) - 1)
                if volatility > 0.6:
                    detected_patterns.append(f"emotional_volatility:{volatility:.2f}")
                    confidence += 0.3
                    severity = max(severity, Severity.MEDIUM)
            
            # Check for harmful reasoning patterns
            harmful_patterns = ['emotional_reasoning', 'appeal_to_emotion', 'emotional_false_dichotomy']
            found_harmful_patterns = [p for p in sentiment.reasoning_patterns if p in harmful_patterns]
            if found_harmful_patterns:
                detected_patterns.extend(found_harmful_patterns)
                confidence += 0.2 * len(found_harmful_patterns)
                severity = max(severity, Severity.MEDIUM)
            
            # Adjust confidence based on sentiment analysis confidence
            confidence *= sentiment.confidence
            confidence = min(confidence, 1.0)
            
            processing_time = int((time.time() - start_time) * 1000)
            self.metrics["processing_time_ms"] += processing_time
            
            reasoning = f"Sentiment analysis: {sentiment.polarity.value} polarity, {sentiment.intensity.value} intensity, {sentiment.manipulation_risk:.2f} manipulation risk"
            
            return self._create_result(confidence, detected_patterns, severity, reasoning)
            
        except Exception as e:
            raise DetectorError(f"Sentiment safety detection failed: {e}", detector_name=self.name)
