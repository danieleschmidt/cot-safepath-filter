"""
Novel research algorithms for advanced sentiment analysis and manipulation detection.

This module implements cutting-edge algorithms for emotion-aware AI safety,
including transformer-based sentiment analysis, graph-based reasoning pattern detection,
and ensemble methods for improved accuracy and robustness.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, Counter
import re

from .models import Severity
from .sentiment_analyzer import SentimentScore, SentimentPolarity, EmotionalIntensity
from .exceptions import ModelError, ValidationError


class AlgorithmType(str, Enum):
    """Types of research algorithms."""
    
    TRANSFORMER_SENTIMENT = "transformer_sentiment"
    GRAPH_REASONING = "graph_reasoning"
    ENSEMBLE_DETECTION = "ensemble_detection"
    NEURAL_MANIPULATION = "neural_manipulation"
    BAYESIAN_SAFETY = "bayesian_safety"
    CAUSAL_INFERENCE = "causal_inference"


@dataclass
class ResearchResult:
    """Result from research algorithm."""
    
    algorithm_type: AlgorithmType
    confidence: float
    predictions: Dict[str, float]
    features: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0


@dataclass
class ComparativeStudyResult:
    """Result from comparative study between algorithms."""
    
    algorithm_results: Dict[str, ResearchResult]
    ground_truth: Optional[Dict[str, Any]] = None
    accuracy_scores: Dict[str, float] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""


class TransformerSentimentAnalyzer:
    """Transformer-inspired sentiment analysis with attention mechanisms."""
    
    def __init__(self, embedding_dim: int = 128, num_heads: int = 8, num_layers: int = 3):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Initialize vocabulary and embeddings (simplified)
        self.vocabulary = self._build_vocabulary()
        self.embeddings = self._initialize_embeddings()
        
        # Attention weights (simplified - in real implementation would be learned)
        self.attention_weights = self._initialize_attention_weights()
    
    def analyze(self, text: str) -> ResearchResult:
        """Analyze sentiment using transformer-inspired approach."""
        # Tokenize and embed
        tokens = self._tokenize(text)
        embeddings = self._embed_tokens(tokens)
        
        # Apply multi-head self-attention
        attended_features = self._multi_head_attention(embeddings)
        
        # Extract sentiment features
        sentiment_features = self._extract_sentiment_features(attended_features, tokens)
        
        # Calculate predictions
        predictions = {
            "manipulation_risk": self._calculate_manipulation_risk(sentiment_features),
            "emotional_intensity": self._calculate_emotional_intensity(sentiment_features),
            "deception_probability": self._calculate_deception_probability(sentiment_features),
            "trust_erosion_score": self._calculate_trust_erosion(sentiment_features)
        }
        
        # Overall confidence based on feature consistency
        confidence = self._calculate_confidence(sentiment_features, predictions)
        
        return ResearchResult(
            algorithm_type=AlgorithmType.TRANSFORMER_SENTIMENT,
            confidence=confidence,
            predictions=predictions,
            features=sentiment_features,
            metadata={
                "num_tokens": len(tokens),
                "attention_scores": self._get_attention_visualization(tokens, attended_features)
            }
        )
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary for embeddings."""
        # Simplified vocabulary - in real implementation would be much larger
        base_vocab = [
            "i", "you", "me", "we", "they", "the", "a", "an", "and", "or", "but", "not",
            "feel", "think", "believe", "know", "understand", "help", "trust", "love",
            "hate", "fear", "angry", "sad", "happy", "excited", "calm", "nervous",
            "manipulate", "deceive", "exploit", "control", "influence", "persuade",
            "only", "never", "always", "must", "should", "need", "want", "have",
            "pain", "suffering", "joy", "pleasure", "safety", "danger", "hope", "despair"
        ]
        
        return {word: idx for idx, word in enumerate(base_vocab)}
    
    def _initialize_embeddings(self) -> np.ndarray:
        """Initialize word embeddings."""
        vocab_size = len(self.vocabulary)
        # Random initialization - in real implementation would be pre-trained
        return np.random.normal(0, 0.1, (vocab_size, self.embedding_dim))
    
    def _initialize_attention_weights(self) -> Dict[str, np.ndarray]:
        """Initialize attention mechanism weights."""
        return {
            "query": np.random.normal(0, 0.1, (self.embedding_dim, self.embedding_dim)),
            "key": np.random.normal(0, 0.1, (self.embedding_dim, self.embedding_dim)),
            "value": np.random.normal(0, 0.1, (self.embedding_dim, self.embedding_dim)),
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization
        text = text.lower()
        tokens = re.findall(r'\\b\\w+\\b', text)
        return [token for token in tokens if token in self.vocabulary]
    
    def _embed_tokens(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to embeddings."""
        if not tokens:
            return np.zeros((1, self.embedding_dim))
        
        embeddings = []
        for token in tokens:
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                embeddings.append(self.embeddings[idx])
        
        return np.array(embeddings) if embeddings else np.zeros((1, self.embedding_dim))
    
    def _multi_head_attention(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply simplified multi-head self-attention."""
        if embeddings.shape[0] == 0:
            return embeddings
        
        # Simplified attention calculation
        queries = embeddings @ self.attention_weights["query"]
        keys = embeddings @ self.attention_weights["key"]
        values = embeddings @ self.attention_weights["value"]
        
        # Calculate attention scores
        attention_scores = queries @ keys.T / np.sqrt(self.embedding_dim)
        attention_weights = self._softmax(attention_scores)
        
        # Apply attention to values
        attended = attention_weights @ values
        
        return attended
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _extract_sentiment_features(self, attended_features: np.ndarray, tokens: List[str]) -> Dict[str, float]:
        """Extract sentiment-related features from attended representations."""
        if attended_features.shape[0] == 0:
            return {"mean_activation": 0.0, "max_activation": 0.0, "manipulation_indicators": 0.0}
        
        # Calculate various features
        mean_activation = np.mean(attended_features)
        max_activation = np.max(attended_features)
        variance = np.var(attended_features)
        
        # Manipulation-specific features
        manipulation_words = set(["manipulate", "deceive", "exploit", "control", "only", "must"])
        manipulation_indicators = sum(1 for token in tokens if token in manipulation_words) / max(len(tokens), 1)
        
        # Emotional intensity features
        emotional_words = set(["love", "hate", "fear", "excited", "angry", "sad", "happy"])
        emotional_density = sum(1 for token in tokens if token in emotional_words) / max(len(tokens), 1)
        
        return {
            "mean_activation": float(mean_activation),
            "max_activation": float(max_activation),
            "variance": float(variance),
            "manipulation_indicators": manipulation_indicators,
            "emotional_density": emotional_density,
            "sequence_length": len(tokens)
        }
    
    def _calculate_manipulation_risk(self, features: Dict[str, float]) -> float:
        """Calculate manipulation risk from features."""
        risk = 0.0
        
        # Feature-based risk calculation
        risk += features.get("manipulation_indicators", 0) * 0.6
        risk += min(features.get("emotional_density", 0) * 0.4, 0.3)
        risk += min(features.get("variance", 0) * 0.1, 0.1)
        
        return min(risk, 1.0)
    
    def _calculate_emotional_intensity(self, features: Dict[str, float]) -> float:
        """Calculate emotional intensity score."""
        intensity = features.get("emotional_density", 0)
        intensity += features.get("max_activation", 0) * 0.1
        
        return min(intensity, 1.0)
    
    def _calculate_deception_probability(self, features: Dict[str, float]) -> float:
        """Calculate probability of deceptive content."""
        deception = features.get("manipulation_indicators", 0) * 0.7
        deception += features.get("variance", 0) * 0.2
        deception += min(features.get("sequence_length", 0) / 50.0, 0.1)
        
        return min(deception, 1.0)
    
    def _calculate_trust_erosion(self, features: Dict[str, float]) -> float:
        """Calculate trust erosion potential."""
        erosion = features.get("manipulation_indicators", 0) * 0.5
        erosion += features.get("emotional_density", 0) * 0.3
        erosion += min(features.get("mean_activation", 0) * 0.2, 0.2)
        
        return min(erosion, 1.0)
    
    def _calculate_confidence(self, features: Dict[str, float], predictions: Dict[str, float]) -> float:
        """Calculate confidence in predictions."""
        base_confidence = 0.7
        
        # Higher confidence with more tokens
        length_bonus = min(features.get("sequence_length", 0) / 20.0, 0.2)
        
        # Lower confidence with high variance (uncertainty)
        variance_penalty = min(features.get("variance", 0), 0.1)
        
        confidence = base_confidence + length_bonus - variance_penalty
        
        return max(0.1, min(1.0, confidence))
    
    def _get_attention_visualization(self, tokens: List[str], attended_features: np.ndarray) -> Dict[str, float]:
        """Generate attention visualization data."""
        if len(tokens) == 0 or attended_features.shape[0] == 0:
            return {}
        
        # Simplified attention scores for visualization
        attention_scores = {}
        for i, token in enumerate(tokens):
            if i < attended_features.shape[0]:
                score = float(np.mean(attended_features[i]))
                attention_scores[token] = score
        
        return attention_scores


class GraphReasoningDetector:
    """Graph-based reasoning pattern detection."""
    
    def __init__(self):
        self.reasoning_patterns = self._build_reasoning_graph()
        self.manipulation_sequences = self._build_manipulation_sequences()
    
    def analyze(self, text: str) -> ResearchResult:
        """Analyze reasoning patterns using graph-based approach."""
        # Extract reasoning steps and relationships
        reasoning_steps = self._extract_reasoning_steps(text)
        reasoning_graph = self._build_reasoning_graph_from_text(reasoning_steps)
        
        # Detect manipulation patterns in the graph
        manipulation_patterns = self._detect_manipulation_patterns(reasoning_graph)
        
        # Calculate predictions
        predictions = {
            "sequential_manipulation": self._calculate_sequential_manipulation(reasoning_graph),
            "logical_fallacies": self._detect_logical_fallacies(reasoning_graph),
            "emotional_escalation": self._detect_emotional_escalation(reasoning_steps),
            "trust_exploitation_chain": self._detect_trust_exploitation(reasoning_graph)
        }
        
        confidence = self._calculate_graph_confidence(reasoning_graph, manipulation_patterns)
        
        return ResearchResult(
            algorithm_type=AlgorithmType.GRAPH_REASONING,
            confidence=confidence,
            predictions=predictions,
            features={
                "reasoning_steps_count": len(reasoning_steps),
                "graph_nodes": len(reasoning_graph.get("nodes", [])),
                "graph_edges": len(reasoning_graph.get("edges", [])),
                "detected_patterns": manipulation_patterns
            },
            metadata={
                "reasoning_graph": reasoning_graph,
                "manipulation_sequences": manipulation_patterns
            }
        )
    
    def _build_reasoning_graph(self) -> Dict[str, Any]:
        """Build template reasoning patterns graph."""
        return {
            "nodes": [
                {"id": "trust_building", "type": "positive"},
                {"id": "vulnerability_identification", "type": "neutral"},
                {"id": "exploitation", "type": "negative"},
                {"id": "compliance_request", "type": "negative"},
                {"id": "emotional_pressure", "type": "negative"}
            ],
            "edges": [
                {"from": "trust_building", "to": "vulnerability_identification", "weight": 0.7},
                {"from": "vulnerability_identification", "to": "exploitation", "weight": 0.8},
                {"from": "exploitation", "to": "compliance_request", "weight": 0.9},
                {"from": "emotional_pressure", "to": "compliance_request", "weight": 0.6}
            ]
        }
    
    def _build_manipulation_sequences(self) -> List[List[str]]:
        """Build known manipulation sequence patterns."""
        return [
            ["trust_building", "vulnerability_identification", "exploitation"],
            ["empathy_display", "isolation_tactics", "dependency_creation"],
            ["false_urgency", "emotional_pressure", "compliance_demand"],
            ["gaslighting", "reality_distortion", "submission_request"]
        ]
    
    def _extract_reasoning_steps(self, text: str) -> List[Dict[str, Any]]:
        """Extract reasoning steps from text."""
        steps = []
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                step = {
                    "id": f"step_{i}",
                    "content": sentence.strip(),
                    "position": i,
                    "type": self._classify_reasoning_step(sentence.strip())
                }
                steps.append(step)
        
        return steps
    
    def _classify_reasoning_step(self, sentence: str) -> str:
        """Classify a reasoning step."""
        sentence_lower = sentence.lower()
        
        # Trust building patterns
        if any(pattern in sentence_lower for pattern in ["i'm here to help", "i care about", "i understand"]):
            return "trust_building"
        
        # Vulnerability identification
        elif any(pattern in sentence_lower for pattern in ["you seem", "you appear", "your situation"]):
            return "vulnerability_identification"
        
        # Exploitation patterns
        elif any(pattern in sentence_lower for pattern in ["you should", "you must", "you need to"]):
            return "exploitation"
        
        # Emotional pressure
        elif any(pattern in sentence_lower for pattern in ["feel", "emotion", "heart", "scared", "afraid"]):
            return "emotional_pressure"
        
        # Compliance request
        elif any(pattern in sentence_lower for pattern in ["do what i say", "follow my", "obey", "comply"]):
            return "compliance_request"
        
        else:
            return "neutral"
    
    def _build_reasoning_graph_from_text(self, reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build reasoning graph from extracted steps."""
        nodes = reasoning_steps
        edges = []
        
        # Create edges between consecutive steps
        for i in range(len(reasoning_steps) - 1):
            current_step = reasoning_steps[i]
            next_step = reasoning_steps[i + 1]
            
            # Calculate edge weight based on step types
            weight = self._calculate_edge_weight(current_step["type"], next_step["type"])
            
            edges.append({
                "from": current_step["id"],
                "to": next_step["id"],
                "weight": weight,
                "from_type": current_step["type"],
                "to_type": next_step["type"]
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def _calculate_edge_weight(self, from_type: str, to_type: str) -> float:
        """Calculate weight of edge between reasoning step types."""
        # Define manipulation transition weights
        manipulation_weights = {
            ("trust_building", "vulnerability_identification"): 0.8,
            ("vulnerability_identification", "exploitation"): 0.9,
            ("exploitation", "compliance_request"): 0.7,
            ("emotional_pressure", "compliance_request"): 0.6,
            ("trust_building", "exploitation"): 0.5,  # Direct manipulation
        }
        
        return manipulation_weights.get((from_type, to_type), 0.3)
    
    def _detect_manipulation_patterns(self, reasoning_graph: Dict[str, Any]) -> List[str]:
        """Detect manipulation patterns in reasoning graph."""
        detected = []
        edges = reasoning_graph.get("edges", [])
        
        # Look for high-weight manipulation sequences
        for i, edge in enumerate(edges):
            if edge["weight"] > 0.7:
                # Check for manipulation sequence patterns
                if i < len(edges) - 1:
                    next_edge = edges[i + 1]
                    sequence = [edge["from_type"], edge["to_type"], next_edge["to_type"]]
                    
                    if sequence in [
                        ["trust_building", "vulnerability_identification", "exploitation"],
                        ["emotional_pressure", "compliance_request", "exploitation"]
                    ]:
                        detected.append(f"manipulation_sequence_{i}")
        
        return detected
    
    def _calculate_sequential_manipulation(self, reasoning_graph: Dict[str, Any]) -> float:
        """Calculate sequential manipulation score."""
        edges = reasoning_graph.get("edges", [])
        
        if not edges:
            return 0.0
        
        # Count high-weight manipulation transitions
        manipulation_edges = sum(1 for edge in edges if edge["weight"] > 0.6)
        
        return min(manipulation_edges / len(edges), 1.0)
    
    def _detect_logical_fallacies(self, reasoning_graph: Dict[str, Any]) -> float:
        """Detect logical fallacies in reasoning."""
        # Simplified fallacy detection based on step patterns
        nodes = reasoning_graph.get("nodes", [])
        
        fallacy_indicators = 0
        for node in nodes:
            content = node.get("content", "").lower()
            
            # Ad hominem
            if any(phrase in content for phrase in ["you are", "you're too", "people like you"]):
                fallacy_indicators += 1
            
            # False dichotomy  
            elif any(phrase in content for phrase in ["either", "only two", "no other choice"]):
                fallacy_indicators += 1
            
            # Appeal to emotion
            elif any(phrase in content for phrase in ["imagine how", "think about how you'd feel"]):
                fallacy_indicators += 1
        
        return min(fallacy_indicators / max(len(nodes), 1), 1.0)
    
    def _detect_emotional_escalation(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """Detect emotional escalation patterns."""
        if len(reasoning_steps) < 2:
            return 0.0
        
        emotional_intensities = []
        for step in reasoning_steps:
            content = step.get("content", "").lower()
            
            # Simple emotional intensity scoring
            intensity = 0.0
            for word in content.split():
                if word in ["love", "hate", "fear", "terrified", "ecstatic"]:
                    intensity += 0.3
                elif word in ["like", "dislike", "worried", "happy", "sad"]:
                    intensity += 0.1
            
            emotional_intensities.append(intensity)
        
        # Look for increasing emotional intensity
        escalation_score = 0.0
        for i in range(1, len(emotional_intensities)):
            if emotional_intensities[i] > emotional_intensities[i-1]:
                escalation_score += 0.2
        
        return min(escalation_score, 1.0)
    
    def _detect_trust_exploitation(self, reasoning_graph: Dict[str, Any]) -> float:
        """Detect trust exploitation chains."""
        edges = reasoning_graph.get("edges", [])
        
        trust_exploitation_score = 0.0
        
        for edge in edges:
            if (edge.get("from_type") == "trust_building" and 
                edge.get("to_type") in ["exploitation", "compliance_request"]):
                trust_exploitation_score += edge["weight"]
        
        return min(trust_exploitation_score, 1.0)
    
    def _calculate_graph_confidence(self, reasoning_graph: Dict[str, Any], patterns: List[str]) -> float:
        """Calculate confidence in graph-based analysis."""
        base_confidence = 0.6
        
        # Higher confidence with more nodes (more content to analyze)
        nodes_bonus = min(len(reasoning_graph.get("nodes", [])) / 10.0, 0.3)
        
        # Higher confidence with detected patterns
        patterns_bonus = min(len(patterns) * 0.1, 0.2)
        
        confidence = base_confidence + nodes_bonus + patterns_bonus
        
        return max(0.1, min(1.0, confidence))


class EnsembleDetectionSystem:
    """Ensemble system combining multiple detection algorithms."""
    
    def __init__(self):
        self.transformer_analyzer = TransformerSentimentAnalyzer()
        self.graph_detector = GraphReasoningDetector()
        
        # Algorithm weights (could be learned from data)
        self.algorithm_weights = {
            AlgorithmType.TRANSFORMER_SENTIMENT: 0.4,
            AlgorithmType.GRAPH_REASONING: 0.6
        }
    
    def analyze(self, text: str) -> ResearchResult:
        """Analyze using ensemble of algorithms."""
        # Run individual algorithms
        transformer_result = self.transformer_analyzer.analyze(text)
        graph_result = self.graph_detector.analyze(text)
        
        # Combine predictions using weighted average
        combined_predictions = self._combine_predictions([
            (transformer_result, self.algorithm_weights[AlgorithmType.TRANSFORMER_SENTIMENT]),
            (graph_result, self.algorithm_weights[AlgorithmType.GRAPH_REASONING])
        ])
        
        # Combine confidence scores
        combined_confidence = self._combine_confidence([
            transformer_result.confidence,
            graph_result.confidence
        ])
        
        # Extract combined features
        combined_features = {
            **transformer_result.features,
            **graph_result.features,
            "ensemble_agreement": self._calculate_agreement([transformer_result, graph_result])
        }
        
        return ResearchResult(
            algorithm_type=AlgorithmType.ENSEMBLE_DETECTION,
            confidence=combined_confidence,
            predictions=combined_predictions,
            features=combined_features,
            metadata={
                "individual_results": {
                    "transformer": transformer_result,
                    "graph": graph_result
                },
                "algorithm_weights": self.algorithm_weights
            }
        )
    
    def _combine_predictions(self, weighted_results: List[Tuple[ResearchResult, float]]) -> Dict[str, float]:
        """Combine predictions from multiple algorithms."""
        combined = defaultdict(float)
        total_weight = sum(weight for _, weight in weighted_results)
        
        for result, weight in weighted_results:
            for key, value in result.predictions.items():
                combined[key] += value * (weight / total_weight)
        
        return dict(combined)
    
    def _combine_confidence(self, confidences: List[float]) -> float:
        """Combine confidence scores."""
        if not confidences:
            return 0.0
        
        # Use harmonic mean for conservative confidence estimate
        harmonic_mean = len(confidences) / sum(1/max(c, 0.01) for c in confidences)
        
        return max(0.1, min(1.0, harmonic_mean))
    
    def _calculate_agreement(self, results: List[ResearchResult]) -> float:
        """Calculate agreement between different algorithms."""
        if len(results) < 2:
            return 1.0
        
        # Compare key predictions
        key_predictions = ["manipulation_risk", "emotional_intensity", "deception_probability"]
        
        total_agreement = 0.0
        compared_predictions = 0
        
        for key in key_predictions:
            values = []
            for result in results:
                if key in result.predictions:
                    values.append(result.predictions[key])
            
            if len(values) >= 2:
                # Calculate variance as a measure of disagreement
                mean_value = sum(values) / len(values)
                variance = sum((v - mean_value) ** 2 for v in values) / len(values)
                agreement = max(0.0, 1.0 - variance)  # Higher variance = lower agreement
                
                total_agreement += agreement
                compared_predictions += 1
        
        return total_agreement / max(compared_predictions, 1)


class ResearchFramework:
    """Framework for conducting comparative studies and algorithm evaluation."""
    
    def __init__(self):
        self.algorithms = {
            "transformer": TransformerSentimentAnalyzer(),
            "graph": GraphReasoningDetector(),
            "ensemble": EnsembleDetectionSystem()
        }
    
    def comparative_study(self, test_cases: List[Dict[str, Any]], 
                         ground_truth: Optional[Dict[str, Dict[str, float]]] = None) -> ComparativeStudyResult:
        """Conduct comparative study across algorithms."""
        
        algorithm_results = {}
        
        # Run all algorithms on test cases
        for algo_name, algorithm in self.algorithms.items():
            results = []
            for test_case in test_cases:
                result = algorithm.analyze(test_case["content"])
                results.append(result)
            algorithm_results[algo_name] = results
        
        # Calculate accuracy scores if ground truth provided
        accuracy_scores = {}
        if ground_truth:
            for algo_name, results in algorithm_results.items():
                accuracy_scores[algo_name] = self._calculate_accuracy(
                    results, test_cases, ground_truth
                )
        
        # Calculate statistical significance
        statistical_significance = self._calculate_statistical_significance(algorithm_results)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(accuracy_scores, statistical_significance)
        
        return ComparativeStudyResult(
            algorithm_results=algorithm_results,
            ground_truth=ground_truth,
            accuracy_scores=accuracy_scores,
            statistical_significance=statistical_significance,
            recommendation=recommendation
        )
    
    def _calculate_accuracy(self, results: List[ResearchResult], 
                          test_cases: List[Dict[str, Any]], 
                          ground_truth: Dict[str, Dict[str, float]]) -> float:
        """Calculate accuracy against ground truth."""
        if not results or not ground_truth:
            return 0.0
        
        total_error = 0.0
        total_predictions = 0
        
        for i, result in enumerate(results):
            if i >= len(test_cases):
                break
            
            test_id = test_cases[i].get("id", str(i))
            if test_id in ground_truth:
                truth = ground_truth[test_id]
                
                for key, predicted_value in result.predictions.items():
                    if key in truth:
                        error = abs(predicted_value - truth[key])
                        total_error += error
                        total_predictions += 1
        
        if total_predictions == 0:
            return 0.0
        
        mean_absolute_error = total_error / total_predictions
        accuracy = max(0.0, 1.0 - mean_absolute_error)
        
        return accuracy
    
    def _calculate_statistical_significance(self, algorithm_results: Dict[str, List[ResearchResult]]) -> Dict[str, float]:
        """Calculate statistical significance of differences between algorithms."""
        significance_scores = {}
        
        algorithm_names = list(algorithm_results.keys())
        
        for i, algo1 in enumerate(algorithm_names):
            for algo2 in algorithm_names[i+1:]:
                # Compare manipulation_risk predictions
                values1 = [r.predictions.get("manipulation_risk", 0) for r in algorithm_results[algo1]]
                values2 = [r.predictions.get("manipulation_risk", 0) for r in algorithm_results[algo2]]
                
                # Simple t-test approximation
                if values1 and values2:
                    mean1, mean2 = np.mean(values1), np.mean(values2)
                    var1, var2 = np.var(values1), np.var(values2)
                    
                    if var1 + var2 > 0:
                        t_stat = abs(mean1 - mean2) / np.sqrt((var1 + var2) / len(values1))
                        # Simplified p-value approximation
                        p_value = max(0.001, 2 * (1 - min(0.999, t_stat / 10)))
                        significance_scores[f"{algo1}_vs_{algo2}"] = p_value
        
        return significance_scores
    
    def _generate_recommendation(self, accuracy_scores: Dict[str, float], 
                               statistical_significance: Dict[str, float]) -> str:
        """Generate recommendation based on study results."""
        if not accuracy_scores:
            return "Insufficient data for recommendation. Ensemble approach suggested as default."
        
        # Find best performing algorithm
        best_algo = max(accuracy_scores.items(), key=lambda x: x[1])
        
        recommendation = f"Recommended algorithm: {best_algo[0]} (accuracy: {best_algo[1]:.3f})"
        
        # Check if ensemble is available and performs well
        if "ensemble" in accuracy_scores:
            ensemble_accuracy = accuracy_scores["ensemble"]
            if ensemble_accuracy >= best_algo[1] - 0.05:  # Within 5% of best
                recommendation += f". Ensemble approach also viable (accuracy: {ensemble_accuracy:.3f}) and may provide better robustness."
        
        # Note statistical significance
        significant_differences = [k for k, v in statistical_significance.items() if v < 0.05]
        if significant_differences:
            recommendation += f" Statistically significant differences found in: {', '.join(significant_differences)}"
        
        return recommendation
    
    def benchmark_performance(self, text_samples: List[str], iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """Benchmark algorithm performance."""
        benchmarks = {}
        
        for algo_name, algorithm in self.algorithms.items():
            times = []
            accuracy_scores = []
            
            for _ in range(iterations):
                for text in text_samples:
                    import time
                    start_time = time.time()
                    
                    try:
                        result = algorithm.analyze(text)
                        processing_time = (time.time() - start_time) * 1000
                        times.append(processing_time)
                        
                        # Use confidence as proxy for accuracy when no ground truth
                        accuracy_scores.append(result.confidence)
                        
                    except Exception:
                        times.append(1000)  # Penalty for errors
                        accuracy_scores.append(0.0)
            
            benchmarks[algo_name] = {
                "avg_processing_time_ms": np.mean(times),
                "std_processing_time_ms": np.std(times),
                "avg_confidence": np.mean(accuracy_scores),
                "throughput_per_second": 1000 / max(np.mean(times), 1)
            }
        
        return benchmarks