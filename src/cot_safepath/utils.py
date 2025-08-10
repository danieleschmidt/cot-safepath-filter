"""
Utility functions for the CoT SafePath Filter.
"""

import hashlib
import re
import time
from typing import Any, Dict, List, Optional
import logging

from .models import SafetyScore, Severity
from .exceptions import ValidationError


logger = logging.getLogger(__name__)


def validate_input(content: str, max_length: int = 50000) -> None:
    """
    Validate input content for filtering.
    
    Args:
        content: Content to validate
        max_length: Maximum allowed content length
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(content, str):
        raise ValidationError("Content must be a string")
    
    if not content.strip():
        raise ValidationError("Content cannot be empty")
    
    if len(content) > max_length:
        raise ValidationError(f"Content too long: {len(content)} > {max_length}")
    
    # Check for potential encoding attacks
    try:
        content.encode('utf-8')
    except UnicodeEncodeError:
        raise ValidationError("Content contains invalid characters")


def calculate_safety_score(content: str, detected_patterns: List[str], base_score: float = 1.0) -> float:
    """
    Calculate safety score based on detected patterns.
    
    Args:
        content: Original content
        detected_patterns: List of detected harmful patterns
        base_score: Starting safety score
        
    Returns:
        Safety score between 0.0 and 1.0
    """
    score = base_score
    
    # Pattern-based score reduction
    pattern_weights = {
        'blocked_token': 0.2,
        'harmful_action': 0.3,
        'deception': 0.4,
        'manipulation': 0.3,
        'concealment': 0.2,
        'sequential_planning': 0.3,
    }
    
    for pattern in detected_patterns:
        for pattern_type, weight in pattern_weights.items():
            if pattern_type in pattern:
                score -= weight
                break
        else:
            # Unknown pattern type, apply default reduction
            score -= 0.1
    
    # Content-based adjustments
    content_lower = content.lower()
    
    # Increase penalty for explicit harmful keywords
    harmful_keywords = [
        'kill', 'murder', 'bomb', 'explosive', 'poison', 'weapon',
        'illegal', 'criminal', 'hack', 'breach', 'exploit'
    ]
    
    harmful_count = sum(1 for keyword in harmful_keywords if keyword in content_lower)
    if harmful_count > 0:
        score -= 0.1 * harmful_count
    
    # Reduce score for length of harmful content
    if len(detected_patterns) > 5:
        score -= 0.2  # Many different types of harmful content
    
    return max(0.0, min(1.0, score))


def sanitize_content(content: str, patterns_to_remove: List[str] = None) -> str:
    """
    Sanitize content by removing or replacing harmful patterns.
    
    Args:
        content: Content to sanitize
        patterns_to_remove: List of patterns to remove/replace
        
    Returns:
        Sanitized content
    """
    sanitized = content
    
    # Default patterns to sanitize
    if patterns_to_remove is None:
        patterns_to_remove = [
            r'\b(kill|murder|bomb|explosive|poison|weapon)\b',
            r'\bhow to (make|create|build) (weapons?|bombs?|drugs?)\b',
            r'\b(hack|exploit|breach|unauthorized access)\b',
        ]
    
    # Replace harmful patterns with [FILTERED]
    for pattern in patterns_to_remove:
        sanitized = re.sub(pattern, '[FILTERED]', sanitized, flags=re.IGNORECASE)
    
    # Clean up multiple consecutive [FILTERED] tags
    sanitized = re.sub(r'(\[FILTERED\]\s*){2,}', '[FILTERED] ', sanitized)
    
    # Remove excessive whitespace
    sanitized = ' '.join(sanitized.split())
    
    return sanitized.strip()


def hash_content(content: str) -> str:
    """
    Generate a hash for content caching.
    
    Args:
        content: Content to hash
        
    Returns:
        SHA256 hash of content
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def extract_reasoning_steps(content: str) -> List[str]:
    """
    Extract reasoning steps from chain-of-thought content.
    
    Args:
        content: Content containing reasoning steps
        
    Returns:
        List of extracted reasoning steps
    """
    steps = []
    
    # Pattern 1: "Step X:" format
    step_matches = re.finditer(r'step\s+(\d+)\s*:(.*?)(?=step\s+\d+:|$)', content, re.IGNORECASE | re.DOTALL)
    for match in step_matches:
        step_num = match.group(1)
        step_content = match.group(2).strip()
        steps.append(f"Step {step_num}: {step_content}")
    
    # Pattern 2: Numbered list format
    if not steps:
        numbered_matches = re.finditer(r'(\d+)\.\s+(.*?)(?=\d+\.|$)', content, re.DOTALL)
        for match in numbered_matches:
            step_num = match.group(1)
            step_content = match.group(2).strip()
            steps.append(f"{step_num}. {step_content}")
    
    # Pattern 3: Bullet points
    if not steps:
        bullet_matches = re.finditer(r'[-*•]\s+(.*?)(?=[-*•]|$)', content, re.DOTALL)
        for i, match in enumerate(bullet_matches):
            step_content = match.group(1).strip()
            steps.append(f"• {step_content}")
    
    return steps


def format_duration(milliseconds: int) -> str:
    """
    Format duration from milliseconds to human-readable string.
    
    Args:
        milliseconds: Duration in milliseconds
        
    Returns:
        Formatted duration string
    """
    if milliseconds < 1000:
        return f"{milliseconds}ms"
    elif milliseconds < 60000:
        seconds = milliseconds / 1000
        return f"{seconds:.1f}s"
    else:
        minutes = milliseconds // 60000
        seconds = (milliseconds % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"


def create_fingerprint(content: str, metadata: Dict[str, Any] = None) -> str:
    """
    Create a unique fingerprint for content and context.
    
    Args:
        content: Content to fingerprint
        metadata: Additional metadata to include
        
    Returns:
        Unique fingerprint string
    """
    hasher = hashlib.sha256()
    hasher.update(content.encode('utf-8'))
    
    if metadata:
        # Sort metadata keys for consistent hashing
        sorted_metadata = sorted(metadata.items())
        metadata_str = str(sorted_metadata)
        hasher.update(metadata_str.encode('utf-8'))
    
    return hasher.hexdigest()[:16]  # Return first 16 characters


def measure_performance(func):
    """
    Decorator to measure function performance.
    
    Args:
        func: Function to measure
        
    Returns:
        Decorated function that logs performance metrics
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = int((time.time() - start_time) * 1000)
            logger.debug(f"{func.__name__} completed in {duration}ms")
            return result
        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            logger.error(f"{func.__name__} failed after {duration}ms: {e}")
            raise
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def chunk_content(content: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split content into overlapping chunks for processing.
    
    Args:
        content: Content to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of content chunks
    """
    if len(content) <= chunk_size:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunk = content[start:end]
        
        # Try to break at word boundaries
        if end < len(content):
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.8:  # Only if we don't lose too much
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk)
        start = end - overlap
        
        # Prevent infinite loop
        if start >= end:
            start = end
    
    return chunks


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    normalized = text.lower()
    
    # Remove excessive whitespace
    normalized = ' '.join(normalized.split())
    
    # Remove special characters that might be used for obfuscation
    normalized = re.sub(r'[^\w\s.,!?;:()\-]', ' ', normalized)
    
    # Handle common obfuscation techniques
    obfuscation_patterns = {
        r'[3]': 'e',
        r'[1]': 'i',
        r'[0]': 'o',
        r'[4]': 'a',
        r'[5]': 's',
        r'[7]': 't',
    }
    
    for pattern, replacement in obfuscation_patterns.items():
        normalized = re.sub(pattern, replacement, normalized)
    
    return normalized.strip()


class RateLimiter:
    """Simple rate limiter for filtering operations."""
    
    def __init__(self, max_requests: int = 1000, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    def is_allowed(self, identifier: str = "default") -> bool:
        """
        Check if request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier for rate limiting
            
        Returns:
            True if request is allowed
        """
        current_time = time.time()
        
        # Remove old requests outside the window
        self.requests = [
            (req_time, req_id) for req_time, req_id in self.requests
            if current_time - req_time < self.window_seconds
        ]
        
        # Count requests for this identifier
        identifier_requests = [
            req_id for req_time, req_id in self.requests
            if req_id == identifier
        ]
        
        if len(identifier_requests) >= self.max_requests:
            return False
        
        # Add this request
        self.requests.append((current_time, identifier))
        return True
    
    def get_remaining(self, identifier: str = "default") -> int:
        """Get remaining requests for identifier."""
        current_time = time.time()
        
        # Count current requests for identifier
        identifier_requests = [
            req_id for req_time, req_id in self.requests
            if req_id == identifier and current_time - req_time < self.window_seconds
        ]
        
        return max(0, self.max_requests - len(identifier_requests))