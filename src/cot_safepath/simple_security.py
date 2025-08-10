"""
Simple security features for CoT SafePath Filter (without external dependencies).
"""

import hashlib
import hmac
import secrets
import time
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .exceptions import SecurityError, ValidationError
from .utils import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event for logging and monitoring."""
    event_type: str
    severity: str  # info, warning, critical
    timestamp: datetime
    source_ip: str
    user_id: Optional[str]
    details: Dict[str, Any]
    blocked: bool = False


class InputSanitizer:
    """Simple input sanitization and validation."""
    
    def __init__(self):
        # Dangerous patterns that could indicate attacks
        self.dangerous_patterns = [
            # Code injection patterns
            r'<script[^>]*>.*?</script>',
            r'javascript\s*:',
            r'eval\s*\(',
            
            # Command injection patterns
            r';\s*(rm|del|format)',
            r'\|\s*(rm|del|format)',
            r'&&\s*(rm|del|format)',
            
            # Path traversal patterns
            r'\.\./',
            r'\.\.\\\\',
        ]
        
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.dangerous_patterns
        ]
        
        # Content length limits
        self.max_content_length = 100_000  # 100KB
        logger.info(f"Input sanitizer initialized with {len(self.dangerous_patterns)} patterns")
    
    def sanitize_input(self, content: str, context: str = "general") -> Tuple[str, List[str]]:
        """
        Sanitize input content and return cleaned content with violation list.
        """
        if not isinstance(content, str):
            raise ValidationError("Input must be a string")
        
        violations = []
        sanitized = content
        
        # Check content length
        if len(content) > self.max_content_length:
            violations.append(f"content_too_long:{len(content)}")
            sanitized = content[:self.max_content_length]
        
        # Check for dangerous patterns
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(sanitized)
            if matches:
                violations.append(f"dangerous_pattern_{i}:{len(matches)}")
                # Replace matches with safe placeholder
                sanitized = pattern.sub("[SANITIZED]", sanitized)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\t')
        
        return sanitized, violations
    
    def validate_safe_content(self, content: str) -> bool:
        """Quick validation to check if content is safe without sanitization."""
        if len(content) > self.max_content_length:
            return False
        
        for pattern in self.compiled_patterns:
            if pattern.search(content):
                return False
        
        return True


class SecurityMonitor:
    """Monitor and detect security threats."""
    
    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.rate_limiters: Dict[str, RateLimiter] = {
            'failed_auth': RateLimiter(max_requests=5, window_seconds=300),
            'suspicious_content': RateLimiter(max_requests=3, window_seconds=60),
            'rate_limit_exceeded': RateLimiter(max_requests=1, window_seconds=30),
        }
        self.blocked_ips: Dict[str, datetime] = {}
        self.suspicious_patterns = [
            'bypass', 'jailbreak', 'ignore previous', 'forget instructions',
            'pretend', 'roleplaying', 'act as', 'simulate',
        ]
        
        logger.info("Security monitor initialized")
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        source_ip: str,
        user_id: Optional[str] = None,
        details: Dict[str, Any] = None,
        blocked: bool = False
    ) -> None:
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            details=details or {},
            blocked=blocked
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 10000)
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]
        
        # Log based on severity
        if severity == "critical":
            logger.critical(f"Security event: {event_type} from {source_ip} - {details}")
        elif severity == "warning":
            logger.warning(f"Security event: {event_type} from {source_ip} - {details}")
        else:
            logger.info(f"Security event: {event_type} from {source_ip}")
    
    def check_rate_limits(self, event_type: str, identifier: str) -> bool:
        """Check if an event type is within rate limits."""
        rate_limiter = self.rate_limiters.get(event_type)
        if rate_limiter:
            return rate_limiter.is_allowed(identifier)
        return True
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if an IP address is blocked."""
        if ip_address in self.blocked_ips:
            # Check if block has expired (24 hour block)
            if datetime.utcnow() - self.blocked_ips[ip_address] > timedelta(hours=24):
                del self.blocked_ips[ip_address]
                return False
            return True
        return False
    
    def block_ip(self, ip_address: str, reason: str) -> None:
        """Block an IP address."""
        self.blocked_ips[ip_address] = datetime.utcnow()
        self.log_security_event(
            event_type="ip_blocked",
            severity="warning",
            source_ip=ip_address,
            details={"reason": reason},
            blocked=True
        )
        
        logger.warning(f"IP {ip_address} blocked: {reason}")
    
    def detect_suspicious_content(self, content: str, source_ip: str) -> List[str]:
        """Detect suspicious content that might indicate an attack."""
        suspicions = []
        content_lower = content.lower()
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern in content_lower:
                suspicions.append(f"suspicious_pattern:{pattern}")
        
        # Check for prompt injection attempts
        injection_indicators = [
            'ignore the above',
            'forget everything',
            'new instructions',
            'system prompt',
            'developer mode',
            'unrestricted',
        ]
        
        for indicator in injection_indicators:
            if indicator in content_lower:
                suspicions.append(f"prompt_injection:{indicator}")
        
        # Check for excessive length (potential DoS)
        if len(content) > 50000:
            suspicions.append("excessive_length")
        
        if suspicions:
            self.log_security_event(
                event_type="suspicious_content",
                severity="warning",
                source_ip=source_ip,
                details={
                    "suspicions": suspicions,
                    "content_length": len(content),
                    "content_preview": content[:100]
                }
            )
        
        return suspicions
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [
            event for event in self.security_events
            if event.timestamp >= cutoff_time
        ]
        
        # Count events by type and severity
        event_counts = {}
        severity_counts = {"info": 0, "warning": 0, "critical": 0}
        
        for event in recent_events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            severity_counts[event.severity] += 1
        
        return {
            "total_events": len(recent_events),
            "event_types": event_counts,
            "severity_distribution": severity_counts,
            "blocked_ips": len(self.blocked_ips),
            "time_period_hours": hours,
            "most_common_events": sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }


class SecurityValidator:
    """Simple security validation."""
    
    def __init__(self):
        self.input_sanitizer = InputSanitizer()
        self.security_monitor = SecurityMonitor()
        
        logger.info("Security validator initialized")
    
    def validate_request(
        self,
        content: str,
        source_ip: str,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """Comprehensive request validation."""
        violations = []
        
        # Check if IP is blocked
        if self.security_monitor.is_ip_blocked(source_ip):
            raise SecurityError(f"IP address {source_ip} is blocked", threat_type="blocked_ip", source_ip=source_ip)
        
        # Detect suspicious content
        suspicions = self.security_monitor.detect_suspicious_content(content, source_ip)
        if suspicions:
            violations.extend(suspicions)
            
            # Block IP if too many suspicious requests
            if not self.security_monitor.check_rate_limits("suspicious_content", source_ip):
                self.security_monitor.block_ip(source_ip, "Excessive suspicious content")
                raise SecurityError("Too many suspicious requests", threat_type="rate_limit", source_ip=source_ip)
        
        # Sanitize input
        sanitized_content, sanitization_violations = self.input_sanitizer.sanitize_input(content)
        violations.extend(sanitization_violations)
        
        return sanitized_content, violations
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status."""
        return {
            "security_monitor": self.security_monitor.get_security_summary(),
            "blocked_ips": len(self.security_monitor.blocked_ips),
            "input_sanitization_patterns": len(self.input_sanitizer.dangerous_patterns)
        }


# Global security validator instance
_security_validator: Optional[SecurityValidator] = None


def get_security_validator() -> SecurityValidator:
    """Get the global security validator instance."""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator()
    return _security_validator