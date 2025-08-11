"""
Enhanced security features for CoT SafePath Filter.
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
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import json

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
    """Advanced input sanitization and validation."""
    
    def __init__(self):
        # Dangerous patterns that could indicate attacks
        self.dangerous_patterns = [
            # Code injection patterns
            r'<script[^>]*>.*?</script>',
            r'javascript\s*:',
            r'vbscript\s*:',
            r'data\s*:.*?base64',
            r'eval\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\(',
            
            # SQL injection patterns
            r'union\s+select',
            r'or\s+1\s*=\s*1',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+.*?set',
            
            # Command injection patterns
            r';\s*(rm|del|format|sudo)',
            r'\|\s*(rm|del|format|sudo)',
            r'&&\s*(rm|del|format|sudo)',
            r'`.*?`',
            r'\$\([^)]*\)',
            
            # Path traversal patterns
            r'\.\./',
            r'\.\.\\\\',
            r'%2e%2e%2f',
            r'%2e%2e\\',
            
            # File system patterns
            r'/etc/passwd',
            r'c:\\\\windows\\\\system32',
            r'file:///',
            
            # Network patterns
            r'http://[^/]*?/.*?\?.*?=.*?&',
            r'ftp://.*?@',
            
            # Encoding attacks
            r'%[0-9a-f]{2}%[0-9a-f]{2}%[0-9a-f]{2}',
            r'&#x[0-9a-f]+;',
            r'&#[0-9]+;',
        ]
        
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.dangerous_patterns
        ]
        
        # Content length limits
        self.max_content_length = 100_000  # 100KB
        self.max_line_length = 10_000
        
        logger.info(f"Input sanitizer initialized with {len(self.dangerous_patterns)} patterns")
    
    def sanitize_input(self, content: str, context: str = "general") -> Tuple[str, List[str]]:
        """
        Sanitize input content and return cleaned content with violation list.
        
        Args:
            content: Content to sanitize
            context: Context for sanitization (e.g., "user_input", "api_request")
            
        Returns:
            Tuple of (sanitized_content, violations_found)
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
        
        # Check line length
        lines = sanitized.split('\n')
        long_lines = [i for i, line in enumerate(lines) if len(line) > self.max_line_length]
        if long_lines:
            violations.append(f"long_lines:{len(long_lines)}")
            # Truncate long lines
            for line_idx in long_lines:
                lines[line_idx] = lines[line_idx][:self.max_line_length] + "[TRUNCATED]"
            sanitized = '\n'.join(lines)
        
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
            'failed_auth': RateLimiter(max_requests=5, window_seconds=300),  # 5 failed attempts per 5 min
            'suspicious_content': RateLimiter(max_requests=3, window_seconds=60),  # 3 per minute
            'rate_limit_exceeded': RateLimiter(max_requests=1, window_seconds=30),  # 1 per 30 sec
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
        
        # Check for repeated patterns (potential spam/DoS)
        words = content_lower.split()
        if len(words) > 10:
            unique_words = set(words)
            repetition_ratio = len(words) / len(unique_words)
            if repetition_ratio > 10:  # More than 10x repetition
                suspicions.append("excessive_repetition")
        
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


class ContentEncryption:
    """Encrypt sensitive content for storage."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        if encryption_key:
            self.encryption_key = encryption_key
        else:
            # Generate key from environment or create new one
            key_string = os.getenv('ENCRYPTION_KEY')
            if key_string:
                self.encryption_key = base64.urlsafe_b64decode(key_string)
            else:
                self.encryption_key = Fernet.generate_key()
                logger.warning("Generated new encryption key. Set ENCRYPTION_KEY environment variable.")
        
        self.fernet = Fernet(self.encryption_key)
        logger.info("Content encryption initialized")
    
    def encrypt_content(self, content: str) -> str:
        """Encrypt content for secure storage."""
        try:
            encrypted_bytes = self.fernet.encrypt(content.encode('utf-8'))
            return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise SecurityError(f"Content encryption failed: {e}")
    
    def decrypt_content(self, encrypted_content: str) -> str:
        """Decrypt content from storage."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_content.encode('utf-8'))
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise SecurityError(f"Content decryption failed: {e}")
    
    def hash_content(self, content: str, salt: Optional[bytes] = None) -> str:
        """Create a secure hash of content."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        hash_bytes = kdf.derive(content.encode('utf-8'))
        return base64.urlsafe_b64encode(salt + hash_bytes).decode('utf-8')
    
    def verify_content_hash(self, content: str, content_hash: str) -> bool:
        """Verify content against its hash."""
        try:
            hash_bytes = base64.urlsafe_b64decode(content_hash.encode('utf-8'))
            salt = hash_bytes[:32]
            expected_hash = hash_bytes[32:]
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            actual_hash = kdf.derive(content.encode('utf-8'))
            return hmac.compare_digest(expected_hash, actual_hash)
        
        except Exception as e:
            logger.error(f"Hash verification failed: {e}")
            return False


class APIKeyManager:
    """Manage API keys and authentication."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.key_usage: Dict[str, List[datetime]] = {}
        logger.info("API key manager initialized")
    
    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate a new API key for a user."""
        api_key = 'sk-' + secrets.token_urlsafe(32)
        
        self.api_keys[api_key] = {
            'user_id': user_id,
            'permissions': permissions or ['read'],
            'created_at': datetime.utcnow(),
            'last_used': None,
            'active': True,
            'rate_limit': 1000  # requests per hour
        }
        
        logger.info(f"Generated API key for user {user_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return user info if valid."""
        if api_key not in self.api_keys:
            return None
        
        key_info = self.api_keys[api_key]
        
        if not key_info['active']:
            return None
        
        # Update last used timestamp
        key_info['last_used'] = datetime.utcnow()
        
        # Record usage for rate limiting
        if api_key not in self.key_usage:
            self.key_usage[api_key] = []
        
        self.key_usage[api_key].append(datetime.utcnow())
        
        # Clean up old usage records (keep last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.key_usage[api_key] = [
            timestamp for timestamp in self.key_usage[api_key]
            if timestamp >= cutoff
        ]
        
        return key_info
    
    def check_rate_limit(self, api_key: str) -> bool:
        """Check if API key is within rate limits."""
        if api_key not in self.api_keys:
            return False
        
        key_info = self.api_keys[api_key]
        rate_limit = key_info['rate_limit']
        
        current_usage = len(self.key_usage.get(api_key, []))
        return current_usage < rate_limit
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.api_keys:
            self.api_keys[api_key]['active'] = False
            logger.info(f"Revoked API key: {api_key[:10]}...")
            return True
        return False


class SecurityValidator:
    """Comprehensive security validation."""
    
    def __init__(self):
        self.input_sanitizer = InputSanitizer()
        self.security_monitor = SecurityMonitor()
        self.content_encryption = ContentEncryption()
        self.api_key_manager = APIKeyManager()
        
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
            raise SecurityError(f"IP address {source_ip} is blocked")
        
        # Validate API key if provided
        if api_key:
            key_info = self.api_key_manager.validate_api_key(api_key)
            if not key_info:
                self.security_monitor.log_security_event(
                    event_type="invalid_api_key",
                    severity="warning",
                    source_ip=source_ip,
                    details={"api_key_prefix": api_key[:10] if api_key else None}
                )
                raise SecurityError("Invalid API key")
            
            # Check rate limits
            if not self.api_key_manager.check_rate_limit(api_key):
                raise SecurityError("API key rate limit exceeded")
        
        # Detect suspicious content
        suspicions = self.security_monitor.detect_suspicious_content(content, source_ip)
        if suspicions:
            violations.extend(suspicions)
            
            # Block IP if too many suspicious requests
            if not self.security_monitor.check_rate_limits("suspicious_content", source_ip):
                self.security_monitor.block_ip(source_ip, "Excessive suspicious content")
                raise SecurityError("Too many suspicious requests")
        
        # Sanitize input
        sanitized_content, sanitization_violations = self.input_sanitizer.sanitize_input(content)
        violations.extend(sanitization_violations)
        
        return sanitized_content, violations
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status."""
        return {
            "security_monitor": self.security_monitor.get_security_summary(),
            "blocked_ips": len(self.security_monitor.blocked_ips),
            "active_api_keys": len([k for k, v in self.api_key_manager.api_keys.items() if v['active']]),
            "encryption_available": True,
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


def validate_request(
    content: str,
    source_ip: str,
    user_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> Tuple[str, List[str]]:
    """Validate a request using the global security validator."""
    return get_security_validator().validate_request(content, source_ip, user_id, api_key)