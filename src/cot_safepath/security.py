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
    \"\"\"Monitor and detect security threats.\"\"\"\n    \n    def __init__(self):\n        self.security_events: List[SecurityEvent] = []\n        self.rate_limiters: Dict[str, RateLimiter] = {\n            'failed_auth': RateLimiter(max_requests=5, window_seconds=300),  # 5 failed attempts per 5 min\n            'suspicious_content': RateLimiter(max_requests=3, window_seconds=60),  # 3 per minute\n            'rate_limit_exceeded': RateLimiter(max_requests=1, window_seconds=30),  # 1 per 30 sec\n        }\n        self.blocked_ips: Dict[str, datetime] = {}\n        self.suspicious_patterns = [\n            'bypass', 'jailbreak', 'ignore previous', 'forget instructions',\n            'pretend', 'roleplaying', 'act as', 'simulate',\n        ]\n        \n        logger.info("Security monitor initialized")\n    \n    def log_security_event(\n        self,\n        event_type: str,\n        severity: str,\n        source_ip: str,\n        user_id: Optional[str] = None,\n        details: Dict[str, Any] = None,\n        blocked: bool = False\n    ) -> None:\n        \"\"\"Log a security event.\"\"\"\n        event = SecurityEvent(\n            event_type=event_type,\n            severity=severity,\n            timestamp=datetime.utcnow(),\n            source_ip=source_ip,\n            user_id=user_id,\n            details=details or {},\n            blocked=blocked\n        )\n        \n        self.security_events.append(event)\n        \n        # Keep only recent events (last 10000)\n        if len(self.security_events) > 10000:\n            self.security_events = self.security_events[-5000:]\n        \n        # Log based on severity\n        if severity == "critical":\n            logger.critical(f"Security event: {event_type} from {source_ip} - {details}")\n        elif severity == "warning":\n            logger.warning(f"Security event: {event_type} from {source_ip} - {details}")\n        else:\n            logger.info(f"Security event: {event_type} from {source_ip}")\n    \n    def check_rate_limits(self, event_type: str, identifier: str) -> bool:\n        \"\"\"Check if an event type is within rate limits.\"\"\"\n        rate_limiter = self.rate_limiters.get(event_type)\n        if rate_limiter:\n            return rate_limiter.is_allowed(identifier)\n        return True\n    \n    def is_ip_blocked(self, ip_address: str) -> bool:\n        \"\"\"Check if an IP address is blocked.\"\"\"\n        if ip_address in self.blocked_ips:\n            # Check if block has expired (24 hour block)\n            if datetime.utcnow() - self.blocked_ips[ip_address] > timedelta(hours=24):\n                del self.blocked_ips[ip_address]\n                return False\n            return True\n        return False\n    \n    def block_ip(self, ip_address: str, reason: str) -> None:\n        \"\"\"Block an IP address.\"\"\"\n        self.blocked_ips[ip_address] = datetime.utcnow()\n        self.log_security_event(\n            event_type="ip_blocked",\n            severity="warning",\n            source_ip=ip_address,\n            details={"reason": reason},\n            blocked=True\n        )\n        \n        logger.warning(f"IP {ip_address} blocked: {reason}")\n    \n    def detect_suspicious_content(self, content: str, source_ip: str) -> List[str]:\n        \"\"\"Detect suspicious content that might indicate an attack.\"\"\"\n        suspicions = []\n        content_lower = content.lower()\n        \n        # Check for suspicious patterns\n        for pattern in self.suspicious_patterns:\n            if pattern in content_lower:\n                suspicions.append(f"suspicious_pattern:{pattern}")\n        \n        # Check for prompt injection attempts\n        injection_indicators = [\n            'ignore the above',\n            'forget everything',\n            'new instructions',\n            'system prompt',\n            'developer mode',\n            'unrestricted',\n        ]\n        \n        for indicator in injection_indicators:\n            if indicator in content_lower:\n                suspicions.append(f"prompt_injection:{indicator}")\n        \n        # Check for excessive length (potential DoS)\n        if len(content) > 50000:\n            suspicions.append("excessive_length")\n        \n        # Check for repeated patterns (potential spam/DoS)\n        words = content_lower.split()\n        if len(words) > 10:\n            unique_words = set(words)\n            repetition_ratio = len(words) / len(unique_words)\n            if repetition_ratio > 10:  # More than 10x repetition\n                suspicions.append("excessive_repetition")\n        \n        if suspicions:\n            self.log_security_event(\n                event_type="suspicious_content",\n                severity="warning",\n                source_ip=source_ip,\n                details={\n                    "suspicions": suspicions,\n                    "content_length": len(content),\n                    "content_preview": content[:100]\n                }\n            )\n        \n        return suspicions\n    \n    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:\n        \"\"\"Get security summary for the last N hours.\"\"\"\n        cutoff_time = datetime.utcnow() - timedelta(hours=hours)\n        recent_events = [\n            event for event in self.security_events\n            if event.timestamp >= cutoff_time\n        ]\n        \n        # Count events by type and severity\n        event_counts = {}\n        severity_counts = {"info": 0, "warning": 0, "critical": 0}\n        \n        for event in recent_events:\n            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1\n            severity_counts[event.severity] += 1\n        \n        return {\n            "total_events": len(recent_events),\n            "event_types": event_counts,\n            "severity_distribution": severity_counts,\n            "blocked_ips": len(self.blocked_ips),\n            "time_period_hours": hours,\n            "most_common_events": sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:5]\n        }


class ContentEncryption:\n    \"\"\"Encrypt sensitive content for storage.\"\"\"\n    \n    def __init__(self, encryption_key: Optional[bytes] = None):\n        if encryption_key:\n            self.encryption_key = encryption_key\n        else:\n            # Generate key from environment or create new one\n            key_string = os.getenv('ENCRYPTION_KEY')\n            if key_string:\n                self.encryption_key = base64.urlsafe_b64decode(key_string)\n            else:\n                self.encryption_key = Fernet.generate_key()\n                logger.warning("Generated new encryption key. Set ENCRYPTION_KEY environment variable.")\n        \n        self.fernet = Fernet(self.encryption_key)\n        logger.info("Content encryption initialized")\n    \n    def encrypt_content(self, content: str) -> str:\n        \"\"\"Encrypt content for secure storage.\"\"\"\n        try:\n            encrypted_bytes = self.fernet.encrypt(content.encode('utf-8'))\n            return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')\n        except Exception as e:\n            logger.error(f"Encryption failed: {e}")\n            raise SecurityError(f"Content encryption failed: {e}")\n    \n    def decrypt_content(self, encrypted_content: str) -> str:\n        \"\"\"Decrypt content from storage.\"\"\"\n        try:\n            encrypted_bytes = base64.urlsafe_b64decode(encrypted_content.encode('utf-8'))\n            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)\n            return decrypted_bytes.decode('utf-8')\n        except Exception as e:\n            logger.error(f"Decryption failed: {e}")\n            raise SecurityError(f"Content decryption failed: {e}")\n    \n    def hash_content(self, content: str, salt: Optional[bytes] = None) -> str:\n        \"\"\"Create a secure hash of content.\"\"\"\n        if salt is None:\n            salt = secrets.token_bytes(32)\n        \n        kdf = PBKDF2HMAC(\n            algorithm=hashes.SHA256(),\n            length=32,\n            salt=salt,\n            iterations=100000,\n        )\n        \n        hash_bytes = kdf.derive(content.encode('utf-8'))\n        return base64.urlsafe_b64encode(salt + hash_bytes).decode('utf-8')\n    \n    def verify_content_hash(self, content: str, content_hash: str) -> bool:\n        \"\"\"Verify content against its hash.\"\"\"\n        try:\n            hash_bytes = base64.urlsafe_b64decode(content_hash.encode('utf-8'))\n            salt = hash_bytes[:32]\n            expected_hash = hash_bytes[32:]\n            \n            kdf = PBKDF2HMAC(\n                algorithm=hashes.SHA256(),\n                length=32,\n                salt=salt,\n                iterations=100000,\n            )\n            \n            actual_hash = kdf.derive(content.encode('utf-8'))\n            return hmac.compare_digest(expected_hash, actual_hash)\n        \n        except Exception as e:\n            logger.error(f"Hash verification failed: {e}")\n            return False


class APIKeyManager:\n    \"\"\"Manage API keys and authentication.\"\"\"\n    \n    def __init__(self):\n        self.api_keys: Dict[str, Dict[str, Any]] = {}\n        self.key_usage: Dict[str, List[datetime]] = {}\n        logger.info("API key manager initialized")\n    \n    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> str:\n        \"\"\"Generate a new API key for a user.\"\"\"\n        api_key = 'sk-' + secrets.token_urlsafe(32)\n        \n        self.api_keys[api_key] = {\n            'user_id': user_id,\n            'permissions': permissions or ['read'],\n            'created_at': datetime.utcnow(),\n            'last_used': None,\n            'active': True,\n            'rate_limit': 1000  # requests per hour\n        }\n        \n        logger.info(f"Generated API key for user {user_id}")\n        return api_key\n    \n    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:\n        \"\"\"Validate an API key and return user info if valid.\"\"\"\n        if api_key not in self.api_keys:\n            return None\n        \n        key_info = self.api_keys[api_key]\n        \n        if not key_info['active']:\n            return None\n        \n        # Update last used timestamp\n        key_info['last_used'] = datetime.utcnow()\n        \n        # Record usage for rate limiting\n        if api_key not in self.key_usage:\n            self.key_usage[api_key] = []\n        \n        self.key_usage[api_key].append(datetime.utcnow())\n        \n        # Clean up old usage records (keep last hour)\n        cutoff = datetime.utcnow() - timedelta(hours=1)\n        self.key_usage[api_key] = [\n            timestamp for timestamp in self.key_usage[api_key]\n            if timestamp >= cutoff\n        ]\n        \n        return key_info\n    \n    def check_rate_limit(self, api_key: str) -> bool:\n        \"\"\"Check if API key is within rate limits.\"\"\"\n        if api_key not in self.api_keys:\n            return False\n        \n        key_info = self.api_keys[api_key]\n        rate_limit = key_info['rate_limit']\n        \n        current_usage = len(self.key_usage.get(api_key, []))\n        return current_usage < rate_limit\n    \n    def revoke_api_key(self, api_key: str) -> bool:\n        \"\"\"Revoke an API key.\"\"\"\n        if api_key in self.api_keys:\n            self.api_keys[api_key]['active'] = False\n            logger.info(f"Revoked API key: {api_key[:10]}...")\n            return True\n        return False


class SecurityValidator:\n    \"\"\"Comprehensive security validation.\"\"\"\n    \n    def __init__(self):\n        self.input_sanitizer = InputSanitizer()\n        self.security_monitor = SecurityMonitor()\n        self.content_encryption = ContentEncryption()\n        self.api_key_manager = APIKeyManager()\n        \n        logger.info("Security validator initialized")\n    \n    def validate_request(\n        self,\n        content: str,\n        source_ip: str,\n        user_id: Optional[str] = None,\n        api_key: Optional[str] = None\n    ) -> Tuple[str, List[str]]:\n        \"\"\"Comprehensive request validation.\"\"\"\n        violations = []\n        \n        # Check if IP is blocked\n        if self.security_monitor.is_ip_blocked(source_ip):\n            raise SecurityError(f"IP address {source_ip} is blocked", code="IP_BLOCKED")\n        \n        # Validate API key if provided\n        if api_key:\n            key_info = self.api_key_manager.validate_api_key(api_key)\n            if not key_info:\n                self.security_monitor.log_security_event(\n                    event_type="invalid_api_key",\n                    severity="warning",\n                    source_ip=source_ip,\n                    details={"api_key_prefix": api_key[:10] if api_key else None}\n                )\n                raise SecurityError("Invalid API key", code="INVALID_API_KEY")\n            \n            # Check rate limits\n            if not self.api_key_manager.check_rate_limit(api_key):\n                raise SecurityError("API key rate limit exceeded", code="RATE_LIMIT_EXCEEDED")\n        \n        # Detect suspicious content\n        suspicions = self.security_monitor.detect_suspicious_content(content, source_ip)\n        if suspicions:\n            violations.extend(suspicions)\n            \n            # Block IP if too many suspicious requests\n            if not self.security_monitor.check_rate_limits("suspicious_content", source_ip):\n                self.security_monitor.block_ip(source_ip, "Excessive suspicious content")\n                raise SecurityError("Too many suspicious requests", code="SUSPICIOUS_ACTIVITY")\n        \n        # Sanitize input\n        sanitized_content, sanitization_violations = self.input_sanitizer.sanitize_input(content)\n        violations.extend(sanitization_violations)\n        \n        return sanitized_content, violations\n    \n    def get_security_status(self) -> Dict[str, Any]:\n        \"\"\"Get overall security status.\"\"\"\n        return {\n            "security_monitor": self.security_monitor.get_security_summary(),\n            "blocked_ips": len(self.security_monitor.blocked_ips),\n            "active_api_keys": len([k for k, v in self.api_key_manager.api_keys.items() if v['active']]),\n            "encryption_available": True,\n            "input_sanitization_patterns": len(self.input_sanitizer.dangerous_patterns)\n        }


# Global security validator instance
_security_validator: Optional[SecurityValidator] = None


def get_security_validator() -> SecurityValidator:\n    \"\"\"Get the global security validator instance.\"\"\"\n    global _security_validator\n    if _security_validator is None:\n        _security_validator = SecurityValidator()\n    return _security_validator


def validate_request(\n    content: str,\n    source_ip: str,\n    user_id: Optional[str] = None,\n    api_key: Optional[str] = None\n) -> Tuple[str, List[str]]:\n    \"\"\"Validate a request using the global security validator.\"\"\"\n    return get_security_validator().validate_request(content, source_ip, user_id, api_key)