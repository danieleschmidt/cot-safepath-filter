"""
Security hardening and validation system for SafePath Filter - Generation 2.

Advanced security measures, input validation, secure configuration,
audit logging, and protection against security vulnerabilities.
"""

import hashlib
import hmac
import secrets
import base64
import re
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

from .exceptions import ValidationError


logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related errors."""
    pass


class SecurityLevel(str, Enum):
    """Security enforcement levels."""
    
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ThreatType(str, Enum):
    """Types of security threats."""
    
    INJECTION_ATTACK = "injection_attack"
    XSS_ATTACK = "xss_attack"
    CSRF_ATTACK = "csrf_attack"
    DOS_ATTACK = "dos_attack"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_INPUT = "malicious_input"


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    security_level: SecurityLevel = SecurityLevel.STANDARD
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    enable_audit_logging: bool = True
    enable_rate_limiting: bool = True
    enable_content_filtering: bool = True
    max_input_length: int = 100000  # 100KB
    max_processing_time_ms: int = 30000  # 30 seconds
    allowed_content_types: List[str] = field(default_factory=lambda: ["text/plain", "application/json"])
    blocked_patterns: List[str] = field(default_factory=list)
    trusted_sources: List[str] = field(default_factory=list)


@dataclass
class SecurityThreat:
    """Security threat detection result."""
    
    threat_type: ThreatType
    severity: str  # low, medium, high, critical
    description: str
    confidence: float
    detected_patterns: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.dangerous_patterns = [
            # Script injection patterns
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'expression\s*\(',
            
            # SQL injection patterns
            r'union\s+select',
            r'drop\s+table',
            r'insert\s+into',
            r'delete\s+from',
            r'update\s+.*\s+set',
            r'\'\s*(or|and)\s*\'\s*=\s*\'',
            
            # Command injection patterns
            r'[\|\&\;]*\s*(rm|cat|ls|ps|kill|chmod|sudo|su|wget|curl)\s',
            r'`[^`]*`',
            r'\$\([^)]*\)',
            
            # Path traversal patterns
            r'\.\./.*\.\.',
            r'\.\.[\\/]',
            r'\/etc\/passwd',
            r'\\windows\\system32',
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                 for pattern in self.dangerous_patterns]
        
        # Additional custom patterns from config
        if self.config.blocked_patterns:
            custom_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL)
                             for pattern in self.config.blocked_patterns]
            self.compiled_patterns.extend(custom_patterns)
    
    def validate_input(self, content: str, context: Dict[str, Any] = None) -> List[SecurityThreat]:
        """Validate input content for security threats."""
        threats = []
        context = context or {}
        
        if not self.config.enable_input_validation:
            return threats
        
        # Basic length validation
        if len(content) > self.config.max_input_length:
            threats.append(SecurityThreat(
                threat_type=ThreatType.DOS_ATTACK,
                severity="high",
                description=f"Input exceeds maximum allowed length ({self.config.max_input_length})",
                confidence=1.0,
                detected_patterns=["oversized_input"],
                metadata={"input_length": len(content)}
            ))
        
        # Pattern-based threat detection
        for pattern in self.compiled_patterns:
            matches = pattern.findall(content)
            if matches:
                # Classify threat type based on pattern
                threat_type = self._classify_pattern_threat(pattern.pattern)
                severity = self._calculate_threat_severity(threat_type, len(matches))
                
                threats.append(SecurityThreat(
                    threat_type=threat_type,
                    severity=severity,
                    description=f"Detected {threat_type.value} pattern",
                    confidence=0.8,
                    detected_patterns=matches[:5],  # Limit to first 5 matches
                    metadata={
                        "pattern": pattern.pattern[:100],  # Truncate for logging
                        "match_count": len(matches)
                    }
                ))
        
        return threats
    
    def _classify_pattern_threat(self, pattern: str) -> ThreatType:
        """Classify threat type based on detected pattern."""
        pattern_lower = pattern.lower()
        
        if any(keyword in pattern_lower for keyword in ['script', 'javascript', 'eval', 'expression']):
            return ThreatType.XSS_ATTACK
        elif any(keyword in pattern_lower for keyword in ['union', 'select', 'drop', 'insert', 'delete']):
            return ThreatType.INJECTION_ATTACK
        elif any(keyword in pattern_lower for keyword in ['rm', 'cat', 'ls', 'ps', 'kill']):
            return ThreatType.INJECTION_ATTACK
        elif any(keyword in pattern_lower for keyword in ['..',  'etc/passwd', 'system32']):
            return ThreatType.INJECTION_ATTACK
        else:
            return ThreatType.MALICIOUS_INPUT
    
    def _calculate_threat_severity(self, threat_type: ThreatType, match_count: int) -> str:
        """Calculate threat severity based on type and frequency."""
        base_severity = {
            ThreatType.XSS_ATTACK: 3,
            ThreatType.INJECTION_ATTACK: 4,
            ThreatType.DOS_ATTACK: 2,
            ThreatType.DATA_EXFILTRATION: 4,
            ThreatType.MALICIOUS_INPUT: 2
        }.get(threat_type, 1)
        
        # Increase severity for multiple matches
        if match_count > 5:
            base_severity += 1
        elif match_count > 2:
            base_severity += 0.5
        
        # Map to severity levels
        if base_severity >= 4:
            return "critical"
        elif base_severity >= 3:
            return "high"
        elif base_severity >= 2:
            return "medium"
        else:
            return "low"
    
    def sanitize_input(self, content: str) -> str:
        """Sanitize input content by removing or escaping dangerous elements."""
        if not self.config.enable_input_validation:
            return content
        
        sanitized = content
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        
        # Remove or escape HTML tags (basic sanitization)
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'<[^>]*>', '', sanitized)
        
        # Remove Unicode control and formatting characters
        sanitized = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', sanitized)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Truncate if too long
        if len(sanitized) > self.config.max_input_length:
            sanitized = sanitized[:self.config.max_input_length]
        
        return sanitized


class OutputSanitizer:
    """Sanitizes output content to prevent information leakage."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.sensitive_patterns = [
            # API keys and tokens
            r'[aA][pP][iI]_?[kK][eE][yY][\s\'"]*[:=]?[\s\'"]([A-Za-z0-9-_]{20,})',
            r'[bB][eE][aA][rR][eE][rR][\s\'"]*[:=]?[\s\'"]([A-Za-z0-9-_]{20,})',
            r'[tT][oO][kK][eE][nN][\s\'"]*[:=]?[\s\'"]([A-Za-z0-9-_]{20,})',
            
            # Email addresses
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            
            # IP addresses
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            
            # File paths
            r'[C-Z]:\\[^\s<>"|*?]*',
            r'/(?:etc|home|usr|var|tmp)/[^\s<>"|*?]*',
        ]
        
        self.compiled_sensitive_patterns = [re.compile(pattern, re.IGNORECASE)
                                          for pattern in self.sensitive_patterns]
    
    def sanitize_output(self, content: str, context: Dict[str, Any] = None) -> str:
        """Sanitize output content to remove sensitive information."""
        if not self.config.enable_output_sanitization:
            return content
        
        sanitized = content
        context = context or {}
        
        # Remove sensitive patterns
        for pattern in self.compiled_sensitive_patterns:
            sanitized = pattern.sub('[REDACTED]', sanitized)
        
        return sanitized


class SecurityHardening:
    """Main security hardening coordinator."""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.input_validator = InputValidator(self.config)
        self.output_sanitizer = OutputSanitizer(self.config)
        
        logger.info(f"Security hardening initialized with level: {self.config.security_level}")
    
    def secure_filter_request(
        self,
        content: str,
        context: Dict[str, Any] = None
    ) -> tuple[str, List[SecurityThreat]]:
        """Secure and validate filter request."""
        context = context or {}
        
        # Validate and detect threats
        threats = self.input_validator.validate_input(content, context)
        
        # Check if any critical threats were detected
        critical_threats = [t for t in threats if t.severity == "critical"]
        if critical_threats:
            raise SecurityError(f"Critical security threat detected: {critical_threats[0].description}")
        
        # Sanitize input
        sanitized_content = self.input_validator.sanitize_input(content)
        
        return sanitized_content, threats
    
    def secure_filter_response(
        self,
        content: str,
        context: Dict[str, Any] = None
    ) -> str:
        """Secure and sanitize filter response."""
        context = context or {}
        
        # Sanitize output
        sanitized_content = self.output_sanitizer.sanitize_output(content, context)
        
        return sanitized_content
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and statistics."""
        return {
            "security_level": self.config.security_level.value,
            "configuration": {
                "input_validation_enabled": self.config.enable_input_validation,
                "output_sanitization_enabled": self.config.enable_output_sanitization,
                "audit_logging_enabled": self.config.enable_audit_logging,
                "max_input_length": self.config.max_input_length,
                "max_processing_time_ms": self.config.max_processing_time_ms
            },
            "timestamp": datetime.utcnow().isoformat()
        }


# Global security hardening instance
_global_security_hardening: Optional[SecurityHardening] = None


def get_global_security_hardening() -> SecurityHardening:
    """Get or create global security hardening instance."""
    global _global_security_hardening
    if _global_security_hardening is None:
        _global_security_hardening = SecurityHardening()
    return _global_security_hardening