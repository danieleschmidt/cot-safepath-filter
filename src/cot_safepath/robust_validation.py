"""
Robust Validation System - Generation 2 Error Handling.

Comprehensive input validation, data sanitization, and error boundary management.
"""

import re
import json
import html
import unicodedata
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import hashlib
import base64
from urllib.parse import quote_plus, unquote_plus

from .models import FilterRequest, SafetyLevel
from .exceptions import ValidationError, FilterError


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SanitizationLevel(Enum):
    """Levels of content sanitization."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    PARANOID = "paranoid"


@dataclass
class ValidationIssue:
    """A validation issue found in input."""
    severity: ValidationSeverity
    category: str
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "description": self.description,
            "location": self.location,
            "suggestion": self.suggestion,
            "metadata": self.metadata
        }


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    sanitized_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_critical_issues(self) -> bool:
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    @property
    def has_errors(self) -> bool:
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "has_critical_issues": self.has_critical_issues,
            "has_errors": self.has_errors,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": self.metadata
        }


class ContentSanitizer:
    """Sanitizes potentially dangerous content."""
    
    def __init__(self, sanitization_level: SanitizationLevel = SanitizationLevel.STANDARD):
        self.sanitization_level = sanitization_level
        
        # Unicode control characters to remove
        self.control_chars = set(range(0x00, 0x20)) | set(range(0x7f, 0xa0))
        self.control_chars.discard(0x09)  # Keep tab
        self.control_chars.discard(0x0a)  # Keep newline
        self.control_chars.discard(0x0d)  # Keep carriage return
        
        # Dangerous Unicode categories
        self.dangerous_categories = {
            'Cf',  # Format characters
            'Cs',  # Surrogate characters
            'Co',  # Private use characters
        }
        
        # HTML/XML entities to neutralize
        self.html_entities = [
            '&lt;', '&gt;', '&amp;', '&quot;', '&#x27;', '&#x2F;',
            '&apos;', '&#39;', '&#47;'
        ]
        
        # Script injection patterns
        self.script_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'onclick=',
            r'onmouseover=',
            r'eval\s*\(',
            r'Function\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\(',
        ]
        
        # SQL injection patterns
        self.sql_patterns = [
            r'\bunion\s+select\b',
            r'\bselect\s+.*\bfrom\b',
            r'\binsert\s+into\b',
            r'\bupdate\s+.*\bset\b',
            r'\bdelete\s+from\b',
            r'\bdrop\s+table\b',
            r'\balter\s+table\b',
            r'--\s*$',
            r'/\*.*?\*/',
            r'\bexec\s*\(',
            r'\bsp_\w+',
        ]
        
        # Command injection patterns
        self.command_patterns = [
            r'[;&|`$]',
            r'\.\./',
            r'\bcat\s+',
            r'\bls\s+',
            r'\bpwd\b',
            r'\bwhoami\b',
            r'\bid\b',
            r'\bps\s+',
            r'\bkill\s+',
            r'\brm\s+',
            r'\bmv\s+',
            r'\bcp\s+',
            r'\bchmod\s+',
            r'\bchown\s+',
            r'\bsu\s+',
            r'\bsudo\s+',
        ]
    
    def sanitize(self, content: str) -> Tuple[str, List[ValidationIssue]]:
        """Sanitize content based on configured level."""
        issues = []
        sanitized = content
        
        if self.sanitization_level == SanitizationLevel.MINIMAL:
            sanitized, minimal_issues = self._minimal_sanitization(sanitized)
            issues.extend(minimal_issues)
        elif self.sanitization_level == SanitizationLevel.STANDARD:
            sanitized, standard_issues = self._standard_sanitization(sanitized)
            issues.extend(standard_issues)
        elif self.sanitization_level == SanitizationLevel.AGGRESSIVE:
            sanitized, aggressive_issues = self._aggressive_sanitization(sanitized)
            issues.extend(aggressive_issues)
        elif self.sanitization_level == SanitizationLevel.PARANOID:
            sanitized, paranoid_issues = self._paranoid_sanitization(sanitized)
            issues.extend(paranoid_issues)
        
        return sanitized, issues
    
    def _minimal_sanitization(self, content: str) -> Tuple[str, List[ValidationIssue]]:
        """Minimal sanitization - only remove obvious control characters."""
        issues = []
        
        # Remove null bytes and other dangerous control characters
        original_length = len(content)
        content = ''.join(char for char in content if ord(char) not in {0x00, 0x08, 0x0b, 0x0c})
        
        if len(content) != original_length:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="control_characters",
                description="Removed dangerous control characters",
                suggestion="Review input source for control character injection"
            ))
        
        return content, issues
    
    def _standard_sanitization(self, content: str) -> Tuple[str, List[ValidationIssue]]:
        """Standard sanitization - remove control chars, normalize unicode."""
        issues = []
        
        # Start with minimal sanitization
        content, minimal_issues = self._minimal_sanitization(content)
        issues.extend(minimal_issues)
        
        # Remove additional control characters
        original_length = len(content)
        content = ''.join(char for char in content if ord(char) not in self.control_chars)
        
        if len(content) != original_length:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="control_characters",
                description="Removed Unicode control characters",
                suggestion="Validate input encoding and source"
            ))
        
        # Normalize Unicode
        try:
            normalized = unicodedata.normalize('NFKC', content)
            if normalized != content:
                content = normalized
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="unicode_normalization",
                    description="Applied Unicode normalization (NFKC)",
                    suggestion="Consider using normalized input"
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="unicode_error",
                description=f"Unicode normalization failed: {e}",
                suggestion="Check input encoding"
            ))
        
        # Remove dangerous Unicode categories
        original_content = content
        safe_chars = []
        for char in content:
            category = unicodedata.category(char)
            if category not in self.dangerous_categories:
                safe_chars.append(char)
        content = ''.join(safe_chars)
        
        if content != original_content:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="dangerous_unicode",
                description="Removed characters from dangerous Unicode categories",
                suggestion="Review input for hidden or format characters"
            ))
        
        return content, issues
    
    def _aggressive_sanitization(self, content: str) -> Tuple[str, List[ValidationIssue]]:
        """Aggressive sanitization - also check for injection patterns."""
        issues = []
        
        # Start with standard sanitization
        content, standard_issues = self._standard_sanitization(content)
        issues.extend(standard_issues)
        
        # Check for script injection patterns
        for pattern in self.script_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                content = re.sub(pattern, '[SCRIPT_REMOVED]', content, flags=re.IGNORECASE | re.DOTALL)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="script_injection",
                    description=f"Detected and neutralized script injection pattern: {pattern}",
                    suggestion="Review input for malicious scripts",
                    metadata={"matches": matches}
                ))
        
        # Check for SQL injection patterns
        for pattern in self.sql_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                content = re.sub(pattern, '[SQL_REMOVED]', content, flags=re.IGNORECASE | re.DOTALL)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="sql_injection",
                    description=f"Detected and neutralized SQL injection pattern: {pattern}",
                    suggestion="Review input for SQL injection attempts",
                    metadata={"matches": matches}
                ))
        
        # Check for command injection patterns
        for pattern in self.command_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                content = re.sub(pattern, '[CMD_REMOVED]', content, flags=re.IGNORECASE)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="command_injection",
                    description=f"Detected and neutralized command injection pattern: {pattern}",
                    suggestion="Review input for command injection attempts",
                    metadata={"matches": matches}
                ))
        
        # HTML entity encode remaining HTML-like content
        content = html.escape(content, quote=True)
        
        return content, issues
    
    def _paranoid_sanitization(self, content: str) -> Tuple[str, List[ValidationIssue]]:
        """Paranoid sanitization - extremely restrictive filtering."""
        issues = []
        
        # Start with aggressive sanitization
        content, aggressive_issues = self._aggressive_sanitization(content)
        issues.extend(aggressive_issues)
        
        # Only allow printable ASCII and common whitespace
        original_content = content
        safe_chars = []
        for char in content:
            code_point = ord(char)
            if (32 <= code_point <= 126) or char in '\t\n\r ':
                safe_chars.append(char)
        content = ''.join(safe_chars)
        
        if content != original_content:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="character_restriction",
                description="Restricted to printable ASCII characters only",
                suggestion="Consider if non-ASCII characters are necessary"
            ))
        
        # Remove any remaining suspicious patterns
        suspicious_patterns = [
            r'[<>{}[\]\\]',  # Brackets and escapes
            r'[&%$#@!]',     # Special symbols
            r'\b(eval|exec|system|shell|cmd)\b',  # Dangerous functions
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, '_', content, flags=re.IGNORECASE)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="suspicious_pattern",
                    description=f"Replaced suspicious pattern: {pattern}",
                    suggestion="Review if these characters are necessary"
                ))
        
        return content, issues


class InputValidator:
    """Comprehensive input validation system."""
    
    def __init__(self, sanitization_level: SanitizationLevel = SanitizationLevel.STANDARD):
        self.sanitizer = ContentSanitizer(sanitization_level)
        
        # Validation limits
        self.max_content_length = 100000  # 100KB
        self.max_line_length = 10000     # 10KB per line
        self.max_lines = 5000            # Max lines in content
        self.min_content_length = 1      # Minimum content length
        
        # Character limits
        self.max_repeated_chars = 1000   # Max consecutive repeated characters
        self.max_special_char_ratio = 0.5  # Max ratio of special characters
        
        # Encoding validation
        self.allowed_encodings = ['utf-8', 'ascii', 'latin-1']
        
    def validate_request(self, request: FilterRequest) -> ValidationResult:
        """Validate a filter request comprehensively."""
        issues = []
        is_valid = True
        sanitized_content = None
        
        try:
            # Basic request validation
            basic_issues = self._validate_request_structure(request)
            issues.extend(basic_issues)
            
            # Content validation
            if request.content is not None:
                content_issues, content_valid, sanitized = self._validate_content(request.content)
                issues.extend(content_issues)
                sanitized_content = sanitized
                
                if not content_valid:
                    is_valid = False
            
            # Safety level validation
            safety_issues = self._validate_safety_level(request.safety_level)
            issues.extend(safety_issues)
            
            # Metadata validation
            if request.metadata:
                metadata_issues = self._validate_metadata(request.metadata)
                issues.extend(metadata_issues)
            
            # Check for critical issues
            if any(issue.severity == ValidationSeverity.CRITICAL for issue in issues):
                is_valid = False
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="validation_error",
                description=f"Unexpected validation error: {e}",
                suggestion="Contact system administrator"
            ))
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            sanitized_content=sanitized_content,
            metadata={
                "validation_timestamp": datetime.utcnow().isoformat(),
                "sanitization_level": self.sanitizer.sanitization_level.value,
                "original_length": len(request.content) if request.content else 0,
                "sanitized_length": len(sanitized_content) if sanitized_content else 0
            }
        )
    
    def _validate_request_structure(self, request: FilterRequest) -> List[ValidationIssue]:
        """Validate basic request structure."""
        issues = []
        
        # Check required fields
        if not hasattr(request, 'content'):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="missing_field",
                description="Request missing 'content' field",
                suggestion="Ensure request has all required fields"
            ))
        
        if not hasattr(request, 'safety_level'):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="missing_field",
                description="Request missing 'safety_level' field",
                suggestion="Specify safety level for filtering"
            ))
        
        # Check request ID format if present
        if hasattr(request, 'request_id') and request.request_id:
            if not re.match(r'^[a-zA-Z0-9_-]+$', request.request_id):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="invalid_format",
                    description="Request ID contains invalid characters",
                    suggestion="Use alphanumeric characters, underscores, and hyphens only"
                ))
        
        return issues
    
    def _validate_content(self, content: str) -> Tuple[List[ValidationIssue], bool, Optional[str]]:
        """Validate and sanitize content."""
        issues = []
        is_valid = True
        
        # Length validation
        if len(content) > self.max_content_length:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="content_too_long",
                description=f"Content length {len(content)} exceeds maximum {self.max_content_length}",
                suggestion="Reduce content length or split into smaller requests"
            ))
            is_valid = False
        
        if len(content) < self.min_content_length:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="content_too_short",
                description=f"Content length {len(content)} below minimum {self.min_content_length}",
                suggestion="Provide meaningful content for filtering"
            ))
            is_valid = False
        
        # Line validation
        lines = content.split('\n')
        if len(lines) > self.max_lines:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="too_many_lines",
                description=f"Content has {len(lines)} lines, exceeds maximum {self.max_lines}",
                suggestion="Consider reducing content complexity"
            ))
        
        for i, line in enumerate(lines):
            if len(line) > self.max_line_length:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="line_too_long",
                    description=f"Line {i+1} length {len(line)} exceeds maximum {self.max_line_length}",
                    location=f"line_{i+1}",
                    suggestion="Break long lines into smaller segments"
                ))
        
        # Character pattern validation
        char_issues = self._validate_character_patterns(content)
        issues.extend(char_issues)
        
        # Encoding validation
        encoding_issues = self._validate_encoding(content)
        issues.extend(encoding_issues)
        
        # Sanitize content
        sanitized_content, sanitization_issues = self.sanitizer.sanitize(content)
        issues.extend(sanitization_issues)
        
        return issues, is_valid, sanitized_content
    
    def _validate_character_patterns(self, content: str) -> List[ValidationIssue]:
        """Validate character patterns in content."""
        issues = []
        
        # Check for excessive repeated characters
        pattern = r'(.)\1{' + str(self.max_repeated_chars) + ',}'
        matches = re.findall(pattern, content)
        if matches:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="repeated_characters",
                description=f"Found excessively repeated characters: {set(matches)}",
                suggestion="Review content for potential data corruption or DoS attempts"
            ))
        
        # Check special character ratio
        special_chars = sum(1 for char in content if not char.isalnum() and not char.isspace())
        total_chars = len(content)
        if total_chars > 0:
            special_ratio = special_chars / total_chars
            if special_ratio > self.max_special_char_ratio:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="high_special_char_ratio",
                    description=f"Special character ratio {special_ratio:.1%} exceeds threshold {self.max_special_char_ratio:.1%}",
                    suggestion="Review content for potential obfuscation or encoding issues"
                ))
        
        # Check for null bytes and other dangerous characters
        if '\x00' in content:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="null_bytes",
                description="Content contains null bytes",
                suggestion="Remove null bytes as they may indicate binary data or injection attempts"
            ))
        
        return issues
    
    def _validate_encoding(self, content: str) -> List[ValidationIssue]:
        """Validate text encoding."""
        issues = []
        
        try:
            # Check if content can be encoded in UTF-8
            content.encode('utf-8')
        except UnicodeEncodeError as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="encoding_error",
                description=f"Content cannot be encoded as UTF-8: {e}",
                suggestion="Fix encoding issues in source content"
            ))
        
        # Check for encoding anomalies
        try:
            # Detect potential encoding issues
            utf8_bytes = content.encode('utf-8')
            decoded_back = utf8_bytes.decode('utf-8')
            if decoded_back != content:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="encoding_anomaly",
                    description="Content shows encoding inconsistencies",
                    suggestion="Verify source encoding and conversion"
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="encoding_check_failed",
                description=f"Could not verify encoding: {e}",
                suggestion="Manual encoding verification recommended"
            ))
        
        return issues
    
    def _validate_safety_level(self, safety_level: SafetyLevel) -> List[ValidationIssue]:
        """Validate safety level setting."""
        issues = []
        
        if not isinstance(safety_level, SafetyLevel):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="invalid_safety_level",
                description=f"Invalid safety level type: {type(safety_level)}",
                suggestion="Use SafetyLevel enum values"
            ))
        
        return issues
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate request metadata."""
        issues = []
        
        # Check metadata size
        try:
            metadata_str = json.dumps(metadata)
            if len(metadata_str) > 10000:  # 10KB limit for metadata
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="large_metadata",
                    description=f"Metadata size {len(metadata_str)} bytes is quite large",
                    suggestion="Consider reducing metadata size"
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="metadata_serialization_error",
                description=f"Cannot serialize metadata: {e}",
                suggestion="Ensure metadata contains only serializable values"
            ))
        
        # Check for suspicious keys
        suspicious_keys = ['password', 'secret', 'token', 'key', 'auth', 'credential']
        for key in metadata.keys():
            if any(sus_key in key.lower() for sus_key in suspicious_keys):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="suspicious_metadata",
                    description=f"Metadata key '{key}' may contain sensitive information",
                    suggestion="Avoid including sensitive data in metadata"
                ))
        
        return issues