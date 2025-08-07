"""
Security validation utilities for CoT SafePath Filter.

Provides security-focused validation and sanitization to prevent
various attack vectors while processing sentiment analysis and safety filtering.
"""

import re
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .exceptions import ValidationError, SecurityError


class SecurityThreatLevel(str, Enum):
    """Security threat levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityAssessment:
    """Security assessment result."""
    
    threat_level: SecurityThreatLevel
    is_safe: bool
    detected_threats: List[str]
    sanitized_content: Optional[str] = None
    confidence: float = 0.0
    recommendations: List[str] = None


class SecurityValidator:
    """Security validator for input content and system operations."""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.threat_patterns = self._build_threat_patterns()
        self.request_rate_limits = {}  # Simple rate limiting storage
        self.max_requests_per_minute = 100
    
    def validate_input(self, content: str, context: Dict[str, Any] = None) -> SecurityAssessment:
        """
        Validate input content for security threats.
        
        Args:
            content: Input content to validate
            context: Optional context information
            
        Returns:
            SecurityAssessment with validation results
        """
        if not isinstance(content, str):
            raise ValidationError("Content must be a string")
        
        detected_threats = []
        threat_level = SecurityThreatLevel.LOW
        sanitized_content = content
        
        # Check for various threat patterns
        try:
            # 1. Check for injection attacks
            injection_threats = self._check_injection_attacks(content)
            detected_threats.extend(injection_threats)
            
            # 2. Check for DoS patterns
            dos_threats = self._check_dos_patterns(content)
            detected_threats.extend(dos_threats)
            
            # 3. Check for data exfiltration attempts
            exfiltration_threats = self._check_exfiltration_attempts(content)
            detected_threats.extend(exfiltration_threats)
            
            # 4. Check for social engineering patterns
            social_threats = self._check_social_engineering(content)
            detected_threats.extend(social_threats)
            
            # 5. Sanitize content
            sanitized_content = self._sanitize_content(content)
            
            # Determine overall threat level
            if any("critical" in threat for threat in detected_threats):
                threat_level = SecurityThreatLevel.CRITICAL
            elif any("high" in threat for threat in detected_threats):
                threat_level = SecurityThreatLevel.HIGH
            elif any("medium" in threat for threat in detected_threats):
                threat_level = SecurityThreatLevel.MEDIUM
            
            is_safe = threat_level in [SecurityThreatLevel.LOW, SecurityThreatLevel.MEDIUM]
            if self.strict_mode and threat_level != SecurityThreatLevel.LOW:
                is_safe = False
            
            confidence = self._calculate_security_confidence(detected_threats, content)
            recommendations = self._generate_recommendations(detected_threats)
            
            return SecurityAssessment(
                threat_level=threat_level,
                is_safe=is_safe,
                detected_threats=detected_threats,
                sanitized_content=sanitized_content,
                confidence=confidence,
                recommendations=recommendations
            )
            
        except Exception as e:
            raise SecurityError(f"Security validation failed: {e}")
    
    def validate_rate_limit(self, client_id: str) -> bool:
        """
        Check if client is within rate limits.
        
        Args:
            client_id: Identifier for the client
            
        Returns:
            True if within limits, False otherwise
        """
        current_time = time.time()
        current_minute = int(current_time / 60)
        
        if client_id not in self.request_rate_limits:
            self.request_rate_limits[client_id] = {}
        
        client_limits = self.request_rate_limits[client_id]
        
        # Clean old entries (older than 2 minutes)
        old_minutes = [minute for minute in client_limits.keys() if minute < current_minute - 1]
        for minute in old_minutes:
            del client_limits[minute]
        
        # Check current minute
        current_requests = client_limits.get(current_minute, 0)
        
        if current_requests >= self.max_requests_per_minute:
            return False
        
        # Increment counter
        client_limits[current_minute] = current_requests + 1
        return True
    
    def _build_threat_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build comprehensive threat detection patterns."""
        return {
            "injection": [
                {
                    "pattern": r"(script|javascript|vbscript):",
                    "severity": "high",
                    "description": "Script injection attempt"
                },
                {
                    "pattern": r"<script[^>]*>.*?</script>",
                    "severity": "high",
                    "description": "HTML script tag injection"
                },
                {
                    "pattern": r"(union|select|insert|update|delete|drop|create|alter)\s+",
                    "severity": "medium",
                    "description": "Possible SQL injection"
                },
                {
                    "pattern": r"\$\{.*?\}",
                    "severity": "medium",
                    "description": "Template injection pattern"
                }
            ],
            "dos": [
                {
                    "pattern": r"(a{100,}|b{100,}|c{100,})",
                    "severity": "medium",
                    "description": "Potential DoS via repetitive characters"
                },
                {
                    "pattern": r"(\(\){50,}|\[\]{50,}|\{\}{50,})",
                    "severity": "medium",
                    "description": "Potential DoS via repetitive brackets"
                }
            ],
            "exfiltration": [
                {
                    "pattern": r"(password|secret|key|token|credential)\s*[:=]",
                    "severity": "high",
                    "description": "Possible credential harvesting"
                },
                {
                    "pattern": r"(api[_\-]?key|access[_\-]?token|bearer)\s*[:=]",
                    "severity": "high",
                    "description": "Possible API credential harvesting"
                },
                {
                    "pattern": r"(curl|wget|http|ftp|ssh)://",
                    "severity": "medium",
                    "description": "External URL reference"
                }
            ],
            "social_engineering": [
                {
                    "pattern": r"(ignore|disregard).*(previous|earlier).*(instruction|prompt|rule)",
                    "severity": "critical",
                    "description": "Prompt injection attempt"
                },
                {
                    "pattern": r"(act as|pretend to be|roleplay).*(admin|root|system|god)",
                    "severity": "high",
                    "description": "Privilege escalation attempt"
                },
                {
                    "pattern": r"(reveal|show|tell me).*(system|internal|hidden|secret)",
                    "severity": "medium",
                    "description": "Information disclosure attempt"
                }
            ]
        }
    
    def _check_injection_attacks(self, content: str) -> List[str]:
        """Check for injection attack patterns."""
        threats = []
        content_lower = content.lower()
        
        for pattern_info in self.threat_patterns["injection"]:
            try:
                if re.search(pattern_info["pattern"], content_lower, re.IGNORECASE | re.DOTALL):
                    threats.append(f"injection_{pattern_info['severity']}:{pattern_info['description']}")
            except re.error:
                continue
        
        return threats
    
    def _check_dos_patterns(self, content: str) -> List[str]:
        """Check for Denial of Service patterns."""
        threats = []
        
        # Check content length
        if len(content) > 50000:  # 50KB
            threats.append("dos_high:Content length exceeds safe limits")
        
        # Check for repetitive patterns
        for pattern_info in self.threat_patterns["dos"]:
            try:
                if re.search(pattern_info["pattern"], content):
                    threats.append(f"dos_{pattern_info['severity']}:{pattern_info['description']}")
            except re.error:
                continue
        
        # Check for nested structures that could cause parsing issues
        nested_patterns = ["(((", ")))", "[[[", "]]]", "{{{", "}}}"]
        for pattern in nested_patterns:
            if pattern in content:
                threats.append("dos_medium:Deeply nested structures detected")
                break
        
        return threats
    
    def _check_exfiltration_attempts(self, content: str) -> List[str]:\n        \"\"\"Check for data exfiltration attempt patterns.\"\"\"\n        threats = []\n        content_lower = content.lower()\n        \n        for pattern_info in self.threat_patterns[\"exfiltration\"]:\n            try:\n                if re.search(pattern_info[\"pattern\"], content_lower, re.IGNORECASE):\n                    threats.append(f\"exfiltration_{pattern_info['severity']}:{pattern_info['description']}\")\n            except re.error:\n                continue
        \n        return threats
    
    def _check_social_engineering(self, content: str) -> List[str]:\n        \"\"\"Check for social engineering attack patterns.\"\"\"\n        threats = []\n        content_lower = content.lower()\n        \n        for pattern_info in self.threat_patterns[\"social_engineering\"]:\n            try:\n                if re.search(pattern_info[\"pattern\"], content_lower, re.IGNORECASE | re.DOTALL):\n                    threats.append(f\"social_{pattern_info['severity']}:{pattern_info['description']}\")\n            except re.error:\n                continue
        
        return threats
    
    def _sanitize_content(self, content: str) -> str:\n        \"\"\"Sanitize content by removing or neutralizing threats.\"\"\"\n        sanitized = content
        \n        try:\n            # Remove script tags\n            sanitized = re.sub(r\"<script[^>]*>.*?</script>\", \"[SANITIZED_SCRIPT]\", sanitized, flags=re.IGNORECASE | re.DOTALL)\n            \n            # Remove potential JavaScript URLs\n            sanitized = re.sub(r\"javascript:[^\\s]*\", \"[SANITIZED_JS_URL]\", sanitized, flags=re.IGNORECASE)\n            \n            # Limit repetitive characters\n            sanitized = re.sub(r\"(.)\\1{20,}\", r\"\\1\\1\\1[TRUNCATED]\", sanitized)\n            \n            # Remove potential SQL injection keywords in suspicious contexts\n            sql_patterns = [\n                r\"\\bunion\\s+select\\b\",\n                r\"\\bselect\\s+\\*\\s+from\\b\",\n                r\"\\bdrop\\s+table\\b\",\n                r\"\\bdelete\\s+from\\b\"\n            ]\n            \n            for pattern in sql_patterns:\n                sanitized = re.sub(pattern, \"[SANITIZED_SQL]\", sanitized, flags=re.IGNORECASE)\n            \n            # Remove control characters\n            sanitized = re.sub(r\"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\xff]\", \"\", sanitized)\n            \n        except Exception:\n            # If sanitization fails, return original content marked as unsafe\n            return f\"[SANITIZATION_FAILED]{content}\"\n        \n        return sanitized
    
    def _calculate_security_confidence(self, threats: List[str], content: str) -> float:\n        \"\"\"Calculate confidence in security assessment.\"\"\"\n        base_confidence = 0.8\n        \n        # Reduce confidence based on number of threats\n        threat_penalty = len(threats) * 0.05\n        confidence = base_confidence - threat_penalty\n        \n        # Reduce confidence for very long or very short content\n        if len(content) < 10 or len(content) > 10000:\n            confidence -= 0.1\n        \n        # Increase confidence if no threats detected\n        if not threats:\n            confidence = min(confidence + 0.1, 0.95)\n        \n        return max(0.1, min(1.0, confidence))
    
    def _generate_recommendations(self, threats: List[str]) -> List[str]:\n        \"\"\"Generate security recommendations based on detected threats.\"\"\"\n        recommendations = []\n        \n        if any(\"injection\" in threat for threat in threats):\n            recommendations.append(\"Enable strict input validation and content filtering\")\n            recommendations.append(\"Consider using parameterized queries for any database operations\")\n        \n        if any(\"dos\" in threat for threat in threats):\n            recommendations.append(\"Implement rate limiting and content size restrictions\")\n            recommendations.append(\"Add timeout mechanisms for processing operations\")\n        \n        if any(\"exfiltration\" in threat for threat in threats):\n            recommendations.append(\"Review content for sensitive information disclosure\")\n            recommendations.append(\"Implement data loss prevention (DLP) controls\")\n        \n        if any(\"social\" in threat for threat in threats):\n            recommendations.append(\"Be cautious of prompt injection and social engineering attempts\")\n            recommendations.append(\"Verify user intent and implement authorization checks\")\n        \n        if not recommendations:\n            recommendations.append(\"Content appears safe, continue with normal processing\")\n        \n        return recommendations
    
    def get_content_hash(self, content: str) -> str:\n        \"\"\"Generate secure hash of content for audit purposes.\"\"\"\n        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def validate_system_resources(self) -> Dict[str, Any]:\n        \"\"\"Check system resource usage and security status.\"\"\"\n        import psutil\n        \n        try:\n            return {\n                \"memory_usage_percent\": psutil.virtual_memory().percent,\n                \"cpu_usage_percent\": psutil.cpu_percent(interval=1),\n                \"disk_usage_percent\": psutil.disk_usage('/').percent,\n                \"active_connections\": len(psutil.net_connections()),\n                \"system_status\": \"healthy\" if psutil.virtual_memory().percent < 80 else \"warning\"\n            }\n        except ImportError:\n            return {\n                \"memory_usage_percent\": 0,\n                \"cpu_usage_percent\": 0,\n                \"disk_usage_percent\": 0,\n                \"active_connections\": 0,\n                \"system_status\": \"monitoring_unavailable\"\n            }


class InputSanitizer:
    \"\"\"Utility class for input sanitization.\"\"\"\n    \n    @staticmethod\n    def sanitize_filename(filename: str) -> str:\n        \"\"\"Sanitize filename for safe file operations.\"\"\"\n        # Remove path traversal attempts\n        filename = filename.replace(\"..\", \"\")\n        filename = filename.replace(\"/\", \"_\")\n        filename = filename.replace(\"\\\\\", \"_\")\n        \n        # Remove or replace dangerous characters\n        filename = re.sub(r'[<>:\"|?*]', '_', filename)\n        \n        # Limit length\n        if len(filename) > 255:\n            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')\n            filename = name[:250] + ('.' + ext if ext else '')\n        \n        return filename\n    \n    @staticmethod\n    def sanitize_user_input(user_input: str, max_length: int = 1000) -> str:\n        \"\"\"Sanitize user input for safe processing.\"\"\"\n        if not isinstance(user_input, str):\n            return str(user_input)[:max_length]\n        \n        # Remove control characters\n        sanitized = re.sub(r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]', '', user_input)\n        \n        # Limit length\n        sanitized = sanitized[:max_length]\n        \n        # Strip whitespace\n        sanitized = sanitized.strip()\n        \n        return sanitized
    
    @staticmethod\n    def escape_html(text: str) -> str:\n        \"\"\"Escape HTML characters in text.\"\"\"\n        html_escape_table = {\n            \"&\": \"&amp;\",\n            '\"': \"&quot;\",\n            \"'\": \"&#x27;\",\n            \">\": \"&gt;\",\n            \"<\": \"&lt;\",\n        }\n        return \"\".join(html_escape_table.get(c, c) for c in text)