"""
Security Hardening System - Generation 2 Security Implementation.

Advanced security measures, threat detection, and protection against malicious inputs.
"""

import hashlib
import hmac
import secrets
import time
import base64
import json
import ipaddress
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
from collections import defaultdict, deque
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding

from .models import FilterRequest
from .exceptions import SecurityError, ValidationError


logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of attacks that can be detected."""
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    DOS = "dos"
    BRUTE_FORCE = "brute_force"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RECONNAISSANCE = "reconnaissance"
    MALFORMED_INPUT = "malformed_input"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"


@dataclass
class SecurityThreat:
    """Detected security threat."""
    threat_id: str
    attack_type: AttackType
    threat_level: ThreatLevel
    source_ip: Optional[str]
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    payload_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "threat_id": self.threat_id,
            "attack_type": self.attack_type.value,
            "threat_level": self.threat_level.value,
            "source_ip": self.source_ip,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "payload_hash": self.payload_hash,
            "metadata": self.metadata
        }


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_request_size: int = 1024 * 1024  # 1MB
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    blocked_user_agents: Set[str] = field(default_factory=set)
    blocked_ips: Set[str] = field(default_factory=set)
    allowed_ip_ranges: List[str] = field(default_factory=list)
    require_authentication: bool = True
    enable_rate_limiting: bool = True
    enable_payload_analysis: bool = True
    enable_threat_detection: bool = True
    log_all_requests: bool = True


class CryptoManager:
    """Manages cryptographic operations for security."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            master_key = secrets.token_bytes(32)
        
        self.master_key = master_key
        self.fernet = Fernet(base64.urlsafe_b64encode(master_key))
        
        # Generate RSA key pair for asymmetric operations
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()
        
        logger.info("Crypto manager initialized")
    
    def encrypt_sensitive_data(self, data: Union[str, bytes]) -> str:
        """Encrypt sensitive data using symmetric encryption."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hash a password with salt using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(password.encode('utf-8'))
        
        salt_b64 = base64.urlsafe_b64encode(salt).decode('utf-8')
        hash_b64 = base64.urlsafe_b64encode(key).decode('utf-8')
        
        return hash_b64, salt_b64
    
    def verify_password(self, password: str, hash_b64: str, salt_b64: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt = base64.urlsafe_b64decode(salt_b64.encode('utf-8'))
            expected_hash = base64.urlsafe_b64decode(hash_b64.encode('utf-8'))
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            kdf.verify(password.encode('utf-8'), expected_hash)
            return True
        except Exception:
            return False
    
    def sign_data(self, data: Union[str, bytes]) -> str:
        """Sign data using RSA private key."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.urlsafe_b64encode(signature).decode('utf-8')
    
    def verify_signature(self, data: Union[str, bytes], signature_b64: str) -> bool:
        """Verify a signature using RSA public key."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            signature = base64.urlsafe_b64decode(signature_b64.encode('utf-8'))
            
            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class RateLimiter:
    """Rate limiting to prevent abuse."""
    
    def __init__(self, max_per_minute: int = 60, max_per_hour: int = 1000):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        
        # Use deques for efficient sliding window
        self.minute_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_per_minute))
        self.hour_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_per_hour))
        
        self._lock = threading.Lock()
        
        logger.info(f"Rate limiter initialized: {max_per_minute}/min, {max_per_hour}/hour")
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed for the identifier."""
        current_time = time.time()
        
        with self._lock:
            # Clean old entries
            minute_cutoff = current_time - 60
            hour_cutoff = current_time - 3600
            
            # Remove old minute entries
            minute_queue = self.minute_requests[identifier]
            while minute_queue and minute_queue[0] < minute_cutoff:
                minute_queue.popleft()
            
            # Remove old hour entries
            hour_queue = self.hour_requests[identifier]
            while hour_queue and hour_queue[0] < hour_cutoff:
                hour_queue.popleft()
            
            # Check limits
            minute_count = len(minute_queue)
            hour_count = len(hour_queue)
            
            if minute_count >= self.max_per_minute:
                return False, {
                    "reason": "minute_limit_exceeded",
                    "current_count": minute_count,
                    "limit": self.max_per_minute,
                    "reset_time": minute_cutoff + 60
                }
            
            if hour_count >= self.max_per_hour:
                return False, {
                    "reason": "hour_limit_exceeded",
                    "current_count": hour_count,
                    "limit": self.max_per_hour,
                    "reset_time": hour_cutoff + 3600
                }
            
            # Record the request
            minute_queue.append(current_time)
            hour_queue.append(current_time)
            
            return True, {
                "minute_count": minute_count + 1,
                "hour_count": hour_count + 1,
                "minute_remaining": self.max_per_minute - minute_count - 1,
                "hour_remaining": self.max_per_hour - hour_count - 1
            }
    
    def get_stats(self, identifier: str) -> Dict[str, Any]:
        """Get rate limiting stats for an identifier."""
        current_time = time.time()
        minute_cutoff = current_time - 60
        hour_cutoff = current_time - 3600
        
        with self._lock:
            minute_queue = self.minute_requests[identifier]
            hour_queue = self.hour_requests[identifier]
            
            # Count recent requests
            minute_count = sum(1 for req_time in minute_queue if req_time >= minute_cutoff)
            hour_count = sum(1 for req_time in hour_queue if req_time >= hour_cutoff)
            
            return {
                "identifier": identifier,
                "minute_count": minute_count,
                "hour_count": hour_count,
                "minute_remaining": max(0, self.max_per_minute - minute_count),
                "hour_remaining": max(0, self.max_per_hour - hour_count)
            }


class PayloadAnalyzer:
    """Analyzes request payloads for security threats."""
    
    def __init__(self):
        # Injection patterns
        self.sql_injection_patterns = [
            r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
            r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
            r"\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
            r"((\%27)|(\'))union",
            r"union(\s*(\/\*.*\*\/)?)*select",
            r"union(\s*)(select|all|distinct|having|delete|drop|update|alter|insert)",
        ]
        
        self.xss_patterns = [
            r"<script[\s\S]*?>[\s\S]*?<\/script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"onmouseover\s*=",
            r"onclick\s*=",
            r"<iframe[\s\S]*?>",
            r"<object[\s\S]*?>",
            r"<embed[\s\S]*?>",
        ]
        
        self.command_injection_patterns = [
            r"[;&|`$]",
            r"\|\s*(cat|ls|pwd|id|whoami)",
            r"(^|\s)(cat|ls|pwd|id|whoami)\s",
            r"\$\([^)]*\)",
            r"`[^`]*`",
            r"\|\s*nc\s",
            r"\|\s*netcat\s",
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\.\/",
            r"\.\.\\",
            r"\/etc\/passwd",
            r"\/etc\/shadow",
            r"\/proc\/",
            r"\/sys\/",
        ]
        
        # Data exfiltration patterns
        self.exfiltration_patterns = [
            r"(password|passwd|secret|key|token|credential)",
            r"(api[_\-]?key|auth[_\-]?token)",
            r"(BEGIN\s+PRIVATE\s+KEY|BEGIN\s+RSA\s+PRIVATE\s+KEY)",
            r"(ssh-rsa|ssh-dss|ssh-ed25519)",
        ]
        
        logger.info("Payload analyzer initialized")
    
    def analyze_payload(self, payload: str, request_metadata: Dict[str, Any] = None) -> List[SecurityThreat]:
        """Analyze payload for security threats."""
        threats = []
        payload_lower = payload.lower()
        
        # Check for SQL injection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, payload_lower, re.IGNORECASE):
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    attack_type=AttackType.INJECTION,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=request_metadata.get('source_ip') if request_metadata else None,
                    description=f"SQL injection pattern detected: {pattern}",
                    payload_hash=self._hash_payload(payload),
                    metadata={"pattern": pattern, "type": "sql_injection"}
                )
                threats.append(threat)
        
        # Check for XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, payload, re.IGNORECASE | re.DOTALL):
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    attack_type=AttackType.XSS,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=request_metadata.get('source_ip') if request_metadata else None,
                    description=f"XSS pattern detected: {pattern}",
                    payload_hash=self._hash_payload(payload),
                    metadata={"pattern": pattern, "type": "xss"}
                )
                threats.append(threat)
        
        # Check for command injection
        for pattern in self.command_injection_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    attack_type=AttackType.INJECTION,
                    threat_level=ThreatLevel.CRITICAL,
                    source_ip=request_metadata.get('source_ip') if request_metadata else None,
                    description=f"Command injection pattern detected: {pattern}",
                    payload_hash=self._hash_payload(payload),
                    metadata={"pattern": pattern, "type": "command_injection"}
                )
                threats.append(threat)
        
        # Check for path traversal
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    attack_type=AttackType.PRIVILEGE_ESCALATION,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=request_metadata.get('source_ip') if request_metadata else None,
                    description=f"Path traversal pattern detected: {pattern}",
                    payload_hash=self._hash_payload(payload),
                    metadata={"pattern": pattern, "type": "path_traversal"}
                )
                threats.append(threat)
        
        # Check for data exfiltration attempts
        for pattern in self.exfiltration_patterns:
            if re.search(pattern, payload_lower, re.IGNORECASE):
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    attack_type=AttackType.DATA_EXFILTRATION,
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip=request_metadata.get('source_ip') if request_metadata else None,
                    description=f"Potential data exfiltration pattern: {pattern}",
                    payload_hash=self._hash_payload(payload),
                    metadata={"pattern": pattern, "type": "data_exfiltration"}
                )
                threats.append(threat)
        
        # Check payload size
        if len(payload) > 1024 * 1024:  # 1MB
            threat = SecurityThreat(
                threat_id=self._generate_threat_id(),
                attack_type=AttackType.DOS,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=request_metadata.get('source_ip') if request_metadata else None,
                description=f"Oversized payload: {len(payload)} bytes",
                payload_hash=self._hash_payload(payload),
                metadata={"payload_size": len(payload), "type": "oversized_payload"}
            )
            threats.append(threat)
        
        return threats
    
    def _generate_threat_id(self) -> str:
        """Generate unique threat ID."""
        return f"threat_{int(time.time())}_{secrets.token_hex(8)}"
    
    def _hash_payload(self, payload: str) -> str:
        """Generate hash of payload for tracking."""
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


class AccessController:
    """Controls access to the filtering system."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.crypto_manager = CryptoManager()
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = 3600  # 1 hour
        
        # IP whitelist/blacklist
        self.blocked_ips = set(policy.blocked_ips)
        self.allowed_ip_ranges = [ipaddress.ip_network(cidr) for cidr in policy.allowed_ip_ranges]
        
        self._lock = threading.Lock()
        
        logger.info("Access controller initialized")
    
    def authenticate_request(self, request_data: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Authenticate a request."""
        if not self.policy.require_authentication:
            return True, None, {}
        
        # Extract authentication data
        auth_header = request_data.get('headers', {}).get('Authorization', '')
        api_key = request_data.get('api_key')
        session_token = request_data.get('session_token')
        
        # Check API key authentication
        if api_key:
            is_valid, user_id, metadata = self._validate_api_key(api_key)
            if is_valid:
                return True, user_id, metadata
        
        # Check session token
        if session_token:
            is_valid, user_id, metadata = self._validate_session_token(session_token)
            if is_valid:
                return True, user_id, metadata
        
        # Check bearer token
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            is_valid, user_id, metadata = self._validate_bearer_token(token)
            if is_valid:
                return True, user_id, metadata
        
        return False, None, {"error": "Authentication required"}
    
    def authorize_request(self, user_id: str, request_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Authorize a request for a specific user."""
        # Basic authorization logic - can be extended
        
        # Check user permissions (placeholder)
        user_permissions = self._get_user_permissions(user_id)
        
        required_permission = "filter_content"
        if required_permission not in user_permissions:
            return False, {"error": "Insufficient permissions"}
        
        return True, {"permissions": user_permissions}
    
    def check_ip_access(self, source_ip: str) -> Tuple[bool, str]:
        """Check if IP address is allowed access."""
        try:
            ip_addr = ipaddress.ip_address(source_ip)
            
            # Check if IP is explicitly blocked
            if source_ip in self.blocked_ips:
                return False, "IP blocked"
            
            # If allow list is configured, check if IP is in allowed ranges
            if self.allowed_ip_ranges:
                for allowed_range in self.allowed_ip_ranges:
                    if ip_addr in allowed_range:
                        return True, "IP allowed"
                return False, "IP not in allowed range"
            
            # If no allow list, allow all except blocked
            return True, "IP allowed"
            
        except ValueError:
            return False, "Invalid IP address"
    
    def create_session(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new session for a user."""
        session_token = self.crypto_manager.generate_secure_token()
        
        with self._lock:
            self.active_sessions[session_token] = {
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "metadata": metadata or {}
            }
        
        return session_token
    
    def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a session."""
        with self._lock:
            if session_token in self.active_sessions:
                del self.active_sessions[session_token]
                return True
        return False
    
    def _validate_api_key(self, api_key: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Validate API key (placeholder implementation)."""
        # In a real implementation, this would check against a database
        # For now, accept any key that looks valid
        if len(api_key) >= 32 and api_key.isalnum():
            return True, f"user_{api_key[:8]}", {"auth_method": "api_key"}
        return False, None, {}
    
    def _validate_session_token(self, session_token: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Validate session token."""
        with self._lock:
            if session_token not in self.active_sessions:
                return False, None, {"error": "Invalid session"}
            
            session = self.active_sessions[session_token]
            
            # Check if session has expired
            if (datetime.utcnow() - session["created_at"]).total_seconds() > self.session_timeout:
                del self.active_sessions[session_token]
                return False, None, {"error": "Session expired"}
            
            # Update last activity
            session["last_activity"] = datetime.utcnow()
            
            return True, session["user_id"], {"auth_method": "session", "session_data": session["metadata"]}
    
    def _validate_bearer_token(self, token: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Validate bearer token."""
        try:
            # In a real implementation, this would validate JWT or similar
            # For now, just check if it's a valid format
            if len(token) >= 32:
                return True, f"user_{token[:8]}", {"auth_method": "bearer"}
        except Exception:
            pass
        
        return False, None, {}
    
    def _get_user_permissions(self, user_id: str) -> Set[str]:
        """Get permissions for a user (placeholder)."""
        # In a real implementation, this would query a permissions database
        return {"filter_content", "view_results", "access_api"}


class SecurityHardeningManager:
    """Central manager for all security hardening features."""
    
    def __init__(self, policy: SecurityPolicy = None):
        self.policy = policy or SecurityPolicy()
        
        self.crypto_manager = CryptoManager()
        self.rate_limiter = RateLimiter(
            max_per_minute=self.policy.max_requests_per_minute,
            max_per_hour=self.policy.max_requests_per_hour
        )
        self.payload_analyzer = PayloadAnalyzer()
        self.access_controller = AccessController(self.policy)
        
        # Threat tracking
        self.detected_threats: List[SecurityThreat] = []
        self.threat_stats = defaultdict(int)
        self.max_threat_history = 10000
        
        # Request logging
        self.request_log: List[Dict[str, Any]] = []
        self.max_request_log = 5000
        
        self._lock = threading.Lock()
        
        logger.info("Security hardening manager initialized")
    
    def validate_request_security(self, request: FilterRequest, 
                                metadata: Dict[str, Any] = None) -> Tuple[bool, List[SecurityThreat], Dict[str, Any]]:
        """Comprehensive security validation of a request."""
        threats = []
        validation_metadata = {}
        
        request_metadata = metadata or {}
        source_ip = request_metadata.get('source_ip', 'unknown')
        user_agent = request_metadata.get('user_agent', '')
        
        # IP access control
        if source_ip != 'unknown':
            ip_allowed, ip_reason = self.access_controller.check_ip_access(source_ip)
            if not ip_allowed:
                threat = SecurityThreat(
                    threat_id=f"ip_block_{int(time.time())}",
                    attack_type=AttackType.RECONNAISSANCE,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=source_ip,
                    description=f"IP access denied: {ip_reason}",
                    metadata={"reason": ip_reason}
                )
                threats.append(threat)
                validation_metadata["ip_blocked"] = True
        
        # Rate limiting
        if self.policy.enable_rate_limiting:
            identifier = source_ip if source_ip != 'unknown' else 'default'
            allowed, rate_info = self.rate_limiter.is_allowed(identifier)
            
            if not allowed:
                threat = SecurityThreat(
                    threat_id=f"rate_limit_{int(time.time())}",
                    attack_type=AttackType.RATE_LIMIT_VIOLATION,
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip=source_ip,
                    description=f"Rate limit exceeded: {rate_info['reason']}",
                    metadata=rate_info
                )
                threats.append(threat)
                validation_metadata["rate_limited"] = True
            
            validation_metadata["rate_info"] = rate_info
        
        # User agent validation
        if user_agent in self.policy.blocked_user_agents:
            threat = SecurityThreat(
                threat_id=f"ua_block_{int(time.time())}",
                attack_type=AttackType.RECONNAISSANCE,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                description=f"Blocked user agent: {user_agent}",
                metadata={"user_agent": user_agent}
            )
            threats.append(threat)
            validation_metadata["user_agent_blocked"] = True
        
        # Payload analysis
        if self.policy.enable_payload_analysis and request.content:
            payload_threats = self.payload_analyzer.analyze_payload(
                request.content, request_metadata
            )
            threats.extend(payload_threats)
            validation_metadata["payload_threats"] = len(payload_threats)
        
        # Request size validation
        content_size = len(request.content) if request.content else 0
        if content_size > self.policy.max_request_size:
            threat = SecurityThreat(
                threat_id=f"size_limit_{int(time.time())}",
                attack_type=AttackType.DOS,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                description=f"Request size {content_size} exceeds limit {self.policy.max_request_size}",
                metadata={"request_size": content_size, "limit": self.policy.max_request_size}
            )
            threats.append(threat)
            validation_metadata["size_exceeded"] = True
        
        # Record threats
        with self._lock:
            self.detected_threats.extend(threats)
            for threat in threats:
                self.threat_stats[threat.attack_type.value] += 1
            
            # Trim threat history
            if len(self.detected_threats) > self.max_threat_history:
                self.detected_threats = self.detected_threats[-self.max_threat_history//2:]
        
        # Log request if enabled
        if self.policy.log_all_requests:
            self._log_request(request, threats, validation_metadata)
        
        # Determine if request should be allowed
        critical_threats = [t for t in threats if t.threat_level == ThreatLevel.CRITICAL]
        high_threats = [t for t in threats if t.threat_level == ThreatLevel.HIGH]
        
        # Block requests with critical threats or too many high threats
        is_allowed = len(critical_threats) == 0 and len(high_threats) < 3
        
        validation_metadata.update({
            "total_threats": len(threats),
            "critical_threats": len(critical_threats),
            "high_threats": len(high_threats),
            "validation_timestamp": datetime.utcnow().isoformat()
        })
        
        return is_allowed, threats, validation_metadata
    
    def _log_request(self, request: FilterRequest, threats: List[SecurityThreat], 
                    metadata: Dict[str, Any]) -> None:
        """Log request for security auditing."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request.request_id,
            "content_hash": hashlib.sha256(request.content.encode('utf-8')).hexdigest()[:16] if request.content else None,
            "content_length": len(request.content) if request.content else 0,
            "safety_level": request.safety_level.value if hasattr(request.safety_level, 'value') else str(request.safety_level),
            "threats_detected": len(threats),
            "threat_levels": [t.threat_level.value for t in threats],
            "attack_types": [t.attack_type.value for t in threats],
            "metadata": metadata
        }
        
        with self._lock:
            self.request_log.append(log_entry)
            
            # Trim log
            if len(self.request_log) > self.max_request_log:
                self.request_log = self.request_log[-self.max_request_log//2:]
    
    def get_security_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate security report for the specified time range."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        with self._lock:
            # Filter recent threats
            recent_threats = [
                t for t in self.detected_threats 
                if t.timestamp >= cutoff_time
            ]
            
            # Filter recent requests
            recent_requests = [
                r for r in self.request_log 
                if datetime.fromisoformat(r["timestamp"]) >= cutoff_time
            ]
        
        # Threat statistics
        threat_by_type = defaultdict(int)
        threat_by_level = defaultdict(int)
        threat_by_ip = defaultdict(int)
        
        for threat in recent_threats:
            threat_by_type[threat.attack_type.value] += 1
            threat_by_level[threat.threat_level.value] += 1
            if threat.source_ip:
                threat_by_ip[threat.source_ip] += 1
        
        # Request statistics
        total_requests = len(recent_requests)
        requests_with_threats = sum(1 for r in recent_requests if r["threats_detected"] > 0)
        
        return {
            "report_period_hours": time_range_hours,
            "report_timestamp": datetime.utcnow().isoformat(),
            "total_requests": total_requests,
            "requests_with_threats": requests_with_threats,
            "threat_detection_rate": requests_with_threats / max(total_requests, 1),
            "total_threats": len(recent_threats),
            "threats_by_type": dict(threat_by_type),
            "threats_by_level": dict(threat_by_level),
            "top_threat_sources": dict(sorted(threat_by_ip.items(), key=lambda x: x[1], reverse=True)[:10]),
            "recent_critical_threats": [
                t.to_dict() for t in recent_threats 
                if t.threat_level == ThreatLevel.CRITICAL
            ][-10:],  # Last 10 critical threats
            "security_policy": {
                "rate_limiting_enabled": self.policy.enable_rate_limiting,
                "payload_analysis_enabled": self.policy.enable_payload_analysis,
                "authentication_required": self.policy.require_authentication,
                "max_request_size": self.policy.max_request_size,
                "blocked_ips_count": len(self.policy.blocked_ips),
                "allowed_ip_ranges_count": len(self.policy.allowed_ip_ranges)
            }
        }
    
    def block_ip(self, ip_address: str, reason: str = "Security violation") -> bool:
        """Block an IP address."""
        try:
            ipaddress.ip_address(ip_address)  # Validate IP
            self.policy.blocked_ips.add(ip_address)
            logger.warning(f"Blocked IP {ip_address}: {reason}")
            return True
        except ValueError:
            logger.error(f"Invalid IP address: {ip_address}")
            return False
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock an IP address."""
        if ip_address in self.policy.blocked_ips:
            self.policy.blocked_ips.remove(ip_address)
            logger.info(f"Unblocked IP {ip_address}")
            return True
        return False
    
    def cleanup(self) -> None:
        """Cleanup security resources."""
        with self._lock:
            self.detected_threats.clear()
            self.request_log.clear()
            self.threat_stats.clear()
        
        # Clear session data
        self.access_controller.active_sessions.clear()
        
        logger.info("Security hardening manager cleaned up")