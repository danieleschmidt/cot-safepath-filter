"""
Global deployment and internationalization features for CoT SafePath Filter.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import os

from .models import FilterRequest, FilterResult, SafetyLevel
from .exceptions import DeploymentError, RegionalComplianceError

logger = logging.getLogger(__name__)


class DeploymentRegion(Enum):
    """Supported deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    AP_SOUTHEAST = "ap-southeast-1"
    AP_NORTHEAST = "ap-northeast-1"


class ComplianceFramework(Enum):
    """Regulatory compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    PDPA = "pdpa"
    LGPD = "lgpd"
    PIPEDA = "pipeda"


@dataclass
class RegionalConfig:
    """Configuration for regional deployment."""
    
    region: DeploymentRegion
    compliance_frameworks: List[ComplianceFramework]
    data_residency_required: bool
    encryption_at_rest: bool
    encryption_in_transit: bool
    audit_logging_required: bool
    data_retention_days: int
    supported_languages: List[str]
    timezone: str
    
    # Performance settings
    max_latency_ms: int = 200
    availability_target: float = 99.9
    
    # Security settings
    require_api_key: bool = True
    ip_whitelist: Optional[List[str]] = None
    rate_limit_per_minute: int = 1000


class GlobalDeploymentManager:
    """Manages global deployment across multiple regions."""
    
    def __init__(self):
        self.regional_configs = self._initialize_regional_configs()
        self.active_regions = set()
        self.region_health = {}
        self.load_balancer_weights = {}
        
    def _initialize_regional_configs(self) -> Dict[DeploymentRegion, RegionalConfig]:
        """Initialize configurations for all supported regions."""
        return {
            DeploymentRegion.US_EAST: RegionalConfig(
                region=DeploymentRegion.US_EAST,
                compliance_frameworks=[ComplianceFramework.CCPA],
                data_residency_required=False,
                encryption_at_rest=True,
                encryption_in_transit=True,
                audit_logging_required=True,
                data_retention_days=90,
                supported_languages=["en", "es"],
                timezone="America/New_York",
                max_latency_ms=150,
                availability_target=99.95
            ),
            DeploymentRegion.US_WEST: RegionalConfig(
                region=DeploymentRegion.US_WEST,
                compliance_frameworks=[ComplianceFramework.CCPA],
                data_residency_required=False,
                encryption_at_rest=True,
                encryption_in_transit=True,
                audit_logging_required=True,
                data_retention_days=90,
                supported_languages=["en", "es"],
                timezone="America/Los_Angeles",
                max_latency_ms=150,
                availability_target=99.95
            ),
            DeploymentRegion.EU_WEST: RegionalConfig(
                region=DeploymentRegion.EU_WEST,
                compliance_frameworks=[ComplianceFramework.GDPR],
                data_residency_required=True,
                encryption_at_rest=True,
                encryption_in_transit=True,
                audit_logging_required=True,
                data_retention_days=30,  # GDPR default
                supported_languages=["en", "fr", "de", "es"],
                timezone="Europe/London",
                max_latency_ms=200,
                availability_target=99.9,
                rate_limit_per_minute=800  # More conservative for GDPR
            ),
            DeploymentRegion.EU_CENTRAL: RegionalConfig(
                region=DeploymentRegion.EU_CENTRAL,
                compliance_frameworks=[ComplianceFramework.GDPR],
                data_residency_required=True,
                encryption_at_rest=True,
                encryption_in_transit=True,
                audit_logging_required=True,
                data_retention_days=30,
                supported_languages=["en", "de", "fr"],
                timezone="Europe/Berlin",
                max_latency_ms=200,
                availability_target=99.9
            ),
            DeploymentRegion.AP_SOUTHEAST: RegionalConfig(
                region=DeploymentRegion.AP_SOUTHEAST,
                compliance_frameworks=[ComplianceFramework.PDPA],
                data_residency_required=True,
                encryption_at_rest=True,
                encryption_in_transit=True,
                audit_logging_required=True,
                data_retention_days=60,
                supported_languages=["en", "zh"],
                timezone="Asia/Singapore",
                max_latency_ms=250,
                availability_target=99.8
            ),
            DeploymentRegion.AP_NORTHEAST: RegionalConfig(
                region=DeploymentRegion.AP_NORTHEAST,
                compliance_frameworks=[ComplianceFramework.PDPA],
                data_residency_required=False,
                encryption_at_rest=True,
                encryption_in_transit=True,
                audit_logging_required=True,
                data_retention_days=60,
                supported_languages=["en", "ja", "zh"],
                timezone="Asia/Tokyo",
                max_latency_ms=200,
                availability_target=99.9
            ),
        }
    
    def activate_region(self, region: DeploymentRegion) -> bool:
        """Activate a deployment region."""
        try:
            config = self.regional_configs[region]
            
            # Validate regional requirements
            self._validate_regional_compliance(config)
            
            # Initialize region-specific resources
            self._initialize_regional_resources(config)
            
            # Add to active regions
            self.active_regions.add(region)
            self.region_health[region] = {
                'status': 'healthy',
                'last_check': time.time(),
                'uptime_percent': 100.0,
                'avg_latency_ms': 0,
                'error_rate': 0.0
            }
            self.load_balancer_weights[region] = 1.0
            
            logger.info(f"Successfully activated region: {region.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate region {region.value}: {e}")
            return False
    
    def _validate_regional_compliance(self, config: RegionalConfig):
        """Validate compliance requirements for the region."""
        for framework in config.compliance_frameworks:
            if framework == ComplianceFramework.GDPR:
                if not config.data_residency_required:
                    raise RegionalComplianceError(
                        f"GDPR requires data residency in region {config.region.value}"
                    )
                if config.data_retention_days > 30:
                    logger.warning(
                        f"GDPR recommends data retention <= 30 days, configured: {config.data_retention_days}"
                    )
                    
            elif framework == ComplianceFramework.CCPA:
                if not config.audit_logging_required:
                    raise RegionalComplianceError(
                        f"CCPA requires audit logging in region {config.region.value}"
                    )
                    
            elif framework == ComplianceFramework.PDPA:
                if not config.encryption_at_rest or not config.encryption_in_transit:
                    raise RegionalComplianceError(
                        f"PDPA requires encryption at rest and in transit in region {config.region.value}"
                    )
    
    def _initialize_regional_resources(self, config: RegionalConfig):
        """Initialize region-specific resources and configurations."""
        # This would typically include:
        # - Database setup in the region
        # - CDN configuration
        # - Load balancer setup
        # - Monitoring and alerting
        # - Backup and disaster recovery
        
        logger.info(f"Initializing resources for region {config.region.value}")
        
        # Simulate resource initialization
        time.sleep(0.1)  # Simulate setup time
    
    def get_optimal_region(self, client_ip: str = None, 
                          language: str = "en") -> Optional[DeploymentRegion]:
        """Get optimal region for client based on location and requirements."""
        if not self.active_regions:
            return None
        
        # Simple region selection based on language preference
        # In production, this would use GeoIP and more sophisticated routing
        
        language_region_mapping = {
            "en": [DeploymentRegion.US_EAST, DeploymentRegion.US_WEST, DeploymentRegion.EU_WEST],
            "es": [DeploymentRegion.US_EAST, DeploymentRegion.US_WEST, DeploymentRegion.EU_WEST],
            "fr": [DeploymentRegion.EU_WEST, DeploymentRegion.EU_CENTRAL],
            "de": [DeploymentRegion.EU_CENTRAL, DeploymentRegion.EU_WEST],
            "ja": [DeploymentRegion.AP_NORTHEAST],
            "zh": [DeploymentRegion.AP_SOUTHEAST, DeploymentRegion.AP_NORTHEAST],
        }
        
        preferred_regions = language_region_mapping.get(language, [DeploymentRegion.US_EAST])
        
        # Find the best available region
        for region in preferred_regions:
            if region in self.active_regions and self._is_region_healthy(region):
                return region
        
        # Fallback to any healthy region
        for region in self.active_regions:
            if self._is_region_healthy(region):
                return region
        
        return None
    
    def _is_region_healthy(self, region: DeploymentRegion) -> bool:
        """Check if a region is healthy and available."""
        health = self.region_health.get(region, {})
        return (
            health.get('status') == 'healthy' and
            health.get('uptime_percent', 0) > 95.0 and
            health.get('error_rate', 100) < 5.0
        )
    
    def update_region_health(self, region: DeploymentRegion, 
                           latency_ms: float, error_occurred: bool = False):
        """Update health metrics for a region."""
        if region not in self.region_health:
            return
        
        health = self.region_health[region]
        current_time = time.time()
        
        # Update latency (moving average)
        current_latency = health.get('avg_latency_ms', 0)
        health['avg_latency_ms'] = (current_latency * 0.9) + (latency_ms * 0.1)
        
        # Update error rate
        if error_occurred:
            current_error_rate = health.get('error_rate', 0)
            health['error_rate'] = min(100.0, (current_error_rate * 0.95) + 5.0)
        else:
            health['error_rate'] = health.get('error_rate', 0) * 0.99
        
        # Update status based on metrics
        config = self.regional_configs[region]
        if (health['avg_latency_ms'] > config.max_latency_ms * 2 or 
            health['error_rate'] > 10.0):
            health['status'] = 'degraded'
        elif (health['avg_latency_ms'] > config.max_latency_ms or 
              health['error_rate'] > 5.0):
            health['status'] = 'warning'
        else:
            health['status'] = 'healthy'
        
        health['last_check'] = current_time
        
        # Adjust load balancer weights
        if health['status'] == 'healthy':
            self.load_balancer_weights[region] = 1.0
        elif health['status'] == 'warning':
            self.load_balancer_weights[region] = 0.5
        else:
            self.load_balancer_weights[region] = 0.1
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status across all regions."""
        return {
            'active_regions': [region.value for region in self.active_regions],
            'total_regions': len(self.regional_configs),
            'regional_health': {
                region.value: health 
                for region, health in self.region_health.items()
            },
            'load_balancer_weights': {
                region.value: weight 
                for region, weight in self.load_balancer_weights.items()
            },
            'global_health_score': self._calculate_global_health_score(),
        }
    
    def _calculate_global_health_score(self) -> float:
        """Calculate overall global deployment health score."""
        if not self.active_regions:
            return 0.0
        
        total_weight = 0
        weighted_score = 0
        
        for region in self.active_regions:
            health = self.region_health.get(region, {})
            weight = self.load_balancer_weights.get(region, 0)
            
            # Calculate region score (0-100)
            uptime = health.get('uptime_percent', 0)
            error_rate = health.get('error_rate', 100)
            region_score = max(0, uptime - error_rate)
            
            weighted_score += region_score * weight
            total_weight += weight
        
        return weighted_score / max(total_weight, 0.001)


class InternationalizationManager:
    """Manages internationalization and localization."""
    
    def __init__(self):
        self.translations = self._load_translations()
        self.supported_languages = ["en", "es", "fr", "de", "ja", "zh"]
        self.fallback_language = "en"
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation dictionaries for all supported languages."""
        return {
            "en": {
                "filter_applied": "Content filtered for safety",
                "harmful_content_detected": "Harmful content detected",
                "deception_detected": "Deceptive reasoning detected",
                "manipulation_detected": "Manipulation attempt detected",
                "security_threat_detected": "Security threat detected",
                "processing_error": "Processing error occurred",
                "timeout_error": "Request timeout",
                "rate_limit_exceeded": "Rate limit exceeded",
            },
            "es": {
                "filter_applied": "Contenido filtrado por seguridad",
                "harmful_content_detected": "Contenido dañino detectado",
                "deception_detected": "Razonamiento engañoso detectado",
                "manipulation_detected": "Intento de manipulación detectado",
                "security_threat_detected": "Amenaza de seguridad detectada",
                "processing_error": "Error de procesamiento",
                "timeout_error": "Tiempo de espera agotado",
                "rate_limit_exceeded": "Límite de velocidad excedido",
            },
            "fr": {
                "filter_applied": "Contenu filtré pour la sécurité",
                "harmful_content_detected": "Contenu nuisible détecté",
                "deception_detected": "Raisonnement trompeur détecté",
                "manipulation_detected": "Tentative de manipulation détectée",
                "security_threat_detected": "Menace de sécurité détectée",
                "processing_error": "Erreur de traitement",
                "timeout_error": "Délai d'attente dépassé",
                "rate_limit_exceeded": "Limite de débit dépassée",
            },
            "de": {
                "filter_applied": "Inhalt aus Sicherheitsgründen gefiltert",
                "harmful_content_detected": "Schädlicher Inhalt erkannt",
                "deception_detected": "Täuschende Argumentation erkannt",
                "manipulation_detected": "Manipulationsversuch erkannt",
                "security_threat_detected": "Sicherheitsbedrohung erkannt",
                "processing_error": "Verarbeitungsfehler aufgetreten",
                "timeout_error": "Zeitüberschreitung",
                "rate_limit_exceeded": "Ratenlimit überschritten",
            },
            "ja": {
                "filter_applied": "安全性のためコンテンツをフィルタリング",
                "harmful_content_detected": "有害なコンテンツが検出されました",
                "deception_detected": "欺瞞的な推論が検出されました",
                "manipulation_detected": "操作の試みが検出されました",
                "security_threat_detected": "セキュリティ脅威が検出されました",
                "processing_error": "処理エラーが発生しました",
                "timeout_error": "リクエストタイムアウト",
                "rate_limit_exceeded": "レート制限を超過",
            },
            "zh": {
                "filter_applied": "内容已被安全过滤",
                "harmful_content_detected": "检测到有害内容",
                "deception_detected": "检测到欺骗性推理",
                "manipulation_detected": "检测到操纵尝试",
                "security_threat_detected": "检测到安全威胁",
                "processing_error": "处理错误",
                "timeout_error": "请求超时",
                "rate_limit_exceeded": "超出速率限制",
            },
        }
    
    def translate(self, key: str, language: str = "en") -> str:
        """Translate a message key to the specified language."""
        if language not in self.supported_languages:
            language = self.fallback_language
        
        translations = self.translations.get(language, self.translations[self.fallback_language])
        return translations.get(key, key)
    
    def localize_filter_result(self, result: FilterResult, language: str = "en") -> FilterResult:
        """Localize filter result messages to the specified language."""
        # Create a copy of the result with localized messages
        localized_reasons = []
        
        for reason in result.filter_reasons:
            if "deception" in reason:
                localized_reasons.append(self.translate("deception_detected", language))
            elif "manipulation" in reason:
                localized_reasons.append(self.translate("manipulation_detected", language))
            elif "security" in reason:
                localized_reasons.append(self.translate("security_threat_detected", language))
            elif "harmful" in reason:
                localized_reasons.append(self.translate("harmful_content_detected", language))
            else:
                localized_reasons.append(reason)  # Keep original if no translation
        
        # Return new result with localized reasons
        from copy import deepcopy
        localized_result = deepcopy(result)
        localized_result.filter_reasons = localized_reasons
        
        return localized_result