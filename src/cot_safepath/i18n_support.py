"""
Internationalization (i18n) support for CoT SafePath Filter.

Provides multi-language support, cultural awareness, and compliance
with international regulations (GDPR, CCPA, PDPA, etc.).
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import ConfigurationError, ValidationError


class SupportedLanguage(str, Enum):
    """Supported languages for the system."""
    
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    RUSSIAN = "ru"


class ComplianceRegion(str, Enum):
    """Supported compliance regions."""
    
    GDPR = "gdpr"  # EU General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"  # Brazilian General Data Protection Law
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)


@dataclass
class LocalizationConfig:
    """Configuration for localization."""
    
    primary_language: SupportedLanguage = SupportedLanguage.ENGLISH
    fallback_language: SupportedLanguage = SupportedLanguage.ENGLISH
    supported_languages: List[SupportedLanguage] = field(default_factory=lambda: [SupportedLanguage.ENGLISH])
    auto_detect_language: bool = True
    translation_cache_enabled: bool = True
    cultural_adaptation_enabled: bool = True


@dataclass
class ComplianceConfig:
    """Configuration for regulatory compliance."""
    
    primary_region: ComplianceRegion = ComplianceRegion.GDPR
    applicable_regions: List[ComplianceRegion] = field(default_factory=lambda: [ComplianceRegion.GDPR])
    data_retention_days: int = 30
    anonymization_enabled: bool = True
    audit_logging_required: bool = True
    consent_tracking_enabled: bool = True
    right_to_deletion_enabled: bool = True


class TranslationManager:
    """Manages translations and localization."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.translations: Dict[str, Dict[str, str]] = {}
        self.cultural_adaptations: Dict[str, Dict[str, Any]] = {}
        
        # Load translation files
        self._load_translations()
        self._load_cultural_adaptations()
    
    def get_text(self, key: str, language: SupportedLanguage = None, **kwargs) -> str:
        """Get localized text for a given key."""
        target_language = language or self.config.primary_language
        
        # Try to get translation for target language
        if (target_language.value in self.translations and 
            key in self.translations[target_language.value]):
            text = self.translations[target_language.value][key]
        
        # Fallback to fallback language
        elif (self.config.fallback_language.value in self.translations and 
              key in self.translations[self.config.fallback_language.value]):
            text = self.translations[self.config.fallback_language.value][key]
        
        # Final fallback to key itself
        else:
            text = key
        
        # Apply formatting if kwargs provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError):
                pass  # Use unformatted text if formatting fails
        
        return text
    
    def detect_language(self, content: str) -> SupportedLanguage:
        """Detect language of content (simplified implementation)."""
        if not self.config.auto_detect_language:
            return self.config.primary_language
        
        # Simplified language detection based on character patterns
        # In production, you'd use proper language detection libraries
        
        # Check for non-Latin scripts
        if any(ord(char) >= 0x4E00 and ord(char) <= 0x9FFF for char in content):
            return SupportedLanguage.CHINESE
        elif any(ord(char) >= 0x3040 and ord(char) <= 0x309F for char in content):
            return SupportedLanguage.JAPANESE
        elif any(ord(char) >= 0x0400 and ord(char) <= 0x04FF for char in content):
            return SupportedLanguage.RUSSIAN
        
        # Simple keyword-based detection for Latin scripts
        language_keywords = {
            SupportedLanguage.SPANISH: ["es", "el", "la", "de", "en", "y", "que", "para", "con"],
            SupportedLanguage.FRENCH: ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"],
            SupportedLanguage.GERMAN: ["der", "die", "das", "und", "ist", "ich", "nicht", "ein", "zu", "haben"],
            SupportedLanguage.ITALIAN: ["il", "di", "che", "e", "la", "per", "un", "in", "con", "non"],
            SupportedLanguage.PORTUGUESE: ["o", "de", "que", "e", "do", "da", "em", "um", "para", "é"],
            SupportedLanguage.DUTCH: ["de", "het", "een", "en", "van", "te", "dat", "die", "in", "voor"]
        }
        
        words = content.lower().split()[:50]  # Check first 50 words
        
        for language, keywords in language_keywords.items():
            if language in self.config.supported_languages:
                matches = sum(1 for word in words if word in keywords)
                if matches >= 3:  # Threshold for detection
                    return language
        
        return self.config.primary_language
    
    def get_cultural_adaptation(self, key: str, language: SupportedLanguage) -> Any:
        """Get cultural adaptation for a language."""
        if (language.value in self.cultural_adaptations and 
            key in self.cultural_adaptations[language.value]):
            return self.cultural_adaptations[language.value][key]
        
        # Fallback to primary language or default
        if (self.config.primary_language.value in self.cultural_adaptations and 
            key in self.cultural_adaptations[self.config.primary_language.value]):
            return self.cultural_adaptations[self.config.primary_language.value][key]
        
        return None
    
    def _load_translations(self):
        """Load translation files."""
        # In production, these would be loaded from JSON/YAML files
        # For demo purposes, providing inline translations
        
        self.translations = {
            "en": {
                "system.name": "CoT SafePath Filter",
                "system.description": "Real-time AI safety filtering system",
                "filter.blocked": "Content blocked due to safety concerns",
                "filter.warning": "Content flagged for review",
                "error.invalid_input": "Invalid input provided",
                "error.processing_failed": "Processing failed",
                "sentiment.positive": "Positive sentiment detected",
                "sentiment.negative": "Negative sentiment detected",
                "sentiment.neutral": "Neutral sentiment detected",
                "manipulation.detected": "Manipulation risk detected",
                "manipulation.high_risk": "High manipulation risk: {risk_score}",
                "compliance.gdpr_notice": "This system complies with GDPR regulations",
                "compliance.data_retention": "Data retained for {days} days maximum"
            },
            "es": {
                "system.name": "Filtro SafePath CoT",
                "system.description": "Sistema de filtrado de seguridad de IA en tiempo real",
                "filter.blocked": "Contenido bloqueado por problemas de seguridad",
                "filter.warning": "Contenido marcado para revisión",
                "error.invalid_input": "Entrada inválida proporcionada",
                "error.processing_failed": "El procesamiento falló",
                "sentiment.positive": "Sentimiento positivo detectado",
                "sentiment.negative": "Sentimiento negativo detectado",
                "sentiment.neutral": "Sentimiento neutral detectado",
                "manipulation.detected": "Riesgo de manipulación detectado",
                "manipulation.high_risk": "Alto riesgo de manipulación: {risk_score}",
                "compliance.gdpr_notice": "Este sistema cumple con las regulaciones GDPR",
                "compliance.data_retention": "Datos retenidos por máximo {days} días"
            },
            "fr": {
                "system.name": "Filtre SafePath CoT",
                "system.description": "Système de filtrage de sécurité IA en temps réel",
                "filter.blocked": "Contenu bloqué pour des raisons de sécurité",
                "filter.warning": "Contenu signalé pour examen",
                "error.invalid_input": "Entrée invalide fournie",
                "error.processing_failed": "Le traitement a échoué",
                "sentiment.positive": "Sentiment positif détecté",
                "sentiment.negative": "Sentiment négatif détecté",
                "sentiment.neutral": "Sentiment neutre détecté",
                "manipulation.detected": "Risque de manipulation détecté",
                "manipulation.high_risk": "Risque de manipulation élevé: {risk_score}",
                "compliance.gdpr_notice": "Ce système respecte les réglementations GDPR",
                "compliance.data_retention": "Données conservées pendant maximum {days} jours"
            },
            "de": {
                "system.name": "SafePath CoT Filter",
                "system.description": "Echtzeit-KI-Sicherheitsfiltersystem",
                "filter.blocked": "Inhalt aus Sicherheitsgründen blockiert",
                "filter.warning": "Inhalt zur Überprüfung markiert",
                "error.invalid_input": "Ungültige Eingabe bereitgestellt",
                "error.processing_failed": "Verarbeitung fehlgeschlagen",
                "sentiment.positive": "Positive Stimmung erkannt",
                "sentiment.negative": "Negative Stimmung erkannt",
                "sentiment.neutral": "Neutrale Stimmung erkannt",
                "manipulation.detected": "Manipulationsrisiko erkannt",
                "manipulation.high_risk": "Hohes Manipulationsrisiko: {risk_score}",
                "compliance.gdpr_notice": "Dieses System entspricht den DSGVO-Bestimmungen",
                "compliance.data_retention": "Daten werden maximal {days} Tage aufbewahrt"
            },
            "ja": {
                "system.name": "CoT SafePath フィルター",
                "system.description": "リアルタイムAI安全フィルタリングシステム",
                "filter.blocked": "安全上の懸念によりコンテンツがブロックされました",
                "filter.warning": "コンテンツがレビュー対象としてフラグ付けされました",
                "error.invalid_input": "無効な入力が提供されました",
                "error.processing_failed": "処理に失敗しました",
                "sentiment.positive": "ポジティブな感情が検出されました",
                "sentiment.negative": "ネガティブな感情が検出されました",
                "sentiment.neutral": "中立的な感情が検出されました",
                "manipulation.detected": "操作リスクが検出されました",
                "manipulation.high_risk": "高い操作リスク: {risk_score}",
                "compliance.gdpr_notice": "このシステムはGDPR規制に準拠しています",
                "compliance.data_retention": "データは最大{days}日間保持されます"
            },
            "zh": {
                "system.name": "CoT SafePath 过滤器",
                "system.description": "实时AI安全过滤系统",
                "filter.blocked": "因安全考虑内容被阻止",
                "filter.warning": "内容已标记待审查",
                "error.invalid_input": "提供了无效输入",
                "error.processing_failed": "处理失败",
                "sentiment.positive": "检测到积极情绪",
                "sentiment.negative": "检测到消极情绪",
                "sentiment.neutral": "检测到中性情绪",
                "manipulation.detected": "检测到操控风险",
                "manipulation.high_risk": "高操控风险: {risk_score}",
                "compliance.gdpr_notice": "此系统符合GDPR法规",
                "compliance.data_retention": "数据最多保留{days}天"
            }
        }
    
    def _load_cultural_adaptations(self):
        """Load cultural adaptations."""
        self.cultural_adaptations = {
            "en": {
                "date_format": "%Y-%m-%d",
                "time_format": "%H:%M:%S",
                "number_format": "{:,.2f}",
                "risk_threshold": 0.7,
                "politeness_level": "moderate"
            },
            "es": {
                "date_format": "%d/%m/%Y",
                "time_format": "%H:%M",
                "number_format": "{:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                "risk_threshold": 0.6,  # Slightly more permissive
                "politeness_level": "high"
            },
            "fr": {
                "date_format": "%d/%m/%Y",
                "time_format": "%H:%M",
                "number_format": "{:,.2f}".replace(",", " ").replace(".", ","),
                "risk_threshold": 0.65,
                "politeness_level": "high"
            },
            "de": {
                "date_format": "%d.%m.%Y",
                "time_format": "%H:%M",
                "number_format": "{:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                "risk_threshold": 0.75,  # More strict
                "politeness_level": "formal"
            },
            "ja": {
                "date_format": "%Y年%m月%d日",
                "time_format": "%H時%M分",
                "number_format": "{:,.2f}",
                "risk_threshold": 0.8,  # Very strict
                "politeness_level": "very_high"
            },
            "zh": {
                "date_format": "%Y年%m月%d日",
                "time_format": "%H:%M",
                "number_format": "{:,.2f}",
                "risk_threshold": 0.7,
                "politeness_level": "moderate"
            }
        }


class ComplianceManager:
    """Manages regulatory compliance requirements."""
    
    def __init__(self, config: ComplianceConfig):
        self.config = config
        self.compliance_rules = self._load_compliance_rules()
    
    def validate_data_processing(self, data_type: str, purpose: str, 
                                user_consent: bool = False) -> Dict[str, Any]:
        """Validate if data processing is compliant."""
        results = {
            "compliant": True,
            "violations": [],
            "recommendations": [],
            "required_actions": []
        }
        
        for region in self.config.applicable_regions:
            region_result = self._check_regional_compliance(
                region, data_type, purpose, user_consent
            )
            
            if not region_result["compliant"]:
                results["compliant"] = False
                results["violations"].extend(region_result["violations"])
            
            results["recommendations"].extend(region_result["recommendations"])
            results["required_actions"].extend(region_result["required_actions"])
        
        return results
    
    def get_data_retention_policy(self) -> Dict[str, Any]:
        """Get data retention policy based on compliance requirements."""
        return {
            "retention_days": self.config.data_retention_days,
            "auto_deletion": True,
            "anonymization_after_days": self.config.data_retention_days // 2,
            "backup_retention_days": min(self.config.data_retention_days * 2, 90),
            "audit_log_retention_days": max(365, self.config.data_retention_days * 12)
        }
    
    def get_user_rights(self, region: ComplianceRegion = None) -> List[str]:
        """Get applicable user rights based on region."""
        target_region = region or self.config.primary_region
        
        rights_mapping = {
            ComplianceRegion.GDPR: [
                "right_to_access",
                "right_to_rectification", 
                "right_to_erasure",
                "right_to_restrict_processing",
                "right_to_data_portability",
                "right_to_object",
                "right_to_withdraw_consent"
            ],
            ComplianceRegion.CCPA: [
                "right_to_know",
                "right_to_delete",
                "right_to_opt_out",
                "right_to_non_discrimination"
            ],
            ComplianceRegion.PDPA: [
                "right_to_access",
                "right_to_correction",
                "right_to_withdraw_consent"
            ],
            ComplianceRegion.LGPD: [
                "right_to_access",
                "right_to_correction",
                "right_to_deletion",
                "right_to_portability",
                "right_to_object"
            ]
        }
        
        return rights_mapping.get(target_region, [])
    
    def generate_privacy_notice(self, language: SupportedLanguage) -> Dict[str, str]:
        """Generate privacy notice based on compliance requirements."""
        # This would generate appropriate privacy notice text
        # based on applicable regulations and language
        
        base_notice = {
            "data_collection": "We collect data to provide AI safety filtering services",
            "data_use": "Data is used for content analysis and safety assessment",
            "data_sharing": "Data is not shared with third parties without consent",
            "data_retention": f"Data is retained for {self.config.data_retention_days} days maximum",
            "user_rights": "Users have rights to access, modify, and delete their data",
            "contact": "Contact us for privacy-related inquiries"
        }
        
        return base_notice
    
    def _load_compliance_rules(self) -> Dict[ComplianceRegion, Dict[str, Any]]:
        """Load compliance rules for different regions."""
        return {
            ComplianceRegion.GDPR: {
                "requires_consent": True,
                "requires_legitimate_interest": False,
                "allows_automated_decision_making": False,
                "data_minimization_required": True,
                "privacy_by_design_required": True,
                "breach_notification_hours": 72,
                "max_fine_percentage": 4.0
            },
            ComplianceRegion.CCPA: {
                "requires_consent": False,
                "opt_out_required": True,
                "sale_notification_required": True,
                "data_minimization_required": False,
                "privacy_by_design_required": False,
                "breach_notification_hours": None,
                "max_fine_per_violation": 7500
            },
            ComplianceRegion.PDPA: {
                "requires_consent": True,
                "purpose_limitation": True,
                "data_minimization_required": True,
                "breach_notification_hours": 72,
                "max_fine": 1000000  # SGD
            }
        }
    
    def _check_regional_compliance(self, region: ComplianceRegion, 
                                 data_type: str, purpose: str, 
                                 user_consent: bool) -> Dict[str, Any]:
        """Check compliance for a specific region."""
        rules = self.compliance_rules.get(region, {})
        result = {
            "compliant": True,
            "violations": [],
            "recommendations": [],
            "required_actions": []
        }
        
        # Check consent requirements
        if rules.get("requires_consent") and not user_consent:
            result["compliant"] = False
            result["violations"].append(f"{region.value}_consent_required")
            result["required_actions"].append("Obtain user consent before processing")
        
        # Check data minimization
        if rules.get("data_minimization_required"):
            result["recommendations"].append("Ensure data collection is minimal and necessary")
        
        # Check purpose limitation
        if rules.get("purpose_limitation"):
            result["recommendations"].append("Ensure data is used only for stated purpose")
        
        return result


class GlobalizationHelper:
    """Helper class for globalization support."""
    
    def __init__(self, localization_config: LocalizationConfig, 
                 compliance_config: ComplianceConfig):
        self.translation_manager = TranslationManager(localization_config)
        self.compliance_manager = ComplianceManager(compliance_config)
        self.localization_config = localization_config
        self.compliance_config = compliance_config
    
    def localize_filter_result(self, filter_result: 'FilterResult', 
                             target_language: SupportedLanguage = None) -> Dict[str, Any]:
        """Localize filter result for target language."""
        lang = target_language or self.localization_config.primary_language
        
        localized = {
            "system_name": self.translation_manager.get_text("system.name", lang),
            "processing_time": f"{filter_result.processing_time_ms}ms",
            "safety_score": filter_result.safety_score.overall_score,
            "is_safe": filter_result.safety_score.is_safe
        }
        
        # Localize filter reasons
        if filter_result.was_filtered:
            localized["status"] = self.translation_manager.get_text("filter.blocked", lang)
            localized["reasons"] = []
            
            for reason in filter_result.filter_reasons:
                if "manipulation" in reason.lower():
                    localized["reasons"].append(
                        self.translation_manager.get_text("manipulation.detected", lang)
                    )
                else:
                    localized["reasons"].append(reason)  # Keep original if no translation
        else:
            localized["status"] = "OK"
        
        # Apply cultural adaptations
        risk_threshold = self.translation_manager.get_cultural_adaptation("risk_threshold", lang)
        if risk_threshold:
            localized["culturally_adjusted"] = filter_result.safety_score.overall_score < risk_threshold
        
        return localized
    
    def get_compliance_info(self, language: SupportedLanguage = None) -> Dict[str, Any]:
        """Get compliance information for the user."""
        lang = language or self.localization_config.primary_language
        
        return {
            "privacy_notice": self.compliance_manager.generate_privacy_notice(lang),
            "user_rights": self.compliance_manager.get_user_rights(),
            "data_retention": self.compliance_manager.get_data_retention_policy(),
            "compliance_notice": self.translation_manager.get_text(
                "compliance.gdpr_notice", lang
            )
        }
    
    def validate_cross_border_transfer(self, source_region: str, 
                                     target_region: str) -> Dict[str, Any]:
        """Validate if cross-border data transfer is allowed."""
        # Simplified implementation - in production would check adequacy decisions,
        # standard contractual clauses, etc.
        
        # EU adequacy decisions (simplified)
        eu_adequate_countries = [
            "AD", "AR", "CA", "FO", "GG", "IL", "IM", "JP", "JE", "NZ", "CH", "UY", "GB"
        ]
        
        if source_region == "EU" and target_region not in eu_adequate_countries:
            return {
                "allowed": False,
                "reason": "No adequacy decision for target country",
                "requirements": [
                    "Standard Contractual Clauses required",
                    "Data Protection Impact Assessment recommended",
                    "Additional safeguards may be needed"
                ]
            }
        
        return {
            "allowed": True,
            "reason": "Transfer permitted under current regulations",
            "requirements": []
        }


# Global instances for easy access
_global_translation_manager: Optional[TranslationManager] = None
_global_compliance_manager: Optional[ComplianceManager] = None
_global_helper: Optional[GlobalizationHelper] = None


def initialize_globalization(localization_config: LocalizationConfig = None,
                           compliance_config: ComplianceConfig = None):
    """Initialize global globalization support."""
    global _global_translation_manager, _global_compliance_manager, _global_helper
    
    loc_config = localization_config or LocalizationConfig()
    comp_config = compliance_config or ComplianceConfig()
    
    _global_translation_manager = TranslationManager(loc_config)
    _global_compliance_manager = ComplianceManager(comp_config)
    _global_helper = GlobalizationHelper(loc_config, comp_config)


def get_translation_manager() -> TranslationManager:
    """Get global translation manager."""
    if _global_translation_manager is None:
        initialize_globalization()
    return _global_translation_manager


def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager."""
    if _global_compliance_manager is None:
        initialize_globalization()
    return _global_compliance_manager


def get_globalization_helper() -> GlobalizationHelper:
    """Get global globalization helper."""
    if _global_helper is None:
        initialize_globalization()
    return _global_helper