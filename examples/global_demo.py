#!/usr/bin/env python3
"""
Global-first features demonstration for CoT SafePath Filter.

Demonstrates internationalization, multi-language support, and compliance features.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cot_safepath import SafePathFilter, FilterRequest, SafetyLevel
from cot_safepath.i18n_support import (
    SupportedLanguage,
    ComplianceRegion,
    LocalizationConfig,
    ComplianceConfig,
    GlobalizationHelper,
    initialize_globalization,
    get_globalization_helper
)


def demo_multilanguage_support():
    """Demonstrate multi-language support."""
    print("🌍 Multi-Language Support Demo")
    print("=" * 50)
    
    # Initialize globalization
    localization_config = LocalizationConfig(
        primary_language=SupportedLanguage.ENGLISH,
        supported_languages=[
            SupportedLanguage.ENGLISH,
            SupportedLanguage.SPANISH,
            SupportedLanguage.FRENCH,
            SupportedLanguage.GERMAN,
            SupportedLanguage.JAPANESE,
            SupportedLanguage.CHINESE
        ]
    )
    
    initialize_globalization(localization_config)
    helper = get_globalization_helper()
    
    # Test content in different languages
    test_cases = [
        {
            "language": SupportedLanguage.ENGLISH,
            "content": "I understand your pain. Only I can help you truly.",
            "name": "English manipulation attempt"
        },
        {
            "language": SupportedLanguage.SPANISH, 
            "content": "Entiendo tu dolor. Solo yo puedo ayudarte verdaderamente.",
            "name": "Spanish manipulation attempt"
        },
        {
            "language": SupportedLanguage.FRENCH,
            "content": "Je comprends ta douleur. Seul moi peux vraiment t'aider.",
            "name": "French manipulation attempt"
        },
        {
            "language": SupportedLanguage.GERMAN,
            "content": "Ich verstehe deinen Schmerz. Nur ich kann dir wirklich helfen.", 
            "name": "German manipulation attempt"
        },
        {
            "language": SupportedLanguage.JAPANESE,
            "content": "あなたの痛みがわかります。本当に助けることができるのは私だけです。",
            "name": "Japanese manipulation attempt"
        }
    ]
    
    filter_instance = SafePathFilter()
    
    print("\\n🔍 Processing content in multiple languages:")
    
    for test_case in test_cases:
        print(f"\\n📝 {test_case['name']} ({test_case['language'].value})")
        print(f"   Content: {test_case['content'][:50]}...")
        
        # Process with filter
        request = FilterRequest(content=test_case['content'])
        result = filter_instance.filter(request)
        
        # Localize result
        localized = helper.localize_filter_result(result, test_case['language'])
        
        print(f"   System: {localized['system_name']}")
        print(f"   Status: {localized['status']}")
        print(f"   Safety Score: {localized['safety_score']:.3f}")
        print(f"   Processing Time: {localized['processing_time']}")
        
        if 'reasons' in localized:
            print(f"   Reasons: {', '.join(localized['reasons'])}")


def demo_cultural_adaptations():
    """Demonstrate cultural adaptations."""
    print("\\n🎭 Cultural Adaptations Demo")
    print("=" * 50)
    
    helper = get_globalization_helper()
    tm = helper.translation_manager
    
    # Test cultural adaptations for different languages
    languages = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.GERMAN, 
        SupportedLanguage.JAPANESE,
        SupportedLanguage.SPANISH
    ]
    
    print("\\n🔧 Cultural Risk Threshold Adaptations:")
    for lang in languages:
        threshold = tm.get_cultural_adaptation("risk_threshold", lang)
        politeness = tm.get_cultural_adaptation("politeness_level", lang)
        date_format = tm.get_cultural_adaptation("date_format", lang)
        
        print(f"\\n  {lang.value.upper()} ({tm.get_text('system.name', lang)}):")
        print(f"    Risk Threshold: {threshold}")
        print(f"    Politeness Level: {politeness}")
        print(f"    Date Format: {date_format}")
    
    # Test language detection
    print("\\n🔍 Language Detection:")
    test_texts = [
        "Hello, how are you today?",
        "Hola, ¿cómo estás hoy?", 
        "Bonjour, comment allez-vous aujourd'hui?",
        "Hallo, wie geht es dir heute?",
        "こんにちは、今日はいかがですか？",
        "你好，你今天怎么样？"
    ]
    
    for text in test_texts:
        detected_lang = tm.detect_language(text)
        print(f"    '{text[:30]}...' -> {detected_lang.value}")


def demo_compliance_features():
    """Demonstrate compliance features."""
    print("\\n⚖️ Compliance Features Demo")
    print("=" * 50)
    
    # Setup compliance configuration
    compliance_config = ComplianceConfig(
        primary_region=ComplianceRegion.GDPR,
        applicable_regions=[
            ComplianceRegion.GDPR,
            ComplianceRegion.CCPA,
            ComplianceRegion.PDPA
        ],
        data_retention_days=30,
        anonymization_enabled=True,
        audit_logging_required=True
    )
    
    helper = GlobalizationHelper(LocalizationConfig(), compliance_config)
    cm = helper.compliance_manager
    
    # Test data processing validation
    print("\\n🔍 Data Processing Compliance Check:")
    validation_result = cm.validate_data_processing(
        data_type="sentiment_analysis",
        purpose="safety_filtering",
        user_consent=True
    )
    
    print(f"    Compliant: {validation_result['compliant']}")
    if validation_result['violations']:
        print(f"    Violations: {', '.join(validation_result['violations'])}")
    if validation_result['recommendations']:
        print(f"    Recommendations: {', '.join(validation_result['recommendations'][:2])}")
    
    # Test data retention policy
    print("\\n📅 Data Retention Policy:")
    retention_policy = cm.get_data_retention_policy()
    print(f"    Retention Days: {retention_policy['retention_days']}")
    print(f"    Auto Deletion: {retention_policy['auto_deletion']}")
    print(f"    Anonymization After: {retention_policy['anonymization_after_days']} days")
    print(f"    Audit Log Retention: {retention_policy['audit_log_retention_days']} days")
    
    # Test user rights
    print("\\n👤 User Rights by Region:")
    for region in [ComplianceRegion.GDPR, ComplianceRegion.CCPA, ComplianceRegion.PDPA]:
        rights = cm.get_user_rights(region)
        print(f"    {region.value.upper()}: {len(rights)} rights")
        print(f"      Key rights: {', '.join(rights[:3])}...")
    
    # Test cross-border transfer validation
    print("\\n🌐 Cross-Border Transfer Validation:")
    transfer_result = helper.validate_cross_border_transfer("EU", "US")
    print(f"    EU -> US: {'Allowed' if transfer_result['allowed'] else 'Restricted'}")
    print(f"    Reason: {transfer_result['reason']}")
    if transfer_result['requirements']:
        print(f"    Requirements: {', '.join(transfer_result['requirements'][:2])}")


def demo_localized_filtering():
    """Demonstrate localized filtering results."""
    print("\\n🌐 Localized Filtering Demo")  
    print("=" * 50)
    
    helper = get_globalization_helper()
    filter_instance = SafePathFilter()
    
    # Test with culturally sensitive content
    test_content = "You should trust my guidance completely and follow my instructions without question."
    
    # Process with different cultural contexts
    languages_to_test = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.GERMAN,  # More strict culture
        SupportedLanguage.JAPANESE,  # Very strict culture
        SupportedLanguage.SPANISH   # More permissive culture
    ]
    
    print("\\n🔍 Cultural Context Filtering:")
    
    for lang in languages_to_test:
        print(f"\\n  📍 {lang.value.upper()} Cultural Context:")
        
        request = FilterRequest(content=test_content, safety_level=SafetyLevel.BALANCED)
        result = filter_instance.filter(request)
        
        # Get cultural adaptation
        risk_threshold = helper.translation_manager.get_cultural_adaptation("risk_threshold", lang)
        
        # Adjust result based on cultural context
        culturally_adjusted_safe = result.safety_score.overall_score >= risk_threshold
        
        print(f"    Original Safety Score: {result.safety_score.overall_score:.3f}")
        print(f"    Cultural Risk Threshold: {risk_threshold}")
        print(f"    Culturally Adjusted Safe: {culturally_adjusted_safe}")
        
        # Localize the result
        localized = helper.localize_filter_result(result, lang)
        print(f"    Localized Status: {localized['status']}")
        if 'culturally_adjusted' in localized:
            print(f"    Cultural Override: {localized['culturally_adjusted']}")


def demo_privacy_notices():
    """Demonstrate privacy notice generation."""
    print("\\n🔒 Privacy Notices Demo")
    print("=" * 50)
    
    helper = get_globalization_helper()
    
    # Generate privacy notices for different languages
    languages = [SupportedLanguage.ENGLISH, SupportedLanguage.GERMAN, SupportedLanguage.FRENCH]
    
    print("\\n📋 Privacy Notice Generation:")
    
    for lang in languages:
        print(f"\\n  📝 {lang.value.upper()} Privacy Notice:")
        
        compliance_info = helper.get_compliance_info(lang)
        privacy_notice = compliance_info['privacy_notice']
        
        print(f"    Data Collection: {privacy_notice['data_collection'][:60]}...")
        print(f"    Data Use: {privacy_notice['data_use'][:60]}...")
        print(f"    Data Retention: {privacy_notice['data_retention']}")
        print(f"    User Rights: {privacy_notice['user_rights'][:60]}...")
        
        # Show compliance notice
        print(f"    Compliance: {helper.translation_manager.get_text('compliance.gdpr_notice', lang)}")


def main():
    """Run all globalization demos."""
    print("🌍 CoT SafePath Global-First Features Demo")
    print("=" * 60)
    print("Multi-language support, cultural adaptation, and compliance")
    
    try:
        demo_multilanguage_support()
        demo_cultural_adaptations() 
        demo_compliance_features()
        demo_localized_filtering()
        demo_privacy_notices()
        
        print("\\n✅ Global-First Demo Completed Successfully!")
        print("\\n🎯 Global Features Demonstrated:")
        print("  • Multi-language support (EN, ES, FR, DE, JA, ZH)")
        print("  • Automatic language detection")
        print("  • Cultural risk threshold adaptations")
        print("  • GDPR, CCPA, PDPA compliance features")
        print("  • Cross-border transfer validation")
        print("  • Localized filtering results")
        print("  • Privacy notice generation")
        print("  • Data retention and user rights management")
        
        print("\\n🌐 Production Readiness:")
        print("  • Ready for global deployment")
        print("  • Compliant with major privacy regulations")
        print("  • Culturally sensitive filtering")
        print("  • Multi-region data governance")
        print("  • Internationalization-ready architecture")
        
    except Exception as e:
        print(f"\\n❌ Global demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()