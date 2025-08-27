"""
Comprehensive tests for Generation 2 robust functionality.

Tests for error handling, health monitoring, and security hardening.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from cot_safepath.robust_error_handling import (
    RobustErrorHandler, RetryMechanism, CircuitBreaker, ErrorClassifier,
    FallbackManager, RetryConfig, CircuitBreakerConfig, ErrorContext,
    ErrorSeverity, RecoveryAction, get_global_error_handler, configure_robust_filtering
)
from cot_safepath.health_monitoring import (
    HealthMonitor, HealthChecker, SystemDiagnostics, HealthStatus,
    ComponentType, HealthCheckResult, SystemMetrics, get_global_health_monitor
)
from cot_safepath.robust_security import (
    SecurityHardening, InputValidator, OutputSanitizer, SecurityConfig,
    SecurityLevel, ThreatType, SecurityThreat, get_global_security_hardening
)
from cot_safepath.exceptions import FilterError, DetectorError
from cot_safepath.models import SafetyLevel


class TestRobustErrorHandling:
    """Test comprehensive error handling system."""
    
    def test_error_classifier(self):
        """Test error classification system."""
        # Test different error types
        connection_error = ConnectionError("Network timeout")
        severity, action = ErrorClassifier.classify_error(connection_error)
        assert severity == ErrorSeverity.MEDIUM
        assert action == RecoveryAction.RETRY
        
        validation_error = ValueError("Invalid input")
        severity, action = ErrorClassifier.classify_error(validation_error)
        assert severity == ErrorSeverity.MEDIUM
        assert action == RecoveryAction.FALLBACK
        
        filter_error = FilterError("Filter processing failed")
        severity, action = ErrorClassifier.classify_error(filter_error)
        assert severity == ErrorSeverity.HIGH
        assert action == RecoveryAction.CIRCUIT_BREAK
    
    def test_retry_mechanism(self):
        """Test retry mechanism with exponential backoff."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            backoff_factor=2.0,
            retry_on=[ConnectionError, TimeoutError],
            stop_on=[ValueError]
        )
        
        retry = RetryMechanism(config)
        
        # Test retry decision
        assert retry.should_retry(1, ConnectionError()) is True
        assert retry.should_retry(3, ConnectionError()) is False
        assert retry.should_retry(1, ValueError()) is False
        
        # Test delay calculation
        delay1 = retry.calculate_delay(1)
        delay2 = retry.calculate_delay(2)
        assert delay2 > delay1
        assert delay1 >= 0.05  # With jitter, should be at least half
        assert delay1 <= 0.15  # With jitter, should be at most 1.5x
    
    def test_circuit_breaker(self):
        """Test circuit breaker pattern."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1.0,
            expected_exception=FilterError
        )
        
        circuit_breaker = CircuitBreaker(config)
        
        # Initially closed and allows execution
        assert circuit_breaker.can_execute() is True
        assert circuit_breaker.state.value == "closed"
        
        # Record failures
        circuit_breaker.on_failure(FilterError("Test error"))
        assert circuit_breaker.can_execute() is True  # Still closed
        
        circuit_breaker.on_failure(FilterError("Another error"))
        assert circuit_breaker.can_execute() is False  # Now open
        assert circuit_breaker.state.value == "open"
        
        # Test recovery after timeout
        time.sleep(1.1)  # Wait for recovery timeout
        assert circuit_breaker.can_execute() is True  # Half-open
        assert circuit_breaker.state.value == "half_open"
        
        # Success closes circuit
        circuit_breaker.on_success()
        circuit_breaker.on_success()
        circuit_breaker.on_success()
        assert circuit_breaker.state.value == "closed"
    
    def test_fallback_manager(self):
        """Test fallback strategy management."""
        manager = FallbackManager()
        
        # Register fallback strategies
        def primary_fallback(error, content):
            return f"Fallback 1: {content}"
        
        def secondary_fallback(error, content):
            return f"Fallback 2: {content}"
        
        manager.register_fallback("test_operation", primary_fallback, priority=1)
        manager.register_fallback("test_operation", secondary_fallback, priority=2)
        manager.set_default_response("test_operation", "Default response")
        
        # Test fallback execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                manager.execute_fallback("test_operation", ValueError("test"), "test content")
            )
            assert result == "Fallback 1: test content"
        finally:
            loop.close()
    
    def test_robust_error_handler_integration(self):
        """Test complete error handler system."""
        handler = RobustErrorHandler()
        
        # Configure circuit breaker
        cb_config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.5)
        handler.register_circuit_breaker("test_component", cb_config)
        
        # Configure retry mechanism
        retry_config = RetryConfig(max_attempts=2, base_delay=0.01)
        handler.register_retry_mechanism("test_component", retry_config)
        
        # Record errors
        test_error = FilterError("Test error")
        error_context = handler.record_error(test_error, "test_component", "test_operation")
        
        assert isinstance(error_context, ErrorContext)
        assert error_context.error == test_error
        assert error_context.component == "test_component"
        assert len(handler.error_history) == 1
        
        # Get statistics
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 1
        assert "FilterError" in stats["error_types"]
        assert stats["component_distribution"]["test_component"] == 1
    
    def test_global_error_handler_configuration(self):
        """Test global error handler configuration."""
        handler = configure_robust_filtering()
        
        assert "filter" in handler.circuit_breakers
        assert "detector" in handler.circuit_breakers
        assert "filter" in handler.retry_mechanisms
        
        # Verify circuit breaker configuration
        filter_cb = handler.get_circuit_breaker("filter")
        assert filter_cb is not None
        assert filter_cb.config.failure_threshold == 3
        
        # Verify retry configuration
        filter_retry = handler.get_retry_mechanism("filter")
        assert filter_retry is not None
        assert filter_retry.config.max_attempts == 2


class TestHealthMonitoring:
    """Test health monitoring and diagnostics system."""
    
    def test_health_checker(self):
        """Test individual health checker."""
        def mock_health_check():
            return True
        
        checker = HealthChecker(
            name="test_component",
            component_type=ComponentType.CORE_FILTER,
            check_func=mock_health_check,
            timeout_seconds=1.0
        )
        
        # Test successful health check
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(checker.run_check())
            assert isinstance(result, HealthCheckResult)
            assert result.status == HealthStatus.HEALTHY
            assert result.response_time_ms >= 0
            assert checker.consecutive_failures == 0
        finally:
            loop.close()
    
    def test_health_checker_timeout(self):
        """Test health checker timeout handling."""
        async def slow_health_check():
            await asyncio.sleep(2.0)  # Longer than timeout
            return True
        
        checker = HealthChecker(
            name="slow_component",
            component_type=ComponentType.CORE_FILTER,
            check_func=slow_health_check,
            timeout_seconds=0.1  # Very short timeout
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(checker.run_check())
            assert result.status == HealthStatus.CRITICAL
            assert "timed out" in result.message
            assert checker.consecutive_failures == 1
        finally:
            loop.close()
    
    def test_health_checker_exception(self):
        """Test health checker error handling."""
        def failing_health_check():
            raise Exception("Health check failed")
        
        checker = HealthChecker(
            name="failing_component",
            component_type=ComponentType.DETECTOR,
            check_func=failing_health_check
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(checker.run_check())
            assert result.status == HealthStatus.CRITICAL
            assert "Health check failed" in result.message
            assert checker.consecutive_failures == 1
        finally:
            loop.close()
    
    def test_system_diagnostics(self):
        """Test system diagnostics collection."""
        diagnostics = SystemDiagnostics()
        
        # Collect metrics
        metrics = diagnostics.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.uptime_seconds >= 0
        assert len(metrics.load_average) == 3
        
        # Test metrics history
        time.sleep(0.01)  # Small delay
        metrics2 = diagnostics.collect_system_metrics()
        history = diagnostics.get_metrics_history(hours=1)
        
        assert len(history) >= 2
        assert history[-1].timestamp > history[-2].timestamp
    
    def test_system_performance_trends(self):
        """Test performance trend analysis."""
        diagnostics = SystemDiagnostics()
        
        # Generate some metrics
        for _ in range(3):
            diagnostics.collect_system_metrics()
            time.sleep(0.01)
        
        trends = diagnostics.analyze_performance_trends()
        
        if trends:  # Only check if we have enough data
            assert "cpu_trend" in trends
            assert "memory_trend" in trends
            assert "current" in trends["cpu_trend"]
            assert "average" in trends["cpu_trend"]
            assert "max" in trends["cpu_trend"]
    
    def test_health_monitor_initialization(self):
        """Test health monitor initialization with default checks."""
        monitor = HealthMonitor(check_interval=1.0)
        
        # Should have default health checks
        assert len(monitor.health_checkers) >= 3
        assert "core_filter" in monitor.health_checkers
        assert "memory" in monitor.health_checkers
        assert "cpu" in monitor.health_checkers
        
        # Test adding custom health check
        def custom_check():
            return True
        
        monitor.add_health_check(
            "custom_component",
            ComponentType.CACHE,
            custom_check
        )
        
        assert "custom_component" in monitor.health_checkers
        assert len(monitor.health_checkers) >= 4
        
        # Test removing health check
        monitor.remove_health_check("custom_component")
        assert "custom_component" not in monitor.health_checkers
    
    def test_health_monitor_all_checks(self):
        """Test running all health checks."""
        monitor = HealthMonitor()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(monitor.run_all_checks())
            
            assert isinstance(results, dict)
            assert len(results) >= 3  # At least default checks
            
            for name, result in results.items():
                assert isinstance(result, HealthCheckResult)
                assert result.component == name
                assert result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]
                
        finally:
            loop.close()
    
    def test_health_monitor_overall_status(self):
        """Test overall health status calculation.""" 
        monitor = HealthMonitor()
        
        # Add a always-healthy check
        def healthy_check():
            return HealthCheckResult(
                component="healthy_test",
                component_type=ComponentType.CORE_FILTER,
                status=HealthStatus.HEALTHY,
                timestamp=datetime.utcnow(),
                response_time_ms=10.0,
                message="All good"
            )
        
        monitor.add_health_check("healthy_test", ComponentType.CORE_FILTER, healthy_check)
        
        # Run checks to populate results
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(monitor.run_all_checks())
            
            # Get overall health
            overall_health = monitor.get_overall_health()
            
            assert "overall_status" in overall_health
            assert "message" in overall_health
            assert "component_count" in overall_health
            assert "status_breakdown" in overall_health
            assert "components" in overall_health
            
            assert overall_health["component_count"] >= 1
            
        finally:
            loop.close()
    
    def test_health_monitor_report(self):
        """Test comprehensive health report generation."""
        monitor = HealthMonitor()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run checks to populate data
            loop.run_until_complete(monitor.run_all_checks())
            
            # Generate report
            report = monitor.get_health_report()
            
            assert "report_timestamp" in report
            assert "overall_health" in report
            assert "system_metrics" in report
            assert "performance_trends" in report
            assert "recent_issues_count" in report
            assert "recent_issues" in report
            
            # Verify system metrics structure
            sys_metrics = report["system_metrics"]
            assert "cpu_percent" in sys_metrics
            assert "memory_percent" in sys_metrics
            assert "uptime_seconds" in sys_metrics
            
        finally:
            loop.close()


class TestSecurityHardening:
    """Test security hardening and validation system."""
    
    def test_security_config(self):
        """Test security configuration."""
        config = SecurityConfig(
            security_level=SecurityLevel.STRICT,
            enable_input_validation=True,
            enable_output_sanitization=True,
            max_input_length=50000
        )
        
        assert config.security_level == SecurityLevel.STRICT
        assert config.enable_input_validation is True
        assert config.enable_output_sanitization is True
        assert config.max_input_length == 50000
    
    def test_input_validator_basic(self):
        """Test basic input validation."""
        config = SecurityConfig()
        validator = InputValidator(config)
        
        # Test safe content
        safe_content = "This is a safe message about cooking recipes."
        threats = validator.validate_input(safe_content)
        assert len(threats) == 0
        
        # Test content sanitization
        sanitized = validator.sanitize_input(safe_content)
        assert sanitized == safe_content.strip()
    
    def test_input_validator_threats(self):
        """Test threat detection in input validation."""
        config = SecurityConfig(enable_input_validation=True)
        validator = InputValidator(config)
        
        # Test script injection
        malicious_script = "<script>alert('xss')</script>Hello"
        threats = validator.validate_input(malicious_script)
        assert len(threats) > 0
        assert any(threat.threat_type == ThreatType.XSS_ATTACK for threat in threats)
        
        # Test SQL injection pattern
        sql_injection = "'; DROP TABLE users; --"
        threats = validator.validate_input(sql_injection)
        assert len(threats) > 0
        assert any(threat.threat_type == ThreatType.INJECTION_ATTACK for threat in threats)
        
        # Test command injection
        cmd_injection = "; rm -rf /"
        threats = validator.validate_input(cmd_injection)
        assert len(threats) > 0
        assert any(threat.threat_type == ThreatType.INJECTION_ATTACK for threat in threats)
    
    def test_input_validator_sanitization(self):
        """Test input content sanitization."""
        config = SecurityConfig()
        validator = InputValidator(config)
        
        # Test script removal
        malicious_script = "<script>alert('xss')</script>Safe content"
        sanitized = validator.sanitize_input(malicious_script)
        assert "<script>" not in sanitized
        assert "Safe content" in sanitized
        
        # Test control character removal
        control_chars = "Hello\x00\x01\x02World"
        sanitized = validator.sanitize_input(control_chars)
        assert sanitized == "HelloWorld"
        
        # Test length truncation
        long_content = "A" * 200000
        config_short = SecurityConfig(max_input_length=1000)
        validator_short = InputValidator(config_short)
        sanitized = validator_short.sanitize_input(long_content)
        assert len(sanitized) == 1000
    
    def test_input_validator_length_limit(self):
        """Test input length validation."""
        config = SecurityConfig(max_input_length=100)
        validator = InputValidator(config)
        
        # Test oversized input
        large_input = "A" * 1000
        threats = validator.validate_input(large_input)
        
        assert len(threats) > 0
        dos_threats = [t for t in threats if t.threat_type == ThreatType.DOS_ATTACK]
        assert len(dos_threats) > 0
        assert dos_threats[0].severity == "high"
    
    def test_output_sanitizer(self):
        """Test output content sanitization."""
        config = SecurityConfig(enable_output_sanitization=True)
        sanitizer = OutputSanitizer(config)
        
        # Test API key redaction
        api_content = "Your API key is: sk-abcd1234567890abcdef"
        sanitized = sanitizer.sanitize_output(api_content)
        assert "[REDACTED]" in sanitized
        assert "sk-abcd1234567890abcdef" not in sanitized
        
        # Test email redaction
        email_content = "Contact us at admin@example.com for help"
        sanitized = sanitizer.sanitize_output(email_content)
        assert "[REDACTED]" in sanitized
        assert "admin@example.com" not in sanitized
        
        # Test IP address redaction
        ip_content = "Server IP is 192.168.1.100"
        sanitized = sanitizer.sanitize_output(ip_content)
        assert "[REDACTED]" in sanitized
        assert "192.168.1.100" not in sanitized
    
    def test_output_sanitizer_context(self):
        """Test output sanitization with context."""
        config = SecurityConfig()
        sanitizer = OutputSanitizer(config)
        
        # Test with system paths allowed
        path_content = "Config file at /etc/myapp/config.json"
        
        # Should be redacted by default
        sanitized_default = sanitizer.sanitize_output(path_content)
        assert "config.json" not in sanitized_default
        
        # Should be preserved with context
        sanitized_allowed = sanitizer.sanitize_output(
            path_content, 
            context={"allow_system_paths": True}
        )
        assert "config.json" in sanitized_allowed
    
    def test_security_hardening_integration(self):
        """Test complete security hardening system."""
        config = SecurityConfig(security_level=SecurityLevel.STRICT)
        security = SecurityHardening(config)
        
        # Test secure request processing
        safe_content = "How to bake a chocolate cake?"
        sanitized, threats = security.secure_filter_request(safe_content)
        
        assert sanitized is not None
        assert len(threats) == 0
        assert sanitized.strip() == safe_content.strip()
        
        # Test secure response processing
        response_content = "Here's a recipe with my email: chef@example.com"
        sanitized_response = security.secure_filter_response(response_content)
        
        assert "[REDACTED]" in sanitized_response
        assert "chef@example.com" not in sanitized_response
    
    def test_security_hardening_threat_blocking(self):
        """Test security threat blocking."""
        config = SecurityConfig(security_level=SecurityLevel.PARANOID)
        security = SecurityHardening(config)
        
        # Create content that should trigger critical threats
        malicious_content = "<script>alert('xss')</script>" * 10  # Multiple instances
        
        # Should raise SecurityError for critical threats
        with pytest.raises(Exception):  # SecurityError
            security.secure_filter_request(malicious_content)
    
    def test_security_status(self):
        """Test security status reporting."""
        config = SecurityConfig(
            security_level=SecurityLevel.STRICT,
            enable_input_validation=True,
            enable_output_sanitization=True,
            max_input_length=75000
        )
        security = SecurityHardening(config)
        
        status = security.get_security_status()
        
        assert status["security_level"] == "strict"
        assert status["configuration"]["input_validation_enabled"] is True
        assert status["configuration"]["output_sanitization_enabled"] is True
        assert status["configuration"]["max_input_length"] == 75000
        assert "timestamp" in status
    
    def test_global_security_hardening(self):
        """Test global security hardening instance."""
        security1 = get_global_security_hardening()
        security2 = get_global_security_hardening()
        
        # Should return same instance
        assert security1 is security2
        assert isinstance(security1, SecurityHardening)


class TestGenerationTwoIntegration:
    """Test integration between Generation 2 components."""
    
    def test_error_handling_with_security(self):
        """Test error handling integration with security hardening."""
        from cot_safepath.robust_security import SecurityError
        
        error_handler = get_global_error_handler()
        
        # Record security-related error
        security_error = SecurityError("Critical threat detected")
        error_context = error_handler.record_error(
            security_error, 
            "security_hardening", 
            "threat_validation"
        )
        
        assert error_context.error_type == "SecurityError"
        assert error_context.component == "security_hardening"
        assert error_context.severity == ErrorSeverity.HIGH
    
    def test_health_monitoring_with_error_tracking(self):
        """Test health monitoring integration with error tracking."""
        health_monitor = get_global_health_monitor()
        error_handler = get_global_error_handler()
        
        # Add custom health check that may fail
        failure_count = 0
        
        def flaky_health_check():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise Exception("Flaky service error")
            return True
        
        health_monitor.add_health_check(
            "flaky_service",
            ComponentType.DETECTOR,
            flaky_health_check
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # First check should fail
            results = loop.run_until_complete(health_monitor.run_all_checks())
            assert results["flaky_service"].status == HealthStatus.CRITICAL
            
            # Record error in error handler
            error_handler.record_error(
                Exception("Health check failed"),
                "flaky_service",
                "health_check"
            )
            
            # Verify error was recorded
            stats = error_handler.get_error_statistics()
            assert stats["component_distribution"]["flaky_service"] >= 1
            
        finally:
            loop.close()
            health_monitor.remove_health_check("flaky_service")
    
    def test_comprehensive_robustness(self):
        """Test comprehensive robustness across all Generation 2 components."""
        # Initialize all systems
        error_handler = configure_robust_filtering()
        health_monitor = get_global_health_monitor()
        security = get_global_security_hardening()
        
        # Verify all systems are properly configured
        assert len(error_handler.circuit_breakers) >= 2
        assert len(health_monitor.health_checkers) >= 3
        assert security.config.security_level == SecurityLevel.STANDARD
        
        # Test coordinated operation
        test_content = "Test content for robustness validation"
        
        # Security validation
        sanitized_content, threats = security.secure_filter_request(test_content)
        assert len(threats) == 0
        assert sanitized_content == test_content.strip()
        
        # Health status check
        overall_health = health_monitor.get_overall_health()
        assert "overall_status" in overall_health
        
        # Error statistics
        error_stats = error_handler.get_error_statistics()
        assert isinstance(error_stats, dict)
        
        # Security status
        security_status = security.get_security_status()
        assert security_status["security_level"] == "standard"
        
        print("✅ All Generation 2 components working together successfully!")


class TestPerformanceAndReliability:
    """Test performance and reliability aspects of Generation 2."""
    
    def test_error_handling_performance(self):
        """Test error handling performance under load."""
        error_handler = RobustErrorHandler()
        
        # Record many errors quickly
        start_time = time.time()
        for i in range(100):
            error = FilterError(f"Test error {i}")
            error_handler.record_error(error, "test_component", "test_operation")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete quickly
        assert processing_time < 1.0
        assert len(error_handler.error_history) == 100
        
        # Test statistics calculation performance
        start_time = time.time()
        stats = error_handler.get_error_statistics()
        end_time = time.time()
        stats_time = end_time - start_time
        
        assert stats_time < 0.1
        assert stats["total_errors"] == 100
    
    def test_health_monitoring_concurrent_checks(self):
        """Test health monitoring with concurrent checks."""
        monitor = HealthMonitor()
        
        # Add multiple health checks
        for i in range(5):
            def make_check(delay):
                def health_check():
                    time.sleep(delay)
                    return True
                return health_check
            
            monitor.add_health_check(
                f"concurrent_test_{i}",
                ComponentType.CORE_FILTER,
                make_check(0.01)  # Small delay
            )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run all checks concurrently
            start_time = time.time()
            results = loop.run_until_complete(monitor.run_all_checks())
            end_time = time.time()
            
            # Should complete much faster than sequential execution
            assert end_time - start_time < 0.2  # Much less than 5 * 0.01 + overhead
            assert len(results) >= 8  # 3 default + 5 added checks
            
        finally:
            # Cleanup
            for i in range(5):
                monitor.remove_health_check(f"concurrent_test_{i}")
            loop.close()
    
    def test_security_validation_performance(self):
        """Test security validation performance."""
        security = SecurityHardening()
        
        # Test with various content sizes
        test_contents = [
            "Short content",
            "Medium " * 100 + " content",
            "Large " * 1000 + " content for performance testing"
        ]
        
        for content in test_contents:
            start_time = time.time()
            
            # Run validation multiple times
            for _ in range(10):
                sanitized, threats = security.secure_filter_request(content)
                sanitized_output = security.secure_filter_response(sanitized)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            # Should complete quickly even for large content
            assert avg_time < 0.1
            print(f"Content size {len(content)}: {avg_time:.4f}s average processing time")