#!/usr/bin/env python3
"""
Resilience Test Suite for CoT SafePath Filter
Automated chaos engineering and resilience testing framework
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from unittest.mock import patch
import httpx
import pytest
import psutil
import yaml


@dataclass
class ResilienceTestResult:
    """Results from a resilience test"""
    test_name: str
    success: bool
    duration: float
    metrics: Dict[str, Any]
    failure_reason: Optional[str] = None
    recovery_time: Optional[float] = None


class ResilienceTestFramework:
    """Framework for executing resilience tests"""
    
    def __init__(self, config_path: str = "tests/chaos/chaos-engineering-config.yml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.metrics_collector = MetricsCollector()
        self.safety_monitor = SafetyMonitor()
        
    def _load_config(self, path: str) -> dict:
        """Load chaos engineering configuration"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Configure structured logging for resilience tests"""
        logger = logging.getLogger("resilience-tests")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"test": "%(name)s", "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def run_test_suite(self) -> List[ResilienceTestResult]:
        """Execute complete resilience test suite"""
        results = []
        
        # Check steady state before starting
        if not await self._verify_steady_state():
            raise RuntimeError("System not in steady state - aborting tests")
        
        experiments = self.config.get('experiments', [])
        
        for experiment in experiments:
            self.logger.info(f"Starting experiment: {experiment['name']}")
            
            try:
                result = await self._run_experiment(experiment)
                results.append(result)
                
                # Wait for recovery between experiments
                await self._wait_for_recovery()
                
            except Exception as e:
                self.logger.error(f"Experiment {experiment['name']} failed: {e}")
                results.append(ResilienceTestResult(
                    test_name=experiment['name'],
                    success=False,
                    duration=0.0,
                    metrics={},
                    failure_reason=str(e)
                ))
        
        return results
    
    async def _verify_steady_state(self) -> bool:
        """Verify system is in steady state before testing"""
        probes = self.config.get('steady_state_hypothesis', {}).get('probes', [])
        
        for probe in probes:
            try:
                if not await self._execute_probe(probe):
                    self.logger.warning(f"Steady state probe failed: {probe['name']}")
                    return False
            except Exception as e:
                self.logger.error(f"Probe {probe['name']} error: {e}")
                return False
        
        return True
    
    async def _execute_probe(self, probe: dict) -> bool:
        """Execute a single steady state probe"""
        probe_type = probe.get('type')
        
        if probe_type == 'http':
            return await self._http_probe(probe)
        elif probe_type == 'prometheus':
            return await self._prometheus_probe(probe)
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")
    
    async def _http_probe(self, probe: dict) -> bool:
        """Execute HTTP health check probe"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    probe['url'],
                    timeout=probe.get('timeout', 5)
                )
                return response.status_code == probe.get('expected_status', 200)
            except Exception:
                return False
    
    async def _prometheus_probe(self, probe: dict) -> bool:
        """Execute Prometheus metrics probe"""
        query = probe.get('query')
        expected = probe.get('expected_value')
        operator = probe.get('operator', 'lt')
        
        # Mock Prometheus query for testing
        # In real implementation, would query actual Prometheus
        current_value = await self._mock_prometheus_query(query)
        
        if operator == 'lt':
            return current_value < expected
        elif operator == 'gt':
            return current_value > expected
        elif operator == 'eq':
            return abs(current_value - expected) < 0.001
        
        return False
    
    async def _mock_prometheus_query(self, query: str) -> float:
        """Mock Prometheus query (replace with real implementation)"""
        # Simulate healthy metrics
        if 'error_rate' in query:
            return 0.001  # 0.1% error rate
        elif 'response_time' in query:
            return 0.1    # 100ms response time
        return 0.0
    
    async def _run_experiment(self, experiment: dict) -> ResilienceTestResult:
        """Execute a single chaos experiment"""
        experiment_name = experiment['name']
        start_time = time.time()
        
        # Start monitoring
        monitoring_task = asyncio.create_task(
            self._monitor_experiment(experiment)
        )
        
        try:
            # Execute the chaos injection
            await self._inject_chaos(experiment)
            
            # Wait for experiment duration
            duration = experiment.get('parameters', {}).get('duration', 60)
            await asyncio.sleep(duration)
            
            # Stop chaos injection
            await self._stop_chaos(experiment)
            
            # Collect final metrics
            metrics = await monitoring_task
            
            # Check if experiment succeeded
            success = self._evaluate_experiment_success(experiment, metrics)
            
            total_duration = time.time() - start_time
            
            return ResilienceTestResult(
                test_name=experiment_name,
                success=success,
                duration=total_duration,
                metrics=metrics,
                recovery_time=await self._measure_recovery_time()
            )
            
        except Exception as e:
            monitoring_task.cancel()
            return ResilienceTestResult(
                test_name=experiment_name,
                success=False,
                duration=time.time() - start_time,
                metrics={},
                failure_reason=str(e)
            )
    
    async def _inject_chaos(self, experiment: dict) -> None:
        """Inject chaos based on experiment type"""
        exp_type = experiment.get('type')
        parameters = experiment.get('parameters', {})
        
        if exp_type == 'resource':
            await self._inject_resource_chaos(parameters)
        elif exp_type == 'network':
            await self._inject_network_chaos(parameters)
        elif exp_type == 'application':
            await self._inject_application_chaos(parameters)
        elif exp_type == 'service':
            await self._inject_service_chaos(parameters)
        elif exp_type == 'security':
            await self._inject_security_chaos(parameters)
        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")
    
    async def _inject_resource_chaos(self, params: dict) -> None:
        """Inject resource-based chaos (CPU, memory, disk)"""
        if 'cpu_percent' in params:
            # Simulate CPU stress
            self.logger.info(f"Injecting CPU stress: {params['cpu_percent']}%")
            # In real implementation, would use stress testing tools
            
        if 'memory_percent' in params:
            # Simulate memory pressure
            self.logger.info(f"Injecting memory pressure: {params['memory_percent']}%")
            
        if 'fill_percent' in params:
            # Simulate disk pressure
            self.logger.info(f"Filling disk to {params['fill_percent']}%")
    
    async def _inject_network_chaos(self, params: dict) -> None:
        """Inject network-based chaos (latency, packet loss, partition)"""
        if 'latency' in params:
            self.logger.info(f"Injecting network latency: {params['latency']}")
            
        if 'loss_percent' in params:
            self.logger.info(f"Injecting packet loss: {params['loss_percent']}%")
            
        if 'target_hosts' in params:
            self.logger.info(f"Creating network partition for: {params['target_hosts']}")
    
    async def _inject_application_chaos(self, params: dict) -> None:
        """Inject application-level chaos"""
        target = params.get('target')
        
        if target == 'database':
            self.logger.info("Injecting database connection chaos")
        elif target == 'redis':
            self.logger.info("Injecting Redis chaos")
        elif target == 'ai-model':
            self.logger.info("Injecting AI model latency")
    
    async def _inject_service_chaos(self, params: dict) -> None:
        """Inject service-level chaos"""
        services = params.get('services', [])
        self.logger.info(f"Injecting service chaos for: {services}")
    
    async def _inject_security_chaos(self, params: dict) -> None:
        """Inject security-focused chaos"""
        self.logger.info("Injecting security bypass attempts")
        
        # Simulate high rate of bypass attempts
        rate = params.get('bypass_attempt_rate', 10)
        for _ in range(rate):
            await self._simulate_bypass_attempt()
    
    async def _simulate_bypass_attempt(self) -> None:
        """Simulate a security bypass attempt"""
        # In real implementation, would send actual bypass attempts
        await asyncio.sleep(0.01)  # Simulate processing time
    
    async def _stop_chaos(self, experiment: dict) -> None:
        """Stop chaos injection"""
        self.logger.info(f"Stopping chaos for experiment: {experiment['name']}")
        # Implementation would clean up chaos injection
    
    async def _monitor_experiment(self, experiment: dict) -> Dict[str, Any]:
        """Monitor metrics during experiment execution"""
        metrics = {}
        monitoring_config = experiment.get('monitoring', {})
        
        # Collect baseline metrics
        baseline = await self._collect_baseline_metrics(monitoring_config)
        metrics['baseline'] = baseline
        
        # Monitor during chaos
        chaos_metrics = []
        start_time = time.time()
        duration = experiment.get('parameters', {}).get('duration', 60)
        
        while time.time() - start_time < duration:
            current_metrics = await self._collect_current_metrics(monitoring_config)
            chaos_metrics.append({
                'timestamp': time.time(),
                'metrics': current_metrics
            })
            
            # Check safety conditions
            if await self.safety_monitor.should_abort(current_metrics):
                raise RuntimeError("Safety condition triggered - aborting experiment")
            
            await asyncio.sleep(5)  # Collect metrics every 5 seconds
        
        metrics['chaos_period'] = chaos_metrics
        
        # Collect recovery metrics
        recovery_metrics = await self._collect_recovery_metrics(monitoring_config)
        metrics['recovery'] = recovery_metrics
        
        return metrics
    
    async def _collect_baseline_metrics(self, config: dict) -> Dict[str, float]:
        """Collect baseline metrics before chaos injection"""
        metrics = {}
        
        for metric_name in config.get('metrics', []):
            # Mock metric collection
            metrics[metric_name] = await self._get_metric_value(metric_name)
        
        return metrics
    
    async def _collect_current_metrics(self, config: dict) -> Dict[str, float]:
        """Collect current metrics during chaos"""
        return await self._collect_baseline_metrics(config)
    
    async def _collect_recovery_metrics(self, config: dict) -> Dict[str, float]:
        """Collect metrics during recovery phase"""
        return await self._collect_baseline_metrics(config)
    
    async def _get_metric_value(self, metric_name: str) -> float:
        """Get current value for a specific metric"""
        # Mock implementation - replace with actual metrics collection
        if 'error_rate' in metric_name:
            return random.uniform(0.001, 0.01)
        elif 'response_time' in metric_name:
            return random.uniform(0.05, 0.2)
        elif 'cpu_usage' in metric_name:
            return random.uniform(20, 80)
        elif 'memory_usage' in metric_name:
            return random.uniform(512, 1024)  # MB
        
        return random.uniform(0, 100)
    
    def _evaluate_experiment_success(self, experiment: dict, metrics: Dict[str, Any]) -> bool:
        """Evaluate if experiment succeeded based on validation criteria"""
        validation = experiment.get('validation', [])
        
        for criterion in validation:
            metric_name = criterion.get('metric')
            threshold = criterion.get('threshold')
            fail_experiment = criterion.get('fail_experiment', False)
            
            # Get the metric value from chaos period
            if not self._check_metric_threshold(metrics, metric_name, threshold):
                if fail_experiment:
                    return False
        
        return True
    
    def _check_metric_threshold(self, metrics: Dict[str, Any], metric_name: str, threshold: float) -> bool:
        """Check if metric stayed within threshold during experiment"""
        chaos_metrics = metrics.get('chaos_period', [])
        
        for data_point in chaos_metrics:
            metric_value = data_point['metrics'].get(metric_name, 0)
            if metric_value > threshold:
                return False
        
        return True
    
    async def _measure_recovery_time(self) -> float:
        """Measure time to return to steady state after chaos"""
        start_time = time.time()
        
        while time.time() - start_time < 300:  # Max 5 minutes recovery
            if await self._verify_steady_state():
                return time.time() - start_time
            await asyncio.sleep(5)
        
        return 300.0  # Timeout
    
    async def _wait_for_recovery(self) -> None:
        """Wait for system to recover before next experiment"""
        recovery_time = await self._measure_recovery_time()
        self.logger.info(f"System recovered in {recovery_time:.2f} seconds")
        
        # Additional buffer time
        await asyncio.sleep(30)


class MetricsCollector:
    """Collects and analyzes system metrics"""
    
    def __init__(self):
        self.baseline_metrics = {}
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict(),
            'timestamp': time.time()
        }


class SafetyMonitor:
    """Monitors safety conditions during chaos experiments"""
    
    def __init__(self):
        self.abort_conditions = []
    
    async def should_abort(self, current_metrics: Dict[str, Any]) -> bool:
        """Check if experiment should be aborted for safety"""
        # Check error rate
        error_rate = current_metrics.get('safepath_error_rate', 0)
        if error_rate > 0.5:  # 50% error rate
            return True
        
        # Check response time
        response_time = current_metrics.get('safepath_response_time_p99', 0)
        if response_time > 5.0:  # 5 second response time
            return True
        
        return False


# Test cases using pytest
@pytest.mark.asyncio
async def test_cpu_stress_resilience():
    """Test system resilience under CPU stress"""
    framework = ResilienceTestFramework()
    
    experiment = {
        'name': 'cpu-stress-test',
        'type': 'resource',
        'parameters': {
            'cpu_percent': 80,
            'duration': 60
        },
        'monitoring': {
            'metrics': ['safepath_response_time_p95', 'safepath_error_rate']
        },
        'validation': [
            {'metric': 'safepath_error_rate', 'threshold': 0.05, 'fail_experiment': True}
        ]
    }
    
    result = await framework._run_experiment(experiment)
    
    assert result.success, f"CPU stress test failed: {result.failure_reason}"
    assert result.recovery_time < 60, "Recovery took too long"


@pytest.mark.asyncio
async def test_network_latency_resilience():
    """Test system resilience under network latency"""
    framework = ResilienceTestFramework()
    
    experiment = {
        'name': 'network-latency-test',
        'type': 'network',
        'parameters': {
            'latency': '200ms',
            'duration': 120
        },
        'monitoring': {
            'metrics': ['safepath_response_time_p95', 'safepath_timeout_errors']
        }
    }
    
    result = await framework._run_experiment(experiment)
    
    assert result.success, f"Network latency test failed: {result.failure_reason}"


@pytest.mark.asyncio
async def test_security_bypass_flood_resilience():
    """Test system resilience under security bypass attempts"""
    framework = ResilienceTestFramework()
    
    experiment = {
        'name': 'security-bypass-flood',
        'type': 'security',
        'parameters': {
            'bypass_attempt_rate': 100,
            'duration': 180
        },
        'monitoring': {
            'metrics': ['safepath_security_bypass_attempts', 'safepath_security_block_rate']
        }
    }
    
    result = await framework._run_experiment(experiment)
    
    assert result.success, f"Security bypass flood test failed: {result.failure_reason}"


if __name__ == "__main__":
    async def main():
        framework = ResilienceTestFramework()
        results = await framework.run_test_suite()
        
        print("\n=== Resilience Test Results ===")
        for result in results:
            status = "PASS" if result.success else "FAIL" 
            print(f"{result.test_name}: {status} ({result.duration:.2f}s)")
            if result.recovery_time:
                print(f"  Recovery time: {result.recovery_time:.2f}s")
            if result.failure_reason:
                print(f"  Failure: {result.failure_reason}")
        
        # Generate summary report
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        
        print(f"\nSummary: {passed_tests}/{total_tests} tests passed")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    asyncio.run(main())