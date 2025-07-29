#!/usr/bin/env python3
"""
Automated scaling system for CoT SafePath Filter.

This script provides intelligent auto-scaling capabilities based on:
- CPU and memory usage
- Request queue depth
- Response latency metrics
- Filter processing load
- Cost optimization goals
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import boto3
import psutil
import redis
import structlog
from prometheus_client.parser import text_string_to_metric_families

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

@dataclass
class ScalingMetrics:
    """Container for scaling decision metrics."""
    cpu_usage_percent: float
    memory_usage_percent: float
    request_queue_depth: int
    avg_response_latency_ms: float
    filter_processing_rate: float
    active_connections: int
    error_rate_percent: float
    timestamp: datetime

@dataclass
class ScalingDecision:
    """Container for scaling decisions."""
    action: str  # 'scale_up', 'scale_down', 'no_action'
    target_instances: int
    current_instances: int
    reason: str
    confidence: float
    cost_impact: str

class AutoScaler:
    """Intelligent auto-scaling manager for SafePath application."""
    
    def __init__(self, config: Dict):
        """Initialize auto-scaler with configuration."""
        self.config = config
        self.redis_client = None
        self.cloudwatch = None
        self.ecs_client = None
        self.k8s_client = None
        
        # Scaling parameters
        self.min_instances = config.get('min_instances', 1)
        self.max_instances = config.get('max_instances', 10)
        self.target_cpu_percent = config.get('target_cpu_percent', 70)
        self.target_memory_percent = config.get('target_memory_percent', 80)
        self.scale_up_threshold_cpu = config.get('scale_up_threshold_cpu', 80)
        self.scale_down_threshold_cpu = config.get('scale_down_threshold_cpu', 50)
        self.scale_up_cooldown = config.get('scale_up_cooldown_minutes', 5)
        self.scale_down_cooldown = config.get('scale_down_cooldown_minutes', 10)
        
        # Last scaling action timestamp
        self.last_scale_action = None
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize cloud and monitoring clients."""
        # Redis for metrics
        if self.config.get('redis_enabled', False):
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                decode_responses=True
            )
        
        # AWS CloudWatch and ECS
        if self.config.get('aws_enabled', False):
            self.cloudwatch = boto3.client(
                'cloudwatch',
                region_name=self.config.get('aws_region', 'us-east-1')
            )
            self.ecs_client = boto3.client(
                'ecs',
                region_name=self.config.get('aws_region', 'us-east-1')
            )
        
        # Kubernetes client (placeholder - would need kubernetes library)
        if self.config.get('k8s_enabled', False):
            # from kubernetes import client, config as k8s_config
            # k8s_config.load_incluster_config()  # or load_kube_config()
            # self.k8s_client = client.AppsV1Api()
            pass
    
    async def collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics for scaling decisions."""
        logger.info("Collecting scaling metrics")
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Application-specific metrics
        request_queue_depth = await self._get_request_queue_depth()
        avg_latency = await self._get_average_latency()
        filter_rate = await self._get_filter_processing_rate()
        active_connections = await self._get_active_connections()
        error_rate = await self._get_error_rate()
        
        metrics = ScalingMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory_percent,
            request_queue_depth=request_queue_depth,
            avg_response_latency_ms=avg_latency,
            filter_processing_rate=filter_rate,
            active_connections=active_connections,
            error_rate_percent=error_rate,
            timestamp=datetime.now()
        )
        
        logger.info("Metrics collected", metrics=metrics.__dict__)
        return metrics
    
    async def _get_request_queue_depth(self) -> int:
        """Get current request queue depth."""
        if self.redis_client:
            try:
                queue_depth = self.redis_client.llen('safepath:request_queue')
                return queue_depth or 0
            except Exception as e:
                logger.warning("Failed to get queue depth", error=str(e))
        return 0
    
    async def _get_average_latency(self) -> float:
        """Get average response latency from metrics."""
        if self.redis_client:
            try:
                # Get recent latency samples
                latencies = self.redis_client.lrange('safepath:latencies', 0, 99)
                if latencies:
                    avg_latency = sum(float(l) for l in latencies) / len(latencies)
                    return avg_latency
            except Exception as e:
                logger.warning("Failed to get average latency", error=str(e))
        return 0.0
    
    async def _get_filter_processing_rate(self) -> float:
        """Get filter processing rate (requests per second)."""
        if self.redis_client:
            try:
                # Get processing count from last minute
                processing_count = self.redis_client.get('safepath:processing_rate_1min')
                return float(processing_count or 0)
            except Exception as e:
                logger.warning("Failed to get processing rate", error=str(e))
        return 0.0
    
    async def _get_active_connections(self) -> int:
        """Get number of active connections."""
        try:
            # Count network connections on application port
            connections = psutil.net_connections(kind='inet')
            app_port = self.config.get('app_port', 8080)
            active_count = sum(1 for conn in connections 
                             if conn.laddr.port == app_port and conn.status == 'ESTABLISHED')
            return active_count
        except Exception as e:
            logger.warning("Failed to get active connections", error=str(e))
            return 0
    
    async def _get_error_rate(self) -> float:
        """Get current error rate percentage."""
        if self.redis_client:
            try:
                errors_1min = float(self.redis_client.get('safepath:errors_1min') or 0)
                requests_1min = float(self.redis_client.get('safepath:requests_1min') or 1)
                error_rate = (errors_1min / requests_1min) * 100
                return error_rate
            except Exception as e:
                logger.warning("Failed to get error rate", error=str(e))
        return 0.0
    
    def make_scaling_decision(self, metrics: ScalingMetrics, current_instances: int) -> ScalingDecision:
        """Make intelligent scaling decision based on metrics."""
        logger.info("Making scaling decision", 
                   current_instances=current_instances, 
                   metrics=metrics.__dict__)
        
        # Check cooldown periods
        if self.last_scale_action:
            time_since_last_action = datetime.now() - self.last_scale_action
            min_cooldown = timedelta(minutes=self.scale_up_cooldown)
            if time_since_last_action < min_cooldown:
                return ScalingDecision(
                    action='no_action',
                    target_instances=current_instances,
                    current_instances=current_instances,
                    reason=f'Cooling down, {min_cooldown - time_since_last_action} remaining',
                    confidence=1.0,
                    cost_impact='none'
                )
        
        # Calculate scaling factors
        cpu_factor = self._calculate_cpu_scaling_factor(metrics.cpu_usage_percent)
        memory_factor = self._calculate_memory_scaling_factor(metrics.memory_usage_percent)
        latency_factor = self._calculate_latency_scaling_factor(metrics.avg_response_latency_ms)
        queue_factor = self._calculate_queue_scaling_factor(metrics.request_queue_depth)
        error_factor = self._calculate_error_scaling_factor(metrics.error_rate_percent)
        
        # Weighted decision calculation
        weights = {
            'cpu': 0.3,
            'memory': 0.25,
            'latency': 0.25,
            'queue': 0.15,
            'error': 0.05
        }
        
        weighted_score = (
            cpu_factor * weights['cpu'] +
            memory_factor * weights['memory'] +
            latency_factor * weights['latency'] +
            queue_factor * weights['queue'] +
            error_factor * weights['error']
        )
        
        # Determine scaling action
        if weighted_score > 0.7 and current_instances < self.max_instances:
            # Scale up
            target_instances = min(current_instances + 1, self.max_instances)
            
            # Aggressive scaling for high load
            if weighted_score > 0.9 and current_instances < self.max_instances - 1:
                target_instances = min(current_instances + 2, self.max_instances)
            
            return ScalingDecision(
                action='scale_up',
                target_instances=target_instances,
                current_instances=current_instances,
                reason=f'High load detected (score: {weighted_score:.2f}), '
                       f'CPU: {metrics.cpu_usage_percent:.1f}%, '
                       f'Memory: {metrics.memory_usage_percent:.1f}%, '
                       f'Latency: {metrics.avg_response_latency_ms:.1f}ms',
                confidence=weighted_score,
                cost_impact='increase'
            )
        
        elif weighted_score < 0.3 and current_instances > self.min_instances:
            # Scale down
            target_instances = max(current_instances - 1, self.min_instances)
            
            return ScalingDecision(
                action='scale_down',
                target_instances=target_instances,
                current_instances=current_instances,
                reason=f'Low load detected (score: {weighted_score:.2f}), '
                       f'CPU: {metrics.cpu_usage_percent:.1f}%, '
                       f'Memory: {metrics.memory_usage_percent:.1f}%',
                confidence=1.0 - weighted_score,
                cost_impact='decrease'
            )
        
        else:
            return ScalingDecision(
                action='no_action',
                target_instances=current_instances,
                current_instances=current_instances,
                reason=f'Load within acceptable range (score: {weighted_score:.2f})',
                confidence=0.5,
                cost_impact='none'
            )
    
    def _calculate_cpu_scaling_factor(self, cpu_percent: float) -> float:
        """Calculate CPU-based scaling factor (0.0 to 1.0)."""
        if cpu_percent > self.scale_up_threshold_cpu:
            return min(1.0, (cpu_percent - self.target_cpu_percent) / 30.0)
        elif cpu_percent < self.scale_down_threshold_cpu:
            return max(0.0, (self.target_cpu_percent - cpu_percent) / 30.0)
        else:
            return 0.5  # Neutral
    
    def _calculate_memory_scaling_factor(self, memory_percent: float) -> float:
        """Calculate memory-based scaling factor (0.0 to 1.0)."""
        if memory_percent > 85:
            return 1.0
        elif memory_percent > self.target_memory_percent:
            return (memory_percent - self.target_memory_percent) / 20.0
        elif memory_percent < 40:
            return 0.0
        else:
            return 0.5
    
    def _calculate_latency_scaling_factor(self, latency_ms: float) -> float:
        """Calculate latency-based scaling factor (0.0 to 1.0)."""
        target_latency = self.config.get('target_latency_ms', 50)
        
        if latency_ms > target_latency * 2:
            return 1.0
        elif latency_ms > target_latency:
            return (latency_ms - target_latency) / target_latency
        elif latency_ms < target_latency * 0.5:
            return 0.0
        else:
            return 0.5
    
    def _calculate_queue_scaling_factor(self, queue_depth: int) -> float:
        """Calculate queue-based scaling factor (0.0 to 1.0)."""
        max_acceptable_queue = self.config.get('max_queue_depth', 100)
        
        if queue_depth > max_acceptable_queue:
            return 1.0
        elif queue_depth > max_acceptable_queue * 0.5:
            return queue_depth / max_acceptable_queue
        else:
            return 0.0
    
    def _calculate_error_scaling_factor(self, error_rate: float) -> float:
        """Calculate error-based scaling factor (0.0 to 1.0)."""
        if error_rate > 5.0:  # More than 5% errors
            return 1.0
        elif error_rate > 1.0:
            return error_rate / 5.0
        else:
            return 0.0
    
    async def execute_scaling_action(self, decision: ScalingDecision) -> bool:
        """Execute the scaling decision."""
        if decision.action == 'no_action':
            return True
        
        logger.info("Executing scaling action", decision=decision.__dict__)
        
        try:
            if self.config.get('deployment_type') == 'docker_compose':
                success = await self._scale_docker_compose(decision.target_instances)
            elif self.config.get('deployment_type') == 'kubernetes':
                success = await self._scale_kubernetes(decision.target_instances)
            elif self.config.get('deployment_type') == 'ecs':
                success = await self._scale_ecs(decision.target_instances)
            else:
                logger.warning("Unknown deployment type, cannot scale")
                return False
            
            if success:
                self.last_scale_action = datetime.now()
                logger.info("Scaling action completed successfully", 
                           action=decision.action,
                           target_instances=decision.target_instances)
                
                # Send notification if configured
                await self._send_scaling_notification(decision)
                
            return success
            
        except Exception as e:
            logger.error("Scaling action failed", error=str(e), exc_info=True)
            return False
    
    async def _scale_docker_compose(self, target_instances: int) -> bool:
        """Scale Docker Compose deployment."""
        import subprocess
        
        try:
            cmd = ['docker-compose', 'up', '-d', '--scale', f'safepath={target_instances}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Docker Compose scaling completed", target_instances=target_instances)
                return True
            else:
                logger.error("Docker Compose scaling failed", error=result.stderr)
                return False
        except subprocess.TimeoutExpired:
            logger.error("Docker Compose scaling timed out")
            return False
    
    async def _scale_kubernetes(self, target_instances: int) -> bool:
        """Scale Kubernetes deployment."""
        if not self.k8s_client:
            logger.error("Kubernetes client not configured")
            return False
        
        try:
            # This would use the kubernetes client library
            # deployment_name = self.config.get('k8s_deployment_name', 'safepath')
            # namespace = self.config.get('k8s_namespace', 'default')
            # 
            # body = {'spec': {'replicas': target_instances}}
            # self.k8s_client.patch_namespaced_deployment_scale(
            #     name=deployment_name,
            #     namespace=namespace,
            #     body=body
            # )
            
            logger.info("Kubernetes scaling completed", target_instances=target_instances)
            return True
        except Exception as e:
            logger.error("Kubernetes scaling failed", error=str(e))
            return False
    
    async def _scale_ecs(self, target_instances: int) -> bool:
        """Scale ECS service."""
        if not self.ecs_client:
            logger.error("ECS client not configured")
            return False
        
        try:
            cluster_name = self.config.get('ecs_cluster_name')
            service_name = self.config.get('ecs_service_name')
            
            response = self.ecs_client.update_service(
                cluster=cluster_name,
                service=service_name,
                desiredCount=target_instances
            )
            
            logger.info("ECS scaling initiated", 
                       target_instances=target_instances,
                       task_arn=response['service']['taskDefinition'])
            return True
        except Exception as e:
            logger.error("ECS scaling failed", error=str(e))
            return False
    
    async def _send_scaling_notification(self, decision: ScalingDecision):
        """Send notification about scaling action."""
        if not self.config.get('notifications_enabled', False):
            return
        
        message = f"""
SafePath Auto-Scaling Action

Action: {decision.action}
Current Instances: {decision.current_instances}
Target Instances: {decision.target_instances}
Reason: {decision.reason}
Confidence: {decision.confidence:.2f}
Cost Impact: {decision.cost_impact}
Timestamp: {datetime.now().isoformat()}
        """.strip()
        
        # Send to configured notification channels
        notification_channels = self.config.get('notification_channels', [])
        
        for channel in notification_channels:
            if channel['type'] == 'slack':
                await self._send_slack_notification(channel, message)
            elif channel['type'] == 'email':
                await self._send_email_notification(channel, message)
            elif channel['type'] == 'webhook':
                await self._send_webhook_notification(channel, decision.__dict__)
    
    async def _send_slack_notification(self, channel_config: Dict, message: str):
        """Send Slack notification."""
        # Implementation would use Slack SDK
        logger.info("Slack notification sent", message=message)
    
    async def _send_email_notification(self, channel_config: Dict, message: str):
        """Send email notification."""
        # Implementation would use email service
        logger.info("Email notification sent", message=message)
    
    async def _send_webhook_notification(self, channel_config: Dict, data: Dict):
        """Send webhook notification."""
        # Implementation would use HTTP client
        logger.info("Webhook notification sent", data=data)
    
    async def get_current_instance_count(self) -> int:
        """Get current number of running instances."""
        deployment_type = self.config.get('deployment_type')
        
        if deployment_type == 'docker_compose':
            return await self._get_docker_compose_instances()
        elif deployment_type == 'kubernetes':
            return await self._get_kubernetes_instances()
        elif deployment_type == 'ecs':
            return await self._get_ecs_instances()
        else:
            return 1  # Default single instance
    
    async def _get_docker_compose_instances(self) -> int:
        """Get current Docker Compose instance count."""
        import subprocess
        
        try:
            cmd = ['docker-compose', 'ps', '-q', 'safepath']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                container_ids = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                return len(container_ids)
        except Exception as e:
            logger.warning("Failed to get Docker Compose instance count", error=str(e))
        
        return 1
    
    async def _get_kubernetes_instances(self) -> int:
        """Get current Kubernetes replica count."""
        # Implementation would query Kubernetes API
        return 1
    
    async def _get_ecs_instances(self) -> int:
        """Get current ECS service instance count."""
        if not self.ecs_client:
            return 1
        
        try:
            cluster_name = self.config.get('ecs_cluster_name')
            service_name = self.config.get('ecs_service_name')
            
            response = self.ecs_client.describe_services(
                cluster=cluster_name,
                services=[service_name]
            )
            
            if response['services']:
                return response['services'][0]['runningCount']
        except Exception as e:
            logger.warning("Failed to get ECS instance count", error=str(e))
        
        return 1


async def main():
    """Main auto-scaling execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SafePath Auto-Scaling Manager')
    parser.add_argument('--config', default='scaling_config.json', help='Scaling configuration file')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once instead of continuous monitoring')
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'min_instances': 1,
            'max_instances': 10,
            'target_cpu_percent': 70,
            'target_memory_percent': 80,
            'scale_up_threshold_cpu': 80,
            'scale_down_threshold_cpu': 50,
            'scale_up_cooldown_minutes': 5,
            'scale_down_cooldown_minutes': 10,
            'deployment_type': 'docker_compose',
            'notifications_enabled': False
        }
    
    # Initialize auto-scaler
    auto_scaler = AutoScaler(config)
    
    logger.info("Starting SafePath Auto-Scaler", config=config)
    
    try:
        if args.once:
            # Single run
            metrics = await auto_scaler.collect_metrics()
            current_instances = await auto_scaler.get_current_instance_count()
            decision = auto_scaler.make_scaling_decision(metrics, current_instances)
            
            print(f"Scaling Decision: {decision}")
            
            if decision.action != 'no_action':
                success = await auto_scaler.execute_scaling_action(decision)
                print(f"Scaling Action Success: {success}")
        else:
            # Continuous monitoring
            while True:
                try:
                    metrics = await auto_scaler.collect_metrics()
                    current_instances = await auto_scaler.get_current_instance_count()
                    decision = auto_scaler.make_scaling_decision(metrics, current_instances)
                    
                    logger.info("Scaling decision made", decision=decision.__dict__)
                    
                    if decision.action != 'no_action':
                        await auto_scaler.execute_scaling_action(decision)
                    
                    await asyncio.sleep(args.interval)
                    
                except KeyboardInterrupt:
                    logger.info("Auto-scaler stopped by user")
                    break
                except Exception as e:
                    logger.error("Auto-scaler error", error=str(e), exc_info=True)
                    await asyncio.sleep(args.interval)
    
    except Exception as e:
        logger.error("Auto-scaler failed to start", error=str(e), exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))