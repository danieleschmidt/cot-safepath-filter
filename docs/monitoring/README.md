# Monitoring & Observability Guide

This guide covers the comprehensive monitoring and observability setup for the CoT SafePath Filter system.

## Overview

The SafePath Filter implements a complete observability stack including:
- **Metrics**: Prometheus metrics collection and Grafana visualization
- **Logging**: Structured logging with centralized collection
- **Tracing**: Distributed tracing for request flow analysis
- **Health Checks**: Application and infrastructure health monitoring
- **Alerting**: Proactive issue detection and notification

## Monitoring Stack

### Core Components

```mermaid
graph TB
    subgraph "Application"
        App[SafePath Filter]
        HealthCheck[Health Checks]
    end
    
    subgraph "Metrics Collection"
        Prometheus[Prometheus]
        NodeExporter[Node Exporter]
        CAdvisor[cAdvisor]
    end
    
    subgraph "Visualization"
        Grafana[Grafana]
        Dashboards[Custom Dashboards]
    end
    
    subgraph "Alerting"
        AlertManager[Alert Manager]
        PagerDuty[PagerDuty]
        Slack[Slack]
    end
    
    subgraph "Logging"
        Logs[Application Logs]
        Loki[Loki]
        LogAggregation[Log Aggregation]
    end
    
    App --> Prometheus
    App --> Logs
    HealthCheck --> Prometheus
    NodeExporter --> Prometheus
    CAdvisor --> Prometheus
    
    Prometheus --> Grafana
    Prometheus --> AlertManager
    
    Grafana --> Dashboards
    AlertManager --> PagerDuty
    AlertManager --> Slack
    
    Logs --> Loki
    Loki --> Grafana
```

## Metrics

### Application Metrics

#### Request Metrics
```python
# HTTP request metrics
http_requests_total = Counter(
    'safepath_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'safepath_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)
```

#### Filter Metrics
```python
# Filter operation metrics
filter_operations_total = Counter(
    'safepath_filter_operations_total',
    'Total filter operations',
    ['safety_level', 'filtered']
)

filter_processing_duration_seconds = Histogram(
    'safepath_filter_processing_duration_seconds',
    'Filter processing duration',
    ['safety_level']
)

safety_score_distribution = Histogram(
    'safepath_safety_score_distribution',
    'Distribution of safety scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
```

#### System Metrics
```python
# System resource metrics
cpu_usage_percent = Gauge(
    'safepath_cpu_usage_percent',
    'CPU usage percentage'
)

memory_usage_bytes = Gauge(
    'safepath_memory_usage_bytes',
    'Memory usage in bytes'
)

active_connections = Gauge(
    'safepath_active_connections',
    'Number of active connections'
)
```

### Infrastructure Metrics

- **Node Exporter**: System-level metrics (CPU, memory, disk, network)
- **cAdvisor**: Container-level metrics (resource usage, performance)
- **Prometheus**: Time-series database and collection engine

## Dashboards

### SafePath Overview Dashboard

Key metrics displayed:
- Request rate and latency percentiles
- Filter operation success/failure rates
- Safety score distributions
- System resource utilization
- Error rates and response codes

### Performance Dashboard

Focus on:
- Response time percentiles (P50, P95, P99)
- Throughput trends
- Resource utilization trends
- Database performance metrics
- Cache hit rates

### Security Dashboard

Monitoring:
- Failed authentication attempts
- Suspicious pattern detections
- Rate limiting violations
- Security scan results
- Anomalous behavior patterns

### Infrastructure Dashboard

Tracking:
- Container health and resource usage
- Database connection pools
- Network traffic patterns
- Disk I/O and storage usage
- Service dependencies

## Health Checks

### Application Health Checks

```python
@app.get("/health")
async def health_check():
    """Application health check endpoint."""
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": app.version,
        "checks": {
            "database": await check_database_health(),
            "redis": await check_redis_health(),
            "models": await check_model_health(),
            "external_apis": await check_external_apis()
        }
    }
    
    # Determine overall health
    all_healthy = all(
        check["status"] == "healthy" 
        for check in checks["checks"].values()
    )
    
    if not all_healthy:
        checks["status"] = "unhealthy"
        return JSONResponse(
            content=checks,
            status_code=503
        )
    
    return checks
```

### Detailed Health Checks

```python
async def check_database_health():
    """Check database connectivity and performance."""
    try:
        start_time = time.time()
        await database.execute("SELECT 1")
        response_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time * 1000, 2),
            "connection_pool": {
                "size": database.pool_size,
                "checked_out": database.checked_out_connections
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

### Readiness and Liveness Probes

```yaml
# Kubernetes health check configuration
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

## Alerting

### Alert Rules

#### Critical Alerts
```yaml
# High error rate
- alert: HighErrorRate
  expr: rate(http_requests_total{status_code=~"5.."}[5m]) > 0.1
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"
    description: "Error rate is {{ $value }} errors per second"

# Service down
- alert: ServiceDown
  expr: up{job="safepath-filter"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "SafePath Filter service is down"
    description: "Service has been down for more than 1 minute"
```

#### Warning Alerts
```yaml
# High latency
- alert: HighLatency
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High latency detected"
    description: "95th percentile latency is {{ $value }} seconds"

# High memory usage
- alert: HighMemoryUsage
  expr: memory_usage_bytes / memory_limit_bytes > 0.8
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High memory usage"
    description: "Memory usage is {{ $value | humanizePercentage }}"
```

### Notification Channels

#### Slack Integration
```yaml
slack_configs:
  - api_url: '{{ .SlackWebhook }}'
    channel: '#safepath-alerts'
    title: 'SafePath Filter Alert'
    text: >-
      {{ range .Alerts }}
      *Alert:* {{ .Annotations.summary }}
      *Description:* {{ .Annotations.description }}
      *Severity:* {{ .Labels.severity }}
      {{ end }}
```

#### PagerDuty Integration
```yaml
pagerduty_configs:
  - routing_key: '{{ .PagerDutyRoutingKey }}'
    description: '{{ .CommonAnnotations.summary }}'
    severity: '{{ .CommonLabels.severity }}'
```

## Logging

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

# Request logging
logger.info(
    "filter_request_processed",
    request_id=request.id,
    user_id=request.user_id,
    processing_time_ms=processing_time,
    safety_score=result.safety_score,
    was_filtered=result.was_filtered
)

# Error logging
logger.error(
    "filter_processing_error",
    request_id=request.id,
    error_type=type(error).__name__,
    error_message=str(error),
    stack_trace=traceback.format_exc()
)
```

### Log Levels and Categories

#### Log Levels
- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about system operation
- **WARNING**: Potentially harmful situations
- **ERROR**: Error events that allow application to continue
- **CRITICAL**: Serious error events that may cause shutdown

#### Log Categories
- **audit**: Security and compliance audit logs
- **performance**: Performance-related logs
- **security**: Security event logs
- **business**: Business logic and filter operation logs
- **system**: System and infrastructure logs

### Log Aggregation

```yaml
# Loki configuration
server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h
```

## Tracing

### Distributed Tracing Setup

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

### Trace Instrumentation

```python
@tracer.start_as_current_span("filter_request")
async def process_filter_request(request: FilterRequest):
    """Process filter request with tracing."""
    with tracer.start_as_current_span("input_validation"):
        validate_input(request)
    
    with tracer.start_as_current_span("safety_analysis"):
        safety_score = await analyze_safety(request.content)
    
    with tracer.start_as_current_span("filter_application"):
        result = await apply_filters(request, safety_score)
    
    return result
```

## Performance Monitoring

### SLA Monitoring

#### Service Level Objectives (SLOs)
- **Availability**: 99.9% uptime
- **Latency**: 95% of requests < 100ms
- **Error Rate**: < 0.1% error rate
- **Throughput**: Support 1000 RPS

#### Service Level Indicators (SLIs)
```promql
# Availability SLI
sum(rate(http_requests_total[5m])) by (status_code) / 
sum(rate(http_requests_total[5m]))

# Latency SLI
histogram_quantile(0.95, 
  rate(http_request_duration_seconds_bucket[5m])
)

# Error rate SLI
sum(rate(http_requests_total{status_code=~"5.."}[5m])) / 
sum(rate(http_requests_total[5m]))
```

### Performance Benchmarking

```python
import time
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def performance_monitor(operation_name: str):
    """Context manager for performance monitoring."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        performance_histogram.labels(
            operation=operation_name
        ).observe(duration)
```

## Deployment and Configuration

### Docker Compose Monitoring Stack

```yaml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager:/etc/alertmanager
```

### Environment Variables

```bash
# Prometheus configuration
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
PROMETHEUS_ENDPOINT=/metrics

# Grafana configuration
GRAFANA_URL=http://localhost:3000
GRAFANA_USERNAME=admin
GRAFANA_PASSWORD=admin

# Alerting configuration
ALERTMANAGER_URL=http://localhost:9093
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
PAGERDUTY_ROUTING_KEY=your-routing-key

# Logging configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOKI_URL=http://localhost:3100

# Tracing configuration
JAEGER_AGENT_HOST=localhost
JAEGER_AGENT_PORT=6831
TRACING_ENABLED=true
```

## Troubleshooting

### Common Issues

#### High Memory Usage
1. Check for memory leaks in application code
2. Monitor garbage collection patterns
3. Analyze heap dumps for large objects
4. Review caching configurations

#### High Latency
1. Check database query performance
2. Monitor external API response times
3. Analyze request processing bottlenecks
4. Review resource allocation

#### Alert Fatigue
1. Tune alert thresholds
2. Group related alerts
3. Implement alert escalation
4. Add contextual information

### Debugging Commands

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Query specific metrics
curl "http://localhost:9090/api/v1/query?query=http_requests_total"

# Check Grafana datasources
curl http://admin:admin@localhost:3000/api/datasources

# View application logs
docker logs safepath-filter

# Check alertmanager status
curl http://localhost:9093/api/v1/status
```

## Best Practices

1. **Metrics Naming**: Use consistent naming conventions
2. **Alert Thresholds**: Set appropriate thresholds to avoid noise
3. **Dashboard Design**: Focus on actionable insights
4. **Log Retention**: Configure appropriate retention periods
5. **Security**: Protect monitoring endpoints and data
6. **Documentation**: Maintain up-to-date runbooks
7. **Testing**: Test alerting and monitoring regularly

---

**Last Updated**: January 27, 2025  
**Version**: 1.0