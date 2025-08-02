# Load Testing Directory

This directory contains load testing configurations and scripts for the CoT SafePath Filter system.

## Overview

Load testing ensures the SafePath Filter can handle expected traffic volumes while maintaining performance and reliability. We use multiple tools and approaches to test different aspects of system performance.

## Testing Tools

### Primary Tools
- **Locust**: Primary load testing framework for HTTP APIs
- **Artillery**: Alternative tool for high-throughput testing
- **k6**: Lightweight tool for API performance testing
- **pytest-benchmark**: Microbenchmarks for individual functions

### Monitoring During Tests
- **Prometheus**: Metrics collection during load tests
- **Grafana**: Real-time visualization of performance metrics
- **cAdvisor**: Container resource monitoring
- **htop/top**: System resource monitoring

## Test Scenarios

### 1. Baseline Performance Tests
- **Single user**: Establish baseline response times
- **Steady load**: Sustained load at normal traffic levels
- **Capacity testing**: Find maximum sustainable load

### 2. Stress Testing
- **Spike testing**: Sudden increases in traffic
- **Volume testing**: Large amounts of data processing
- **Endurance testing**: Extended duration under load

### 3. Scalability Testing
- **Horizontal scaling**: Adding more instances
- **Vertical scaling**: Increasing resources per instance
- **Database scaling**: Testing database performance under load

## Test Configuration

### Load Test Profiles

#### Light Load (Development)
- **Users**: 10 concurrent users
- **Duration**: 5 minutes
- **Ramp-up**: 30 seconds
- **Target**: Development environment validation

#### Normal Load (Staging)
- **Users**: 100 concurrent users
- **Duration**: 15 minutes
- **Ramp-up**: 2 minutes
- **Target**: Staging environment validation

#### Heavy Load (Production)
- **Users**: 1000 concurrent users
- **Duration**: 30 minutes
- **Ramp-up**: 5 minutes
- **Target**: Production capacity testing

#### Stress Test
- **Users**: 2000 concurrent users
- **Duration**: 10 minutes
- **Ramp-up**: 1 minute
- **Target**: Breaking point identification

## Performance Targets

### Response Time Targets
- **P50**: < 50ms
- **P95**: < 100ms
- **P99**: < 200ms
- **P99.9**: < 500ms

### Throughput Targets
- **Normal load**: 1,000 requests/second
- **Peak load**: 5,000 requests/second
- **Burst capacity**: 10,000 requests/second

### Resource Utilization Targets
- **CPU utilization**: < 70% under normal load
- **Memory utilization**: < 80% under normal load
- **Disk I/O**: < 60% capacity
- **Network I/O**: < 40% capacity

### Error Rate Targets
- **Error rate**: < 0.1% under normal load
- **Timeout rate**: < 0.01%
- **5xx errors**: < 0.05%

## Running Load Tests

### Prerequisites
```bash
# Install load testing tools
pip install locust artillery k6
npm install -g artillery-engine-k6

# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d
```

### Basic Load Test
```bash
# Run light load test
locust -f tests/load/locustfile.py --host=http://localhost:8080 \
       --users=10 --spawn-rate=2 --run-time=5m

# Run with web UI
locust -f tests/load/locustfile.py --host=http://localhost:8080
```

### Advanced Load Testing
```bash
# Run specific test scenario
locust -f tests/load/scenarios/normal_load.py --host=http://localhost:8080 \
       --users=100 --spawn-rate=10 --run-time=15m

# Run headless with CSV output
locust -f tests/load/locustfile.py --host=http://localhost:8080 \
       --users=1000 --spawn-rate=50 --run-time=30m --headless \
       --csv=results/load_test_results
```

### Performance Monitoring
```bash
# Monitor during test execution
docker exec -it safepath-prometheus prometheus \
    --query.timeout=30s \
    --web.console.libraries=/usr/share/prometheus/console_libraries \
    --web.console.templates=/usr/share/prometheus/consoles

# View real-time metrics
open http://localhost:3000  # Grafana dashboard
```

## Test Scenarios

### API Endpoint Testing
1. **Filter endpoint**: Primary filtering functionality
2. **Health check**: System health monitoring
3. **Metrics endpoint**: Performance metrics collection
4. **Authentication**: User authentication flows

### Data Scenarios
1. **Small payloads**: < 1KB chain-of-thought content
2. **Medium payloads**: 1-10KB content
3. **Large payloads**: 10-100KB content
4. **Mixed payloads**: Realistic distribution of sizes

### Traffic Patterns
1. **Steady traffic**: Constant request rate
2. **Spike traffic**: Sudden traffic increases
3. **Burst traffic**: Short bursts of high traffic
4. **Gradual ramp**: Slowly increasing traffic

## Results Analysis

### Key Metrics to Track
- **Response time percentiles**: P50, P95, P99, P99.9
- **Throughput**: Requests per second
- **Error rates**: 4xx and 5xx error percentages
- **Resource utilization**: CPU, memory, disk, network
- **Concurrency**: Active connections and users

### Performance Bottleneck Identification
1. **Database performance**: Query execution times
2. **Model inference**: ML model processing times
3. **Memory usage**: Memory leaks or excessive usage
4. **Network I/O**: Bandwidth limitations
5. **CPU utilization**: Processing bottlenecks

### Reporting
Load test results should include:
- **Executive summary**: High-level performance assessment
- **Detailed metrics**: All performance measurements
- **Bottleneck analysis**: Identified performance issues
- **Recommendations**: Optimization suggestions
- **Comparison**: Performance vs. previous tests

## Continuous Performance Testing

### CI/CD Integration
- **PR testing**: Light load tests on pull requests
- **Staging tests**: Normal load tests on staging deployment
- **Production tests**: Scheduled performance regression tests

### Performance Monitoring
- **Baseline establishment**: Regular baseline performance tests
- **Regression detection**: Automated detection of performance degradation
- **Alerting**: Notifications for performance threshold breaches

### Optimization Workflow
1. **Identify bottlenecks**: Through load testing and monitoring
2. **Implement optimizations**: Code, configuration, or infrastructure changes
3. **Validate improvements**: Re-run load tests to confirm improvements
4. **Monitor production**: Ensure optimizations work in production

## Best Practices

1. **Test environment parity**: Ensure test environments match production
2. **Realistic data**: Use production-like data volumes and patterns
3. **Incremental testing**: Start with small loads and gradually increase
4. **Monitor everything**: Track all relevant metrics during testing
5. **Document results**: Maintain detailed records of all tests
6. **Regular testing**: Establish regular performance testing schedules
7. **Automation**: Automate tests and result analysis where possible

## Troubleshooting

### Common Issues
1. **Test environment differences**: Ensure consistent configurations
2. **Network limitations**: Account for network bandwidth and latency
3. **Resource constraints**: Ensure adequate test environment resources
4. **Tool limitations**: Understand load testing tool limitations
5. **Monitoring overhead**: Account for monitoring system impact

### Debug Commands
```bash
# Check system resources during test
htop
iostat 1
sar -u 1

# Monitor application logs
tail -f /var/log/safepath/app.log

# Check database performance
# (Database-specific monitoring commands)

# Monitor network traffic
iftop
netstat -i
```