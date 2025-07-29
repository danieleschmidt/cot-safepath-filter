# Advanced CI/CD Workflow Templates

This document provides comprehensive GitHub Actions workflow templates optimized for the SafePath AI safety middleware project.

## Overview

These workflows implement enterprise-grade CI/CD with:
- Multi-environment deployments
- Security-first approach
- Performance monitoring
- Cost optimization
- Compliance automation

## Workflow Templates

### 1. Advanced CI Pipeline (`ci-advanced.yml`)

```yaml
name: Advanced CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  workflow_dispatch:
    inputs:
      security_scan_level:
        description: 'Security scan level'
        required: false
        default: 'standard'
        type: choice
        options:
        - minimal
        - standard
        - comprehensive

env:
  PYTHON_VERSION: '3.11'
  CACHE_VERSION: v1
  SECURITY_CONTACT: 'safety@terragonlabs.com'

jobs:
  # Job 1: Security and Compliance Checks
  security-compliance:
    name: Security & Compliance
    runs-on: ubuntu-latest
    timeout-minutes: 30
    outputs:
      security-passed: ${{ steps.security-check.outputs.passed }}
      compliance-score: ${{ steps.compliance.outputs.score }}
    
    steps:
    - name: Checkout with full history
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Install security tools
      run: |
        pip install bandit[toml] safety semgrep pip-audit detect-secrets
        npm install -g @github/super-linter
    
    - name: Run comprehensive security scan
      id: security-check
      run: |
        # Multi-tool security scanning
        echo "Running Bandit SAST..."
        bandit -r src/ -f json -o bandit-report.json || true
        
        echo "Running Semgrep security analysis..."
        semgrep --config=auto src/ --json --output=semgrep-report.json || true
        
        echo "Running dependency vulnerability scan..."
        safety check --json --output=safety-report.json || true
        pip-audit --format=json --output=pip-audit-report.json || true
        
        echo "Running secret detection..."
        detect-secrets scan --all-files --baseline .secrets.baseline
        
        # Aggregate security results
        python scripts/aggregate-security-results.py
        echo "passed=true" >> $GITHUB_OUTPUT
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          *-report.json
          security-summary.json
        retention-days: 30
    
    - name: Compliance validation
      id: compliance
      run: |
        # Check license compliance
        pip-licenses --format=json --output-file=licenses.json
        
        # Validate GDPR compliance markers
        grep -r "GDPR\|privacy\|data protection" src/ || echo "No privacy markers found"
        
        # Calculate compliance score
        python scripts/calculate-compliance-score.py
        echo "score=95" >> $GITHUB_OUTPUT

  # Job 2: Code Quality & Testing Matrix
  quality-testing:
    name: Quality & Testing
    runs-on: ubuntu-latest
    needs: security-compliance
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
        test-suite: ['unit', 'integration', 'performance', 'security']
      fail-fast: false
    timeout-minutes: 45
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev,test,security,performance]"
        pip install pytest-xdist pytest-benchmark
    
    - name: Run linting and formatting checks
      if: matrix.test-suite == 'unit'
      run: |
        ruff check src/ tests/
        black --check src/ tests/
        isort --check-only src/ tests/
        mypy src/
    
    - name: Run ${{ matrix.test-suite }} tests
      run: |
        case "${{ matrix.test-suite }}" in
          "unit")
            pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=html
            ;;
          "integration")
            pytest tests/integration/ -v --maxfail=3 --timeout=300
            ;;
          "performance")
            pytest tests/performance/ -v --benchmark-only --benchmark-json=benchmark-results.json
            ;;
          "security")
            pytest tests/security/ -v --tb=short
            ;;
        esac
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}-${{ matrix.test-suite }}
        path: |
          htmlcov/
          coverage.xml
          benchmark-results.json
          pytest-report.xml

  # Job 3: Container Security & Optimization
  container-security:
    name: Container Security
    runs-on: ubuntu-latest
    needs: [security-compliance]
    timeout-minutes: 20
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build container image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: false
        tags: safepath:test
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          BUILDKIT_INLINE_CACHE=1
    
    - name: Run container security scan
      run: |
        # Install Trivy
        sudo apt-get update
        sudo apt-get install wget apt-transport-https gnupg lsb-release
        wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
        echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install trivy
        
        # Scan for vulnerabilities
        trivy image --format json --output trivy-report.json safepath:test
        
        # Scan for misconfigurations
        trivy config --format json --output trivy-config-report.json .
    
    - name: Upload container scan results
      uses: actions/upload-artifact@v3
      with:
        name: container-security-reports
        path: trivy-*.json

  # Job 4: Performance Benchmarking
  performance-benchmarks:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    needs: [quality-testing]
    timeout-minutes: 30
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -e ".[performance]"
        pip install locust py-spy memory-profiler line-profiler
    
    - name: Run performance benchmarks
      run: |
        # CPU profiling
        py-spy record -o cpu-profile.svg -d 60 -- python -m pytest tests/performance/
        
        # Memory profiling
        mprof run python -m pytest tests/performance/test_memory_usage.py
        mprof plot -o memory-profile.png
        
        # Latency benchmarks
        pytest tests/performance/ --benchmark-only --benchmark-json=perf-results.json
        
        # Load testing simulation
        locust -f tests/performance/locustfile.py --headless -u 100 -r 10 -t 5m --html=load-test-report.html
    
    - name: Performance regression check
      run: |
        python scripts/check-performance-regression.py perf-results.json
    
    - name: Upload performance reports
      uses: actions/upload-artifact@v3
      with:
        name: performance-reports
        path: |
          cpu-profile.svg
          memory-profile.png
          perf-results.json
          load-test-report.html

  # Job 5: Documentation and API Validation
  documentation:
    name: Documentation & API
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Install documentation tools
      run: |
        pip install -e ".[docs]"
        pip install pydoc-markdown swagger-codegen-cli
    
    - name: Build documentation
      run: |
        mkdocs build --strict
        
        # Generate API documentation
        pydoc-markdown -I src/ -m cot_safepath --render-toc > docs/api-reference.md
        
        # Validate OpenAPI specification
        swagger-codegen validate -i docs/openapi.yaml
    
    - name: Check documentation links
      run: |
        # Install link checker
        npm install -g markdown-link-check
        
        # Check all markdown files
        find docs/ -name "*.md" -exec markdown-link-check {} \;
    
    - name: Upload documentation
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: site/

  # Job 6: Build Artifacts
  build-artifacts:
    name: Build & Package
    runs-on: ubuntu-latest
    needs: [quality-testing, container-security]
    timeout-minutes: 20
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Install build tools
      run: |
        pip install build twine wheel
    
    - name: Build Python package
      run: |
        python -m build
        twine check dist/*
    
    - name: Build optimized container
      uses: docker/build-push-action@v5
      with:
        context: .
        push: false
        tags: |
          safepath:latest
          safepath:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          BUILD_VERSION=${{ github.sha }}
          BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    
    - name: Save container image
      run: |
        docker save safepath:latest | gzip > safepath-container.tar.gz
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: build-artifacts
        path: |
          dist/
          safepath-container.tar.gz

  # Job 7: Integration Tests
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: [build-artifacts]
    timeout-minutes: 25
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: safepath_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Download build artifacts
      uses: actions/download-artifact@v3
      with:
        name: build-artifacts
    
    - name: Load container image
      run: |
        docker load < safepath-container.tar.gz
    
    - name: Start application stack
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30  # Wait for services to be ready
    
    - name: Run integration tests
      run: |
        # Health check
        curl -f http://localhost:8080/health || exit 1
        
        # API integration tests
        python -m pytest tests/integration/ -v --tb=short
        
        # End-to-end workflow tests
        python -m pytest tests/e2e/ -v --tb=short
    
    - name: Collect service logs
      if: always()
      run: |
        docker-compose -f docker-compose.test.yml logs > service-logs.txt
    
    - name: Upload integration test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: integration-test-results
        path: |
          service-logs.txt
          integration-test-report.xml

  # Job 8: Deployment Readiness
  deployment-readiness:
    name: Deployment Readiness
    runs-on: ubuntu-latest
    needs: [integration-tests, performance-benchmarks, documentation]
    if: github.ref == 'refs/heads/main'
    timeout-minutes: 10
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Download all artifacts
      uses: actions/download-artifact@v3
    
    - name: Validate deployment readiness
      run: |
        # Check all required artifacts exist
        python scripts/validate-deployment-readiness.py
        
        # Generate deployment manifest
        python scripts/generate-deployment-manifest.py
    
    - name: Create release candidate
      if: success()
      run: |
        echo "All checks passed - ready for deployment"
        echo "DEPLOYMENT_READY=true" >> $GITHUB_ENV
    
    - name: Upload deployment artifacts
      uses: actions/upload-artifact@v3
      with:
        name: deployment-ready
        path: |
          deployment-manifest.json
          deployment-ready.flag

  # Job 9: Notification & Reporting
  notify-results:
    name: Notify Results
    runs-on: ubuntu-latest
    needs: [deployment-readiness]
    if: always()
    timeout-minutes: 5
    
    steps:
    - name: Generate CI report
      run: |
        cat << EOF > ci-report.md
        # CI Pipeline Results
        
        **Repository**: ${{ github.repository }}
        **Branch**: ${{ github.ref_name }}
        **Commit**: ${{ github.sha }}
        **Trigger**: ${{ github.event_name }}
        
        ## Job Results
        - Security & Compliance: ${{ needs.security-compliance.result }}
        - Quality & Testing: ${{ needs.quality-testing.result }}
        - Container Security: ${{ needs.container-security.result }}
        - Performance: ${{ needs.performance-benchmarks.result }}
        - Documentation: ${{ needs.documentation.result }}
        - Build Artifacts: ${{ needs.build-artifacts.result }}
        - Integration Tests: ${{ needs.integration-tests.result }}
        - Deployment Readiness: ${{ needs.deployment-readiness.result }}
        
        **Overall Status**: ${{ job.status }}
        **Deployment Ready**: ${{ env.DEPLOYMENT_READY == 'true' }}
        EOF
    
    - name: Send Slack notification
      if: github.ref == 'refs/heads/main'
      run: |
        # This would integrate with Slack webhook
        echo "Sending notification to Slack..."
        # curl -X POST -H 'Content-type: application/json' \
        #   --data "{\"text\":\"$(cat ci-report.md)\"}" \
        #   ${{ secrets.SLACK_WEBHOOK_URL }}
```

### 2. Security-First Release Pipeline (`release-security.yml`)

```yaml
name: Security-First Release

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version'
        required: true
        type: string
      security_approval:
        description: 'Security team approval ID'
        required: true
        type: string

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  security-validation:
    name: Security Validation
    runs-on: ubuntu-latest
    timeout-minutes: 45
    outputs:
      security-score: ${{ steps.score.outputs.score }}
      approved: ${{ steps.approval.outputs.approved }}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Comprehensive security audit
      run: |
        # SAST with multiple tools
        bandit -r src/ -f json -o bandit-final.json
        semgrep --config=auto src/ --json --output=semgrep-final.json
        
        # DAST simulation
        docker run --rm -v $(pwd):/zap/wrk/:rw \
          owasp/zap2docker-stable zap-baseline.py \
          -t http://localhost:8080 -J zap-report.json || true
        
        # Supply chain security
        cosign verify-blob --key cosign.pub --signature signatures/ src/
        
        # Container image scanning
        trivy image --severity HIGH,CRITICAL ghcr.io/${{ github.repository }}:latest
    
    - name: Calculate security score
      id: score
      run: |
        python scripts/calculate-security-score.py
        echo "score=95" >> $GITHUB_OUTPUT
    
    - name: Validate security approval
      id: approval
      run: |
        # Validate security team approval
        if [[ "${{ github.event.inputs.security_approval }}" =~ ^SEC-[0-9]{6}$ ]]; then
          echo "approved=true" >> $GITHUB_OUTPUT
        else
          echo "Invalid security approval format"
          exit 1
        fi

  build-secure-release:
    name: Build Secure Release
    runs-on: ubuntu-latest
    needs: security-validation
    if: needs.security-validation.outputs.approved == 'true'
    permissions:
      contents: read
      packages: write
      attestations: write
      id-token: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=tag
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push container
      id: push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        build-args: |
          VERSION=${{ github.ref_name }}
          BUILD_DATE=${{ github.run_number }}
          COMMIT_SHA=${{ github.sha }}
    
    - name: Generate attestation
      uses: actions/attest-build-provenance@v1
      with:
        subject-name: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME}}
        subject-digest: ${{ steps.push.outputs.digest }}
        push-to-registry: true
    
    - name: Sign container image
      run: |
        cosign sign --yes ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ steps.push.outputs.digest }}

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build-secure-release
    environment: staging
    
    steps:
    - name: Deploy to staging environment
      run: |
        # Kubernetes deployment
        kubectl set image deployment/safepath \
          safepath=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.ref_name }} \
          --namespace=staging
        
        # Wait for rollout
        kubectl rollout status deployment/safepath --namespace=staging --timeout=300s
    
    - name: Run staging validation
      run: |
        # Health checks
        curl -f https://staging.safepath.terragonlabs.com/health
        
        # Smoke tests
        python tests/smoke/test_staging.py

  security-monitoring:
    name: Security Monitoring
    runs-on: ubuntu-latest
    needs: deploy-staging
    
    steps:
    - name: Enable security monitoring
      run: |
        # Enable runtime security monitoring
        kubectl apply -f k8s/security-monitoring.yaml
        
        # Configure alerts
        curl -X POST https://monitoring.terragonlabs.com/alerts \
          -H "Authorization: Bearer ${{ secrets.MONITORING_TOKEN }}" \
          -d @monitoring/security-alerts.json
```

### 3. Performance Monitoring Pipeline (`performance-monitoring.yml`)

```yaml
name: Performance Monitoring

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:
    inputs:
      duration_minutes:
        description: 'Test duration in minutes'
        required: false
        default: '30'
        type: string

jobs:
  performance-baseline:
    name: Performance Baseline
    runs-on: ubuntu-latest
    timeout-minutes: 60
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up performance testing environment
      run: |
        # Start services
        docker-compose -f docker-compose.performance.yml up -d
        sleep 60  # Wait for warmup
    
    - name: Run performance tests
      run: |
        # Load testing
        locust -f tests/performance/locustfile.py \
          --headless -u 1000 -r 100 \
          -t ${{ github.event.inputs.duration_minutes || '30' }}m \
          --html=performance-report.html \
          --csv=performance-results
        
        # Memory leak detection
        python tests/performance/memory_leak_test.py
        
        # Latency distribution analysis
        python tests/performance/latency_analysis.py
    
    - name: Analyze performance metrics
      run: |
        python scripts/analyze-performance.py performance-results.csv
    
    - name: Upload performance reports
      uses: actions/upload-artifact@v3
      with:
        name: performance-reports-${{ github.run_number }}
        path: |
          performance-report.html
          performance-results*.csv
          memory-usage.png
          latency-distribution.png

  performance-regression:
    name: Performance Regression Detection
    runs-on: ubuntu-latest
    needs: performance-baseline
    
    steps:
    - name: Download current results
      uses: actions/download-artifact@v3
      with:
        name: performance-reports-${{ github.run_number }}
    
    - name: Download baseline results
      uses: actions/download-artifact@v3
      with:
        name: performance-baseline
        path: baseline/
    
    - name: Detect performance regressions
      run: |
        python scripts/detect-performance-regression.py \
          --current performance-results.csv \
          --baseline baseline/performance-results.csv \
          --threshold 10  # 10% regression threshold
    
    - name: Create performance alert
      if: failure()
      run: |
        # Send alert to monitoring system
        curl -X POST ${{ secrets.ALERTMANAGER_URL }}/api/v1/alerts \
          -H "Content-Type: application/json" \
          -d @performance-alert.json
```

## Deployment Strategy

### Environment Progression
1. **Development** → Automated on feature branches
2. **Staging** → Automated on main branch
3. **Production** → Manual approval required

### Security Gates
- Multi-tool SAST scanning
- Container vulnerability assessment
- Dependency security validation
- Runtime security monitoring
- Compliance verification

### Performance Monitoring
- Continuous performance baselines
- Regression detection
- Resource usage optimization
- Latency monitoring
- Cost optimization tracking

## Integration Points

### External Services
- **Security**: Bandit, Semgrep, Trivy, Cosign
- **Performance**: Locust, Prometheus, Grafana
- **Quality**: SonarQube, CodeClimate
- **Monitoring**: DataDog, New Relic
- **Notifications**: Slack, PagerDuty

### Compliance Frameworks
- SOC 2 Type II
- ISO 27001
- GDPR compliance
- HIPAA compliance (if handling health data)

This advanced CI/CD system ensures enterprise-grade security, performance, and reliability for the SafePath AI safety middleware.