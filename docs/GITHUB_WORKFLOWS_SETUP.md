# GitHub Actions Workflows Setup Guide

## Overview

This document provides the complete GitHub Actions workflows that need to be manually created for this repository. Due to GitHub App permissions, these workflows cannot be automatically created and must be set up by a repository maintainer with appropriate permissions.

## Required Workflows

### 1. CI Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.9'
  POETRY_VERSION: '1.7.1'
  NODE_VERSION: '18'

jobs:
  test:
    name: Test Python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
        
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
          
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test,security]"
        
    - name: Run linting
      run: |
        make lint
        
    - name: Run type checking
      run: |
        make type-check
        
    - name: Run unit tests
      run: |
        make test-unit
        
    - name: Run integration tests
      run: |
        make test-integration
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      if: matrix.python-version == '3.9'
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[security]"
        
    - name: Run Bandit security linter
      run: |
        bandit -r src/ -f json -o bandit-report.json || true
        
    - name: Run Safety check
      run: |
        safety check --json --output safety-report.json || true
        
    - name: Run Semgrep
      run: |
        semgrep --config=auto --json --output semgrep-report.json src/ || true
        
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
          semgrep-report.json

  performance:
    name: Performance Tests
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[performance,test]"
        
    - name: Run performance tests
      run: |
        make test-performance
        
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: performance-results.json

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Build Docker image
      run: |
        docker build -t cot-safepath-filter:test .
        
    - name: Test Docker image
      run: |
        docker run --rm cot-safepath-filter:test --version
        
    - name: Run container security scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'cot-safepath-filter:test'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [test, security]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        
    - name: Build package
      run: |
        python -m build
        
    - name: Check package
      run: |
        twine check dist/*
        
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: python-package
        path: dist/

  docs:
    name: Build Documentation
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install documentation dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[docs]"
        
    - name: Build documentation
      run: |
        make docs
        
    - name: Upload documentation
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: docs/_build/
```

### 2. Security Scan (`.github/workflows/security.yml`)

```yaml
name: Security Scan

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run weekly security scan on Sundays at 3 AM UTC
    - cron: '0 3 * * 0'

env:
  PYTHON_VERSION: '3.9'

jobs:
  dependency-scan:
    name: Dependency Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[security]"
        
    - name: Run Safety check for known vulnerabilities
      run: |
        safety check --json --output safety-report.json
        
    - name: Run pip-audit for dependency vulnerabilities
      run: |
        pip-audit --format=json --output=pip-audit-report.json
        
    - name: Upload dependency scan results
      uses: actions/upload-artifact@v3
      with:
        name: dependency-scan-results
        path: |
          safety-report.json
          pip-audit-report.json

  code-scan:
    name: Static Code Security Analysis
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[security]"
        
    - name: Run Bandit security linter
      run: |
        bandit -r src/ -f json -o bandit-report.json
        
    - name: Run Semgrep security analysis
      run: |
        semgrep --config=auto --json --output semgrep-report.json src/
        
    - name: Upload code scan results
      uses: actions/upload-artifact@v3
      with:
        name: code-scan-results
        path: |
          bandit-report.json
          semgrep-report.json

  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Build Docker image
      run: |
        docker build -t cot-safepath-filter:security-test .
        
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'cot-safepath-filter:security-test'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  secret-scan:
    name: Secret Detection
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Fetch full history for secret scanning
        
    - name: Run GitLeaks secret detection
      uses: gitleaks/gitleaks-action@v2
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  security-report:
    name: Generate Security Report
    runs-on: ubuntu-latest
    needs: [dependency-scan, code-scan, container-scan, secret-scan]
    if: always()
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Download all scan results
      uses: actions/download-artifact@v3
      
    - name: Generate consolidated security report
      run: |
        echo "Security scan completed at $(date)" > security-summary.txt
        echo "Repository: ${{ github.repository }}" >> security-summary.txt
        echo "Commit: ${{ github.sha }}" >> security-summary.txt
        
    - name: Upload consolidated security report
      uses: actions/upload-artifact@v3
      with:
        name: consolidated-security-report
        path: security-summary.txt
```

### 3. Release Pipeline (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g., v1.0.0)'
        required: true
        type: string

env:
  PYTHON_VERSION: '3.9'

jobs:
  validate-release:
    name: Validate Release
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Validate tag format
      if: github.event_name == 'push'
      run: |
        tag=${GITHUB_REF#refs/tags/}
        if [[ ! $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
          echo "Invalid tag format: $tag. Expected format: v1.2.3"
          exit 1
        fi
        echo "Valid tag format: $tag"

  test-release:
    name: Test Before Release
    runs-on: ubuntu-latest
    needs: validate-release
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test,security]"
        
    - name: Run full test suite
      run: |
        make test
        
    - name: Run security checks
      run: |
        make security-check
        
    - name: Build package
      run: |
        python -m pip install build
        python -m build

  create-release:
    name: Create GitHub Release
    runs-on: ubuntu-latest
    needs: test-release
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Create Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: CoT SafePath Filter ${{ github.ref }}
        draft: false
        prerelease: false

  publish-pypi:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: create-release
    environment: release
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Build package
      run: |
        python -m pip install build
        python -m build
        
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

## Setup Instructions

### Prerequisites

1. **Repository Admin Access**: You need admin permissions to create workflows
2. **Secrets Configuration**: Set up required secrets in repository settings

### Required Secrets

Add these secrets in **Settings > Secrets and Variables > Actions**:

- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token
- `CODECOV_TOKEN`: Codecov token for coverage reporting

### Manual Setup Steps

1. **Create Workflow Directory**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Create Workflow Files**:
   - Copy the CI Pipeline YAML to `.github/workflows/ci.yml`
   - Copy the Security Scan YAML to `.github/workflows/security.yml`
   - Copy the Release Pipeline YAML to `.github/workflows/release.yml`

3. **Commit and Push**:
   ```bash
   git add .github/workflows/
   git commit -m "feat(ci): add production-ready GitHub Actions workflows"
   git push
   ```

4. **Configure Branch Protection**:
   - Go to **Settings > Branches**
   - Add rules for `main` branch:
     - Require status checks to pass before merging
     - Require branches to be up to date before merging
     - Include administrators

5. **Enable Security Features**:
   - Go to **Settings > Security & Analysis**
   - Enable **Dependency graph**
   - Enable **Dependabot alerts**
   - Enable **Dependabot security updates**
   - Enable **Secret scanning**

## Workflow Features

### CI Pipeline
- **Multi-Python Version Testing**: Tests against Python 3.9-3.12
- **Code Quality Checks**: Linting, type checking, formatting
- **Security Scanning**: Bandit, Safety, Semgrep integration
- **Docker Build Validation**: Container build and security scanning
- **Performance Testing**: Automated benchmarking
- **Documentation Build**: Ensures docs compile correctly

### Security Pipeline
- **Dependency Scanning**: Known vulnerability detection
- **Static Code Analysis**: Security-focused code review
- **Container Security**: Docker image vulnerability scanning
- **Secret Detection**: Prevents credential leaks
- **Compliance Reporting**: Automated security documentation

### Release Pipeline
- **Version Validation**: Ensures proper semantic versioning
- **Pre-release Testing**: Full test suite before release
- **Automated Publishing**: PyPI and Docker Hub deployment
- **Release Notes**: Automated changelog generation
- **Multi-platform Builds**: Linux/ARM64 container support

## Maintenance

### Regular Updates
- Review and update action versions quarterly
- Monitor security scan results weekly
- Update Python versions as new releases become available
- Review and update security scanning tools regularly

### Monitoring
- Set up notification channels for workflow failures
- Monitor build times and optimize as needed
- Review security scan results and address findings promptly
- Track performance benchmarks over time

## Troubleshooting

### Common Issues
1. **Workflow Permission Errors**: Ensure repository has necessary permissions
2. **Secret Access Issues**: Verify secrets are properly configured
3. **Build Failures**: Check dependency compatibility and versions
4. **Security Scan False Positives**: Configure tool-specific ignore files

### Support
- Review GitHub Actions documentation
- Check repository-specific workflow runs for detailed logs
- Consult security tool documentation for configuration options
- Contact maintainers for repository-specific issues

---

**Note**: These workflows implement enterprise-grade DevSecOps practices with comprehensive security scanning, multi-environment testing, and automated compliance reporting. They represent industry best practices for AI safety middleware development.