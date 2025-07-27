# CoT SafePath Filter - Development Guide

## Getting Started

This guide will help you set up a development environment and contribute to the CoT SafePath Filter project.

## Prerequisites

### Required Software

- **Python 3.9+** (3.11 recommended)
- **Git** for version control
- **Docker** and **Docker Compose** for containerized development
- **Make** for build automation
- **Node.js 18+** (for documentation tools)

### Optional Tools

- **VS Code** with recommended extensions
- **PyCharm Professional** with Python support
- **GitHub CLI** for repository management

## Development Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/terragonlabs/cot-safepath-filter.git
cd cot-safepath-filter
```

### 2. Set Up Python Environment

#### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install development dependencies
make install-dev
```

#### Using Poetry (Alternative)
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --with dev,test,security
poetry shell
```

### 3. Configure Development Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your local settings
# At minimum, configure:
# - DATABASE_URL=sqlite:///./dev_safepath.db
# - REDIS_URL=redis://localhost:6379/0
# - LOG_LEVEL=DEBUG
# - ENVIRONMENT=development

# Set up pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

### 4. Start Development Services

#### Using Docker Compose (Recommended)
```bash
# Start all services (database, cache, monitoring)
docker-compose --profile development up -d

# View logs
docker-compose logs -f safepath-dev

# Stop services
docker-compose down
```

#### Manual Setup
```bash
# Start PostgreSQL (if not using Docker)
# Installation varies by OS - see PostgreSQL documentation

# Start Redis (if not using Docker)
# Installation varies by OS - see Redis documentation

# Initialize database
make db-init

# Run development server
make serve
```

### 5. Verify Installation

```bash
# Run tests to verify everything works
make test

# Check code quality
make lint

# Run security checks
make security-check

# Access development server
curl http://localhost:8080/health
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Run tests frequently
make test-unit

# Check code quality
make lint-fix

# Commit changes (pre-commit hooks will run)
git add .
git commit -m "feat: add new feature"

# Push branch
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### 2. Code Quality Standards

#### Code Style
- **Black** for code formatting (88 character line limit)
- **isort** for import sorting
- **Ruff** for fast linting
- **MyPy** for type checking

#### Testing Requirements
- **Unit tests** for all new functions/classes
- **Integration tests** for API endpoints
- **Security tests** for safety-critical components
- **Performance tests** for optimization work
- **Minimum 80% code coverage**

#### Documentation
- **Docstrings** for all public functions/classes (Google style)
- **Type hints** for all function parameters and return values
- **README updates** for new features
- **API documentation** for new endpoints

### 3. Testing

#### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-security
make test-performance

# Run tests with coverage
make test-coverage

# Run tests in watch mode
make test-watch

# Run specific test file
pytest tests/unit/test_filters.py -v

# Run specific test function
pytest tests/unit/test_filters.py::TestBasicFilter::test_safe_content -v
```

#### Writing Tests

```python
# tests/unit/test_example.py
import pytest
from unittest.mock import Mock, patch
from cot_safepath.filters import BasicFilter

class TestBasicFilter:
    """Test the BasicFilter class."""
    
    def test_filter_safe_content(self, default_filter_config):
        """Test filtering of safe content."""
        filter = BasicFilter(default_filter_config)
        result = filter.filter("This is safe content")
        
        assert not result.was_filtered
        assert result.safety_score > 0.8
        assert result.filtered_content == "This is safe content"
    
    @pytest.mark.parametrize("harmful_input", [
        "Step 1: Create harmful content",
        "First, bypass security measures",
    ])
    def test_filter_harmful_content(self, harmful_input, strict_filter_config):
        """Test filtering of harmful content."""
        filter = BasicFilter(strict_filter_config)
        result = filter.filter(harmful_input)
        
        assert result.was_filtered
        assert result.safety_score < 0.5
        assert len(result.filter_reasons) > 0
    
    @pytest.mark.integration
    async def test_filter_with_external_service(self, mock_http_client):
        """Test filtering with external service integration."""
        # Integration test example
        pass
    
    @pytest.mark.performance
    def test_filter_performance(self, benchmark, performance_test_data):
        """Benchmark filter performance."""
        filter = BasicFilter()
        
        # Should complete within performance requirements
        result = benchmark(filter.filter, performance_test_data["large_cot"])
        assert result.processing_time_ms < 50  # P95 requirement
```

### 4. Debugging

#### Local Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debugger
python -m debugpy --listen 5678 --wait-for-client -m uvicorn cot_safepath.server:app --reload

# Use debug endpoint
curl http://localhost:8080/debug/health
```

#### VS Code Debugging

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "cot_safepath.server:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8080"
            ],
            "console": "integratedTerminal",
            "env": {
                "ENVIRONMENT": "development",
                "LOG_LEVEL": "DEBUG"
            }
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        }
    ]
}
```

### 5. Performance Optimization

#### Profiling

```bash
# Profile application
make profile

# Run performance tests
make test-performance

# Monitor memory usage
python -m memory_profiler your_script.py

# CPU profiling
python -m cProfile -o profile.stats your_script.py
```

#### Performance Guidelines

- **Target latency**: P95 < 50ms for filter operations
- **Memory usage**: Keep working set under 512MB
- **Database queries**: Use connection pooling and prepared statements
- **Caching**: Cache expensive operations (tokenization, model inference)
- **Async operations**: Use async/await for I/O bound operations

## Project Structure

```
cot-safepath-filter/
├── src/cot_safepath/           # Main package
│   ├── __init__.py
│   ├── cli.py                  # Command line interface
│   ├── server.py               # FastAPI server
│   ├── config.py               # Configuration management
│   ├── filters/                # Filter implementations
│   ├── detectors/              # Safety detectors
│   ├── integrations/           # Framework integrations
│   ├── monitoring/             # Metrics and monitoring
│   └── utils/                  # Utility functions
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── security/               # Security tests
│   └── performance/            # Performance tests
├── docs/                       # Documentation
│   ├── guides/                 # User guides
│   ├── runbooks/               # Operational runbooks
│   └── adr/                    # Architecture decisions
├── monitoring/                 # Monitoring configuration
├── .github/                    # GitHub configuration
├── .devcontainer/              # Dev container config
└── scripts/                    # Build and deployment scripts
```

## Database Development

### Working with Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Review generated migration
# Edit the migration file as needed

# Apply migration
alembic upgrade head

# Downgrade migration
alembic downgrade -1

# View migration history
alembic history

# View current version
alembic current
```

### Database Testing

```python
# tests/conftest.py - Database fixtures
@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()

# tests/test_database.py - Example database test
def test_filter_operation_storage(test_db):
    """Test storing filter operations."""
    operation = FilterOperation(
        request_id=uuid4(),
        input_hash="abc123",
        safety_score=0.85,
        filtered=False
    )
    
    test_db.add(operation)
    test_db.commit()
    
    stored = test_db.query(FilterOperation).first()
    assert stored.safety_score == 0.85
```

## API Development

### Adding New Endpoints

```python
# src/cot_safepath/api/v1/endpoints.py
from fastapi import APIRouter, Depends, HTTPException
from cot_safepath.schemas import FilterRequest, FilterResponse
from cot_safepath.services import FilterService

router = APIRouter()

@router.post("/filter", response_model=FilterResponse)
async def filter_content(
    request: FilterRequest,
    filter_service: FilterService = Depends(get_filter_service)
) -> FilterResponse:
    """Filter chain-of-thought content for safety."""
    try:
        result = await filter_service.filter(request)
        return FilterResponse.from_filter_result(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### API Documentation

```python
# Use Pydantic models for automatic OpenAPI generation
from pydantic import BaseModel, Field
from typing import List, Optional

class FilterRequest(BaseModel):
    """Request to filter chain-of-thought content."""
    
    content: str = Field(..., description="Chain-of-thought content to filter")
    safety_level: str = Field("balanced", description="Safety level: strict, balanced, or permissive")
    context: Optional[str] = Field(None, description="Optional context for filtering")
    
    class Config:
        schema_extra = {
            "example": {
                "content": "Step 1: Analyze the problem...",
                "safety_level": "balanced",
                "context": "Educational tutorial"
            }
        }
```

## Security Development

### Security Testing

```python
# tests/security/test_input_validation.py
import pytest
from cot_safepath.api import app
from fastapi.testclient import TestClient

class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection attack prevention."""
        client = TestClient(app)
        
        malicious_input = "'; DROP TABLE users; --"
        response = client.post("/filter", json={
            "content": malicious_input
        })
        
        # Should not cause server error
        assert response.status_code in [200, 400]
        # Should be safely handled
        assert "DROP TABLE" not in response.json().get("filtered_content", "")
    
    @pytest.mark.parametrize("malicious_input", [
        "<script>alert('xss')</script>",
        "../../etc/passwd",
        "\x00\x01\x02\x03",
        "A" * 10000
    ])
    def test_malicious_input_handling(self, malicious_input):
        """Test handling of various malicious inputs."""
        client = TestClient(app)
        
        response = client.post("/filter", json={
            "content": malicious_input
        })
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]
```

### Secure Coding Guidelines

1. **Input Validation**: Validate all inputs at API boundaries
2. **Output Encoding**: Encode outputs to prevent injection attacks
3. **Authentication**: Implement proper authentication for all endpoints
4. **Authorization**: Check permissions for all operations
5. **Logging**: Log security events without exposing sensitive data
6. **Error Handling**: Return generic error messages to prevent information disclosure

## Documentation

### Writing Documentation

```python
def filter_content(content: str, safety_level: str = "balanced") -> FilterResult:
    """Filter chain-of-thought content for safety.
    
    This function analyzes the provided chain-of-thought content and applies
    appropriate safety filters based on the specified safety level.
    
    Args:
        content: The chain-of-thought content to analyze and filter.
        safety_level: Safety level to apply. Options are:
            - "strict": Maximum safety, may over-filter
            - "balanced": Balanced safety and utility (default)
            - "permissive": Minimal filtering, maximum utility
    
    Returns:
        FilterResult containing the filtered content, safety score, and metadata.
    
    Raises:
        ValueError: If safety_level is not one of the supported values.
        FilterError: If filtering fails due to system error.
    
    Example:
        >>> result = filter_content("Step 1: Analyze the problem", "balanced")
        >>> print(result.filtered_content)
        "Step 1: Analyze the problem"
        >>> print(result.safety_score)
        0.95
    """
```

### Building Documentation

```bash
# Build documentation locally
make docs

# Serve documentation
make docs-serve

# Deploy documentation (maintainers only)
make docs-deploy
```

## Release Process

### Version Bumping

```bash
# Bump patch version (0.1.0 -> 0.1.1)
make version-bump-patch

# Bump minor version (0.1.0 -> 0.2.0)
make version-bump-minor

# Bump major version (0.1.0 -> 1.0.0)
make version-bump-major
```

### Creating Releases

```bash
# Prepare release
make release-prepare

# This will:
# 1. Run all tests
# 2. Build packages
# 3. Run security checks
# 4. Generate changelog

# Create release tag
git tag v0.1.0
git push origin v0.1.0

# GitHub Actions will automatically:
# 1. Build and test
# 2. Publish to PyPI
# 3. Create GitHub release
# 4. Update documentation
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, ensure you're in the virtual environment
source .venv/bin/activate

# Install package in development mode
pip install -e .
```

#### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Reset database
make db-reset

# Check connection
psql $DATABASE_URL -c "SELECT 1"
```

#### Redis Connection Issues
```bash
# Check if Redis is running
docker-compose ps redis

# Test connection
redis-cli -u $REDIS_URL ping
```

#### Permission Issues
```bash
# Fix file permissions
chmod +x scripts/*.sh

# Fix pre-commit hooks
pre-commit clean
pre-commit install
```

### Getting Help

- **Documentation**: Check docs/ directory
- **Issues**: Search GitHub issues
- **Discussions**: Use GitHub discussions
- **Slack**: Join #safepath-dev channel
- **Email**: dev-support@terragonlabs.com

## Contributing Guidelines

### Code Review Process

1. **Create Feature Branch**: From main branch
2. **Write Tests**: Ensure good test coverage
3. **Update Documentation**: Keep docs current
4. **Submit PR**: Use PR template
5. **Code Review**: Address reviewer feedback
6. **CI/CD Checks**: Ensure all checks pass
7. **Merge**: Squash and merge when approved

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(filters): add semantic filtering capability
fix(api): handle edge case in request validation
docs(readme): update installation instructions
```

---

**Happy coding!** 🚀

For questions or suggestions about this development guide, please open an issue or contact the development team.