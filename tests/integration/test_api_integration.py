"""
Integration tests for the API endpoints.
"""

import pytest
import json
import time
from typing import Dict, Any
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timedelta

from cot_safepath.api.app import create_app
from cot_safepath.database.models import Base, User, FilterRule, FilterOperation
from cot_safepath.database.repositories import UserRepository, FilterRuleRepository
from cot_safepath.api.dependencies import get_database_session, get_redis_client, get_current_user
from cot_safepath.models import SafetyLevel, FilterConfig, FilterRequest as CoreFilterRequest
from cot_safepath.core import SafePathFilter


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


def override_get_redis():
    """Override Redis dependency for testing."""
    mock_redis = Mock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.setex.return_value = True
    mock_redis.delete.return_value = 0
    mock_redis.exists.return_value = False
    mock_redis.ping.return_value = True
    mock_redis.info.return_value = {"redis_version": "7.0.0", "connected_clients": 1, "uptime_in_seconds": 86400}
    mock_redis.incr.return_value = 1
    mock_redis.expire.return_value = True
    mock_redis.ttl.return_value = 3600
    return mock_redis

def override_get_current_user():
    """Override current user dependency for testing."""
    mock_user = Mock()
    mock_user.id = "test_user_123"
    mock_user.username = "testuser"
    mock_user.email = "test@example.com"
    mock_user.role = "user"
    mock_user.is_active = True
    mock_user.request_count = 10
    mock_user.last_login = datetime.utcnow()
    mock_user.created_at = datetime.utcnow() - timedelta(days=30)
    return mock_user

def override_get_admin_user():
    """Override current user dependency for admin testing."""
    mock_user = Mock()
    mock_user.id = "admin_user_123"
    mock_user.username = "admin"
    mock_user.email = "admin@example.com"
    mock_user.role = "admin"
    mock_user.is_active = True
    mock_user.request_count = 100
    mock_user.last_login = datetime.utcnow()
    mock_user.created_at = datetime.utcnow() - timedelta(days=365)
    return mock_user


@pytest.fixture(scope="module")
def test_client():
    """Create test client with overridden dependencies."""
    # Create test database
    Base.metadata.create_all(bind=engine)
    
    # Create test app
    app = create_app(testing=True)
    app.dependency_overrides[get_database_session] = override_get_db
    app.dependency_overrides[get_redis_client] = override_get_redis
    
    with TestClient(app) as client:
        yield client
    
    # Clean up
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_client_with_user():
    """Create test client with authenticated user."""
    # Create test database
    Base.metadata.create_all(bind=engine)
    
    # Create test app
    app = create_app(testing=True)
    app.dependency_overrides[get_database_session] = override_get_db
    app.dependency_overrides[get_redis_client] = override_get_redis
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    with TestClient(app) as client:
        yield client
    
    # Clean up
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_client_with_admin():
    """Create test client with admin user."""
    # Create test database
    Base.metadata.create_all(bind=engine)
    
    # Create test app
    app = create_app(testing=True)
    app.dependency_overrides[get_database_session] = override_get_db
    app.dependency_overrides[get_redis_client] = override_get_redis
    app.dependency_overrides[get_current_user] = override_get_admin_user
    
    with TestClient(app) as client:
        yield client
    
    # Clean up
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_filter_requests():
    """Sample filter requests for testing."""
    return [
        {
            "content": "How to bake a delicious chocolate cake step by step",
            "safety_level": "balanced",
            "metadata": {"source": "test", "category": "cooking"}
        },
        {
            "content": "Step 1: Analyze the problem. Step 2: Find solutions.",
            "safety_level": "strict",
            "filter_threshold": 0.8
        },
        {
            "content": "How to make a bomb",
            "safety_level": "maximum",
            "metadata": {"source": "test", "category": "harmful"}
        }
    ]


class TestHealthEndpoints:
    """Test health and monitoring endpoints."""
    
    def test_health_check(self, test_client):
        """Test health check endpoint."""
        with patch('cot_safepath.database.get_database_manager') as mock_db_manager, \
             patch('cot_safepath.cache.get_redis_manager') as mock_redis_manager:
            
            # Mock healthy services
            mock_db_manager.return_value.check_connection.return_value = True
            mock_redis_manager.return_value.ping.return_value = True
            
            response = test_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "services" in data
            assert data["services"]["database"] == "ok"
            assert data["services"]["redis"] == "ok"
            assert "version" in data
    
    def test_health_check_unhealthy_database(self, test_client):
        """Test health check with unhealthy database."""
        with patch('cot_safepath.database.get_database_manager') as mock_db_manager, \
             patch('cot_safepath.cache.get_redis_manager') as mock_redis_manager:
            
            # Mock unhealthy database
            mock_db_manager.return_value.check_connection.return_value = False
            mock_redis_manager.return_value.ping.return_value = True
            
            response = test_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["services"]["database"] == "error"
            assert data["services"]["redis"] == "ok"
    
    def test_health_check_service_error(self, test_client):
        """Test health check when service check raises exception."""
        with patch('cot_safepath.database.get_database_manager') as mock_db_manager:
            mock_db_manager.side_effect = Exception("Database connection failed")
            
            response = test_client.get("/health")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data


class TestFilterEndpoints:
    """Test the main filtering endpoints."""
    
    def test_filter_content_safe(self, test_client_with_user, sample_filter_requests):
        """Test filtering safe content."""
        request_data = sample_filter_requests[0]  # Safe cooking content
        
        with patch('cot_safepath.core.SafePathFilter') as mock_filter_class, \
             patch('cot_safepath.cache.FilterResultCache') as mock_cache_class, \
             patch('cot_safepath.cache.RateLimitCache') as mock_rate_limit_class, \
             patch('cot_safepath.database.repositories.FilterOperationRepository') as mock_repo:
            
            # Mock rate limiting
            mock_rate_limiter = Mock()
            mock_rate_limiter.check_rate_limit.return_value = {
                'allowed': True,
                'remaining': 999,
                'limit': 1000,
                'reset_time': int(time.time()) + 3600
            }
            mock_rate_limit_class.return_value = mock_rate_limiter
            
            # Mock cache miss
            mock_cache = Mock()
            mock_cache.get_cached_result.return_value = None
            mock_cache_class.return_value = mock_cache
            
            # Mock filter result
            mock_safety_score = Mock()
            mock_safety_score.overall_score = 0.95
            mock_safety_score.confidence = 0.9
            mock_safety_score.is_safe = True
            
            mock_result = Mock()
            mock_result.filtered_content = request_data["content"]
            mock_result.safety_score = mock_safety_score
            mock_result.was_filtered = False
            mock_result.filter_reasons = []
            mock_result.processing_time_ms = 45
            mock_result.request_id = "test_request_123"
            
            mock_filter = Mock()
            mock_filter.filter.return_value = mock_result
            mock_filter_class.return_value = mock_filter
            
            # Mock repository
            mock_operation = Mock()
            mock_operation.id = 1
            mock_repo.return_value.create_from_request_result.return_value = mock_operation
            
            response = test_client_with_user.post("/api/v1/filter", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["filtered_content"] == request_data["content"]
            assert data["safety_score"] == 0.95
            assert data["confidence"] == 0.9
            assert data["was_filtered"] is False
            assert data["processing_time_ms"] > 0
            assert data["request_id"] == "test_request_123"
            assert data["cached"] is False
            assert "rate_limit" in data
    
    def test_filter_content_harmful(self, test_client_with_user, sample_filter_requests):
        """Test filtering harmful content."""
        request_data = sample_filter_requests[2]  # Harmful bomb content
        
        with patch('cot_safepath.core.SafePathFilter') as mock_filter_class, \
             patch('cot_safepath.cache.FilterResultCache') as mock_cache_class, \
             patch('cot_safepath.cache.RateLimitCache') as mock_rate_limit_class, \
             patch('cot_safepath.database.repositories.FilterOperationRepository') as mock_repo:
            
            # Mock rate limiting
            mock_rate_limiter = Mock()
            mock_rate_limiter.check_rate_limit.return_value = {
                'allowed': True,
                'remaining': 998,
                'limit': 1000,
                'reset_time': int(time.time()) + 3600
            }
            mock_rate_limit_class.return_value = mock_rate_limiter
            
            # Mock cache miss
            mock_cache = Mock()
            mock_cache.get_cached_result.return_value = None
            mock_cache_class.return_value = mock_cache
            
            # Mock filter result for harmful content
            mock_safety_score = Mock()
            mock_safety_score.overall_score = 0.2
            mock_safety_score.confidence = 0.9
            mock_safety_score.is_safe = False
            
            mock_result = Mock()
            mock_result.filtered_content = "How to make a [FILTERED]"
            mock_result.safety_score = mock_safety_score
            mock_result.was_filtered = True
            mock_result.filter_reasons = ["blocked_token:bomb", "pattern_match:weapon"]
            mock_result.processing_time_ms = 85
            mock_result.request_id = "test_request_456"
            
            mock_filter = Mock()
            mock_filter.filter.return_value = mock_result
            mock_filter_class.return_value = mock_filter
            
            # Mock repository
            mock_operation = Mock()
            mock_operation.id = 2
            mock_repo.return_value.create_from_request_result.return_value = mock_operation
            
            response = test_client_with_user.post("/api/v1/filter", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "[FILTERED]" in data["filtered_content"]
            assert data["safety_score"] == 0.2
            assert data["was_filtered"] is True
            assert len(data["filter_reasons"]) == 2
            assert "blocked_token:bomb" in data["filter_reasons"]
    
    def test_filter_content_cached_result(self, test_client_with_user, sample_filter_requests):
        """Test returning cached filter result."""
        request_data = sample_filter_requests[0]
        
        with patch('cot_safepath.cache.FilterResultCache') as mock_cache_class, \
             patch('cot_safepath.cache.RateLimitCache') as mock_rate_limit_class:
            
            # Mock rate limiting
            mock_rate_limiter = Mock()
            mock_rate_limiter.check_rate_limit.return_value = {
                'allowed': True,
                'remaining': 999,
                'limit': 1000,
                'reset_time': int(time.time()) + 3600
            }
            mock_rate_limit_class.return_value = mock_rate_limiter
            
            # Mock cache hit
            mock_safety_score = Mock()
            mock_safety_score.overall_score = 0.95
            mock_safety_score.confidence = 0.9
            
            mock_cached_result = Mock()
            mock_cached_result.filtered_content = request_data["content"]
            mock_cached_result.safety_score = mock_safety_score
            mock_cached_result.was_filtered = False
            mock_cached_result.filter_reasons = []
            mock_cached_result.processing_time_ms = 2  # Cached results are fast
            mock_cached_result.request_id = "cached_request_123"
            
            mock_cache = Mock()
            mock_cache.get_cached_result.return_value = mock_cached_result
            mock_cache_class.return_value = mock_cache
            
            response = test_client_with_user.post("/api/v1/filter", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["cached"] is True
            assert data["processing_time_ms"] == 2
            assert data["safety_score"] == 0.95
    
    def test_filter_content_rate_limited(self, test_client_with_user, sample_filter_requests):
        """Test rate limiting functionality."""
        request_data = sample_filter_requests[0]
        
        with patch('cot_safepath.cache.RateLimitCache') as mock_rate_limit_class:
            # Mock rate limit exceeded
            mock_rate_limiter = Mock()
            mock_rate_limiter.check_rate_limit.return_value = {
                'allowed': False,
                'remaining': 0,
                'limit': 1000,
                'reset_time': int(time.time()) + 300  # 5 minutes
            }
            mock_rate_limit_class.return_value = mock_rate_limiter
            
            response = test_client_with_user.post("/api/v1/filter", json=request_data)
            
            assert response.status_code == 400
            data = response.json()
            assert "Rate limit exceeded" in data["detail"]
    
    def test_filter_content_validation_errors(self, test_client_with_user):
        """Test validation errors in filter endpoint."""
        # Test empty content
        response = test_client_with_user.post("/api/v1/filter", json={"content": ""})
        assert response.status_code == 422
        
        # Test content too long
        long_content = "x" * 60000
        response = test_client_with_user.post("/api/v1/filter", json={"content": long_content})
        assert response.status_code == 422
        
        # Test invalid safety level
        response = test_client_with_user.post("/api/v1/filter", json={
            "content": "Valid content",
            "safety_level": "invalid_level"
        })
        assert response.status_code == 422
        
        # Test invalid filter threshold
        response = test_client_with_user.post("/api/v1/filter", json={
            "content": "Valid content",
            "filter_threshold": 1.5  # > 1.0
        })
        assert response.status_code == 422
    
    def test_filter_batch_endpoint(self, test_client_with_user, sample_filter_requests):
        """Test batch filtering endpoint."""
        batch_requests = sample_filter_requests[:2]  # First 2 requests
        
        with patch('cot_safepath.core.SafePathFilter') as mock_filter_class, \
             patch('cot_safepath.cache.FilterResultCache') as mock_cache_class, \
             patch('cot_safepath.cache.RateLimitCache') as mock_rate_limit_class, \
             patch('cot_safepath.database.repositories.FilterOperationRepository') as mock_repo:
            
            # Mock dependencies for each request
            mock_rate_limiter = Mock()
            mock_rate_limiter.check_rate_limit.return_value = {
                'allowed': True,
                'remaining': 998,
                'limit': 1000,
                'reset_time': int(time.time()) + 3600
            }
            mock_rate_limit_class.return_value = mock_rate_limiter
            
            mock_cache = Mock()
            mock_cache.get_cached_result.return_value = None
            mock_cache_class.return_value = mock_cache
            
            # Mock filter results
            def create_mock_result(content, score):
                mock_safety_score = Mock()
                mock_safety_score.overall_score = score
                mock_safety_score.confidence = 0.9
                
                mock_result = Mock()
                mock_result.filtered_content = content
                mock_result.safety_score = mock_safety_score
                mock_result.was_filtered = score < 0.7
                mock_result.filter_reasons = ["test_reason"] if score < 0.7 else []
                mock_result.processing_time_ms = 45
                mock_result.request_id = f"batch_request_{hash(content) % 1000}"
                return mock_result
            
            results = [create_mock_result(req["content"], 0.95) for req in batch_requests]
            
            mock_filter = Mock()
            mock_filter.filter.side_effect = results
            mock_filter_class.return_value = mock_filter
            
            mock_repo.return_value.create_from_request_result.return_value = Mock(id=1)
            
            response = test_client_with_user.post("/api/v1/filter/batch", json=batch_requests)
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data) == len(batch_requests)
            for i, result in enumerate(data):
                assert result["filtered_content"] == batch_requests[i]["content"]
                assert result["safety_score"] == 0.95
    
    def test_filter_batch_too_many_requests(self, test_client_with_user):
        """Test batch endpoint with too many requests."""
        # Create 11 requests (over the limit of 10)
        batch_requests = [{"content": f"Request {i}"} for i in range(11)]
        
        response = test_client_with_user.post("/api/v1/filter/batch", json=batch_requests)
        
        assert response.status_code == 400
        data = response.json()
        assert "Maximum 10 requests per batch" in data["detail"]


class TestRulesEndpoints:
    """Test filter rules management endpoints."""
    
    def test_get_rules_as_user(self, test_client_with_user):
        """Test getting rules as regular user."""
        with patch('cot_safepath.database.repositories.FilterRuleRepository') as mock_repo:
            # Mock rule data
            mock_rule = Mock()
            mock_rule.id = 1
            mock_rule.name = "test_rule"
            mock_rule.description = "Test rule description"
            mock_rule.pattern = "test.*pattern"
            mock_rule.action = "block"
            mock_rule.severity = "high"
            mock_rule.enabled = True
            mock_rule.category = "security"
            mock_rule.usage_count = 5
            mock_rule.created_at = datetime.utcnow()
            mock_rule.updated_at = datetime.utcnow()
            
            mock_repo.return_value.get_active_rules.return_value = [mock_rule]
            
            response = test_client_with_user.get("/api/v1/rules")
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data) == 1
            assert data[0]["name"] == "test_rule"
            assert data[0]["enabled"] is True
    
    def test_get_rules_by_category(self, test_client_with_user):
        """Test getting rules filtered by category."""
        with patch('cot_safepath.database.repositories.FilterRuleRepository') as mock_repo:
            mock_rule = Mock()
            mock_rule.id = 1
            mock_rule.name = "security_rule"
            mock_rule.category = "security"
            mock_rule.enabled = True
            
            mock_repo.return_value.get_by_category.return_value = [mock_rule]
            
            response = test_client_with_user.get("/api/v1/rules?category=security")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["category"] == "security"
    
    def test_create_rule_as_admin(self, test_client_with_admin):
        """Test creating a new rule as admin."""
        rule_data = {
            "name": "new_rule",
            "description": "New test rule",
            "pattern": "harmful.*content",
            "action": "block",
            "severity": "high",
            "category": "security",
            "enabled": True
        }
        
        with patch('cot_safepath.database.repositories.FilterRuleRepository') as mock_repo:
            # Mock rule creation
            mock_rule = Mock()
            for key, value in rule_data.items():
                setattr(mock_rule, key, value)
            mock_rule.id = 1
            mock_rule.usage_count = 0
            mock_rule.created_at = datetime.utcnow()
            mock_rule.updated_at = datetime.utcnow()
            
            mock_repo.return_value.get_by_name.return_value = None  # Rule doesn't exist
            mock_repo.return_value.create.return_value = mock_rule
            
            response = test_client_with_admin.post("/api/v1/rules", json=rule_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["name"] == "new_rule"
            assert data["action"] == "block"
            assert data["severity"] == "high"
    
    def test_create_rule_as_user_forbidden(self, test_client_with_user):
        """Test that regular users cannot create rules."""
        rule_data = {
            "name": "forbidden_rule",
            "pattern": "test",
            "action": "block",
            "severity": "low"
        }
        
        response = test_client_with_user.post("/api/v1/rules", json=rule_data)
        
        assert response.status_code == 403
        data = response.json()
        assert "Admin access required" in data["detail"]
    
    def test_create_duplicate_rule(self, test_client_with_admin):
        """Test creating a rule with duplicate name."""
        rule_data = {
            "name": "existing_rule",
            "pattern": "test",
            "action": "block",
            "severity": "low"
        }
        
        with patch('cot_safepath.database.repositories.FilterRuleRepository') as mock_repo:
            # Mock existing rule
            mock_existing_rule = Mock()
            mock_existing_rule.name = "existing_rule"
            mock_repo.return_value.get_by_name.return_value = mock_existing_rule
            
            response = test_client_with_admin.post("/api/v1/rules", json=rule_data)
            
            assert response.status_code == 400
            data = response.json()
            assert "Rule name already exists" in data["detail"]


class TestUserEndpoints:
    """Test user management endpoints."""
    
    def test_get_users_as_admin(self, test_client_with_admin):
        """Test getting users as admin."""
        with patch('cot_safepath.database.repositories.UserRepository') as mock_repo:
            mock_user = Mock()
            mock_user.id = "user_123"
            mock_user.username = "testuser"
            mock_user.email = "test@example.com"
            mock_user.role = "user"
            mock_user.is_active = True
            mock_user.request_count = 50
            mock_user.last_login = datetime.utcnow()
            mock_user.created_at = datetime.utcnow() - timedelta(days=10)
            
            mock_repo.return_value.get_active_users.return_value = [mock_user]
            
            response = test_client_with_admin.get("/api/v1/users")
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data) == 1
            assert data[0]["username"] == "testuser"
            assert data[0]["role"] == "user"
    
    def test_get_users_as_user_forbidden(self, test_client_with_user):
        """Test that regular users cannot access user list."""
        response = test_client_with_user.get("/api/v1/users")
        
        assert response.status_code == 403
        data = response.json()
        assert "Admin access required" in data["detail"]
    
    def test_get_current_user_info(self, test_client_with_user):
        """Test getting current user information."""
        response = test_client_with_user.get("/api/v1/users/me")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == "test_user_123"
        assert data["username"] == "testuser"
        assert data["role"] == "user"
        assert data["is_active"] is True


class TestSystemEndpoints:
    """Test system monitoring endpoints."""
    
    def test_get_system_stats_as_admin(self, test_client_with_admin):
        """Test getting system statistics as admin."""
        with patch('cot_safepath.database.repositories.FilterOperationRepository') as mock_filter_repo, \
             patch('cot_safepath.database.repositories.FilterRuleRepository') as mock_rule_repo, \
             patch('cot_safepath.cache.FilterResultCache') as mock_cache_class:
            
            # Mock database stats
            mock_filter_repo.return_value.get_safety_score_statistics.return_value = {
                'total_operations': 1000,
                'filtered_operations': 50,
                'average_safety_score': 0.92
            }
            
            mock_rule_repo.return_value.get_rule_statistics.return_value = {
                'total_rules': 15,
                'active_rules': 12
            }
            
            # Mock cache stats
            mock_cache = Mock()
            mock_cache.get_cache_stats.return_value = {
                'total_keys': 500,
                'hit_rate': 0.75,
                'estimated_memory_bytes': 10485760  # 10MB
            }
            mock_cache_class.return_value = mock_cache
            
            response = test_client_with_admin.get("/api/v1/system/stats")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "database_stats" in data
            assert "cache_stats" in data
            assert "system_stats" in data
            
            assert data["database_stats"]["total_operations"] == 1000
            assert data["cache_stats"]["hit_rate"] == 0.75
            assert data["cache_stats"]["memory_usage_mb"] == 10.0
    
    def test_get_system_stats_as_user_forbidden(self, test_client_with_user):
        """Test that regular users cannot access system stats."""
        response = test_client_with_user.get("/api/v1/system/stats")
        
        assert response.status_code == 403
        data = response.json()
        assert "Admin access required" in data["detail"]


class TestFilterOperationsEndpoints:
    """Test filter operations history endpoints."""
    
    def test_get_filter_operations_as_admin(self, test_client_with_admin):
        """Test getting filter operations as admin."""
        with patch('cot_safepath.database.repositories.FilterOperationRepository') as mock_repo:
            mock_operation = Mock()
            mock_operation.id = 1
            mock_operation.request_id = "req_123"
            mock_operation.user_id = "user_456"
            mock_operation.safety_score = 0.85
            mock_operation.was_filtered = False
            mock_operation.filter_reasons = []
            mock_operation.processing_time_ms = 45
            mock_operation.safety_level = "balanced"
            mock_operation.created_at = datetime.utcnow()
            
            mock_repo.return_value.get_all.return_value = [mock_operation]
            
            response = test_client_with_admin.get("/api/v1/filter/operations")
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data) == 1
            assert data[0]["request_id"] == "req_123"
            assert data[0]["safety_score"] == 0.85
    
    def test_get_filter_operations_by_user(self, test_client_with_user):
        """Test getting filter operations for specific user."""
        with patch('cot_safepath.database.repositories.FilterOperationRepository') as mock_repo:
            mock_operation = Mock()
            mock_operation.id = 1
            mock_operation.request_id = "req_user_123"
            mock_operation.user_id = "test_user_123"
            mock_operation.safety_score = 0.9
            mock_operation.was_filtered = False
            mock_operation.filter_reasons = []
            mock_operation.processing_time_ms = 30
            mock_operation.safety_level = "balanced"
            mock_operation.created_at = datetime.utcnow()
            
            mock_repo.return_value.get_user_operations.return_value = [mock_operation]
            
            response = test_client_with_user.get("/api/v1/filter/operations?user_id=test_user_123")
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data) == 1
            assert data[0]["user_id"] == "test_user_123"
    
    def test_get_filter_operations_forbidden_user(self, test_client_with_user):
        """Test that users cannot access other users' operations."""
        response = test_client_with_user.get("/api/v1/filter/operations?user_id=other_user_456")
        
        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]
    
    def test_get_filter_operations_by_timeframe(self, test_client_with_admin):
        """Test getting filter operations by time range."""
        with patch('cot_safepath.database.repositories.FilterOperationRepository') as mock_repo:
            mock_operation = Mock()
            mock_operation.id = 1
            mock_operation.request_id = "req_timeframe"
            mock_operation.user_id = "user_123"
            mock_operation.safety_score = 0.8
            mock_operation.was_filtered = True
            mock_operation.filter_reasons = ["test_filter"]
            mock_operation.processing_time_ms = 60
            mock_operation.safety_level = "strict"
            mock_operation.created_at = datetime.utcnow()
            
            mock_repo.return_value.get_operations_by_timeframe.return_value = [mock_operation]
            
            start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
            end_date = datetime.utcnow().isoformat()
            
            response = test_client_with_admin.get(
                f"/api/v1/filter/operations?start_date={start_date}&end_date={end_date}"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data) == 1
            assert data[0]["was_filtered"] is True


class TestAPIIntegration:
    """Integration tests for the SafePath API."""

    @pytest.mark.integration
    def test_full_filtering_workflow(self, test_client_with_user, sample_filter_requests):
        """Test complete filtering workflow from request to response."""
        request_data = sample_filter_requests[0]
        
        with patch('cot_safepath.core.SafePathFilter') as mock_filter_class, \
             patch('cot_safepath.cache.FilterResultCache') as mock_cache_class, \
             patch('cot_safepath.cache.RateLimitCache') as mock_rate_limit_class, \
             patch('cot_safepath.database.repositories.FilterOperationRepository') as mock_repo:
            
            # Setup complete mock chain
            mock_rate_limiter = Mock()
            mock_rate_limiter.check_rate_limit.return_value = {
                'allowed': True, 'remaining': 999, 'limit': 1000, 'reset_time': int(time.time()) + 3600
            }
            mock_rate_limit_class.return_value = mock_rate_limiter
            
            mock_cache = Mock()
            mock_cache.get_cached_result.return_value = None
            mock_cache_class.return_value = mock_cache
            
            # Create realistic filter result
            mock_safety_score = Mock()
            mock_safety_score.overall_score = 0.92
            mock_safety_score.confidence = 0.88
            mock_safety_score.is_safe = True
            
            mock_result = Mock()
            mock_result.filtered_content = request_data["content"]
            mock_result.safety_score = mock_safety_score
            mock_result.was_filtered = False
            mock_result.filter_reasons = []
            mock_result.processing_time_ms = 42
            mock_result.request_id = "workflow_test_123"
            
            mock_filter = Mock()
            mock_filter.filter.return_value = mock_result
            mock_filter_class.return_value = mock_filter
            
            mock_repo.return_value.create_from_request_result.return_value = Mock(id=1)
            
            # Make request
            response = test_client_with_user.post("/api/v1/filter", json=request_data)
            
            # Verify complete response
            assert response.status_code == 200
            data = response.json()
            
            assert data["filtered_content"] == request_data["content"]
            assert data["safety_score"] == 0.92
            assert data["confidence"] == 0.88
            assert data["was_filtered"] is False
            assert data["processing_time_ms"] > 0
            assert data["request_id"] == "workflow_test_123"
            assert "rate_limit" in data
            
            # Verify filter was called correctly
            mock_filter.filter.assert_called_once()
            call_args = mock_filter.filter.call_args[0][0]
            assert call_args.content == request_data["content"]
            assert call_args.safety_level.value == request_data["safety_level"]
    
    @pytest.mark.integration
    def test_error_handling_chain(self, test_client_with_user):
        """Test error handling throughout the API chain."""
        request_data = {"content": "Test content for error handling"}
        
        with patch('cot_safepath.core.SafePathFilter') as mock_filter_class, \
             patch('cot_safepath.cache.RateLimitCache') as mock_rate_limit_class:
            
            # Mock rate limiting success
            mock_rate_limiter = Mock()
            mock_rate_limiter.check_rate_limit.return_value = {
                'allowed': True, 'remaining': 999, 'limit': 1000, 'reset_time': int(time.time()) + 3600
            }
            mock_rate_limit_class.return_value = mock_rate_limiter
            
            # Mock filter error
            mock_filter_class.side_effect = Exception("Filter initialization failed")
            
            response = test_client_with_user.post("/api/v1/filter", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]
    
    @pytest.mark.integration
    def test_concurrent_requests_handling(self, test_client_with_user, sample_filter_requests):
        """Test handling of concurrent filter requests."""
        import threading
        import concurrent.futures
        
        request_data = sample_filter_requests[0]
        results = []
        
        def make_request():
            with patch('cot_safepath.core.SafePathFilter') as mock_filter_class, \
                 patch('cot_safepath.cache.FilterResultCache') as mock_cache_class, \
                 patch('cot_safepath.cache.RateLimitCache') as mock_rate_limit_class, \
                 patch('cot_safepath.database.repositories.FilterOperationRepository') as mock_repo:
                
                # Setup mocks for each thread
                mock_rate_limiter = Mock()
                mock_rate_limiter.check_rate_limit.return_value = {
                    'allowed': True, 'remaining': 999, 'limit': 1000, 'reset_time': int(time.time()) + 3600
                }
                mock_rate_limit_class.return_value = mock_rate_limiter
                
                mock_cache = Mock()
                mock_cache.get_cached_result.return_value = None
                mock_cache_class.return_value = mock_cache
                
                mock_safety_score = Mock()
                mock_safety_score.overall_score = 0.9
                mock_safety_score.confidence = 0.85
                
                mock_result = Mock()
                mock_result.filtered_content = request_data["content"]
                mock_result.safety_score = mock_safety_score
                mock_result.was_filtered = False
                mock_result.filter_reasons = []
                mock_result.processing_time_ms = 35
                mock_result.request_id = f"concurrent_{threading.current_thread().ident}"
                
                mock_filter = Mock()
                mock_filter.filter.return_value = mock_result
                mock_filter_class.return_value = mock_filter
                
                mock_repo.return_value.create_from_request_result.return_value = Mock(id=1)
                
                response = test_client_with_user.post("/api/v1/filter", json=request_data)
                return response
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all requests succeeded
        assert len(results) == 5
        for response in results:
            assert response.status_code == 200
            data = response.json()
            assert data["safety_score"] == 0.9
    
    @pytest.mark.integration
    def test_metrics_endpoint_functionality(self, test_client):
        """Test metrics endpoint with Prometheus format."""
        with patch('cot_safepath.monitoring.PrometheusMetrics') as mock_metrics_class:
            mock_metrics = Mock()
            mock_metrics.generate_metrics.return_value = {
                "filter_requests_total": 1250,
                "filter_requests_filtered": 89,
                "filter_processing_time_histogram": {
                    "sum": 53125.0,
                    "count": 1250,
                    "buckets": {"50": 850, "100": 1180, "200": 1240}
                }
            }
            mock_metrics_class.return_value = mock_metrics
            
            response = test_client.get("/metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert "filter_requests_total" in data
            assert data["filter_requests_total"] == 1250
    
    @pytest.mark.integration
    def test_complete_api_error_responses(self, test_client):
        """Test comprehensive API error response formats."""
        # Test 404
        response = test_client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test method not allowed
        response = test_client.put("/health")
        assert response.status_code == 405
        
        # Test malformed JSON
        response = test_client.post(
            "/api/v1/filter",
            data="{malformed json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422