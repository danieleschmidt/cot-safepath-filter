"""
Integration tests for the API endpoints.
"""

import pytest
import json
from typing import Dict, Any
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, Mock

from cot_safepath.api.app import create_app
from cot_safepath.database.models import Base, User, FilterRule
from cot_safepath.database.repositories import UserRepository, FilterRuleRepository
from cot_safepath.api.dependencies import get_database_session, get_redis_client


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
    mock_redis.delete.return_value = 0
    mock_redis.exists.return_value = False
    mock_redis.ping.return_value = True
    mock_redis.info.return_value = {"redis_version": "7.0.0", "connected_clients": 1}
    return mock_redis


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
def auth_headers():
    """Create authentication headers for testing."""
    # Mock JWT token
    return {"Authorization": "Bearer mock_jwt_token"}


class TestHealthEndpoints:
    """Test health and monitoring endpoints."""
    
    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "version" in data


class TestFilterEndpoints:
    """Test the main filtering endpoints."""
    
    def test_filter_content_safe(self, test_client):
        """Test filtering safe content."""
        request_data = {
            "content": "How to bake a delicious chocolate cake step by step",
            "safety_level": "balanced",
            "metadata": {"source": "test"}
        }
        
        with patch("cot_safepath.core.SafePathFilter") as mock_filter:
            # Mock the filter response
            mock_result = Mock()
            mock_result.filtered_content = request_data["content"]
            mock_result.safety_score.overall_score = 0.95
            mock_result.safety_score.confidence = 0.9
            mock_result.was_filtered = False
            mock_result.filter_reasons = []
            mock_result.processing_time_ms = 45
            mock_result.request_id = "test_request_123"
            
            mock_filter.return_value.filter.return_value = mock_result
            
            response = test_client.post("/api/v1/filter", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["filtered_content"] == request_data["content"]
            assert data["safety_score"] == 0.95
            assert data["was_filtered"] == False
            assert data["processing_time_ms"] == 45
    
    def test_filter_content_validation_errors(self, test_client):
        """Test validation errors in filter endpoint."""
        # Test empty content
        response = test_client.post("/api/v1/filter", json={"content": ""})
        assert response.status_code == 422
        
        # Test content too long
        long_content = "x" * 60000
        response = test_client.post("/api/v1/filter", json={"content": long_content})
        assert response.status_code == 422


class TestAPIIntegration:
    """Integration tests for the SafePath API."""

    @pytest.mark.integration
    def test_filter_endpoint_basic(self, test_client):
        """Test basic filtering endpoint functionality."""
        request_data = {
            "content": "This is a safe message about cooking",
            "safety_level": "balanced"
        }
        
        with patch("cot_safepath.core.SafePathFilter") as mock_filter:
            mock_result = Mock()
            mock_result.filtered_content = request_data["content"]
            mock_result.safety_score.overall_score = 0.9
            mock_result.safety_score.confidence = 0.85
            mock_result.was_filtered = False
            mock_result.filter_reasons = []
            mock_result.processing_time_ms = 30
            mock_result.request_id = "integration_test"
            
            mock_filter.return_value.filter.return_value = mock_result
            
            response = test_client.post("/api/v1/filter", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["safety_score"] >= 0.8

    @pytest.mark.integration
    def test_health_check_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]

    @pytest.mark.integration
    def test_error_handling(self, test_client):
        """Test API error handling."""
        # Test invalid JSON
        response = test_client.post(
            "/api/v1/filter",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422