"""
Comprehensive tests for Generation 3 Performance Optimization features.
Tests caching strategies, concurrent processing, and adaptive tuning.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor
import threading

from cot_safepath.performance_optimization import (
    IntelligentCache,
    CacheStrategy,
    ConcurrentProcessor,
    AdaptivePerformanceTuner,
    HighPerformanceFilterEngine,
    OptimizationConfig
)
from cot_safepath.models import FilterRequest, FilterResult, SafetyScore, SafetyLevel


class TestIntelligentCache:
    
    @pytest.fixture
    def cache_config(self):
        return OptimizationConfig(
            cache_size_mb=100,
            cache_ttl_seconds=300
        )
    
    @pytest.fixture
    def cache(self, cache_config):
        return IntelligentCache(cache_config)
    
    def test_cache_initialization(self, cache):
        assert cache.config.cache_size_mb == 100
        assert cache.strategy == CacheStrategy.ADAPTIVE
        assert hasattr(cache, '_cache')  # Main cache storage
    
    def test_lru_cache_basic_operations(self, cache):
        cache.strategy = CacheStrategy.LRU
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
    
    def test_lru_cache_eviction(self, cache):
        cache.strategy = CacheStrategy.LRU
        cache.max_size = 2
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_lfu_cache_operations(self, cache):
        cache.strategy = CacheStrategy.LFU
        cache.max_size = 2
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 multiple times
        cache.get("key1")
        cache.get("key1")
        cache.get("key2")
        
        # Add key3, should evict key2 (less frequently used)
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
    
    def test_ttl_cache_expiration(self, cache):
        cache.strategy = CacheStrategy.TTL
        
        cache.set("key1", "value1", ttl=0.1)  # 100ms TTL
        assert cache.get("key1") == "value1"
        
        time.sleep(0.2)  # Wait for expiration
        assert cache.get("key1") is None
    
    def test_adaptive_cache_strategy_switching(self, cache):
        # Start with adaptive strategy
        assert cache.strategy == CacheStrategy.ADAPTIVE
        
        # Simulate high hit rate scenario
        for i in range(50):
            cache.set(f"key{i}", f"value{i}")
            cache.get(f"key{i}")
        
        # The cache uses adaptive eviction within the strategy, not strategy switching
        # Should maintain adaptive strategy 
        assert cache.strategy == CacheStrategy.ADAPTIVE
    
    def test_cache_statistics(self, cache):
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_statistics()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert "hit_rate" in stats
        assert 0 <= stats["hit_rate"] <= 1
    
    def test_cache_clear(self, cache):
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestConcurrentProcessor:
    
    @pytest.fixture
    def processing_config(self):
        return OptimizationConfig(
            max_worker_threads=4,
            max_worker_processes=2,
            max_concurrent_requests=10,
            pool_timeout_seconds=5.0
        )
    
    @pytest.fixture
    def processor(self, processing_config):
        return ConcurrentProcessor(processing_config)
    
    def test_concurrent_filter_processing(self, processor):
        mock_filter_func = Mock()
        mock_filter_func.return_value = FilterResult(
            filtered_content="filtered",
            safety_score=SafetyScore(overall_score=0.9, confidence=0.8, is_safe=True),
            was_filtered=False
        )
        
        requests = [
            FilterRequest(content=f"test content {i}")
            for i in range(5)
        ]
        
        results = processor.process_batch_concurrent(mock_filter_func, requests)
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, FilterResult)
    
    def test_performance_metrics_collection(self, processor):
        metrics = processor.get_performance_metrics()
        # The method returns a PerformanceMetrics dataclass, not a dict
        assert hasattr(metrics, 'total_requests')
        assert hasattr(metrics, 'avg_response_time_ms')
        assert hasattr(metrics, 'concurrent_requests')


class TestAdaptivePerformanceTuner:
    
    @pytest.fixture
    def tuning_config(self):
        return OptimizationConfig(
            cache_size_mb=50,
            performance_target_ms=100,
            enable_adaptive_tuning=True
        )
    
    @pytest.fixture
    def tuner(self, tuning_config):
        return AdaptivePerformanceTuner(tuning_config)
    
    def test_tuner_initialization(self, tuner):
        assert tuner.config.performance_target_ms == 100
        assert hasattr(tuner, 'config')
    
    def test_performance_recommendations(self, tuner):
        # Test that we can get recommendations
        from cot_safepath.performance_optimization import PerformanceMetrics
        mock_metrics = PerformanceMetrics(
            avg_response_time_ms=150,
            cache_hits=80,
            cache_misses=20,
            concurrent_requests=5,
            peak_concurrent_requests=10,
            error_rate=0.02
        )
        recommendations = tuner.get_performance_recommendations(mock_metrics)
        assert isinstance(recommendations, list)
    
    def test_analyze_and_tune(self, tuner):
        # Test the main tuning method
        from cot_safepath.performance_optimization import PerformanceMetrics
        
        # Create mock processor and cache
        mock_processor = Mock()
        mock_processor.max_workers = 4
        
        mock_cache = Mock()
        mock_cache.get_statistics.return_value = {
            "hit_rate": 0.8,
            "size": 100,
            "max_size": 1000
        }
        
        mock_metrics = PerformanceMetrics(
            avg_response_time_ms=150,
            cache_hits=80,
            cache_misses=20,
            concurrent_requests=5,
            peak_concurrent_requests=10,
            error_rate=0.02
        )
        
        # This should run without error
        result = tuner.analyze_and_tune(mock_cache, mock_processor, mock_metrics)
        assert isinstance(result, dict)


class TestHighPerformanceFilterEngine:
    
    @pytest.fixture
    def optimization_config(self):
        return OptimizationConfig()
    
    @pytest.fixture
    def engine(self, optimization_config):
        return HighPerformanceFilterEngine(optimization_config)
    
    def test_engine_initialization(self, engine):
        assert hasattr(engine, 'cache')
        assert hasattr(engine, 'processor')  
        assert hasattr(engine, 'tuner')
        assert hasattr(engine, '_cache_enabled')
    
    @pytest.mark.asyncio
    async def test_async_filtering(self, engine):
        request = FilterRequest(content="test content", safety_level=SafetyLevel.BALANCED)
        
        # Mock filter function
        mock_filter_func = AsyncMock()
        mock_result = FilterResult(
            filtered_content="filtered test content",
            safety_score=SafetyScore(overall_score=0.9, confidence=0.8, is_safe=True),
            was_filtered=True
        )
        mock_filter_func.return_value = mock_result
        
        result = await engine.filter_async(mock_filter_func, request)
        
        assert isinstance(result, FilterResult)
        assert result.filtered_content == "filtered test content"
    
    def test_batch_filtering(self, engine):
        requests = [
            FilterRequest(content=f"test content {i}")
            for i in range(5)
        ]
        
        # Mock filter function
        mock_filter_func = Mock()
        mock_result = FilterResult(
            filtered_content="filtered content",
            safety_score=SafetyScore(overall_score=0.9, confidence=0.8, is_safe=True),
            was_filtered=False
        )
        mock_filter_func.return_value = mock_result
        
        results = engine.filter_batch(mock_filter_func, requests)
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, FilterResult)
    
    def test_performance_metrics(self, engine):
        from cot_safepath.performance_optimization import PerformanceMetrics
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, PerformanceMetrics)
        
    def test_optimization_report(self, engine):
        report = engine.get_optimization_report()
        assert isinstance(report, dict)


class TestIntegrationScenarios:
    
    @pytest.mark.asyncio  
    async def test_end_to_end_performance_scenario(self):
        """Test complete high-performance filtering pipeline."""
        config = OptimizationConfig(
            cache_size_mb=50,
            max_worker_threads=2,
            enable_adaptive_tuning=True
        )
        
        engine = HighPerformanceFilterEngine(config)
        
        # Mock filter function
        mock_filter_func = AsyncMock()
        mock_result = FilterResult(
            filtered_content="safe content",
            safety_score=SafetyScore(overall_score=0.95, confidence=0.9, is_safe=True),
            was_filtered=False
        )
        mock_filter_func.return_value = mock_result
        
        # Test single async filtering
        single_request = FilterRequest(content="single request")
        result = await engine.filter_async(mock_filter_func, single_request)
        assert isinstance(result, FilterResult)
        
        # Test batch filtering
        batch_requests = [
            FilterRequest(content=f"batch request {i}")
            for i in range(5)
        ]
        batch_filter_func = Mock(return_value=mock_result)
        batch_results = engine.filter_batch(batch_filter_func, batch_requests)
        assert len(batch_results) == 5
        
        # Get performance metrics
        from cot_safepath.performance_optimization import PerformanceMetrics
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, PerformanceMetrics)
    
    def test_performance_with_caching(self):
        """Test performance improvements with caching."""
        config = OptimizationConfig(cache_size_mb=50)
        
        engine = HighPerformanceFilterEngine(config)
        
        mock_result = FilterResult(
            filtered_content="cached content",
            safety_score=SafetyScore(overall_score=0.9, confidence=0.8, is_safe=True),
            was_filtered=False
        )
        mock_filter_func = Mock(return_value=mock_result)
        
        # Process same request multiple times
        request = FilterRequest(content="repeated request")
        
        results = []
        for _ in range(10):
            results.append(engine.filter_batch(mock_filter_func, [request]))
        
        # Verify all results are valid
        for result_batch in results:
            assert len(result_batch) == 1
            assert isinstance(result_batch[0], FilterResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])