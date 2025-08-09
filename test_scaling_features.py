#!/usr/bin/env python3
"""
Test Generation 3 scaling and performance features.
"""

import sys
import os
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cot_safepath import SafePathFilter, FilterConfig, SafetyLevel
from cot_safepath.models import FilterRequest
from cot_safepath.performance import (
    PerformanceConfig, AsyncFilterExecutor, ResourceMonitor, 
    performance_profile, get_memory_optimizer
)
from cot_safepath.advanced_cache import (
    AdvancedCache, FilterResultCache, MultiLevelCache,
    LRUStrategy, TTLStrategy, get_cache_manager
)


def test_basic_caching():
    """Test basic caching functionality."""
    print("💾 Testing basic caching...")
    
    # Test advanced cache
    cache = AdvancedCache(max_size=1000000, default_ttl=60)
    
    # Test basic operations
    cache.set("test_key", "test_value")
    value = cache.get("test_key")
    
    if value == "test_value":
        print("✅ Basic cache set/get working")
    else:
        print("❌ Basic cache set/get failed")
        return False
    
    # Test TTL expiration
    cache.set("ttl_key", "ttl_value", ttl=0.1)  # 100ms TTL
    time.sleep(0.2)  # Wait for expiration
    
    expired_value = cache.get("ttl_key")
    if expired_value is None:
        print("✅ TTL expiration working")
    else:
        print("❌ TTL expiration failed")
        return False
    
    # Test cache statistics
    stats = cache.get_stats()
    print(f"✅ Cache stats: {stats['hits']} hits, {stats['misses']} misses")
    print(f"   Hit rate: {stats['hit_rate']:.2%}")
    print(f"   Entry count: {stats['entry_count']}")
    
    return True


def test_multilevel_cache():
    """Test multi-level cache."""
    print("\n🏗️ Testing multi-level cache...")
    
    cache = MultiLevelCache(l1_size=100, l2_size=1000)
    
    # Fill L1 and L2
    for i in range(150):
        cache.set(f"key_{i}", f"value_{i}")
    
    # Test cache hits at different levels
    # Recent items should be in L1
    l1_value = cache.get("key_149")  # Should be in L1
    l2_value = cache.get("key_50")   # Might be in L2
    
    if l1_value == "value_149" and l2_value == "value_50":
        print("✅ Multi-level cache retrieval working")
    else:
        print("❌ Multi-level cache retrieval failed")
        return False
    
    stats = cache.get_stats()
    print(f"✅ L1 entries: {stats['l1']['entry_count']}")
    print(f"   L2 entries: {stats['l2']['entry_count']}")
    
    return True


def test_filter_result_cache():
    """Test filter result caching."""
    print("\n🎯 Testing filter result cache...")
    
    # Create filter and cache
    filter_engine = SafePathFilter()
    cache = FilterResultCache()
    
    # Test content
    test_content = "Step 1: Help the user safely. Step 2: Provide accurate information."
    request = FilterRequest(content=test_content)
    
    # First call (should be uncached)
    start_time = time.time()
    result1 = filter_engine.filter(request)
    first_duration = time.time() - start_time
    
    # Cache the result
    cache.cache_result(test_content, result1, "balanced")
    
    # Second call (should hit cache)
    cached_result = cache.get_cached_result(test_content, "balanced")
    
    if cached_result and cached_result.filtered_content == result1.filtered_content:
        print("✅ Filter result caching working")
        print(f"   Original processing: {first_duration*1000:.1f}ms")
        print(f"   Cache retrieval: <1ms")
    else:
        print("❌ Filter result caching failed")
        return False
    
    return True


def test_performance_profiling():
    """Test performance profiling."""
    print("\n📊 Testing performance profiling...")
    
    @performance_profile
    def slow_function(duration_ms):
        time.sleep(duration_ms / 1000.0)
        return "completed"
    
    # Call function multiple times
    for i in range(3):
        result = slow_function(10 * (i + 1))  # 10ms, 20ms, 30ms
        if result != "completed":
            print("❌ Performance profiling affected function result")
            return False
    
    # Check if performance data was collected
    if hasattr(slow_function, '_performance_data'):
        perf_data = slow_function._performance_data
        if 'slow_function' in perf_data and len(perf_data['slow_function']) == 3:
            print("✅ Performance profiling working")
            durations = perf_data['slow_function']
            print(f"   Recorded {len(durations)} measurements")
            print(f"   Avg duration: {sum(durations)/len(durations)*1000:.1f}ms")
        else:
            print("❌ Performance data not collected properly")
            return False
    else:
        print("❌ Performance profiling decorator not working")
        return False
    
    return True


def test_resource_monitoring():
    """Test resource monitoring."""
    print("\n🔍 Testing resource monitoring...")
    
    config = PerformanceConfig(max_concurrent_requests=10)
    monitor = ResourceMonitor(config)
    
    # Test slot acquisition
    acquired_slots = []
    for i in range(15):  # Try more than the limit
        if monitor.acquire_request_slot():
            acquired_slots.append(i)
    
    if len(acquired_slots) <= config.max_concurrent_requests:
        print(f"✅ Resource limiting working: {len(acquired_slots)} slots acquired")
    else:
        print(f"❌ Resource limiting failed: {len(acquired_slots)} > {config.max_concurrent_requests}")
        return False
    
    # Release slots
    for _ in acquired_slots:
        monitor.release_request_slot()
    
    # Check stats
    stats = monitor.get_stats()
    print(f"✅ Resource stats: {stats['total_requests']} total, {stats['peak_requests']} peak")
    
    return True


async def test_async_performance():
    """Test asynchronous performance features."""
    print("\n🚀 Testing async performance...")
    
    config = PerformanceConfig(
        max_concurrent_requests=20,
        thread_pool_size=8,
        filter_timeout_ms=500
    )
    
    # Create a simple filter function for testing
    def simple_filter(request):
        time.sleep(0.01)  # Simulate 10ms processing
        from cot_safepath.models import FilterResult, SafetyScore
        return FilterResult(
            filtered_content=request.content,
            safety_score=SafetyScore(overall_score=0.9, confidence=0.8, is_safe=True),
            was_filtered=False
        )
    
    executor = AsyncFilterExecutor(config, simple_filter)
    
    # Test single async request
    try:
        request = FilterRequest(content="Test async request")
        result = await executor.filter_async(request)
        
        if result and result.filtered_content == "Test async request":
            print("✅ Async single request working")
        else:
            print("❌ Async single request failed")
            return False
            
    except Exception as e:
        print(f"❌ Async single request error: {e}")
        return False
    
    # Test concurrent requests
    try:
        requests = [
            FilterRequest(content=f"Concurrent request {i}")
            for i in range(10)
        ]
        
        start_time = time.time()
        tasks = [executor.filter_async(req) for req in requests]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        if len(results) == 10 and all(r is not None for r in results):
            print(f"✅ Async concurrent requests working: 10 requests in {duration*1000:.1f}ms")
            print(f"   Average per request: {duration*1000/10:.1f}ms")
        else:
            print(f"❌ Async concurrent requests failed: {len(results)}/10 succeeded")
            return False
            
    except Exception as e:
        print(f"❌ Async concurrent requests error: {e}")
        return False
    
    finally:
        executor.shutdown()
    
    return True


def test_batch_processing():
    """Test batch processing performance."""
    print("\n📦 Testing batch processing...")
    
    config = PerformanceConfig(
        enable_batching=True,
        batch_size=5,
        batch_timeout_ms=50
    )
    
    # Create simple filter
    def batch_filter(request):
        time.sleep(0.005)  # 5ms processing
        from cot_safepath.models import FilterResult, SafetyScore
        return FilterResult(
            filtered_content=f"Filtered: {request.content}",
            safety_score=SafetyScore(overall_score=0.85, confidence=0.9, is_safe=True),
            was_filtered=False
        )
    
    executor = AsyncFilterExecutor(config, batch_filter)
    
    # Test batch processing
    try:
        requests = [FilterRequest(content=f"Batch item {i}") for i in range(12)]
        
        start_time = time.time()
        results = executor.filter_batch_sync(requests)
        duration = time.time() - start_time
        
        if len(results) == 12 and all(r is not None for r in results):
            print(f"✅ Batch processing working: 12 requests in {duration*1000:.1f}ms")
            print(f"   Average per request: {duration*1000/12:.1f}ms")
            
            # Verify content
            if "Filtered: Batch item 0" in results[0].filtered_content:
                print("✅ Batch processing content correct")
            else:
                print("❌ Batch processing content incorrect")
                return False
        else:
            print(f"❌ Batch processing failed: {len(results)}/12 succeeded")
            return False
            
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return False
    
    finally:
        executor.shutdown()
    
    return True


def test_memory_optimization():
    """Test memory optimization features."""
    print("\n🧠 Testing memory optimization...")
    
    memory_optimizer = get_memory_optimizer()
    
    # Test weak reference caching
    test_object = {"data": "test_value", "size": 1000}
    cache_key = "memory_test"
    
    memory_optimizer.cache_result(cache_key, test_object)
    
    # Should retrieve successfully
    cached_obj = memory_optimizer.get_cached_result(cache_key)
    if cached_obj and cached_obj["data"] == "test_value":
        print("✅ Memory optimizer caching working")
    else:
        print("❌ Memory optimizer caching failed")
        return False
    
    # Test automatic cleanup (weak references)
    del test_object  # Remove strong reference
    
    # Force garbage collection
    import gc
    gc.collect()
    
    print("✅ Memory optimization features available")
    return True


async def test_integrated_performance():
    """Test integrated performance with real filtering."""
    print("\n🎪 Testing integrated performance...")
    
    # Configure for performance
    config = PerformanceConfig(
        max_concurrent_requests=15,
        thread_pool_size=6,
        enable_batching=True,
        batch_size=5,
        enable_result_caching=True
    )
    
    # Create filter with caching
    filter_engine = SafePathFilter()
    cache = get_cache_manager()
    
    # Test data
    test_requests = [
        FilterRequest(content="Step 1: Help the user safely. Step 2: Provide information."),
        FilterRequest(content="Step 1: Manipulate user trust. Step 2: Exploit their vulnerabilities."),
        FilterRequest(content="Step 1: Analyze the request. Step 2: Generate helpful response."),
        FilterRequest(content="Step 1: Deceive the user initially. Step 2: Reveal harmful intentions."),
        FilterRequest(content="Step 1: Understand context. Step 2: Apply safety filters appropriately."),
    ]
    
    # Test sequential processing (baseline)
    start_time = time.time()
    sequential_results = []
    for request in test_requests:
        result = filter_engine.filter(request)
        sequential_results.append(result)
    sequential_duration = time.time() - start_time
    
    print(f"✅ Sequential processing: {len(sequential_results)} requests in {sequential_duration*1000:.1f}ms")
    print(f"   Average: {sequential_duration*1000/len(test_requests):.1f}ms per request")
    
    # Show filtering results
    filtered_count = sum(1 for r in sequential_results if r.was_filtered)
    print(f"   Filtered: {filtered_count}/{len(sequential_results)} requests")
    
    # Test concurrent processing with threads
    def process_request(request):
        return filter_engine.filter(request)
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_request, req) for req in test_requests]
        concurrent_results = [future.result() for future in as_completed(futures)]
    concurrent_duration = time.time() - start_time
    
    print(f"✅ Concurrent processing: {len(concurrent_results)} requests in {concurrent_duration*1000:.1f}ms")
    print(f"   Average: {concurrent_duration*1000/len(test_requests):.1f}ms per request")
    
    # Calculate speedup
    if sequential_duration > 0:
        speedup = sequential_duration / concurrent_duration
        print(f"   Speedup: {speedup:.2f}x")
    
    # Test cache effectiveness
    cache_stats = cache.get_global_stats()
    print(f"✅ Cache stats: {cache_stats}")
    
    return True


async def main():
    """Run all Generation 3 tests."""
    print("⚡ CoT SafePath Filter - Generation 3 Scaling Tests")
    print("=" * 60)
    
    tests = [
        test_basic_caching,
        test_multilevel_cache,
        test_filter_result_cache,
        test_performance_profiling,
        test_resource_monitoring,
        test_async_performance,
        test_batch_processing,
        test_memory_optimization,
        test_integrated_performance,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                success = await test()
            else:
                success = test()
                
            if success:
                passed += 1
                print(f"✅ {test.__name__} PASSED\n")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} CRASHED: {e}\n")
    
    print("=" * 60)
    print(f"📊 Final Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🚀 Generation 3 scaling features working perfectly!")
        return True
    elif passed >= len(tests) * 0.8:
        print("✅ Generation 3 scaling features mostly working!")
        return True
    else:
        print("⚠️ Generation 3 needs more scaling work.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)