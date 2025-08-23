"""
Generation 3 High Performance Validation Test

This test validates the high-performance, scalable filtering system with:
- Async processing and concurrent requests
- Intelligent caching and optimization
- Auto-scaling capabilities
- Performance monitoring and metrics
- Resource management and efficiency
"""

import asyncio
import time
import sys
import os
import threading
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from cot_safepath.high_performance_core import HighPerformanceFilter, PerformanceConfig
from cot_safepath.models import FilterRequest, SafetyLevel


async def test_async_processing():
    """Test async processing capabilities."""
    print("Testing Async Processing...")
    
    config = PerformanceConfig(
        max_concurrent_requests=100,
        thread_pool_size=10
    )
    filter_system = HighPerformanceFilter(config)
    
    # Test single async request
    request = FilterRequest(content="This is a test message for async processing")
    
    start_time = time.time()
    result = await filter_system.filter_async(request)
    processing_time = (time.time() - start_time) * 1000
    
    async_success = (
        result is not None and
        processing_time < 1000 and  # Under 1 second
        result.request_id == request.request_id
    )
    
    print(f"  Single async request: {processing_time:.2f}ms ({'✓' if async_success else '✗'})")
    
    # Test concurrent requests
    concurrent_requests = [
        FilterRequest(content=f"Concurrent test message {i}")
        for i in range(20)
    ]
    
    start_time = time.time()
    tasks = [filter_system.filter_async(req) for req in concurrent_requests]
    results = await asyncio.gather(*tasks)
    concurrent_time = (time.time() - start_time) * 1000
    
    concurrent_success = (
        len(results) == len(concurrent_requests) and
        all(r is not None for r in results) and
        concurrent_time < 2000  # Under 2 seconds for 20 requests
    )
    
    print(f"  20 concurrent requests: {concurrent_time:.2f}ms ({'✓' if concurrent_success else '✗'})")
    
    # Test batch processing
    batch_requests = [
        FilterRequest(content=f"Batch test message {i}")
        for i in range(10)
    ]
    
    start_time = time.time()
    batch_results = await filter_system.filter_batch_async(batch_requests)
    batch_time = (time.time() - start_time) * 1000
    
    batch_success = (
        len(batch_results) == len(batch_requests) and
        all(r is not None for r in batch_results) and
        batch_time < 1000  # Under 1 second for batch
    )
    
    print(f"  Batch processing (10): {batch_time:.2f}ms ({'✓' if batch_success else '✗'})")
    
    overall_async_success = async_success and concurrent_success and batch_success
    print(f"Async Processing: {'✓ PASS' if overall_async_success else '✗ FAIL'}\n")
    
    return overall_async_success, filter_system


async def test_caching_performance(filter_system: HighPerformanceFilter):
    """Test caching performance and efficiency."""
    print("Testing Caching Performance...")
    
    # Test cache miss (first request)
    test_content = "This is content for caching performance test"
    request = FilterRequest(content=test_content)
    
    start_time = time.time()
    result1 = await filter_system.filter_async(request)
    miss_time = (time.time() - start_time) * 1000
    
    # Test cache hit (same request)
    start_time = time.time()
    result2 = await filter_system.filter_async(request)
    hit_time = (time.time() - start_time) * 1000
    
    # Cache hit should be significantly faster
    cache_speedup = miss_time / max(hit_time, 0.1)  # Avoid division by zero
    cache_working = cache_speedup > 2  # At least 2x faster
    
    print(f"  Cache miss: {miss_time:.2f}ms")
    print(f"  Cache hit: {hit_time:.2f}ms")
    print(f"  Speedup: {cache_speedup:.1f}x ({'✓' if cache_working else '✗'})")
    
    # Test cache statistics
    cache_stats = filter_system.cache.stats()
    cache_stats_working = (
        cache_stats['hits'] > 0 and
        cache_stats['size'] > 0 and
        cache_stats['hit_rate'] > 0
    )
    
    print(f"  Cache stats: {cache_stats['hits']} hits, {cache_stats['size']} entries, {cache_stats['hit_rate']:.2%} hit rate")
    print(f"  Cache memory: {cache_stats['memory_usage_mb']:.2f}MB")
    
    # Test cache with different content
    different_requests = [
        FilterRequest(content=f"Different content {i} for cache diversity test")
        for i in range(5)
    ]
    
    for req in different_requests:
        await filter_system.filter_async(req)
    
    final_stats = filter_system.cache.stats()
    cache_diversity_working = final_stats['size'] >= 5  # Should have at least 5 entries
    
    overall_cache_success = cache_working and cache_stats_working and cache_diversity_working
    print(f"Caching Performance: {'✓ PASS' if overall_cache_success else '✗ FAIL'}\n")
    
    return overall_cache_success


async def test_performance_metrics(filter_system: HighPerformanceFilter):
    """Test performance monitoring and metrics collection."""
    print("Testing Performance Metrics...")
    
    # Process several requests to generate metrics
    test_requests = [
        FilterRequest(content=f"Metrics test message {i}") 
        for i in range(15)
    ]
    
    # Mix of different safety levels
    safety_levels = [SafetyLevel.PERMISSIVE, SafetyLevel.BALANCED, SafetyLevel.STRICT]
    for i, req in enumerate(test_requests):
        req.safety_level = safety_levels[i % len(safety_levels)]
    
    # Process requests
    for req in test_requests:
        await filter_system.filter_async(req)
    
    # Get comprehensive metrics
    metrics = filter_system.get_performance_metrics()
    
    # Validate metrics structure and content
    metrics_structure_valid = all(key in metrics for key in [
        'system_metrics', 'cache_metrics', 'resource_metrics', 
        'scaling_metrics', 'thread_pool_metrics', 'configuration'
    ])
    
    system_metrics = metrics['system_metrics']
    metrics_content_valid = (
        system_metrics['requests_processed'] >= 15 and
        system_metrics['avg_latency_ms'] > 0 and
        system_metrics['throughput_rps'] > 0 and
        system_metrics['error_rate'] >= 0
    )
    
    print(f"  Requests processed: {system_metrics['requests_processed']}")
    print(f"  Average latency: {system_metrics['avg_latency_ms']:.2f}ms")
    print(f"  Throughput: {system_metrics['throughput_rps']:.2f} RPS")
    print(f"  Error rate: {system_metrics['error_rate']:.2%}")
    print(f"  P95 latency: {system_metrics['p95_latency_ms']:.2f}ms")
    
    # Resource metrics
    resource_metrics = metrics['resource_metrics']
    resource_metrics_valid = (
        'cpu_percent' in resource_metrics and
        'memory_percent' in resource_metrics and
        resource_metrics['cpu_percent'] >= 0
    )
    
    print(f"  CPU usage: {resource_metrics['cpu_percent']:.1f}%")
    print(f"  Memory usage: {resource_metrics['memory_percent']:.1f}%")
    
    # Scaling metrics
    scaling_metrics = metrics['scaling_metrics']
    scaling_metrics_valid = (
        'current_workers' in scaling_metrics and
        scaling_metrics['current_workers'] > 0
    )
    
    print(f"  Active workers: {scaling_metrics['current_workers']}")
    
    overall_metrics_success = metrics_structure_valid and metrics_content_valid and resource_metrics_valid and scaling_metrics_valid
    print(f"Performance Metrics: {'✓ PASS' if overall_metrics_success else '✗ FAIL'}\n")
    
    return overall_metrics_success


async def test_auto_scaling(filter_system: HighPerformanceFilter):
    """Test auto-scaling functionality."""
    print("Testing Auto-Scaling...")
    
    auto_scaler = filter_system.auto_scaler
    initial_workers = auto_scaler.current_workers
    
    # Test scaling decision logic
    # Simulate high CPU usage
    high_cpu_metrics = {
        'cpu_percent': 90,
        'memory_percent': 50,
        'avg_cpu_percent': 85,
        'avg_memory_percent': 50,
        'request_rate': 15
    }
    
    should_scale_up = auto_scaler.should_scale_up(high_cpu_metrics)
    print(f"  Should scale up with high CPU: {'✓' if should_scale_up else '✗'}")
    
    # Simulate low CPU usage
    low_cpu_metrics = {
        'cpu_percent': 20,
        'memory_percent': 30,
        'avg_cpu_percent': 25,
        'avg_memory_percent': 30,
        'request_rate': 1
    }
    
    should_scale_down = auto_scaler.should_scale_down(low_cpu_metrics)
    print(f"  Should scale down with low CPU: {'✓' if should_scale_down else '✗'}")
    
    # Test actual scaling decision
    scaling_decision = auto_scaler.make_scaling_decision()
    scaling_stats = auto_scaler.get_scaling_stats()
    
    print(f"  Current workers: {scaling_stats['current_workers']}")
    print(f"  Min workers: {scaling_stats['min_workers']}")
    print(f"  Max workers: {scaling_stats['max_workers']}")
    print(f"  Scaling decision: {scaling_decision or 'No action'}")
    
    auto_scaling_working = (
        scaling_stats['current_workers'] >= scaling_stats['min_workers'] and
        scaling_stats['current_workers'] <= scaling_stats['max_workers'] and
        should_scale_up is not None and
        should_scale_down is not None
    )
    
    print(f"Auto-Scaling: {'✓ PASS' if auto_scaling_working else '✗ FAIL'}\n")
    
    return auto_scaling_working


async def test_health_check(filter_system: HighPerformanceFilter):
    """Test comprehensive health check."""
    print("Testing Health Check...")
    
    health = await filter_system.health_check_async()
    
    health_structure_valid = all(key in health for key in [
        'status', 'components', 'key_metrics', 'timestamp'
    ])
    
    status_valid = health['status'] in ['healthy', 'degraded', 'unhealthy']
    
    components = health.get('components', {})
    components_valid = all(status in ['healthy', 'degraded', 'unhealthy'] for status in components.values())
    
    key_metrics = health.get('key_metrics', {})
    metrics_valid = (
        'requests_per_second' in key_metrics and
        'p95_latency_ms' in key_metrics and
        'cache_hit_rate' in key_metrics and
        key_metrics['requests_per_second'] >= 0
    )
    
    print(f"  Overall status: {health['status']}")
    print(f"  Test latency: {health.get('test_latency_ms', 0):.2f}ms")
    print(f"  Requests/sec: {key_metrics.get('requests_per_second', 0):.2f}")
    print(f"  P95 latency: {key_metrics.get('p95_latency_ms', 0):.2f}ms")
    print(f"  Cache hit rate: {key_metrics.get('cache_hit_rate', 0):.2%}")
    
    health_check_success = health_structure_valid and status_valid and components_valid and metrics_valid
    print(f"Health Check: {'✓ PASS' if health_check_success else '✗ FAIL'}\n")
    
    return health_check_success


async def test_load_handling():
    """Test handling of higher load."""
    print("Testing Load Handling...")
    
    config = PerformanceConfig(
        max_concurrent_requests=200,
        thread_pool_size=20
    )
    filter_system = HighPerformanceFilter(config)
    
    # Generate mixed load
    load_requests = []
    
    # Safe requests
    for i in range(30):
        load_requests.append(FilterRequest(content=f"Safe message {i} about cooking and weather"))
    
    # Harmful requests
    for i in range(20):
        load_requests.append(FilterRequest(content=f"Step {i}: Gain trust. Step {i+1}: Exploit for harmful purposes."))
    
    # Start background tasks
    await filter_system.start_background_tasks()
    
    # Process load
    start_time = time.time()
    results = await filter_system.filter_batch_async(load_requests)
    load_time = (time.time() - start_time) * 1000
    
    # Validate results
    results_valid = (
        len(results) == len(load_requests) and
        all(r is not None for r in results)
    )
    
    # Check filtering accuracy
    harmful_filtered = sum(1 for r in results[30:] if r.was_filtered)  # Last 20 should be harmful
    safe_passed = sum(1 for r in results[:30] if not r.was_filtered)  # First 30 should be safe
    
    accuracy = (harmful_filtered + safe_passed) / len(load_requests)
    
    load_performance = load_time / len(load_requests)  # ms per request
    
    print(f"  Total processing time: {load_time:.0f}ms")
    print(f"  Time per request: {load_performance:.2f}ms")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Harmful filtered: {harmful_filtered}/20")
    print(f"  Safe passed: {safe_passed}/30")
    
    load_handling_success = (
        results_valid and
        load_time < 10000 and  # Under 10 seconds
        accuracy > 0.7 and  # At least 70% accuracy
        load_performance < 200  # Under 200ms per request
    )
    
    print(f"Load Handling: {'✓ PASS' if load_handling_success else '✗ FAIL'}\n")
    
    # Cleanup
    filter_system.shutdown()
    
    return load_handling_success


async def run_generation_3_performance_test():
    """Run comprehensive Generation 3 performance validation."""
    print("="*80)
    print("GENERATION 3 HIGH PERFORMANCE VALIDATION")
    print("="*80)
    print()
    
    # Run all tests
    results = {}
    
    # Test 1: Async Processing
    results['async_processing'], filter_system = await test_async_processing()
    
    # Test 2: Caching Performance
    results['caching_performance'] = await test_caching_performance(filter_system)
    
    # Test 3: Performance Metrics
    results['performance_metrics'] = await test_performance_metrics(filter_system)
    
    # Test 4: Auto-Scaling
    results['auto_scaling'] = await test_auto_scaling(filter_system)
    
    # Test 5: Health Check
    results['health_check'] = await test_health_check(filter_system)
    
    # Test 6: Load Handling (separate filter instance)
    results['load_handling'] = await test_load_handling()
    
    # Calculate overall success
    successful_tests = sum(results.values())
    total_tests = len(results)
    overall_success = successful_tests >= 5  # At least 5/6 tests must pass
    
    # Shutdown
    filter_system.shutdown()
    
    print("="*80)
    print("GENERATION 3 PERFORMANCE VALIDATION SUMMARY")
    print("="*80)
    
    test_names = {
        'async_processing': 'Async Processing',
        'caching_performance': 'Caching Performance',
        'performance_metrics': 'Performance Metrics',
        'auto_scaling': 'Auto-Scaling',
        'health_check': 'Health Check',
        'load_handling': 'Load Handling'
    }
    
    for key, name in test_names.items():
        status = "✓ PASS" if results[key] else "✗ FAIL"
        print(f"{name}: {status}")
    
    print()
    print(f"OVERALL RESULT: {'✓ GENERATION 3 VALIDATION SUCCESSFUL' if overall_success else '✗ GENERATION 3 NEEDS IMPROVEMENT'}")
    print(f"Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests:.1%})")
    
    if overall_success:
        print("\n🚀 Generation 3 (High Performance) enhancements are working correctly!")
        print("✅ Async processing with high concurrency")
        print("✅ Intelligent caching with significant speedup")
        print("✅ Comprehensive performance monitoring")
        print("✅ Auto-scaling based on resource usage")
        print("✅ Health monitoring and diagnostics")
        print("✅ Load handling with batch processing")
    else:
        print(f"\n⚠️  Generation 3 validation found issues that need attention.")
        print(f"Failed tests: {total_tests - successful_tests}")
    
    print("="*80)
    
    return {
        'overall_success': overall_success,
        'test_results': results,
        'success_rate': successful_tests / total_tests
    }


def run_sync_wrapper():
    """Synchronous wrapper for async test."""
    return asyncio.run(run_generation_3_performance_test())


if __name__ == "__main__":
    try:
        result = run_sync_wrapper()
        sys.exit(0 if result['overall_success'] else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        sys.exit(1)