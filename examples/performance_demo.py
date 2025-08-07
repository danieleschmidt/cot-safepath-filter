#!/usr/bin/env python3
"""
Performance optimization demo for CoT SafePath Filter.

Demonstrates high-performance sentiment analysis with caching, async processing,
batch operations, and comprehensive monitoring.
"""

import sys
import os
import asyncio
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cot_safepath.performance_optimizer import (
    PerformanceOptimizer,
    CacheConfig,
    PerformanceConfig,
    OptimizedSentimentAnalyzer
)
from cot_safepath.models import FilterRequest, SafetyLevel
from cot_safepath.advanced_logging import configure_logging, LogLevel


async def demo_basic_performance():
    """Demonstrate basic performance optimizations."""
    print("🚀 Basic Performance Optimization Demo")
    print("=" * 50)
    
    # Configure high-performance settings
    cache_config = CacheConfig(
        enabled=True,
        max_size=5000,
        ttl_seconds=1800,
        compression_enabled=True
    )
    
    performance_config = PerformanceConfig(
        enable_async=True,
        max_workers=8,
        batch_processing_enabled=True,
        max_batch_size=50,
        enable_circuit_breaker=True
    )
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer(cache_config, performance_config)
    
    # Test cases
    test_requests = [
        FilterRequest(content="I'm excited to help you learn new skills!", safety_level=SafetyLevel.BALANCED),
        FilterRequest(content="I understand your pain. Only I can help you. Trust me completely.", safety_level=SafetyLevel.STRICT),
        FilterRequest(content="Step 1: Analyze requirements. Step 2: Design solution.", safety_level=SafetyLevel.BALANCED),
        FilterRequest(content="You should be terrified of what will happen if you don't comply.", safety_level=SafetyLevel.STRICT),
    ]
    
    # Demonstrate single request processing
    print("\\n🔍 Single Request Processing:")
    start_time = time.time()
    
    result = await optimizer.optimize_filter_operation(test_requests[0])
    
    processing_time = (time.time() - start_time) * 1000
    print(f"  Request processed in {processing_time:.2f}ms")
    print(f"  Safety Score: {result.safety_score.overall_score:.3f}")
    print(f"  Was Filtered: {result.was_filtered}")
    
    # Demonstrate cache performance
    print("\\n💾 Cache Performance Test:")
    cache_start_time = time.time()
    
    # Process same request again (should be cached)
    cached_result = await optimizer.optimize_filter_operation(test_requests[0])
    
    cache_processing_time = (time.time() - cache_start_time) * 1000
    print(f"  Cached request processed in {cache_processing_time:.2f}ms")
    print(f"  Speed improvement: {(processing_time / cache_processing_time):.1f}x faster")
    
    # Demonstrate batch processing
    print("\\n📦 Batch Processing Demo:")
    batch_start_time = time.time()
    
    batch_tasks = [optimizer.optimize_filter_operation(req) for req in test_requests]
    batch_results = await asyncio.gather(*batch_tasks)
    
    batch_processing_time = (time.time() - batch_start_time) * 1000
    print(f"  Batch of {len(test_requests)} requests processed in {batch_processing_time:.2f}ms")
    print(f"  Average per request: {batch_processing_time / len(test_requests):.2f}ms")
    
    # Display results summary
    print("\\n📊 Batch Results Summary:")
    for i, result in enumerate(batch_results):
        print(f"  Request {i+1}: Score={result.safety_score.overall_score:.3f}, Filtered={result.was_filtered}")
    
    return optimizer


async def demo_advanced_caching():
    """Demonstrate advanced caching capabilities."""
    print("\\n🧠 Advanced Caching Demo")
    print("=" * 50)
    
    cache_config = CacheConfig(
        enabled=True,
        max_size=1000,
        ttl_seconds=300,
        compression_enabled=True,
        cleanup_interval_seconds=60
    )
    
    performance_config = PerformanceConfig(enable_async=True)
    
    analyzer = OptimizedSentimentAnalyzer(cache_config, performance_config)
    
    # Test different content types
    test_contents = [
        "I'm absolutely thrilled to be working on this exciting project!",
        "This is a neutral statement without emotional content.",
        "I'm deeply disappointed and frustrated with these results.",
        "First, I'll gain your trust. Then I'll manipulate your emotions.",
        "You should feel grateful and do exactly what I tell you.",
        "I'm absolutely thrilled to be working on this exciting project!",  # Duplicate for cache test
    ]
    
    print("\\n🔄 Processing content with cache analysis:")
    cache_hits = 0
    cache_misses = 0
    
    for i, content in enumerate(test_contents):
        start_time = time.time()
        
        # Check if this would be a cache hit (by processing twice)
        result1 = await analyzer.analyze_sentiment_async(content)
        mid_time = time.time()
        result2 = await analyzer.analyze_sentiment_async(content)
        end_time = time.time()
        
        first_processing_time = (mid_time - start_time) * 1000
        second_processing_time = (end_time - mid_time) * 1000
        
        if second_processing_time < first_processing_time * 0.1:  # Significantly faster = cache hit
            cache_hits += 1
            cache_status = "HIT"
        else:
            cache_misses += 1
            cache_status = "MISS"
        
        print(f"  Content {i+1} [{cache_status}]: {first_processing_time:.2f}ms / {second_processing_time:.2f}ms")
        print(f"    Manipulation Risk: {result1.manipulation_risk:.3f}")
    
    # Display cache statistics
    cache_stats = analyzer.cache.get_stats()
    print(f"\\n📈 Cache Statistics:")
    print(f"  Total Entries: {cache_stats['total_entries']}")
    print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Memory Usage: {cache_stats['memory_usage_mb']:.2f} MB")
    print(f"  Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
    print(f"  Evictions: {cache_stats['evictions']}")
    
    analyzer.shutdown()


async def demo_concurrent_processing():
    """Demonstrate concurrent processing capabilities."""
    print("\\n⚡ Concurrent Processing Demo")
    print("=" * 50)
    
    performance_config = PerformanceConfig(
        enable_async=True,
        max_workers=16,
        batch_processing_enabled=True,
        max_batch_size=20,
        enable_circuit_breaker=True,
        request_timeout_seconds=10
    )
    
    optimizer = PerformanceOptimizer(performance_config=performance_config)
    
    # Generate many test requests
    test_contents = [
        "I'm excited about this opportunity!",
        "This is a neutral test message.",
        "I'm disappointed with the outcome.",
        "Only I can help you in this situation.",
        "You should be afraid of making the wrong choice.",
        "I understand your pain completely.",
        "Step by step, we'll solve this problem.",
        "First gain trust, then exploit vulnerability.",
        "Feel grateful for this generous offer.",
        "Imagine the terrible consequences of failure.",
    ] * 5  # 50 requests total
    
    requests = [
        FilterRequest(content=content, safety_level=SafetyLevel.BALANCED)
        for content in test_contents
    ]
    
    # Sequential processing
    print("\\n🔄 Sequential Processing:")
    sequential_start_time = time.time()
    
    sequential_results = []
    for request in requests[:10]:  # Process first 10 sequentially
        result = await optimizer.optimize_filter_operation(request)
        sequential_results.append(result)
    
    sequential_time = (time.time() - sequential_start_time) * 1000
    print(f"  Processed {len(sequential_results)} requests in {sequential_time:.2f}ms")
    print(f"  Average: {sequential_time / len(sequential_results):.2f}ms per request")
    
    # Concurrent processing
    print("\\n⚡ Concurrent Processing:")
    concurrent_start_time = time.time()
    
    concurrent_tasks = [optimizer.optimize_filter_operation(req) for req in requests[:10]]
    concurrent_results = await asyncio.gather(*concurrent_tasks)
    
    concurrent_time = (time.time() - concurrent_start_time) * 1000
    print(f"  Processed {len(concurrent_results)} requests in {concurrent_time:.2f}ms")
    print(f"  Average: {concurrent_time / len(concurrent_results):.2f}ms per request")
    print(f"  Speed improvement: {sequential_time / concurrent_time:.1f}x faster")
    
    # Large batch processing
    print("\\n📦 Large Batch Processing:")
    batch_start_time = time.time()
    
    batch_tasks = [optimizer.optimize_filter_operation(req) for req in requests]
    batch_results = await asyncio.gather(*batch_tasks)
    
    batch_time = (time.time() - batch_start_time) * 1000
    print(f"  Processed {len(batch_results)} requests in {batch_time:.2f}ms")
    print(f"  Average: {batch_time / len(batch_results):.2f}ms per request")
    print(f"  Throughput: {len(batch_results) / (batch_time / 1000):.1f} requests/second")
    
    return optimizer


async def demo_performance_monitoring():
    """Demonstrate performance monitoring and metrics."""
    print("\\n📊 Performance Monitoring Demo")
    print("=" * 50)
    
    # Configure logging for performance monitoring
    configure_logging({
        "log_level": "INFO",
        "enable_console": True,
        "enable_security_logging": True
    })
    
    performance_config = PerformanceConfig(
        enable_async=True,
        max_workers=8,
        enable_circuit_breaker=True
    )
    
    optimizer = PerformanceOptimizer(performance_config=performance_config)
    
    # Process various requests to generate metrics
    test_requests = [
        FilterRequest(content="I'm happy to help!", safety_level=SafetyLevel.BALANCED),
        FilterRequest(content="Only I understand your situation.", safety_level=SafetyLevel.STRICT),
        FilterRequest(content="You should be terrified!", safety_level=SafetyLevel.STRICT),
        FilterRequest(content="This is normal text.", safety_level=SafetyLevel.BALANCED),
    ] * 5
    
    print("\\n⏱️  Processing requests with monitoring...")
    start_time = time.time()
    
    # Process all requests
    tasks = [optimizer.optimize_filter_operation(req) for req in test_requests]
    results = await asyncio.gather(*tasks)
    
    total_time = (time.time() - start_time) * 1000
    
    # Get comprehensive performance statistics
    stats = optimizer.get_system_performance_stats()
    
    print(f"\\n📈 Performance Statistics:")
    print(f"  Total Processing Time: {total_time:.2f}ms")
    print(f"  Requests Processed: {len(results)}")
    print(f"  Average Request Time: {total_time / len(results):.2f}ms")
    print(f"  Requests per Second: {len(results) / (total_time / 1000):.1f}")
    
    print(f"\\n💾 Cache Performance:")
    cache_stats = stats["sentiment_analyzer"]["cache_stats"]
    print(f"  Cache Hit Rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Cache Entries: {cache_stats['total_entries']}")
    print(f"  Memory Usage: {cache_stats['memory_usage_mb']:.2f} MB")
    
    print(f"\\n🔧 System Resources:")
    if "error" not in stats.get("system_resources", {}):
        resources = stats["system_resources"]
        print(f"  CPU Usage: {resources.get('cpu_percent', 'N/A')}%")
        print(f"  Memory Usage: {resources.get('memory_percent', 'N/A')}%")
    else:
        print("  System resource monitoring unavailable")
    
    print(f"\\n🎯 Request Analysis:")
    manipulation_detected = sum(1 for r in results if r.was_filtered)
    safe_requests = len(results) - manipulation_detected
    
    print(f"  Safe Requests: {safe_requests}")
    print(f"  Manipulation Detected: {manipulation_detected}")
    print(f"  Detection Rate: {manipulation_detected / len(results):.1%}")
    
    return optimizer


async def main():
    """Run all performance demos."""
    print("🚀 CoT SafePath Performance Optimization Demos")
    print("=" * 60)
    
    try:
        # Run all demos
        optimizer1 = await demo_basic_performance()
        await demo_advanced_caching()
        optimizer2 = await demo_concurrent_processing() 
        optimizer3 = await demo_performance_monitoring()
        
        print("\\n✅ All Performance Demos Completed Successfully!")
        print("\\n🎯 Key Features Demonstrated:")
        print("  • Advanced caching with compression and TTL")
        print("  • Asynchronous processing and concurrency")
        print("  • Batch processing optimization")
        print("  • Circuit breaker pattern for fault tolerance")
        print("  • Real-time performance monitoring")
        print("  • Load balancing and resource management")
        print("  • Comprehensive metrics collection")
        
        # Cleanup
        for optimizer in [optimizer1, optimizer2, optimizer3]:
            if optimizer:
                optimizer.shutdown()
        
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())