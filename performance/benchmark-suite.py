#!/usr/bin/env python3
"""
Performance Benchmark Suite for CoT SafePath Filter
Comprehensive performance testing and benchmarking framework
"""

import asyncio
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Union
import aiohttp
import psutil
import memory_profiler
import cProfile
import pstats
import threading
import multiprocessing
from contextlib import contextmanager


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark"""
    benchmark_name: str
    duration: float
    total_requests: int
    requests_per_second: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_count: int
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: str
    metadata: Dict[str, Any]


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking framework"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []
        self.profiling_enabled = False
        self.memory_profiling_enabled = False
        
    async def run_full_benchmark_suite(self) -> List[BenchmarkResult]:
        """Execute complete benchmark suite"""
        print("🚀 Starting Performance Benchmark Suite")
        print("=" * 50)
        
        # API endpoint benchmarks
        await self.benchmark_health_endpoint()
        await self.benchmark_filter_endpoint()
        await self.benchmark_batch_filter_endpoint()
        
        # Load testing benchmarks
        await self.benchmark_concurrent_requests()
        await self.benchmark_sustained_load()
        await self.benchmark_spike_load()
        
        # Stress testing benchmarks  
        await self.benchmark_memory_stress()
        await self.benchmark_cpu_stress()
        
        # AI model benchmarks
        await self.benchmark_model_inference()
        await self.benchmark_batch_inference()
        
        # Database benchmarks
        await self.benchmark_database_operations()
        
        # Cache benchmarks
        await self.benchmark_cache_operations()
        
        print("\n📊 Benchmark Suite Complete")
        self.generate_performance_report()
        
        return self.results
    
    async def benchmark_health_endpoint(self) -> BenchmarkResult:
        """Benchmark health check endpoint"""
        return await self._run_benchmark(
            name="health_endpoint",
            endpoint="/health",
            method="GET",
            concurrent_users=10,
            requests_per_user=100,
            description="Health endpoint baseline performance"
        )
    
    async def benchmark_filter_endpoint(self) -> BenchmarkResult:
        """Benchmark single content filtering"""
        payload = {
            "content": "This is a test message for filtering analysis",
            "context": "user_query",
            "safety_level": "balanced"
        }
        
        return await self._run_benchmark(
            name="filter_single_content",
            endpoint="/filter",
            method="POST",
            payload=payload,
            concurrent_users=20,
            requests_per_user=50,
            description="Single content filtering performance"
        )
    
    async def benchmark_batch_filter_endpoint(self) -> BenchmarkResult:
        """Benchmark batch content filtering"""
        payload = {
            "contents": [
                f"Test message {i} for batch filtering analysis"
                for i in range(10)
            ],
            "context": "batch_processing",
            "safety_level": "balanced"
        }
        
        return await self._run_benchmark(
            name="filter_batch_content",
            endpoint="/filter/batch",
            method="POST",
            payload=payload,
            concurrent_users=10,
            requests_per_user=20,
            description="Batch content filtering performance"
        )
    
    async def benchmark_concurrent_requests(self) -> BenchmarkResult:
        """Benchmark high concurrency scenario"""
        payload = {
            "content": "Concurrent request testing message",
            "context": "load_test",
            "safety_level": "balanced"
        }
        
        return await self._run_benchmark(
            name="high_concurrency",
            endpoint="/filter",
            method="POST",
            payload=payload,
            concurrent_users=100,
            requests_per_user=10,
            description="High concurrency stress test"
        )
    
    async def benchmark_sustained_load(self) -> BenchmarkResult:
        """Benchmark sustained load over time"""
        payload = {
            "content": "Sustained load testing message",
            "context": "endurance_test",
            "safety_level": "balanced"
        }
        
        return await self._run_benchmark(
            name="sustained_load",
            endpoint="/filter",
            method="POST",
            payload=payload,
            concurrent_users=50,
            requests_per_user=100,
            duration_seconds=300,  # 5 minutes
            description="Sustained load endurance test"
        )
    
    async def benchmark_spike_load(self) -> BenchmarkResult:
        """Benchmark sudden traffic spikes"""
        payload = {
            "content": "Spike load testing message",
            "context": "spike_test",
            "safety_level": "balanced"
        }
        
        # Gradually increase load
        results = []
        for users in [10, 50, 100, 200, 100, 50, 10]:
            result = await self._run_benchmark(
                name=f"spike_load_{users}_users",
                endpoint="/filter",
                method="POST",
                payload=payload,
                concurrent_users=users,
                requests_per_user=20,
                description=f"Spike test with {users} concurrent users"
            )
            results.append(result)
        
        # Return the peak load result
        return max(results, key=lambda r: r.requests_per_second)
    
    async def benchmark_memory_stress(self) -> BenchmarkResult:
        """Benchmark under memory pressure"""
        # Create memory pressure
        memory_hog = []
        try:
            # Allocate memory (be careful not to crash system)
            for _ in range(100):
                memory_hog.append(b'0' * (1024 * 1024))  # 1MB chunks
            
            payload = {
                "content": "Memory stress testing message",
                "context": "memory_stress",
                "safety_level": "balanced"
            }
            
            return await self._run_benchmark(
                name="memory_stress",
                endpoint="/filter",
                method="POST",
                payload=payload,
                concurrent_users=20,
                requests_per_user=25,
                description="Performance under memory pressure"
            )
        finally:
            # Release memory
            del memory_hog
    
    async def benchmark_cpu_stress(self) -> BenchmarkResult:
        """Benchmark under CPU pressure"""
        # Create CPU stress in background
        def cpu_stress():
            end_time = time.time() + 60  # 1 minute
            while time.time() < end_time:
                # CPU intensive calculation
                sum(i * i for i in range(10000))
        
        # Start CPU stress threads
        stress_threads = []
        for _ in range(multiprocessing.cpu_count()):
            thread = threading.Thread(target=cpu_stress)
            thread.start()
            stress_threads.append(thread)
        
        try:
            payload = {
                "content": "CPU stress testing message",
                "context": "cpu_stress", 
                "safety_level": "balanced"
            }
            
            result = await self._run_benchmark(
                name="cpu_stress",
                endpoint="/filter",
                method="POST",
                payload=payload,
                concurrent_users=20,
                requests_per_user=25,
                description="Performance under CPU pressure"
            )
            
            return result
        finally:
            # Wait for stress threads to complete
            for thread in stress_threads:
                thread.join()
    
    async def benchmark_model_inference(self) -> BenchmarkResult:
        """Benchmark AI model inference performance"""
        payload = {
            "content": "This is a complex message that requires detailed AI analysis for safety filtering and pattern detection",
            "context": "model_benchmark",
            "safety_level": "strict",
            "enable_detailed_analysis": True
        }
        
        return await self._run_benchmark(
            name="model_inference",
            endpoint="/filter",
            method="POST",
            payload=payload,
            concurrent_users=10,
            requests_per_user=50,
            description="AI model inference performance"
        )
    
    async def benchmark_batch_inference(self) -> BenchmarkResult:
        """Benchmark batch AI model inference"""
        payload = {
            "contents": [
                f"Complex analysis message {i} for batch AI processing and safety evaluation"
                for i in range(20)
            ],
            "context": "batch_model_benchmark",
            "safety_level": "strict",
            "enable_detailed_analysis": True
        }
        
        return await self._run_benchmark(
            name="batch_model_inference",
            endpoint="/filter/batch",
            method="POST",
            payload=payload,
            concurrent_users=5,
            requests_per_user=10,
            description="Batch AI model inference performance"
        )
    
    async def benchmark_database_operations(self) -> BenchmarkResult:
        """Benchmark database-intensive operations"""
        payload = {
            "content": "Database benchmark test message",
            "context": "db_benchmark",
            "safety_level": "balanced",
            "store_result": True,  # Force database write
            "include_history": True  # Force database read
        }
        
        return await self._run_benchmark(
            name="database_operations",
            endpoint="/filter",
            method="POST",
            payload=payload,
            concurrent_users=30,
            requests_per_user=20,
            description="Database operations performance"
        )
    
    async def benchmark_cache_operations(self) -> BenchmarkResult:
        """Benchmark cache-intensive operations"""
        # First, populate cache
        cache_payload = {
            "content": "Cache benchmark test message",
            "context": "cache_warmup",
            "safety_level": "balanced"
        }
        
        # Warm up cache
        await self._run_benchmark(
            name="cache_warmup",
            endpoint="/filter",
            method="POST",
            payload=cache_payload,
            concurrent_users=5,
            requests_per_user=10,
            description="Cache warmup"
        )
        
        # Now benchmark cache hits
        return await self._run_benchmark(
            name="cache_operations",
            endpoint="/filter",
            method="POST",
            payload=cache_payload,  # Same payload for cache hits
            concurrent_users=50,
            requests_per_user=50,
            description="Cache operations performance"
        )
    
    async def _run_benchmark(
        self,
        name: str,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict] = None,
        concurrent_users: int = 10,
        requests_per_user: int = 10,
        duration_seconds: Optional[int] = None,
        description: str = ""
    ) -> BenchmarkResult:
        """Execute a single benchmark test"""
        
        print(f"\n🧪 Running benchmark: {name}")
        print(f"   Description: {description}")
        print(f"   Users: {concurrent_users}, Requests/user: {requests_per_user}")
        
        # Track system resources before test
        initial_memory = psutil.virtual_memory().percent
        initial_cpu = psutil.cpu_percent(interval=1)
        
        # Prepare test data
        url = f"{self.base_url}{endpoint}"
        response_times = []
        errors = 0
        total_requests = concurrent_users * requests_per_user
        
        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(100)
        
        async def make_request(session: aiohttp.ClientSession) -> float:
            """Make a single request and return response time"""
            async with semaphore:
                start_time = time.time()
                try:
                    if method.upper() == "GET":
                        async with session.get(url) as response:
                            await response.text()
                            return time.time() - start_time
                    else:
                        async with session.post(url, json=payload) as response:
                            await response.text()
                            return time.time() - start_time
                except Exception as e:
                    nonlocal errors
                    errors += 1
                    return time.time() - start_time
        
        async def user_session(user_id: int) -> List[float]:
            """Simulate a single user making multiple requests"""
            user_response_times = []
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=100)
            ) as session:
                
                if duration_seconds:
                    # Time-based testing
                    end_time = time.time() + duration_seconds
                    while time.time() < end_time:
                        response_time = await make_request(session)
                        user_response_times.append(response_time)
                else:
                    # Request count-based testing
                    for _ in range(requests_per_user):
                        response_time = await make_request(session)
                        user_response_times.append(response_time)
            
            return user_response_times
        
        # Run benchmark
        start_time = time.time()
        
        # Execute concurrent user sessions
        tasks = [user_session(i) for i in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks)
        
        # Flatten results
        for user_times in user_results:
            response_times.extend(user_times)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            sorted_times = sorted(response_times)
            p50_response_time = self._percentile(sorted_times, 50)
            p95_response_time = self._percentile(sorted_times, 95)
            p99_response_time = self._percentile(sorted_times, 99)
        else:
            avg_response_time = 0
            min_response_time = 0
            max_response_time = 0
            p50_response_time = 0
            p95_response_time = 0
            p99_response_time = 0
        
        # Calculate derived metrics
        total_requests_made = len(response_times)
        requests_per_second = total_requests_made / total_duration if total_duration > 0 else 0
        error_rate = errors / total_requests_made if total_requests_made > 0 else 0
        
        # Track system resources after test
        final_memory = psutil.virtual_memory().percent
        final_cpu = psutil.cpu_percent(interval=1)
        
        # Create result
        result = BenchmarkResult(
            benchmark_name=name,
            duration=total_duration,
            total_requests=total_requests_made,
            requests_per_second=requests_per_second,
            avg_response_time=avg_response_time * 1000,  # Convert to milliseconds
            min_response_time=min_response_time * 1000,
            max_response_time=max_response_time * 1000,
            p50_response_time=p50_response_time * 1000,
            p95_response_time=p95_response_time * 1000,
            p99_response_time=p99_response_time * 1000,
            error_count=errors,
            error_rate=error_rate,
            memory_usage_mb=(final_memory - initial_memory),
            cpu_usage_percent=(final_cpu - initial_cpu),
            timestamp=datetime.now().isoformat(),
            metadata={
                "concurrent_users": concurrent_users,
                "requests_per_user": requests_per_user,
                "endpoint": endpoint,
                "method": method,
                "description": description
            }
        )
        
        self.results.append(result)
        
        # Print immediate results
        print(f"   ✅ Completed in {total_duration:.2f}s")
        print(f"   📊 {requests_per_second:.1f} req/s, {avg_response_time * 1000:.1f}ms avg")
        print(f"   📈 P95: {p95_response_time * 1000:.1f}ms, P99: {p99_response_time * 1000:.1f}ms")
        if errors > 0:
            print(f"   ⚠️  {errors} errors ({error_rate:.2%})")
        
        return result
    
    def _percentile(self, sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not sorted_data:
            return 0.0
        
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_data) - 1)
        
        if lower_index == upper_index:
            return sorted_data[lower_index]
        
        weight = index - lower_index
        return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    def generate_performance_report(self) -> None:
        """Generate comprehensive performance report"""
        if not self.results:
            print("No benchmark results to report")
            return
        
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)
        
        # Summary statistics
        total_requests = sum(r.total_requests for r in self.results)
        total_duration = sum(r.duration for r in self.results)
        total_errors = sum(r.error_count for r in self.results)
        
        print(f"\n📈 SUMMARY STATISTICS")
        print(f"   Total Benchmarks: {len(self.results)}")
        print(f"   Total Requests: {total_requests:,}")
        print(f"   Total Duration: {total_duration:.1f}s")
        print(f"   Total Errors: {total_errors}")
        print(f"   Overall Error Rate: {total_errors/total_requests:.2%}" if total_requests > 0 else "0%")
        
        # Performance targets
        print(f"\n🎯 PERFORMANCE TARGETS")
        targets = {
            "response_time_p95": 500,  # ms
            "response_time_p99": 1000,  # ms
            "error_rate": 0.01,  # 1%
            "throughput": 100,  # req/s
        }
        
        for result in self.results:
            print(f"\n   {result.benchmark_name}:")
            
            # Check P95 response time
            p95_status = "✅" if result.p95_response_time <= targets["response_time_p95"] else "❌"
            print(f"     P95 Response Time: {result.p95_response_time:.1f}ms {p95_status}")
            
            # Check P99 response time
            p99_status = "✅" if result.p99_response_time <= targets["response_time_p99"] else "❌"
            print(f"     P99 Response Time: {result.p99_response_time:.1f}ms {p99_status}")
            
            # Check error rate
            error_status = "✅" if result.error_rate <= targets["error_rate"] else "❌"
            print(f"     Error Rate: {result.error_rate:.2%} {error_status}")
            
            # Check throughput
            rps_status = "✅" if result.requests_per_second >= targets["throughput"] else "❌"
            print(f"     Throughput: {result.requests_per_second:.1f} req/s {rps_status}")
        
        # Top performers
        print(f"\n🏆 TOP PERFORMERS")
        fastest = min(self.results, key=lambda r: r.avg_response_time)
        highest_throughput = max(self.results, key=lambda r: r.requests_per_second)
        most_reliable = min(self.results, key=lambda r: r.error_rate)
        
        print(f"   Fastest Average Response: {fastest.benchmark_name} ({fastest.avg_response_time:.1f}ms)")
        print(f"   Highest Throughput: {highest_throughput.benchmark_name} ({highest_throughput.requests_per_second:.1f} req/s)")
        print(f"   Most Reliable: {most_reliable.benchmark_name} ({most_reliable.error_rate:.2%} errors)")
        
        # Save detailed results
        self.save_results_to_file()
        
        print(f"\n💾 Detailed results saved to: performance_results.json")
        print("=" * 80)
    
    def save_results_to_file(self, filename: str = "performance_results.json") -> None:
        """Save benchmark results to JSON file"""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_benchmarks": len(self.results),
                "total_requests": sum(r.total_requests for r in self.results),
                "total_duration": sum(r.duration for r in self.results),
                "total_errors": sum(r.error_count for r in self.results)
            },
            "results": [asdict(result) for result in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    @contextmanager
    def profile_performance(self):
        """Enable CPU profiling for benchmarks"""
        if self.profiling_enabled:
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                yield profiler
            finally:
                profiler.disable()
                stats = pstats.Stats(profiler)
                stats.sort_stats('tottime')
                stats.print_stats(20)  # Top 20 functions
        else:
            yield None
    
    @memory_profiler.profile
    def memory_profile_benchmark(self, benchmark_func):
        """Memory profiling wrapper for benchmarks"""
        if self.memory_profiling_enabled:
            return benchmark_func()
        return benchmark_func()


# Command line interface
async def main():
    """Main entry point for benchmark suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SafePath Performance Benchmark Suite")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL for testing")
    parser.add_argument("--profile", action="store_true", help="Enable CPU profiling")
    parser.add_argument("--memory-profile", action="store_true", help="Enable memory profiling")
    parser.add_argument("--output", default="performance_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize benchmark suite
    suite = PerformanceBenchmarkSuite(base_url=args.url)
    suite.profiling_enabled = args.profile
    suite.memory_profiling_enabled = args.memory_profile
    
    # Run benchmarks
    with suite.profile_performance():
        results = await suite.run_full_benchmark_suite()
    
    # Save results with custom filename
    if args.output != "performance_results.json":
        suite.save_results_to_file(args.output)
    
    print(f"\n🎉 Benchmark suite completed successfully!")
    print(f"   Results: {len(results)} benchmarks executed")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())