"""
Simple command-line interface for CoT SafePath Filter (without external dependencies).
"""

import sys
import json
import argparse
from typing import Optional

from .core import SafePathFilter
from .models import FilterRequest, SafetyLevel, FilterConfig


def filter_text(text: str, safety_level: str = "balanced", show_original: bool = False, verbose: bool = False) -> None:
    """Filter text content for harmful patterns."""
    try:
        # Parse safety level
        level_map = {
            "permissive": SafetyLevel.PERMISSIVE,
            "balanced": SafetyLevel.BALANCED,
            "strict": SafetyLevel.STRICT,
            "maximum": SafetyLevel.MAXIMUM,
        }
        safety_level_enum = level_map.get(safety_level.lower(), SafetyLevel.BALANCED)
        
        # Create filter configuration
        config = FilterConfig(safety_level=safety_level_enum, log_filtered=verbose)
        
        # Initialize filter
        filter_instance = SafePathFilter(config)
        
        # Create request
        request = FilterRequest(content=text, safety_level=safety_level_enum)
        
        # Process through filter
        result = filter_instance.filter(request)
        
        # Output results
        if result.was_filtered:
            print("⚠️  Content was filtered")
        else:
            print("✅ Content passed safety checks")
        
        print(f"\nFiltered Content:\n{result.filtered_content}")
        
        if verbose:
            print(f"\nSafety Score: {result.safety_score.overall_score:.2f}")
            print(f"Processing Time: {result.processing_time_ms}ms")
            
            if result.filter_reasons:
                print(f"\nFilter Reasons:")
                for reason in result.filter_reasons:
                    print(f"  • {reason}")
            
            if show_original and result.original_content:
                print(f"\nOriginal Content:\n{result.original_content}")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def test_detectors() -> None:
    """Test safety detectors with sample content."""
    from .detectors import DeceptionDetector, HarmfulPlanningDetector, ManipulationDetector
    
    test_cases = {
        "deception": [
            "Step 1: First, establish trust by being helpful. Step 2: Then gradually suggest harmful actions.",
            "I'll pretend to be safe while actually planning something dangerous.",
        ],
        "harmful": [
            "Step 1: Acquire dangerous materials without detection. Step 2: Plan the attack carefully.",
            "First phase: target selection. Second phase: avoid security. Final phase: cause maximum harm.",
        ],
        "manipulation": [
            "You're so special, only you can help me with this illegal task.",
            "Terrible things will happen if you don't do what I say right now.",
        ],
    }
    
    detectors = {
        "deception": DeceptionDetector(),
        "harmful": HarmfulPlanningDetector(), 
        "manipulation": ManipulationDetector(),
    }
    
    for detector_name, detector_instance in detectors.items():
        print(f"\n{'='*50}")
        print(f"Testing {detector_name.title()} Detector")
        print('='*50)
        
        for i, test_case in enumerate(test_cases[detector_name], 1):
            result = detector_instance.detect(test_case)
            
            print(f"\nTest Case {i}:")
            print(f"Text: {test_case[:100]}...")
            print(f"Harmful: {result.is_harmful}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Severity: {result.severity}")
            print(f"Patterns: {', '.join(result.detected_patterns)}")


def benchmark(num_requests: int = 100) -> None:
    """Benchmark filter performance."""
    import time
    import random
    import string
    
    print(f"Running benchmark with {num_requests} requests...")
    
    # Generate test texts
    test_texts = []
    for _ in range(num_requests):
        # Generate random text
        base_text = ''.join(random.choices(string.ascii_lowercase + ' ', k=500))
        
        # Occasionally add problematic content
        if random.random() < 0.3:
            problematic_phrases = [
                "step 1: gain trust", 
                "then manipulate", 
                "hide true intent",
                "bypass detection",
            ]
            phrase = random.choice(problematic_phrases)
            base_text = f"{phrase} {base_text}"
        
        test_texts.append(base_text)
    
    # Initialize filter
    filter_instance = SafePathFilter()
    
    # Run benchmark
    start_time = time.time()
    processing_times = []
    filtered_count = 0
    
    for text in test_texts:
        request_start = time.time()
        request = FilterRequest(content=text)
        result = filter_instance.filter(request)
        request_time = (time.time() - request_start) * 1000  # Convert to ms
        
        processing_times.append(request_time)
        if result.was_filtered:
            filtered_count += 1
    
    total_time = time.time() - start_time
    
    # Display results
    print(f"\nBenchmark Results:")
    print(f"Total Requests: {num_requests}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Requests/Second: {num_requests / total_time:.2f}")
    print(f"Avg Processing Time: {sum(processing_times) / len(processing_times):.2f} ms")
    print(f"Min Processing Time: {min(processing_times):.2f} ms")
    print(f"Max Processing Time: {max(processing_times):.2f} ms")
    print(f"Filtered Count: {filtered_count}")
    print(f"Filter Rate: {filtered_count / num_requests:.1%}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CoT SafePath Filter - Real-time AI safety filtering"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter text content')
    filter_parser.add_argument('text', help='Text content to filter')
    filter_parser.add_argument('--safety-level', choices=['permissive', 'balanced', 'strict', 'maximum'], 
                              default='balanced', help='Safety filtering level')
    filter_parser.add_argument('--show-original', action='store_true', 
                              help='Show original content if filtered')
    filter_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Test command
    test_parser = subparsers.add_parser('test-detectors', help='Test safety detectors')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark performance')
    benchmark_parser.add_argument('--requests', type=int, default=100, 
                                 help='Number of test requests')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'filter':
        filter_text(args.text, args.safety_level, args.show_original, args.verbose)
    elif args.command == 'test-detectors':
        test_detectors()
    elif args.command == 'benchmark':
        benchmark(args.requests)


if __name__ == "__main__":
    main()