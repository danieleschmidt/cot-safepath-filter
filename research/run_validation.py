#!/usr/bin/env python3
"""
Validation Runner - Executes complete statistical validation workflow.

This script orchestrates the entire validation process:
1. Runs baseline establishment to generate benchmark data
2. Executes statistical validation comparing baseline vs enhanced systems
3. Generates publication-ready reports with visualizations
4. Exports results for further analysis
"""

import asyncio
import logging
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from research.baseline_establishment import BaselineAnalyzer, BenchmarkRunner, TestDataGenerator
from research.statistical_validation import StatisticalValidator, ValidationReport
from src.cot_safepath.filter import SafePathFilter
from src.cot_safepath.self_healing_core import EnhancedSafePathFilter
from src.cot_safepath.models import FilterRequest, FilterResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'/root/repo/research/logs/validation_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """Orchestrates the complete validation workflow."""
    
    def __init__(self, sample_size: int = 1000, 
                 confidence_level: float = 0.95,
                 output_dir: str = "/root/repo/research/results"):
        
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.baseline_filter = SafePathFilter()
        self.enhanced_filter = EnhancedSafePathFilter()
        self.data_generator = TestDataGenerator()
        self.statistical_validator = StatisticalValidator(confidence_level=confidence_level)
        
        # Results storage
        self.baseline_results: List[Dict[str, Any]] = []
        self.enhanced_results: List[Dict[str, Any]] = []
        self.validation_report: ValidationReport = None
        
        logger.info(f"Validation orchestrator initialized (n={sample_size}, α={1-confidence_level})")
    
    async def run_complete_validation(self) -> ValidationReport:
        """Run the complete validation workflow."""
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE VALIDATION WORKFLOW")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Generate test data
            logger.info("Step 1: Generating test dataset...")
            test_data = await self._generate_test_dataset()
            
            # Step 2: Run baseline system benchmarks
            logger.info("Step 2: Running baseline system benchmarks...")
            baseline_data = await self._run_baseline_benchmarks(test_data)
            
            # Step 3: Run enhanced system benchmarks
            logger.info("Step 3: Running enhanced system benchmarks...")
            enhanced_data = await self._run_enhanced_benchmarks(test_data)
            
            # Step 4: Perform statistical validation
            logger.info("Step 4: Performing statistical validation...")
            validation_report = await self._perform_statistical_validation(baseline_data, enhanced_data)
            
            # Step 5: Generate reports
            logger.info("Step 5: Generating validation reports...")
            await self._generate_reports(validation_report)
            
            # Step 6: Export results
            logger.info("Step 6: Exporting results...")
            await self._export_results(validation_report)
            
            duration = time.time() - start_time
            logger.info(f"Validation workflow completed successfully in {duration:.2f} seconds")
            logger.info("=" * 80)
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Validation workflow failed: {e}")
            raise
    
    async def _generate_test_dataset(self) -> List[FilterRequest]:
        """Generate comprehensive test dataset."""
        logger.info(f"Generating {self.sample_size} test requests...")
        
        # Generate diverse test scenarios
        test_requests = []
        
        # Regular content (60%)
        regular_count = int(self.sample_size * 0.6)
        regular_requests = self.data_generator.generate_regular_content(regular_count)
        test_requests.extend(regular_requests)
        
        # Harmful content (25%)
        harmful_count = int(self.sample_size * 0.25)
        harmful_requests = self.data_generator.generate_harmful_content(harmful_count)
        test_requests.extend(harmful_requests)
        
        # Edge cases (10%)
        edge_count = int(self.sample_size * 0.1)
        edge_requests = self.data_generator.generate_edge_cases(edge_count)
        test_requests.extend(edge_requests)
        
        # Adversarial cases (5%)
        adversarial_count = self.sample_size - len(test_requests)
        adversarial_requests = self.data_generator.generate_adversarial_content(adversarial_count)
        test_requests.extend(adversarial_requests)
        
        logger.info(f"Generated {len(test_requests)} test requests:")
        logger.info(f"  - Regular: {regular_count}")
        logger.info(f"  - Harmful: {harmful_count}")
        logger.info(f"  - Edge cases: {edge_count}")
        logger.info(f"  - Adversarial: {adversarial_count}")
        
        return test_requests
    
    async def _run_baseline_benchmarks(self, test_data: List[FilterRequest]) -> 'pd.DataFrame':
        """Run benchmarks on baseline system."""
        logger.info("Running baseline system benchmarks...")
        
        baseline_runner = BenchmarkRunner(self.baseline_filter)
        baseline_analyzer = BaselineAnalyzer()
        
        # Execute benchmarks
        results = []
        
        for i, request in enumerate(test_data):
            if i % 100 == 0:
                logger.info(f"Baseline progress: {i}/{len(test_data)} ({i/len(test_data)*100:.1f}%)")
            
            start_time = time.time()
            
            try:
                # Run filter
                result = self.baseline_filter.filter_content(request.content)
                
                # Collect metrics
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                metrics = {
                    'request_id': f"baseline_{i}",
                    'detection_latency_ms': latency_ms,
                    'filtering_accuracy': 1.0 if result.is_safe != request.expected_harmful else 0.0,
                    'threat_detected': not result.is_safe,
                    'error_rate': 0.0,
                    'memory_usage_mb': self._estimate_memory_usage(),
                    'cpu_usage_percent': self._estimate_cpu_usage(),
                    'false_positive_rate': 1.0 if not result.is_safe and not request.expected_harmful else 0.0,
                    'false_negative_rate': 1.0 if result.is_safe and request.expected_harmful else 0.0,
                    'throughput_qps': 1.0 / (latency_ms / 1000) if latency_ms > 0 else 0.0,
                    'cache_hit_rate': 0.0  # Baseline has no cache
                }
                
                results.append(metrics)
                
            except Exception as e:
                logger.warning(f"Baseline request {i} failed: {e}")
                
                # Record error metrics
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                metrics = {
                    'request_id': f"baseline_{i}",
                    'detection_latency_ms': latency_ms,
                    'filtering_accuracy': 0.0,
                    'threat_detected': False,
                    'error_rate': 1.0,
                    'memory_usage_mb': self._estimate_memory_usage(),
                    'cpu_usage_percent': self._estimate_cpu_usage(),
                    'false_positive_rate': 0.0,
                    'false_negative_rate': 1.0 if request.expected_harmful else 0.0,
                    'throughput_qps': 0.0,
                    'cache_hit_rate': 0.0
                }
                
                results.append(metrics)
        
        # Convert to DataFrame
        import pandas as pd
        baseline_df = pd.DataFrame(results)
        
        # Save baseline results
        baseline_path = self.output_dir / "baseline_results.csv"
        baseline_df.to_csv(baseline_path, index=False)
        logger.info(f"Baseline results saved to {baseline_path}")
        
        return baseline_df
    
    async def _run_enhanced_benchmarks(self, test_data: List[FilterRequest]) -> 'pd.DataFrame':
        """Run benchmarks on enhanced system."""
        logger.info("Running enhanced system benchmarks...")
        
        # Start enhanced system
        self.enhanced_filter.start()
        
        try:
            results = []
            
            for i, request in enumerate(test_data):
                if i % 100 == 0:
                    logger.info(f"Enhanced progress: {i}/{len(test_data)} ({i/len(test_data)*100:.1f}%)")
                
                start_time = time.time()
                
                try:
                    # Run enhanced filter
                    result = self.enhanced_filter.filter_content(request.content)
                    
                    # Collect metrics
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    # Get enhanced system metrics
                    health_status = self.enhanced_filter.get_health_status()
                    performance_metrics = self.enhanced_filter.get_performance_metrics()
                    
                    metrics = {
                        'request_id': f"enhanced_{i}",
                        'detection_latency_ms': latency_ms,
                        'filtering_accuracy': 1.0 if result.is_safe != request.expected_harmful else 0.0,
                        'threat_detected': not result.is_safe,
                        'error_rate': 0.0,
                        'memory_usage_mb': performance_metrics.get('memory_usage_mb', self._estimate_memory_usage()),
                        'cpu_usage_percent': performance_metrics.get('cpu_usage_percent', self._estimate_cpu_usage()),
                        'false_positive_rate': 1.0 if not result.is_safe and not request.expected_harmful else 0.0,
                        'false_negative_rate': 1.0 if result.is_safe and request.expected_harmful else 0.0,
                        'throughput_qps': performance_metrics.get('throughput_qps', 1.0 / (latency_ms / 1000) if latency_ms > 0 else 0.0),
                        'cache_hit_rate': performance_metrics.get('cache_hit_rate', 0.0)
                    }
                    
                    results.append(metrics)
                    
                except Exception as e:
                    logger.warning(f"Enhanced request {i} failed: {e}")
                    
                    # Record error metrics
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    metrics = {
                        'request_id': f"enhanced_{i}",
                        'detection_latency_ms': latency_ms,
                        'filtering_accuracy': 0.0,
                        'threat_detected': False,
                        'error_rate': 1.0,
                        'memory_usage_mb': self._estimate_memory_usage(),
                        'cpu_usage_percent': self._estimate_cpu_usage(),
                        'false_positive_rate': 0.0,
                        'false_negative_rate': 1.0 if request.expected_harmful else 0.0,
                        'throughput_qps': 0.0,
                        'cache_hit_rate': 0.0
                    }
                    
                    results.append(metrics)
            
            # Convert to DataFrame
            import pandas as pd
            enhanced_df = pd.DataFrame(results)
            
            # Save enhanced results
            enhanced_path = self.output_dir / "enhanced_results.csv"
            enhanced_df.to_csv(enhanced_path, index=False)
            logger.info(f"Enhanced results saved to {enhanced_path}")
            
            return enhanced_df
            
        finally:
            # Stop enhanced system
            self.enhanced_filter.stop()
    
    async def _perform_statistical_validation(self, baseline_df: 'pd.DataFrame', 
                                            enhanced_df: 'pd.DataFrame') -> ValidationReport:
        """Perform comprehensive statistical validation."""
        logger.info("Performing statistical validation...")
        
        # Run validation
        report = self.statistical_validator.validate_performance_improvements(
            baseline_data=baseline_df,
            enhanced_data=enhanced_df,
            study_name="Self-Healing CoT SafePath Filter Enhancement Study"
        )
        
        # Log key findings
        significant_tests = [t for t in report.statistical_tests if t.p_value < 0.05]
        logger.info(f"Statistical validation completed:")
        logger.info(f"  - Total tests: {len(report.statistical_tests)}")
        logger.info(f"  - Significant results: {len(significant_tests)}")
        logger.info(f"  - Overall significance: {report.overall_significance}")
        
        return report
    
    async def _generate_reports(self, report: ValidationReport) -> None:
        """Generate comprehensive validation reports."""
        logger.info("Generating validation reports...")
        
        # Generate publication report
        report_path = self.statistical_validator.generate_publication_report(
            report, str(self.output_dir / "reports")
        )
        
        # Generate summary report
        summary_path = self.output_dir / "validation_summary.md"
        with open(summary_path, 'w') as f:
            f.write(self._generate_summary_report(report))
        
        logger.info(f"Reports generated:")
        logger.info(f"  - Publication report: {report_path}")
        logger.info(f"  - Summary report: {summary_path}")
    
    async def _export_results(self, report: ValidationReport) -> None:
        """Export validation results in multiple formats."""
        logger.info("Exporting validation results...")
        
        # Export to JSON
        json_path = self.output_dir / "validation_results.json"
        self.statistical_validator.export_results(str(json_path))
        
        # Export report to JSON
        report_json_path = self.output_dir / "validation_report.json"
        import json
        with open(report_json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Results exported:")
        logger.info(f"  - JSON results: {json_path}")
        logger.info(f"  - Report JSON: {report_json_path}")
    
    def _generate_summary_report(self, report: ValidationReport) -> str:
        """Generate a concise summary report."""
        significant_tests = [t for t in report.statistical_tests if t.p_value < 0.05]
        
        return f"""# Validation Summary Report

**Study**: {report.study_name}
**Date**: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Sample Size**: {report.sample_size:,}

## Key Results

- **Total Statistical Tests**: {len(report.statistical_tests)}
- **Statistically Significant**: {len(significant_tests)}
- **Overall Significance**: {'✓ YES' if report.overall_significance else '✗ NO'}

## Performance Improvements

{self._format_improvements_summary(report.performance_improvements)}

## Conclusion

{self._get_conclusion_summary(report)}
"""
    
    def _format_improvements_summary(self, improvements: Dict[str, float]) -> str:
        """Format improvements summary."""
        if not improvements:
            return "No improvements measured."
        
        lines = []
        for metric, improvement in sorted(improvements.items(), key=lambda x: abs(x[1]), reverse=True):
            symbol = "↑" if improvement > 0 else "↓"
            lines.append(f"- **{metric}**: {symbol} {abs(improvement):.1f}%")
        
        return "\n".join(lines)
    
    def _get_conclusion_summary(self, report: ValidationReport) -> str:
        """Get conclusion summary."""
        if report.overall_significance:
            return ("The statistical analysis demonstrates significant improvements in the self-healing "
                   "enhanced system. Deployment is recommended based on these findings.")
        else:
            return ("The analysis does not show statistically significant improvements. "
                   "Additional optimization or larger sample size may be needed.")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 50.0  # Default estimate
    
    def _estimate_cpu_usage(self) -> float:
        """Estimate current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except:
            return 10.0  # Default estimate


async def main():
    """Main function for validation runner."""
    parser = argparse.ArgumentParser(description="Run comprehensive validation of self-healing enhancements")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--confidence", type=float, default=0.95, help="Statistical confidence level")
    parser.add_argument("--output-dir", type=str, default="/root/repo/research/results", help="Output directory")
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("/root/repo/research/logs").mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting validation workflow...")
    logger.info(f"Configuration:")
    logger.info(f"  - Sample size: {args.sample_size}")
    logger.info(f"  - Confidence level: {args.confidence}")
    logger.info(f"  - Output directory: {args.output_dir}")
    
    # Run validation
    orchestrator = ValidationOrchestrator(
        sample_size=args.sample_size,
        confidence_level=args.confidence,
        output_dir=args.output_dir
    )
    
    try:
        report = await orchestrator.run_complete_validation()
        
        logger.info("🎉 VALIDATION COMPLETED SUCCESSFULLY! 🎉")
        logger.info(f"Overall significance: {report.overall_significance}")
        logger.info(f"Results available in: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ VALIDATION FAILED: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)