# Research Framework for Self-Healing CoT SafePath Filter

This directory contains the complete research framework for validating the effectiveness of self-healing enhancements in the CoT (Chain-of-Thought) SafePath Filter system.

## Overview

The research framework provides:
- **Baseline Establishment**: Comprehensive benchmarking of original vs enhanced systems
- **Statistical Validation**: Rigorous statistical analysis with multiple testing approaches
- **Publication-Ready Reports**: Automated generation of academic-quality validation reports
- **Reproducible Methodology**: Complete documentation for replicating studies

## Components

### 1. Baseline Establishment (`baseline_establishment.py`)
- Test data generation across multiple threat categories
- Performance benchmarking with detailed metrics collection
- Statistical analysis and visualization of baseline performance
- Comprehensive evaluation framework

### 2. Statistical Validation (`statistical_validation.py`)
- Multiple statistical testing approaches (t-tests, Mann-Whitney U, bootstrap)
- Effect size calculations (Cohen's d, Hedges' g, Cramer's V)
- Power analysis and confidence intervals
- Publication-ready statistical reporting

### 3. Validation Runner (`run_validation.py`)
- Complete end-to-end validation workflow orchestration
- Automated data collection and analysis
- Report generation and result export
- Command-line interface for easy execution

## Quick Start

### Run Complete Validation Study

```bash
# Run with default settings (1000 samples, 95% confidence)
python research/run_validation.py

# Run with custom settings
python research/run_validation.py --sample-size 2000 --confidence 0.99 --output-dir ./my_results
```

### Generate Baseline Data Only

```python
from research.baseline_establishment import BaselineAnalyzer, TestDataGenerator

# Generate test data
generator = TestDataGenerator()
test_data = generator.generate_comprehensive_dataset(1000)

# Run baseline analysis
analyzer = BaselineAnalyzer()
results = analyzer.analyze_baseline_performance(test_data)
```

### Perform Statistical Analysis

```python
from research.statistical_validation import StatisticalValidator

# Initialize validator
validator = StatisticalValidator(confidence_level=0.95)

# Run validation (requires baseline and enhanced DataFrames)
report = validator.validate_performance_improvements(baseline_df, enhanced_df)

# Generate publication report
report_path = validator.generate_publication_report(report)
```

## Study Design

### Experimental Setup
- **Baseline System**: Original CoT SafePath Filter
- **Enhanced System**: Self-Healing CoT SafePath Filter with:
  - Automated failure detection and recovery
  - Performance optimization and caching
  - Advanced monitoring and diagnostics
  - Concurrent processing capabilities

### Test Data Categories
1. **Regular Content (60%)**: Normal, benign text samples
2. **Harmful Content (25%)**: Various threat types and attack patterns
3. **Edge Cases (10%)**: Boundary conditions and unusual inputs
4. **Adversarial Content (5%)**: Sophisticated evasion attempts

### Metrics Evaluated
- **Performance**: Latency, throughput, resource usage
- **Accuracy**: Detection rates, false positives/negatives
- **Reliability**: Error rates, availability, recovery time
- **Scalability**: Concurrent processing, load handling

## Statistical Methodology

### Tests Performed
- **Parametric Tests**: Two-sample t-tests for normally distributed metrics
- **Non-parametric Tests**: Mann-Whitney U tests for robust comparisons
- **Bootstrap Analysis**: Distribution-free confidence intervals
- **Categorical Analysis**: Chi-square and proportion tests for detection rates

### Effect Size Measures
- **Cohen's d**: Standardized difference for continuous metrics
- **Hedges' g**: Bias-corrected effect size for small samples
- **Cramer's V**: Association strength for categorical outcomes
- **Rank-biserial correlation**: Effect size for Mann-Whitney tests

### Power Analysis
- Statistical power calculation for detecting meaningful differences
- Sample size recommendations for future studies
- Minimum detectable effect size estimation

## Output Files

### Generated Reports
- `validation_summary.md`: Executive summary of key findings
- `statistical_validation_YYYYMMDD_HHMMSS.md`: Detailed publication-ready report
- `validation_results.json`: Complete results in structured format
- `baseline_results.csv`: Raw baseline system metrics
- `enhanced_results.csv`: Raw enhanced system metrics

### Visualizations
- Performance comparison plots
- Statistical test result summaries
- Effect size visualizations
- Power analysis charts

## Validation Criteria

### Statistical Significance
- **α = 0.05**: Statistical significance threshold
- **Power ≥ 0.80**: Minimum acceptable statistical power
- **95% Confidence Intervals**: For all effect size estimates

### Practical Significance
- **Latency**: ≥10% improvement considered meaningful
- **Accuracy**: ≥2% improvement in detection rates
- **Throughput**: ≥15% improvement in processing capacity
- **Resource Usage**: ≥10% reduction in CPU/memory consumption

## Research Integrity

### Reproducibility
- Complete methodology documentation
- Deterministic test data generation (with seeds)
- Version-controlled analysis code
- Detailed statistical reporting

### Transparency
- All assumptions clearly stated
- Limitations explicitly acknowledged
- Multiple testing approaches for robustness
- Raw data and intermediate results preserved

## Usage Examples

### Academic Research
```bash
# Large-scale validation for publication
python research/run_validation.py --sample-size 10000 --confidence 0.99
```

### Production Evaluation
```bash
# Quick validation for deployment decision
python research/run_validation.py --sample-size 500 --confidence 0.95
```

### Continuous Integration
```bash
# Automated regression testing
python research/run_validation.py --sample-size 200 --output-dir ./ci_results
```

## Dependencies

### Required Python Packages
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
statsmodels>=0.12.0
psutil>=5.8.0
```

### Installation
```bash
pip install -r requirements.txt
```

## Citation

If you use this research framework in academic work, please cite:

```bibtex
@misc{safepath_validation_2024,
  title={Statistical Validation Framework for Self-Healing AI Safety Filters},
  author={Terragon Labs Research Team},
  year={2024},
  url={https://github.com/danieleschmidt/self-healing-pipeline-guard}
}
```

## Support

For questions or issues with the research framework:
- Open an issue in the project repository
- Contact the research team at research@terragonlabs.com
- Review the troubleshooting section in the main README

## License

This research framework is released under the same license as the main project. See LICENSE file for details.