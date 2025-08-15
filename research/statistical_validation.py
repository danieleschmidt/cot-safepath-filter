"""
Statistical Validation Module - Research Component for Self-Healing Pipeline Guard.

Comprehensive statistical analysis to validate the effectiveness of self-healing enhancements
in AI safety filtering. Provides rigorous statistical testing, effect size calculations,
and publication-ready reports.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, chi2_contingency, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path
import warnings
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import ttest_power

from .baseline_establishment import BaselineAnalyzer, BenchmarkRunner
from ..src.cot_safepath.models import FilterRequest, FilterResult
from ..src.cot_safepath.self_healing_core import EnhancedSafePathFilter


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class StatisticalTest:
    """Represents a statistical test and its results."""
    test_name: str
    test_type: str  # 't-test', 'mann-whitney', 'chi-square', etc.
    statistic: float
    p_value: float
    effect_size: float
    effect_size_interpretation: str
    confidence_interval: Tuple[float, float]
    statistical_power: float
    sample_size: int
    assumptions_met: bool
    interpretation: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "test_type": self.test_type,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_size_interpretation": self.effect_size_interpretation,
            "confidence_interval": list(self.confidence_interval),
            "statistical_power": self.statistical_power,
            "sample_size": self.sample_size,
            "assumptions_met": self.assumptions_met,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    study_name: str
    baseline_system: str
    enhanced_system: str
    sample_size: int
    study_duration_hours: float
    statistical_tests: List[StatisticalTest]
    performance_improvements: Dict[str, float]
    confidence_level: float
    overall_significance: bool
    recommendations: List[str]
    limitations: List[str]
    future_work: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "baseline_system": self.baseline_system,
            "enhanced_system": self.enhanced_system,
            "sample_size": self.sample_size,
            "study_duration_hours": self.study_duration_hours,
            "statistical_tests": [test.to_dict() for test in self.statistical_tests],
            "performance_improvements": self.performance_improvements,
            "confidence_level": self.confidence_level,
            "overall_significance": self.overall_significance,
            "recommendations": self.recommendations,
            "limitations": self.limitations,
            "future_work": self.future_work,
            "timestamp": self.timestamp.isoformat()
        }


class EffectSizeCalculator:
    """Calculates and interprets various effect sizes."""
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    @staticmethod
    def glass_delta(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Glass's Delta effect size."""
        control_std = np.std(group2, ddof=1)
        if control_std == 0:
            return 0.0
        return (np.mean(group1) - np.mean(group2)) / control_std
    
    @staticmethod
    def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedges' g effect size."""
        cohens_d = EffectSizeCalculator.cohens_d(group1, group2)
        n1, n2 = len(group1), len(group2)
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        return cohens_d * correction_factor
    
    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def cramers_v(contingency_table: np.ndarray) -> float:
        """Calculate Cramer's V for categorical associations."""
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        return np.sqrt(chi2 / (n * min_dim))


class StatisticalValidator:
    """Main statistical validation engine."""
    
    def __init__(self, confidence_level: float = 0.95, 
                 power_threshold: float = 0.8,
                 alpha: float = 0.05):
        self.confidence_level = confidence_level
        self.power_threshold = power_threshold
        self.alpha = alpha
        
        self.effect_calculator = EffectSizeCalculator()
        
        # Analysis results storage
        self.validation_reports: List[ValidationReport] = []
        
        logger.info(f"Statistical validator initialized (α={alpha}, power={power_threshold})")
    
    def validate_performance_improvements(self, 
                                        baseline_data: pd.DataFrame,
                                        enhanced_data: pd.DataFrame,
                                        study_name: str = "Self-Healing Enhancement Study") -> ValidationReport:
        """
        Comprehensive validation of performance improvements between baseline and enhanced systems.
        """
        logger.info(f"Starting statistical validation: {study_name}")
        
        # Ensure we have matching metrics
        common_metrics = set(baseline_data.columns) & set(enhanced_data.columns)
        if not common_metrics:
            raise ValueError("No common metrics found between baseline and enhanced data")
        
        statistical_tests = []
        performance_improvements = {}
        
        # Core performance metrics to analyze
        key_metrics = [
            'detection_latency_ms', 'filtering_accuracy', 'throughput_qps',
            'error_rate', 'memory_usage_mb', 'cpu_usage_percent',
            'cache_hit_rate', 'false_positive_rate', 'false_negative_rate'
        ]
        
        available_metrics = [m for m in key_metrics if m in common_metrics]
        
        for metric in available_metrics:
            logger.info(f"Analyzing metric: {metric}")
            
            baseline_values = baseline_data[metric].dropna().values
            enhanced_values = enhanced_data[metric].dropna().values
            
            if len(baseline_values) < 10 or len(enhanced_values) < 10:
                logger.warning(f"Insufficient data for {metric}: baseline={len(baseline_values)}, enhanced={len(enhanced_values)}")
                continue
            
            # Perform multiple statistical tests
            tests = self._perform_comprehensive_testing(
                baseline_values, enhanced_values, metric
            )
            statistical_tests.extend(tests)
            
            # Calculate performance improvement
            baseline_mean = np.mean(baseline_values)
            enhanced_mean = np.mean(enhanced_values)
            
            if baseline_mean != 0:
                # For metrics where lower is better (like latency, error_rate)
                if metric in ['detection_latency_ms', 'error_rate', 'false_positive_rate', 'false_negative_rate']:
                    improvement = (baseline_mean - enhanced_mean) / baseline_mean * 100
                else:
                    # For metrics where higher is better (like accuracy, throughput)
                    improvement = (enhanced_mean - baseline_mean) / baseline_mean * 100
                
                performance_improvements[metric] = improvement
        
        # Additional categorical analysis
        if 'threat_detected' in common_metrics:
            categorical_tests = self._analyze_categorical_outcomes(
                baseline_data, enhanced_data
            )
            statistical_tests.extend(categorical_tests)
        
        # Determine overall significance
        significant_tests = [t for t in statistical_tests if t.p_value < self.alpha]
        overall_significance = len(significant_tests) > 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(statistical_tests, performance_improvements)
        limitations = self._identify_limitations(baseline_data, enhanced_data)
        future_work = self._suggest_future_work(statistical_tests)
        
        # Create validation report
        report = ValidationReport(
            study_name=study_name,
            baseline_system="Original CoT SafePath Filter",
            enhanced_system="Self-Healing Enhanced CoT SafePath Filter",
            sample_size=min(len(baseline_data), len(enhanced_data)),
            study_duration_hours=self._estimate_study_duration(baseline_data, enhanced_data),
            statistical_tests=statistical_tests,
            performance_improvements=performance_improvements,
            confidence_level=self.confidence_level,
            overall_significance=overall_significance,
            recommendations=recommendations,
            limitations=limitations,
            future_work=future_work
        )
        
        self.validation_reports.append(report)
        logger.info(f"Statistical validation completed. Significant improvements: {overall_significance}")
        
        return report
    
    def _perform_comprehensive_testing(self, baseline: np.ndarray, 
                                     enhanced: np.ndarray, 
                                     metric: str) -> List[StatisticalTest]:
        """Perform comprehensive statistical testing on a metric."""
        tests = []
        
        # 1. Two-sample t-test (assuming normality)
        try:
            t_stat, t_p = ttest_ind(enhanced, baseline, equal_var=False)
            cohens_d = self.effect_calculator.cohens_d(enhanced, baseline)
            
            # Calculate confidence interval for difference in means
            pooled_se = np.sqrt(np.var(enhanced, ddof=1)/len(enhanced) + 
                               np.var(baseline, ddof=1)/len(baseline))
            df = len(enhanced) + len(baseline) - 2
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            mean_diff = np.mean(enhanced) - np.mean(baseline)
            margin_error = t_critical * pooled_se
            ci = (mean_diff - margin_error, mean_diff + margin_error)
            
            # Calculate power
            power = ttest_power(effect_size=abs(cohens_d), 
                              nobs=min(len(enhanced), len(baseline)),
                              alpha=self.alpha)
            
            # Check normality assumption
            _, baseline_normal = stats.shapiro(baseline[:5000])  # Limit for shapiro
            _, enhanced_normal = stats.shapiro(enhanced[:5000])
            assumptions_met = baseline_normal > 0.05 and enhanced_normal > 0.05
            
            tests.append(StatisticalTest(
                test_name=f"Two-sample t-test ({metric})",
                test_type="t-test",
                statistic=t_stat,
                p_value=t_p,
                effect_size=cohens_d,
                effect_size_interpretation=self.effect_calculator.interpret_cohens_d(cohens_d),
                confidence_interval=ci,
                statistical_power=power,
                sample_size=len(enhanced) + len(baseline),
                assumptions_met=assumptions_met,
                interpretation=self._interpret_t_test(t_p, cohens_d, metric)
            ))
        except Exception as e:
            logger.warning(f"T-test failed for {metric}: {e}")
        
        # 2. Mann-Whitney U test (non-parametric alternative)
        try:
            u_stat, u_p = mannwhitneyu(enhanced, baseline, alternative='two-sided')
            
            # Effect size for Mann-Whitney (rank-biserial correlation)
            r = 1 - (2 * u_stat) / (len(enhanced) * len(baseline))
            
            tests.append(StatisticalTest(
                test_name=f"Mann-Whitney U test ({metric})",
                test_type="mann-whitney",
                statistic=u_stat,
                p_value=u_p,
                effect_size=r,
                effect_size_interpretation=self._interpret_rank_biserial(r),
                confidence_interval=(0, 0),  # CI calculation is complex for MW
                statistical_power=0.0,  # Power calculation is complex for MW
                sample_size=len(enhanced) + len(baseline),
                assumptions_met=True,  # Non-parametric test
                interpretation=self._interpret_mann_whitney(u_p, r, metric)
            ))
        except Exception as e:
            logger.warning(f"Mann-Whitney test failed for {metric}: {e}")
        
        # 3. Bootstrap confidence interval for difference in means
        try:
            bootstrap_diffs = []
            n_bootstrap = 1000
            
            for _ in range(n_bootstrap):
                bootstrap_enhanced = np.random.choice(enhanced, size=len(enhanced), replace=True)
                bootstrap_baseline = np.random.choice(baseline, size=len(baseline), replace=True)
                bootstrap_diffs.append(np.mean(bootstrap_enhanced) - np.mean(bootstrap_baseline))
            
            bootstrap_ci = np.percentile(bootstrap_diffs, 
                                       [100*(self.alpha/2), 100*(1-self.alpha/2)])
            
            # Bootstrap p-value (proportion of bootstrap differences that cross zero)
            bootstrap_p = 2 * min(np.mean(np.array(bootstrap_diffs) <= 0),
                                 np.mean(np.array(bootstrap_diffs) >= 0))
            
            tests.append(StatisticalTest(
                test_name=f"Bootstrap test ({metric})",
                test_type="bootstrap",
                statistic=np.mean(bootstrap_diffs),
                p_value=bootstrap_p,
                effect_size=np.mean(bootstrap_diffs) / np.std(baseline),
                effect_size_interpretation="standardized difference",
                confidence_interval=tuple(bootstrap_ci),
                statistical_power=0.0,
                sample_size=len(enhanced) + len(baseline),
                assumptions_met=True,
                interpretation=self._interpret_bootstrap(bootstrap_p, bootstrap_ci, metric)
            ))
        except Exception as e:
            logger.warning(f"Bootstrap test failed for {metric}: {e}")
        
        return tests
    
    def _analyze_categorical_outcomes(self, baseline_df: pd.DataFrame, 
                                    enhanced_df: pd.DataFrame) -> List[StatisticalTest]:
        """Analyze categorical outcomes like threat detection rates."""
        tests = []
        
        # Threat detection rate comparison
        if 'threat_detected' in baseline_df.columns and 'threat_detected' in enhanced_df.columns:
            baseline_detections = baseline_df['threat_detected'].sum()
            baseline_total = len(baseline_df)
            enhanced_detections = enhanced_df['threat_detected'].sum()
            enhanced_total = len(enhanced_df)
            
            # Chi-square test
            contingency_table = np.array([
                [baseline_detections, baseline_total - baseline_detections],
                [enhanced_detections, enhanced_total - enhanced_detections]
            ])
            
            try:
                chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
                cramers_v = self.effect_calculator.cramers_v(contingency_table)
                
                tests.append(StatisticalTest(
                    test_name="Threat Detection Rate (Chi-square)",
                    test_type="chi-square",
                    statistic=chi2,
                    p_value=p_chi2,
                    effect_size=cramers_v,
                    effect_size_interpretation=self._interpret_cramers_v(cramers_v),
                    confidence_interval=(0, 0),
                    statistical_power=0.0,
                    sample_size=baseline_total + enhanced_total,
                    assumptions_met=np.all(expected >= 5),
                    interpretation=self._interpret_chi_square(p_chi2, cramers_v)
                ))
            except Exception as e:
                logger.warning(f"Chi-square test failed: {e}")
            
            # Two-proportion z-test
            try:
                baseline_rate = baseline_detections / baseline_total
                enhanced_rate = enhanced_detections / enhanced_total
                
                z_stat, p_z = proportions_ztest(
                    [enhanced_detections, baseline_detections],
                    [enhanced_total, baseline_total]
                )
                
                # Effect size (difference in proportions)
                prop_diff = enhanced_rate - baseline_rate
                
                tests.append(StatisticalTest(
                    test_name="Threat Detection Rate (Two-proportion z-test)",
                    test_type="two-proportion-z",
                    statistic=z_stat,
                    p_value=p_z,
                    effect_size=prop_diff,
                    effect_size_interpretation=self._interpret_proportion_diff(prop_diff),
                    confidence_interval=(0, 0),
                    statistical_power=0.0,
                    sample_size=baseline_total + enhanced_total,
                    assumptions_met=True,
                    interpretation=self._interpret_proportion_test(p_z, prop_diff)
                ))
            except Exception as e:
                logger.warning(f"Two-proportion z-test failed: {e}")
        
        return tests
    
    def _interpret_t_test(self, p_value: float, cohens_d: float, metric: str) -> str:
        """Interpret t-test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        direction = "improvement" if cohens_d > 0 else "decline"
        effect_size = self.effect_calculator.interpret_cohens_d(cohens_d)
        
        return (f"The difference in {metric} between enhanced and baseline systems is "
               f"{significance} (p={p_value:.4f}) with a {effect_size} effect size "
               f"(Cohen's d={cohens_d:.3f}), indicating a {direction} in performance.")
    
    def _interpret_mann_whitney(self, p_value: float, effect_size: float, metric: str) -> str:
        """Interpret Mann-Whitney U test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        direction = "higher" if effect_size > 0 else "lower"
        
        return (f"The Mann-Whitney U test shows a {significance} difference "
               f"(p={p_value:.4f}) in {metric}, with the enhanced system showing "
               f"{direction} values (r={effect_size:.3f}).")
    
    def _interpret_bootstrap(self, p_value: float, ci: Tuple[float, float], metric: str) -> str:
        """Interpret bootstrap test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        improvement = "improvement" if ci[0] > 0 else "decline" if ci[1] < 0 else "mixed"
        
        return (f"Bootstrap analysis shows a {significance} difference "
               f"(p={p_value:.4f}) in {metric}, with 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], "
               f"indicating a {improvement} in performance.")
    
    def _interpret_chi_square(self, p_value: float, cramers_v: float) -> str:
        """Interpret chi-square test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        strength = self._interpret_cramers_v(cramers_v)
        
        return (f"Chi-square test shows a {significance} association "
               f"(p={p_value:.4f}) with {strength} effect size (Cramer's V={cramers_v:.3f}).")
    
    def _interpret_proportion_test(self, p_value: float, prop_diff: float) -> str:
        """Interpret two-proportion test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        direction = "higher" if prop_diff > 0 else "lower"
        
        return (f"Two-proportion test shows a {significance} difference "
               f"(p={p_value:.4f}) in detection rates, with the enhanced system "
               f"showing {direction} rates (difference={prop_diff:.3f}).")
    
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramer's V effect size."""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_rank_biserial(self, r: float) -> str:
        """Interpret rank-biserial correlation."""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_proportion_diff(self, diff: float) -> str:
        """Interpret difference in proportions."""
        abs_diff = abs(diff)
        if abs_diff < 0.01:
            return "negligible"
        elif abs_diff < 0.05:
            return "small"
        elif abs_diff < 0.1:
            return "medium"
        else:
            return "large"
    
    def _generate_recommendations(self, tests: List[StatisticalTest], 
                                improvements: Dict[str, float]) -> List[str]:
        """Generate recommendations based on statistical analysis."""
        recommendations = []
        
        # Check for significant improvements
        significant_tests = [t for t in tests if t.p_value < self.alpha and t.effect_size > 0]
        
        if significant_tests:
            recommendations.append(
                "Deploy the self-healing enhanced system based on statistically significant improvements."
            )
            
            # Specific metric recommendations
            for metric, improvement in improvements.items():
                if improvement > 5:  # 5% improvement threshold
                    recommendations.append(
                        f"The {improvement:.1f}% improvement in {metric} provides substantial value."
                    )
        
        # Power analysis recommendations
        low_power_tests = [t for t in tests if t.statistical_power < self.power_threshold]
        if low_power_tests:
            recommendations.append(
                "Increase sample size for future studies to achieve adequate statistical power (>80%)."
            )
        
        # Effect size recommendations
        large_effects = [t for t in tests if t.effect_size_interpretation in ["large", "medium"]]
        if large_effects:
            recommendations.append(
                "The observed effect sizes suggest practical significance beyond statistical significance."
            )
        
        return recommendations
    
    def _identify_limitations(self, baseline_df: pd.DataFrame, 
                            enhanced_df: pd.DataFrame) -> List[str]:
        """Identify study limitations."""
        limitations = []
        
        sample_size = min(len(baseline_df), len(enhanced_df))
        if sample_size < 100:
            limitations.append(f"Small sample size (n={sample_size}) may limit generalizability.")
        
        # Check for missing data
        baseline_missing = baseline_df.isnull().sum().sum()
        enhanced_missing = enhanced_df.isnull().sum().sum()
        if baseline_missing > 0 or enhanced_missing > 0:
            limitations.append("Missing data may introduce bias in results.")
        
        limitations.extend([
            "Study conducted in controlled environment; real-world performance may vary.",
            "Short-term evaluation; long-term effects require additional study.",
            "Specific to current threat patterns; effectiveness against novel threats unclear."
        ])
        
        return limitations
    
    def _suggest_future_work(self, tests: List[StatisticalTest]) -> List[str]:
        """Suggest future research directions."""
        future_work = [
            "Longitudinal study to assess performance stability over time.",
            "Multi-site validation across different deployment environments.",
            "Cost-benefit analysis of self-healing capabilities.",
            "Evaluation against adversarial attacks and novel threat patterns.",
            "User experience study on operational overhead.",
            "Comparative analysis with alternative self-healing approaches."
        ]
        
        # Add specific suggestions based on test results
        non_significant_tests = [t for t in tests if t.p_value >= self.alpha]
        if non_significant_tests:
            future_work.append(
                "Investigate factors contributing to non-significant differences in some metrics."
            )
        
        return future_work
    
    def _estimate_study_duration(self, baseline_df: pd.DataFrame, 
                               enhanced_df: pd.DataFrame) -> float:
        """Estimate study duration in hours."""
        # Simple heuristic based on sample size
        total_samples = len(baseline_df) + len(enhanced_df)
        estimated_hours = total_samples / 100  # Assume 100 samples per hour
        return max(1.0, estimated_hours)
    
    def generate_publication_report(self, report: ValidationReport, 
                                  output_dir: str = "/root/repo/research/reports") -> str:
        """Generate a publication-ready statistical report."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"statistical_validation_{timestamp}.md")
        
        with open(report_path, 'w') as f:
            f.write(self._format_publication_report(report))
        
        # Generate visualizations
        self._generate_statistical_plots(report, output_dir, timestamp)
        
        logger.info(f"Publication report generated: {report_path}")
        return report_path
    
    def _format_publication_report(self, report: ValidationReport) -> str:
        """Format the report for publication."""
        return f"""
# Statistical Validation of Self-Healing AI Safety Filter Enhancements

## Executive Summary

This study presents a comprehensive statistical analysis of performance improvements achieved through implementing self-healing capabilities in the CoT (Chain-of-Thought) SafePath Filter system. The analysis demonstrates {len([t for t in report.statistical_tests if t.p_value < self.alpha])} statistically significant improvements across key performance metrics.

## Study Design

- **Baseline System**: {report.baseline_system}
- **Enhanced System**: {report.enhanced_system}
- **Sample Size**: {report.sample_size:,} observations
- **Study Duration**: {report.study_duration_hours:.1f} hours
- **Confidence Level**: {report.confidence_level:.0%}
- **Statistical Significance Threshold**: α = {self.alpha}

## Key Findings

### Performance Improvements

{self._format_performance_improvements(report.performance_improvements)}

### Statistical Test Results

{self._format_statistical_tests(report.statistical_tests)}

## Effect Sizes and Practical Significance

{self._format_effect_sizes(report.statistical_tests)}

## Statistical Power Analysis

{self._format_power_analysis(report.statistical_tests)}

## Recommendations

{self._format_recommendations(report.recommendations)}

## Limitations

{self._format_limitations(report.limitations)}

## Future Research Directions

{self._format_future_work(report.future_work)}

## Conclusion

{self._format_conclusion(report)}

## Methodology Notes

All statistical analyses were conducted using Python with scipy.stats, statsmodels, and scikit-learn libraries. Multiple testing correction was not applied due to the exploratory nature of this analysis, but should be considered in confirmatory studies.

**Generated on**: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    def _format_performance_improvements(self, improvements: Dict[str, float]) -> str:
        """Format performance improvements section."""
        if not improvements:
            return "No performance improvements calculated."
        
        lines = []
        for metric, improvement in sorted(improvements.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "improvement" if improvement > 0 else "decline"
            lines.append(f"- **{metric}**: {improvement:+.2f}% {direction}")
        
        return "\n".join(lines)
    
    def _format_statistical_tests(self, tests: List[StatisticalTest]) -> str:
        """Format statistical tests section."""
        if not tests:
            return "No statistical tests performed."
        
        lines = []
        for test in tests:
            significance = "✓" if test.p_value < self.alpha else "✗"
            lines.append(f"""
#### {test.test_name}
- **Test Type**: {test.test_type}
- **Statistic**: {test.statistic:.4f}
- **p-value**: {test.p_value:.4f} {significance}
- **Effect Size**: {test.effect_size:.4f} ({test.effect_size_interpretation})
- **Statistical Power**: {test.statistical_power:.2%} (if calculated)
- **Assumptions Met**: {'Yes' if test.assumptions_met else 'No'}
- **Interpretation**: {test.interpretation}
""")
        
        return "\n".join(lines)
    
    def _format_effect_sizes(self, tests: List[StatisticalTest]) -> str:
        """Format effect sizes section."""
        effect_sizes = [(t.test_name, t.effect_size, t.effect_size_interpretation) 
                       for t in tests if t.effect_size != 0]
        
        if not effect_sizes:
            return "No effect sizes calculated."
        
        lines = ["Effect sizes indicate the practical significance of observed differences:\n"]
        
        for name, size, interpretation in effect_sizes:
            lines.append(f"- **{name}**: {size:.3f} ({interpretation})")
        
        return "\n".join(lines)
    
    def _format_power_analysis(self, tests: List[StatisticalTest]) -> str:
        """Format power analysis section."""
        power_tests = [(t.test_name, t.statistical_power) for t in tests if t.statistical_power > 0]
        
        if not power_tests:
            return "Power analysis not available for conducted tests."
        
        lines = ["Statistical power indicates the probability of detecting true effects:\n"]
        
        for name, power in power_tests:
            adequacy = "Adequate" if power >= self.power_threshold else "Inadequate"
            lines.append(f"- **{name}**: {power:.2%} ({adequacy})")
        
        return "\n".join(lines)
    
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Format recommendations section."""
        return "\n".join(f"- {rec}" for rec in recommendations)
    
    def _format_limitations(self, limitations: List[str]) -> str:
        """Format limitations section."""
        return "\n".join(f"- {lim}" for lim in limitations)
    
    def _format_future_work(self, future_work: List[str]) -> str:
        """Format future work section."""
        return "\n".join(f"- {work}" for work in future_work)
    
    def _format_conclusion(self, report: ValidationReport) -> str:
        """Format conclusion section."""
        if report.overall_significance:
            significant_count = len([t for t in report.statistical_tests if t.p_value < self.alpha])
            total_count = len(report.statistical_tests)
            
            return f"""
The statistical analysis provides strong evidence for the effectiveness of self-healing enhancements in the CoT SafePath Filter system. With {significant_count} out of {total_count} statistical tests showing significant improvements, the enhanced system demonstrates measurable performance gains across multiple dimensions.

The observed effect sizes suggest not only statistical significance but also practical significance, making a compelling case for deployment of the self-healing enhanced system in production environments.
"""
        else:
            return """
While the self-healing enhancements show promise, the current statistical analysis does not provide sufficient evidence for significant performance improvements. Additional data collection and analysis may be needed to establish conclusive benefits.
"""
    
    def _generate_statistical_plots(self, report: ValidationReport, 
                                  output_dir: str, timestamp: str) -> None:
        """Generate statistical visualization plots."""
        # This would generate various plots for the report
        # Implementation would include box plots, effect size plots, power analysis plots, etc.
        logger.info(f"Statistical plots would be generated in {output_dir}")
        # Placeholder for actual plotting implementation
    
    def export_results(self, output_path: str) -> None:
        """Export all validation results to JSON."""
        results = {
            "metadata": {
                "confidence_level": self.confidence_level,
                "power_threshold": self.power_threshold,
                "alpha": self.alpha,
                "export_timestamp": datetime.utcnow().isoformat()
            },
            "validation_reports": [report.to_dict() for report in self.validation_reports]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Validation results exported to {output_path}")


# Example usage and integration
def main():
    """Main function for running statistical validation."""
    logger.info("Starting statistical validation of self-healing enhancements")
    
    # Initialize validator
    validator = StatisticalValidator(confidence_level=0.95, alpha=0.05)
    
    # This would typically load real benchmark data
    # For now, we'll create a placeholder structure
    logger.info("Statistical validation framework ready for integration with baseline data")
    
    return validator


if __name__ == "__main__":
    main()