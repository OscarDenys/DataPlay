"""
Implementation of paired statistical tests including Wilcoxon signed-rank test.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@dataclass
class PairedTestResult:
    """Results from paired statistical tests"""
    test_name: str
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Optional[Tuple[float, float]]
    additional_metrics: Dict[str, Any]
    figures: List[go.Figure]
    interpretation: str
    assumptions_summary: Dict[str, Any]

class PairedTestBase:
    """Base class for paired statistical tests"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def check_sample_size(self, n_samples: int) -> Dict[str, Any]:
        """Check if sample size is adequate"""
        return {
            'passed': n_samples >= 20,
            'warning': "Small sample size may affect reliability" if n_samples < 20 else None,
            'details': f"Sample size: {n_samples} (recommended: ≥20)"
        }
    
    def check_missing_values(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """Check for missing values in paired data"""
        missing1 = np.isnan(data1).sum()
        missing2 = np.isnan(data2).sum()
        return {
            'passed': missing1 == 0 and missing2 == 0,
            'warning': f"Found {missing1 + missing2} missing values" if missing1 + missing2 > 0 else None,
            'details': f"Missing values: {missing1} in first group, {missing2} in second group"
        }
    
    def create_difference_plot(self, 
                             data1: np.ndarray, 
                             data2: np.ndarray, 
                             labels: Tuple[str, str]) -> go.Figure:
        """Create plot showing differences between paired measurements"""
        differences = data2 - data1
        
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Paired Measurements', 'Differences'))
        
        # Paired measurements plot
        fig.add_trace(
            go.Scatter(x=list(range(len(data1))), 
                      y=data1,
                      name=labels[0],
                      mode='lines+markers'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(data2))), 
                      y=data2,
                      name=labels[1],
                      mode='lines+markers'),
            row=1, col=1
        )
        
        # Differences plot
        fig.add_trace(
            go.Box(y=differences, name='Differences',
                  boxpoints='all', jitter=0.3, pointpos=-1.8),
            row=2, col=1
        )
        
        # Add horizontal line at zero for differences
        fig.add_hline(y=0, line_dash="dash", line_color="red",
                     annotation_text="No difference",
                     row=2, col=1)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            title_text="Paired Data Analysis"
        )
        
        return fig

    def create_qq_plot(self, differences: np.ndarray) -> go.Figure:
        """Create Q-Q plot for differences"""
        fig = go.Figure()
        
        # Calculate theoretical quantiles
        sorted_data = np.sort(differences)
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(differences))
        )
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_data,
            mode='markers',
            name='Data Points'
        ))
        
        # Add reference line
        min_val = min(theoretical_quantiles)
        max_val = max(theoretical_quantiles)
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val * np.std(differences) + np.mean(differences),
               max_val * np.std(differences) + np.mean(differences)],
            mode='lines',
            name='Reference Line',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Q-Q Plot of Differences',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            showlegend=True
        )
        
        return fig

class WilcoxonTest(PairedTestBase):
    """Implementation of Wilcoxon signed-rank test"""
    
    def check_assumptions(self, 
                         data1: np.ndarray, 
                         data2: np.ndarray) -> Dict[str, Any]:
        """Check assumptions for Wilcoxon signed-rank test"""
        n_samples = len(data1)
        differences = data2 - data1
        
        assumptions = {
            'sample_size': self.check_sample_size(n_samples),
            'missing_values': self.check_missing_values(data1, data2),
            'paired_data': {
                'passed': len(data1) == len(data2),
                'warning': "Unequal sample sizes" if len(data1) != len(data2) else None,
                'details': f"Group 1: {len(data1)}, Group 2: {len(data2)}"
            },
            'symmetry': {
                'passed': self._check_symmetry(differences),
                'warning': "Differences may not be symmetric around median" 
                          if not self._check_symmetry(differences) else None,
                'details': "Checked symmetry of differences around median"
            }
        }
        
        all_passed = all(check['passed'] for check in assumptions.values())
        warnings = [check['warning'] for check in assumptions.values() 
                   if check['warning'] is not None]
        
        return {
            'assumptions': assumptions,
            'all_passed': all_passed,
            'warnings': warnings
        }
    
    def _check_symmetry(self, differences: np.ndarray) -> bool:
        """Check if differences are approximately symmetric around median"""
        median = np.median(differences)
        centered = differences - median
        pos_mean = np.mean(np.abs(centered[centered > 0]))
        neg_mean = np.mean(np.abs(centered[centered < 0]))
        
        if np.isnan(pos_mean) or np.isnan(neg_mean):
            return False
        
        # Allow for some asymmetry
        return abs(pos_mean - neg_mean) / (pos_mean + neg_mean) < 0.2
    
    def run_test(self,
                 data1: np.ndarray,
                 data2: np.ndarray,
                 labels: Tuple[str, str] = ('Group 1', 'Group 2')) -> PairedTestResult:
        """
        Run Wilcoxon signed-rank test on paired data.
        
        Parameters:
        -----------
        data1, data2 : np.ndarray
            Paired measurements
        labels : tuple of str
            Labels for the two groups
        
        Returns:
        --------
        PairedTestResult
            Complete test results including statistics and visualizations
        """
        # Check assumptions first
        assumptions = self.check_assumptions(data1, data2)
        
        # Calculate differences
        differences = data2 - data1
        n_samples = len(differences)
        
        # Run Wilcoxon test
        statistic, p_value = stats.wilcoxon(differences)
        
        # Calculate effect size (r = Z / sqrt(N))
        z_score = (statistic - ((n_samples * (n_samples + 1)) / 4)) / np.sqrt(
            (n_samples * (n_samples + 1) * (2 * n_samples + 1)) / 24
        )
        effect_size = abs(z_score) / np.sqrt(n_samples)
        
        # Calculate additional metrics
        additional_metrics = {
            'n_samples': n_samples,
            'median_difference': np.median(differences),
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            'positive_differences': np.sum(differences > 0),
            'negative_differences': np.sum(differences < 0),
            'zero_differences': np.sum(differences == 0)
        }
        
        # Create visualizations
        figures = [
            self.create_difference_plot(data1, data2, labels),
            self.create_qq_plot(differences)
        ]
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            p_value, effect_size, additional_metrics, labels
        )
        
        # Calculate confidence interval using bootstrapping
        ci = self._bootstrap_confidence_interval(differences)
        
        return PairedTestResult(
            test_name="Wilcoxon Signed-Rank Test",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            additional_metrics=additional_metrics,
            figures=figures,
            interpretation=interpretation,
            assumptions_summary=assumptions
        )
    
    def _bootstrap_confidence_interval(self, 
                                    differences: np.ndarray,
                                    n_bootstrap: int = 1000,
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for median difference"""
        rng = np.random.default_rng()
        n_samples = len(differences)
        
        # Generate bootstrap samples
        bootstrap_medians = np.array([
            np.median(rng.choice(differences, size=n_samples, replace=True))
            for _ in range(n_bootstrap)
        ])
        
        # Calculate confidence interval
        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(bootstrap_medians, alpha * 100)
        ci_upper = np.percentile(bootstrap_medians, (1 - alpha) * 100)
        
        return ci_lower, ci_upper
    
    def _generate_interpretation(self,
                               p_value: float,
                               effect_size: float,
                               metrics: Dict[str, Any],
                               labels: Tuple[str, str]) -> str:
        """Generate detailed interpretation of test results"""
        parts = []
        
        # Significance interpretation
        if p_value < self.alpha:
            parts.append(
                f"There is a statistically significant difference between {labels[0]} "
                f"and {labels[1]} (p = {p_value:.4f})."
            )
        else:
            parts.append(
                f"There is no statistically significant difference between {labels[0]} "
                f"and {labels[1]} (p = {p_value:.4f})."
            )
        
        # Effect size interpretation
        effect_size_desc = (
            "negligible" if effect_size < 0.1 else
            "small" if effect_size < 0.3 else
            "medium" if effect_size < 0.5 else
            "large"
        )
        parts.append(f"The effect size is {effect_size_desc} (r = {effect_size:.3f}).")
        
        # Direction and magnitude
        median_diff = metrics['median_difference']
        if median_diff != 0:
            direction = "higher" if median_diff > 0 else "lower"
            parts.append(
                f"The median difference is {abs(median_diff):.3f} units "
                f"({labels[1]} is {direction} than {labels[0]})."
            )
        
        # Pattern of differences
        parts.append(
            f"Out of {metrics['n_samples']} pairs, {metrics['positive_differences']} showed "
            f"increases, {metrics['negative_differences']} showed decreases, and "
            f"{metrics['zero_differences']} showed no change."
        )
        
        return " ".join(parts)

# Helper functions for data validation and preprocessing
def validate_paired_data(data1: np.ndarray, 
                        data2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and preprocess paired data"""
    if len(data1) != len(data2):
        raise ValueError("Paired data must have equal lengths")
    
    # Convert to numpy arrays if needed
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    
    # Check data types
    if not (np.issubdtype(data1.dtype, np.number) and 
            np.issubdtype(data2.dtype, np.number)):
        raise ValueError("Data must be numeric")
    
    # Remove paired missing values
    mask = ~(np.isnan(data1) | np.isnan(data2))
    data1 = data1[mask]
    data2 = data2[mask]
    
    if len(data1) == 0:
        raise ValueError("No valid paired data after removing missing values")
    
    return data1, data2

class PairedTTest(PairedTestBase):
    """Implementation of paired t-test"""
    
    def check_assumptions(self, 
                         data1: np.ndarray, 
                         data2: np.ndarray) -> Dict[str, Any]:
        """Check assumptions for paired t-test"""
        n_samples = len(data1)
        differences = data2 - data1
        
        # Test normality of differences
        _, normality_p = stats.shapiro(differences)
        
        assumptions = {
            'sample_size': self.check_sample_size(n_samples),
            'missing_values': self.check_missing_values(data1, data2),
            'paired_data': {
                'passed': len(data1) == len(data2),
                'warning': "Unequal sample sizes" if len(data1) != len(data2) else None,
                'details': f"Group 1: {len(data1)}, Group 2: {len(data2)}"
            },
            'normality': {
                'passed': normality_p >= 0.05,
                'warning': "Differences are not normally distributed" 
                          if normality_p < 0.05 else None,
                'details': f"Shapiro-Wilk test p-value: {normality_p:.4f}"
            }
        }
        
        all_passed = all(check['passed'] for check in assumptions.values())
        warnings = [check['warning'] for check in assumptions.values() 
                   if check['warning'] is not None]
        
        if not assumptions['normality']['passed']:
            warnings.append(
                "Consider using Wilcoxon signed-rank test as non-parametric alternative"
            )
        
        return {
            'assumptions': assumptions,
            'all_passed': all_passed,
            'warnings': warnings
        }
    
    def run_test(self,
                 data1: np.ndarray,
                 data2: np.ndarray,
                 labels: Tuple[str, str] = ('Group 1', 'Group 2')) -> PairedTestResult:
        """
        Run paired t-test.
        
        Parameters:
        -----------
        data1, data2 : np.ndarray
            Paired measurements
        labels : tuple of str
            Labels for the two groups
        
        Returns:
        --------
        PairedTestResult
            Complete test results including statistics and visualizations
        """
        # Check assumptions first
        assumptions = self.check_assumptions(data1, data2)
        
        # Calculate differences
        differences = data2 - data1
        n_samples = len(differences)
        
        # Run paired t-test
        statistic, p_value = stats.ttest_rel(data1, data2)
        
        # Calculate effect size (Cohen's d for paired samples)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # Calculate confidence interval
        ci = stats.t.interval(
            1 - self.alpha,
            df=n_samples - 1,
            loc=np.mean(differences),
            scale=stats.sem(differences)
        )
        
        # Calculate additional metrics
        additional_metrics = {
            'n_samples': n_samples,
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences, ddof=1),
            'sem_difference': stats.sem(differences),
            'degrees_of_freedom': n_samples - 1,
            'positive_differences': np.sum(differences > 0),
            'negative_differences': np.sum(differences < 0),
            'zero_differences': np.sum(differences == 0)
        }
        
        # Create visualizations
        figures = [
            self.create_difference_plot(data1, data2, labels),
            self.create_qq_plot(differences)
        ]
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            p_value, effect_size, additional_metrics, ci, labels
        )
        
        return PairedTestResult(
            test_name="Paired t-test",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            additional_metrics=additional_metrics,
            figures=figures,
            interpretation=interpretation,
            assumptions_summary=assumptions
        )
    
    def _generate_interpretation(self,
                               p_value: float,
                               effect_size: float,
                               metrics: Dict[str, Any],
                               ci: Tuple[float, float],
                               labels: Tuple[str, str]) -> str:
        """Generate detailed interpretation of test results"""
        parts = []
        
        # Significance interpretation
        if p_value < self.alpha:
            parts.append(
                f"There is a statistically significant difference between {labels[0]} "
                f"and {labels[1]} (t({metrics['degrees_of_freedom']}) = "
                f"{metrics['test_statistic']:.3f}, p = {p_value:.4f})."
            )
        else:
            parts.append(
                f"There is no statistically significant difference between {labels[0]} "
                f"and {labels[1]} (t({metrics['degrees_of_freedom']}) = "
                f"{metrics['test_statistic']:.3f}, p = {p_value:.4f})."
            )
        
        # Effect size interpretation
        effect_size_desc = (
            "negligible" if abs(effect_size) < 0.2 else
            "small" if abs(effect_size) < 0.5 else
            "medium" if abs(effect_size) < 0.8 else
            "large"
        )
        parts.append(
            f"The effect size is {effect_size_desc} (Cohen's d = {effect_size:.3f})."
        )
        
        # Mean difference and confidence interval
        mean_diff = metrics['mean_difference']
        direction = "higher" if mean_diff > 0 else "lower"
        parts.append(
            f"On average, {labels[1]} is {abs(mean_diff):.3f} units {direction} than "
            f"{labels[0]} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])."
        )
        
        # Pattern of differences
        parts.append(
            f"Out of {metrics['n_samples']} pairs, {metrics['positive_differences']} showed "
            f"increases, {metrics['negative_differences']} showed decreases, and "
            f"{metrics['zero_differences']} showed no change."
        )
        
        return " ".join(parts)

def get_test_description(test_type: str) -> Dict[str, Any]:
    """Get comprehensive description of paired statistical tests"""
    descriptions = {
        "Paired t-test": {
            "description": "Compare means between two related groups",
            "assumptions": [
                "Paired observations",
                "Normally distributed differences",
                "Continuous measurements",
                "No significant outliers"
            ],
            "use_cases": [
                "Before-after measurements",
                "Matched pairs analysis",
                "Repeated measurements on same subjects",
                "Comparing methods/techniques on same samples"
            ],
            "example": (
                "Testing whether a training program improves test scores by comparing "
                "each student's score before and after training"
            ),
            "interpretation_guide": {
                "p_value": "Probability of observing such differences by chance",
                "effect_size": "Cohen's d: <0.2 (negligible), 0.2-0.5 (small), "
                             "0.5-0.8 (medium), >0.8 (large)",
                "confidence_interval": "Range of plausible values for true mean difference"
            },
            "alternatives": [
                "Wilcoxon signed-rank test (for non-normal differences)",
                "Sign test (for highly skewed differences)"
            ]
        },
        "Wilcoxon Signed-Rank Test": {
            "description": "Compare paired measurements without normality assumption",
            "assumptions": [
                "Paired observations",
                "Symmetric distribution of differences",
                "Ordinal or continuous measurements",
                "Independent pairs"
            ],
            "use_cases": [
                "Non-normally distributed differences",
                "Ordinal measurements",
                "Small sample sizes",
                "Presence of outliers"
            ],
            "example": (
                "Comparing pain levels before and after treatment, where pain is "
                "measured on an ordinal scale"
            ),
            "interpretation_guide": {
                "p_value": "Probability of observing such differences by chance",
                "effect_size": "r: <0.1 (negligible), 0.1-0.3 (small), "
                             "0.3-0.5 (medium), >0.5 (large)",
                "confidence_interval": "Range of plausible values for median difference"
            },
            "alternatives": [
                "Paired t-test (for normal differences)",
                "Sign test (for non-symmetric differences)"
            ]
        }
    }
    
    return descriptions.get(test_type, {})

def recommend_paired_test(data1: np.ndarray, 
                        data2: np.ndarray) -> str:
    """
    Recommend appropriate paired test based on data characteristics.
    
    Parameters:
    -----------
    data1, data2 : np.ndarray
        Paired measurements
    
    Returns:
    --------
    str
        Recommended test type with explanation
    """
    try:
        # Validate and clean data
        data1, data2 = validate_paired_data(data1, data2)
        differences = data2 - data1
        
        # Check sample size
        n_samples = len(differences)
        if n_samples < 10:
            return (
                "Sign Test (small sample size). However, with such a small sample, "
                "consider collecting more data if possible."
            )
        
        # Check normality of differences
        _, normality_p = stats.shapiro(differences)
        
        # Check symmetry of differences
        median = np.median(differences)
        centered = differences - median
        pos_mean = np.mean(np.abs(centered[centered > 0]))
        neg_mean = np.mean(np.abs(centered[centered < 0]))
        is_symmetric = abs(pos_mean - neg_mean) / (pos_mean + neg_mean) < 0.2
        
        if normality_p >= 0.05:
            return (
                "Paired t-test (differences appear normally distributed). "
                "This test will provide maximum statistical power."
            )
        elif is_symmetric:
            return (
                "Wilcoxon Signed-Rank Test (differences not normal but symmetric). "
                "This test provides a robust alternative to the paired t-test."
            )
        else:
            return (
                "Sign Test (differences not normal and not symmetric). "
                "This test makes minimal assumptions but may have less statistical power."
            )
            
    except Exception as e:
        return f"Unable to make recommendation due to error: {str(e)}"

class SignTest(PairedTestBase):
    """Implementation of Sign test for paired data"""
    
    def check_assumptions(self, 
                         data1: np.ndarray, 
                         data2: np.ndarray) -> Dict[str, Any]:
        """Check assumptions for Sign test"""
        n_samples = len(data1)
        differences = data2 - data1
        
        assumptions = {
            'sample_size': self.check_sample_size(n_samples),
            'missing_values': self.check_missing_values(data1, data2),
            'paired_data': {
                'passed': len(data1) == len(data2),
                'warning': "Unequal sample sizes" if len(data1) != len(data2) else None,
                'details': f"Group 1: {len(data1)}, Group 2: {len(data2)}"
            },
            'zero_differences': {
                'passed': np.sum(differences == 0) / len(differences) < 0.25,
                'warning': "Large proportion of zero differences" 
                          if np.sum(differences == 0) / len(differences) >= 0.25 else None,
                'details': f"Zero differences: {np.sum(differences == 0)} "
                          f"({np.sum(differences == 0) / len(differences):.1%})"
            }
        }
        
        all_passed = all(check['passed'] for check in assumptions.values())
        warnings = [check['warning'] for check in assumptions.values() 
                   if check['warning'] is not None]
        
        return {
            'assumptions': assumptions,
            'all_passed': all_passed,
            'warnings': warnings
        }
    
    def run_test(self,
                 data1: np.ndarray,
                 data2: np.ndarray,
                 labels: Tuple[str, str] = ('Group 1', 'Group 2')) -> PairedTestResult:
        """
        Run Sign test on paired data.
        
        Parameters:
        -----------
        data1, data2 : np.ndarray
            Paired measurements
        labels : tuple of str
            Labels for the two groups
        
        Returns:
        --------
        PairedTestResult
            Complete test results including statistics and visualizations
        """
        # Check assumptions first
        assumptions = self.check_assumptions(data1, data2)
        
        # Calculate differences
        differences = data2 - data1
        n_samples = len(differences)
        
        # Get signs of differences (excluding zeros)
        nonzero_diffs = differences[differences != 0]
        pos_signs = np.sum(nonzero_diffs > 0)
        n_nonzero = len(nonzero_diffs)
        
        # Run Sign test using binomial test
        p_value = 2 * stats.binom.cdf(min(pos_signs, n_nonzero - pos_signs), 
                                    n_nonzero, 0.5)
        
        # Calculate effect size (proportion of positive differences - 0.5)
        effect_size = abs(pos_signs/n_nonzero - 0.5)
        
        # Calculate confidence interval for median difference using bootstrap
        ci = self._bootstrap_confidence_interval(differences)
        
        # Calculate additional metrics
        additional_metrics = {
            'n_samples': n_samples,
            'n_nonzero': n_nonzero,
            'median_difference': np.median(differences),
            'positive_differences': np.sum(differences > 0),
            'negative_differences': np.sum(differences < 0),
            'zero_differences': np.sum(differences == 0),
            'proportion_positive': pos_signs/n_nonzero if n_nonzero > 0 else 0
        }
        
        # Create visualizations
        figures = [
            self.create_difference_plot(data1, data2, labels),
            self._create_sign_distribution_plot(differences)
        ]
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            p_value, effect_size, additional_metrics, ci, labels
        )
        
        return PairedTestResult(
            test_name="Sign Test",
            test_statistic=pos_signs,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            additional_metrics=additional_metrics,
            figures=figures,
            interpretation=interpretation,
            assumptions_summary=assumptions
        )
    
    def _create_sign_distribution_plot(self, differences: np.ndarray) -> go.Figure:
        """Create visualization of difference signs"""
        signs = np.sign(differences)
        sign_counts = {
            'Positive': np.sum(signs > 0),
            'Negative': np.sum(signs < 0),
            'Zero': np.sum(signs == 0)
        }
        
        fig = go.Figure()
        
        # Add bars
        colors = {'Positive': 'green', 'Negative': 'red', 'Zero': 'gray'}
        for sign, count in sign_counts.items():
            fig.add_trace(go.Bar(
                x=[sign],
                y=[count],
                name=sign,
                marker_color=colors[sign]
            ))
        
        fig.update_layout(
            title='Distribution of Difference Signs',
            xaxis_title='Sign of Difference',
            yaxis_title='Count',
            showlegend=False,
            bargap=0.2
        )
        
        return fig
    
    def _generate_interpretation(self,
                               p_value: float,
                               effect_size: float,
                               metrics: Dict[str, Any],
                               ci: Tuple[float, float],
                               labels: Tuple[str, str]) -> str:
        """Generate detailed interpretation of test results"""
        parts = []
        
        # Significance interpretation
        if p_value < self.alpha:
            parts.append(
                f"There is a statistically significant difference between {labels[0]} "
                f"and {labels[1]} (p = {p_value:.4f})."
            )
        else:
            parts.append(
                f"There is no statistically significant difference between {labels[0]} "
                f"and {labels[1]} (p = {p_value:.4f})."
            )
        
        # Effect size interpretation
        effect_size_desc = (
            "negligible" if effect_size < 0.1 else
            "small" if effect_size < 0.3 else
            "medium" if effect_size < 0.5 else
            "large"
        )
        prop_positive = metrics['proportion_positive']
        parts.append(
            f"The effect size is {effect_size_desc} "
            f"({prop_positive:.1%} of non-zero differences were positive)."
        )
        
        # Difference pattern
        median_diff = metrics['median_difference']
        direction = "higher" if median_diff > 0 else "lower"
        parts.append(
            f"The median difference is {abs(median_diff):.3f} units "
            f"({labels[1]} is {direction} than {labels[0]}, "
            f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])."
        )
        
        # Detailed counts
        parts.append(
            f"Out of {metrics['n_samples']} pairs, {metrics['positive_differences']} showed "
            f"increases, {metrics['negative_differences']} showed decreases, and "
            f"{metrics['zero_differences']} showed no change."
        )
        
        return " ".join(parts)

def compare_paired_tests(data1: np.ndarray,
                        data2: np.ndarray,
                        labels: Tuple[str, str] = ('Group 1', 'Group 2'),
                        alpha: float = 0.05) -> Dict[str, Any]:
    """
    Compare results from different paired tests.
    
    Parameters:
    -----------
    data1, data2 : np.ndarray
        Paired measurements
    labels : tuple of str
        Labels for the two groups
    alpha : float
        Significance level
    
    Returns:
    --------
    Dict[str, Any]
        Comparison of test results and recommendations
    """
    # Initialize tests
    tests = {
        'Paired t-test': PairedTTest(alpha=alpha),
        'Wilcoxon Signed-Rank Test': WilcoxonTest(alpha=alpha),
        'Sign Test': SignTest(alpha=alpha)
    }
    
    # Run all tests
    results = {}
    for name, test in tests.items():
        try:
            results[name] = test.run_test(data1, data2, labels)
        except Exception as e:
            results[name] = f"Error running {name}: {str(e)}"
    
    # Get test recommendation
    recommended_test = recommend_paired_test(data1, data2)
    
    # Create comparison summary
    summary = {
        'recommendation': recommended_test,
        'results': results,
        'agreement': _check_test_agreement(results),
        'sensitivity': _compare_test_sensitivity(results)
    }
    
    return summary

def _check_test_agreement(results: Dict[str, PairedTestResult]) -> str:
    """Check if different tests agree on significance"""
    try:
        significant = {name: result.p_value < 0.05 
                      for name, result in results.items()
                      if isinstance(result, PairedTestResult)}
        
        if len(set(significant.values())) == 1:
            return (
                "All tests agree on " + 
                ("significance" if list(significant.values())[0] else "non-significance")
            )
        else:
            disagreements = [
                f"{name1} and {name2} disagree"
                for name1, sig1 in significant.items()
                for name2, sig2 in significant.items()
                if sig1 != sig2 and name1 < name2
            ]
            return "Tests disagree: " + "; ".join(disagreements)
    except Exception:
        return "Unable to check test agreement"

def _compare_test_sensitivity(results: Dict[str, PairedTestResult]) -> str:
    """Compare sensitivity of different tests"""
    try:
        p_values = {name: result.p_value 
                   for name, result in results.items()
                   if isinstance(result, PairedTestResult)}
        
        if not p_values:
            return "No valid test results for comparison"
        
        min_p = min(p_values.items(), key=lambda x: x[1])
        
        return (
            f"{min_p[0]} shows the strongest evidence "
            f"(p = {min_p[1]:.4f})"
        )
    except Exception:
        return "Unable to compare test sensitivity"

# Utility function for loading paired data from streamlit uploaded files
def load_paired_data(file, column1: str, column2: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load paired data from uploaded file in Streamlit.
    
    Parameters:
    -----------
    file : streamlit.UploadedFile
        Uploaded file object
    column1, column2 : str
        Names of columns containing paired measurements
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Paired data arrays
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel file.")
        
        if column1 not in df.columns or column2 not in df.columns:
            raise ValueError(f"Columns {column1} and {column2} not found in file")
        
        data1 = df[column1].values
        data2 = df[column2].values
        
        return validate_paired_data(data1, data2)
    
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")