"""
Implementation of statistical tests for group comparisons (t-test, ANOVA).
Provides robust implementations with comprehensive error checking and user feedback.
"""

import numpy as np
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, levene
from scikit_posthocs import posthoc_dunn
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from core import TestResult, StatisticalTestBase
from assumption_checking import (
    AssumptionChecker,
    AssumptionResult,
    AssumptionCheck  
)
from visualizations import TestVisualizations

class StudentTTest(StatisticalTestBase):
    """Implementation of Student's t-test with comprehensive analysis."""
    
    def run_test(self, data: Dict[str, Any]) -> TestResult:
        """
        Run Student's t-test with effect size calculation and visualization.
        
        Args:
            data: Dictionary containing:
                - group1: First group's data
                - group2: Second group's data
                - group_labels: Optional group names
                - alpha: Optional significance level
        
        Returns:
            TestResult object containing test results and visualizations
        """
        # Extract and validate data
        group1 = data['group1']
        group2 = data['group2']
        group_labels = data.get('group_labels', ['Group 1', 'Group 2'])
        alpha = data.get('alpha', 0.05)
        
        # Remove missing values
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        # Perform t-test
        stat, p_val = ttest_ind(group1, group2)
        
        # Calculate Cohen's d effect size
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        effect_size = abs(np.mean(group1) - np.mean(group2)) / pooled_se
        
        # Calculate confidence interval
        dof = n1 + n2 - 2
        mean_diff = np.mean(group1) - np.mean(group2)
        se_diff = np.sqrt(var1/n1 + var2/n2)
        t_crit = stats.t.ppf(1 - alpha/2, dof)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        
        # Generate visualizations
        viz = TestVisualizations()
        plots = []
        
        # Distribution comparison
        plots.append(viz.create_distribution_comparison(
            {group_labels[0]: group1, group_labels[1]: group2},
            "Group Distributions"
        ))
        
        # Box plot
        plots.append(viz.create_box_plot(
            {group_labels[0]: group1, group_labels[1]: group2},
            "Group Comparison"
        ))
        
        # Effect size visualization
        plots.append(viz.create_effect_size_plot(
            effect_size,
            "Cohen's d",
            ci_lower,
            ci_upper
        ))
        
        # Significance visualization
        plots.append(viz.create_significance_plot(
            p_val,
            alpha,
            "Student's t-test"
        ))
        
        # Prepare additional metrics
        additional_metrics = {
            'mean_difference': mean_diff,
            'confidence_interval': (ci_lower, ci_upper),
            'degrees_of_freedom': dof,
            'group_statistics': {
                group_labels[0]: {
                    'n': n1,
                    'mean': np.mean(group1),
                    'std': np.std(group1, ddof=1),
                    'sem': stats.sem(group1)
                },
                group_labels[1]: {
                    'n': n2,
                    'mean': np.mean(group2),
                    'std': np.std(group2, ddof=1),
                    'sem': stats.sem(group2)
                }
            }
        }
        
        # Generate interpretation
        if p_val < alpha:
            direction = "greater" if mean_diff > 0 else "less"
            interpretation = (
                f"There is a significant difference between {group_labels[0]} and "
                f"{group_labels[1]} (t({dof:.0f}) = {stat:.2f}, p = {p_val:.4f}). "
                f"The mean of {group_labels[0]} is {direction} than {group_labels[1]} "
                f"by {abs(mean_diff):.2f} units. The effect size is {self._get_effect_size_interpretation(effect_size)}. "
                f"We can be {(1-alpha)*100:.0f}% confident that the true difference lies "
                f"between {ci_lower:.2f} and {ci_upper:.2f}."
            )
        else:
            interpretation = (
                f"There is no significant difference between {group_labels[0]} and "
                f"{group_labels[1]} (t({dof:.0f}) = {stat:.2f}, p = {p_val:.4f}). "
                f"The observed mean difference of {mean_diff:.2f} units is not "
                f"statistically significant at α = {alpha}."
            )
        
        return TestResult(
            test_statistic=stat,
            p_value=p_val,
            effect_size=effect_size,
            interpretation=interpretation,
            additional_metrics=additional_metrics,
            plots=plots
        )
    
    def check_assumptions(self, data: Dict[str, Any]) -> AssumptionCheck:
        """
        Check t-test assumptions.
        
        Args:
            data: Dictionary containing group data
            
        Returns:
            AssumptionCheck object with results and recommendations
        """
        group1 = data['group1']
        group2 = data['group2']
        group_labels = data.get('group_labels', ['Group 1', 'Group 2'])
        
        checker = AssumptionChecker()
        
        # Initialize lists for results
        checks = []
        warnings = []
        recommendations = []
        
        # Check normality for each group
        for group, label in zip([group1, group2], group_labels):
            norm_check = checker.check_normality(group, group_name=label)
            checks.append(norm_check)
            
            if norm_check.result != "Pass":
                warnings.append(f"Non-normal distribution in {label}")
                recommendations.append("Consider Mann-Whitney U test")
        
        # Check homogeneity of variance
        var_check = checker.check_homogeneity(group1, group2)
        checks.append(var_check)
        
        if var_check.result != "Pass":
            warnings.append("Unequal variances between groups")
            recommendations.append("Consider Welch's t-test")
        
        # Check sample sizes
        for group, label in zip([group1, group2], group_labels):
            size_check = checker.check_sample_size(group, group_name=label)
            checks.append(size_check)
            
            if size_check.result != "Pass":
                warnings.append(f"Small sample size in {label}")
                recommendations.append("Consider non-parametric alternatives")
        
        # Check outliers
        for group, label in zip([group1, group2], group_labels):
            outlier_check = checker.check_outliers(group, group_name=label)
            checks.append(outlier_check)
            
            if outlier_check.result != "Pass":
                warnings.append(f"Potential outliers detected in {label}")
                recommendations.append("Investigate outliers and consider their impact")
        
        # Determine if test should proceed
        proceed = True  # Allow proceeding with warnings
        
        return AssumptionCheck(
            checks=checks,
            recommendations=list(set(recommendations)),
            warnings=warnings,
            proceed=proceed
        )
    
    def create_visualizations(self, data: Dict[str, Any], result: TestResult) -> List[Any]:
        """Return visualizations already created in run_test."""
        return result.plots
    
    def _get_effect_size_interpretation(self, effect_size: float) -> str:
        """Get interpretation of Cohen's d effect size."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"

class AnovaTest(StatisticalTestBase):
    """Implementation of One-way ANOVA with comprehensive analysis."""
    
    def run_test(self, data: Dict[str, Any]) -> TestResult:
        """
        Run one-way ANOVA with post-hoc tests and visualizations.
        
        Args:
            data: Dictionary containing:
                - groups: List of arrays containing group data
                - group_labels: Optional list of group names
                - alpha: Optional significance level
        
        Returns:
            TestResult object containing test results and visualizations
        """
        # Extract and validate data
        groups = data['groups']
        group_labels = data.get('group_labels', [f"Group {i+1}" for i in range(len(groups))])
        alpha = data.get('alpha', 0.05)
        
        # Remove missing values from each group
        groups = [group[~np.isnan(group)] for group in groups]
        
        # Perform ANOVA
        stat, p_val = f_oneway(*groups)
        
        # Calculate effect size (eta-squared)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        # Between groups sum of squares
        between_ss = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        
        # Total sum of squares
        total_ss = sum((x - grand_mean)**2 for g in groups for x in g)
        
        # Calculate effect size
        effect_size = between_ss / total_ss if total_ss != 0 else 0
        
        # Perform post-hoc tests if ANOVA is significant
        posthoc_results = None
        if p_val < alpha:
            try:
                # Perform Dunn's test with Bonferroni correction
                posthoc_results = posthoc_dunn(
                    all_data,
                    np.concatenate([[i] * len(g) for i, g in enumerate(groups)]),
                    p_adjust='bonferroni'
                )
                
                # Replace numeric indices with group labels
                posthoc_results.index = group_labels
                posthoc_results.columns = group_labels
                
            except Exception as e:
                posthoc_results = f"Post-hoc analysis failed: {str(e)}"
        
        # Generate visualizations
        viz = TestVisualizations()
        plots = []
        
        # Distribution comparison
        group_dict = dict(zip(group_labels, groups))
        plots.append(viz.create_distribution_comparison(
            group_dict,
            "Group Distributions"
        ))
        
        # Box plot
        plots.append(viz.create_box_plot(
            group_dict,
            "Group Comparison"
        ))
        
        # Effect size visualization
        plots.append(viz.create_effect_size_plot(
            effect_size,
            "Eta-squared"
        ))
        
        # Significance visualization
        plots.append(viz.create_significance_plot(
            p_val,
            alpha,
            "One-way ANOVA"
        ))
        
        # Add post-hoc visualization if available
        if isinstance(posthoc_results, pd.DataFrame):
            import plotly.graph_objects as go
            
            fig = go.Figure(data=go.Heatmap(
                z=posthoc_results.values,
                x=posthoc_results.columns,
                y=posthoc_results.index,
                colorscale='RdBu',
                zmid=alpha,
                text=np.round(posthoc_results.values, 3),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
                colorbar=dict(
                    title='p-value',
                    titleside='right'
                )
            ))
            
            fig.update_layout(
                title='Post-hoc Analysis p-values',
                xaxis_title='Group',
                yaxis_title='Group',
                template='plotly_white'
            )
            
            plots.append(fig)
        
        # Calculate group statistics
        group_stats = {}
        for label, group in zip(group_labels, groups):
            group_stats[label] = {
                'n': len(group),
                'mean': np.mean(group),
                'std': np.std(group, ddof=1),
                'sem': stats.sem(group),
                'ci_95': stats.t.interval(
                    0.95,
                    len(group)-1,
                    loc=np.mean(group),
                    scale=stats.sem(group)
                )
            }
        
        # Prepare additional metrics
        additional_metrics = {
            'group_statistics': group_stats,
            'between_ss': between_ss,
            'total_ss': total_ss,
            'degrees_of_freedom': {
                'between': len(groups) - 1,
                'within': len(all_data) - len(groups),
                'total': len(all_data) - 1
            },
            'posthoc_results': posthoc_results
        }
        
        # Generate interpretation
        if p_val < alpha:
            interpretation = (
                f"There are significant differences between groups "
                f"(F({len(groups)-1}, {len(all_data)-len(groups)}) = {stat:.2f}, "
                f"p = {p_val:.4f}). The effect size (η²) of {effect_size:.3f} indicates a "
                f"{self._get_effect_size_interpretation(effect_size)} effect. "
            )
            
            if isinstance(posthoc_results, pd.DataFrame):
                # Find significant pairwise differences
                sig_pairs = []
                for i in range(len(group_labels)):
                    for j in range(i+1, len(group_labels)):
                        if posthoc_results.iloc[i, j] < alpha:
                            sig_pairs.append(f"{group_labels[i]} vs {group_labels[j]}")
                
                if sig_pairs:
                    interpretation += (
                        f"Post-hoc analysis reveals significant differences between: "
                        f"{', '.join(sig_pairs)}."
                    )
        else:
            interpretation = (
                f"No significant differences were found between groups "
                f"(F({len(groups)-1}, {len(all_data)-len(groups)}) = {stat:.2f}, "
                f"p = {p_val:.4f}). The effect size (η²) of {effect_size:.3f} indicates a "
                f"{self._get_effect_size_interpretation(effect_size)} effect."
            )
        
        return TestResult(
            test_statistic=stat,
            p_value=p_val,
            effect_size=effect_size,
            interpretation=interpretation,
            additional_metrics=additional_metrics,
            plots=plots
        )
    
    def check_assumptions(self, data: Dict[str, Any]) -> AssumptionCheck:
        """
        Check ANOVA assumptions.
        
        Args:
            data: Dictionary containing group data
            
        Returns:
            AssumptionCheck object with results and recommendations
        """
        groups = data['groups']
        group_labels = data.get('group_labels', [f"Group {i+1}" for i in range(len(groups))])
        
        checker = AssumptionChecker()
        
        # Initialize lists for results
        checks = []
        warnings = []
        recommendations = []
        
        # Check normality for each group
        for group, label in zip(groups, group_labels):
            norm_check = checker.check_normality(group, group_name=label)
            checks.append(norm_check)
            
            if norm_check.result != "Pass":
                warnings.append(f"Non-normal distribution in {label}")
                recommendations.append("Consider Kruskal-Wallis test")
        
        # Check homogeneity of variance
        var_check = checker.check_homogeneity(*groups)
        checks.append(var_check)
        
        if var_check.result != "Pass":
            warnings.append("Unequal variances between groups")
            recommendations.append("Consider Welch's ANOVA")
        
        # Check sample sizes
        for group, label in zip(groups, group_labels):
            size_check = checker.check_sample_size(group, group_name=label)
            checks.append(size_check)
            
            if size_check.result != "Pass":
                warnings.append(f"Small sample size in {label}")
                recommendations.append("Consider non-parametric alternatives")
        
        # Check for outliers
        for group, label in zip(groups, group_labels):
            outlier_check = checker.check_outliers(group, group_name=label)
            checks.append(outlier_check)
            
            if outlier_check.result != "Pass":
                warnings.append(f"Potential outliers detected in {label}")
                recommendations.append("Investigate outliers and consider their impact")
        
        # Determine if test should proceed
        proceed = True  # Allow proceeding with warnings
        
        return AssumptionCheck(
            checks=checks,
            recommendations=list(set(recommendations)),
            warnings=warnings,
            proceed=proceed
        )
    
    def create_visualizations(self, data: Dict[str, Any], result: TestResult) -> List[Any]:
        """Return visualizations already created in run_test."""
        return result.plots
    
    def _get_effect_size_interpretation(self, effect_size: float) -> str:
        """Get interpretation of eta-squared effect size."""
        if effect_size < 0.01:
            return "negligible"
        elif effect_size < 0.06:
            return "small"
        elif effect_size < 0.14:
            return "medium"
        else:
            return "large"
