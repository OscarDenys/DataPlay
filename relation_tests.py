"""
Implementation of statistical tests for analyzing relationships between variables.
Provides comprehensive analysis of correlations and associations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, chi2_contingency, fisher_exact, spearmanr
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px

from core import TestResult, StatisticalTestBase
from assumption_checking import AssumptionChecker, AssumptionResult, AssumptionCheck
from visualizations import TestVisualizations

class PearsonCorrelation(StatisticalTestBase):
    """Implementation of Pearson correlation with comprehensive analysis."""
    
    def run_test(self, data: Dict[str, Any]) -> TestResult:
        """
        Run Pearson correlation analysis with visualization.
        
        Args:
            data: Dictionary containing:
                - x: First variable data
                - y: Second variable data
                - x_label: Optional label for x variable
                - y_label: Optional label for y variable
                - alpha: Optional significance level
        
        Returns:
            TestResult object containing correlation results and visualizations
        """
        # Extract and validate data
        x = data['x']
        y = data['y']
        x_label = data.get('x_label', 'Variable 1')
        y_label = data.get('y_label', 'Variable 2')
        alpha = data.get('alpha', 0.05)
        
        # Remove any pairs with missing values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]
        
        # Calculate Pearson correlation
        stat, p_val = pearsonr(x, y)
        
        # Calculate confidence interval using Fisher's Z transformation
        z = np.arctanh(stat)
        se = 1/np.sqrt(len(x)-3)
        ci_lower = np.tanh(z - stats.norm.ppf(1-alpha/2) * se)
        ci_upper = np.tanh(z + stats.norm.ppf(1-alpha/2) * se)
        
        # Generate visualizations
        viz = TestVisualizations()
        plots = []
        
        # Scatter plot with regression line
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            name='Data Points',
            marker=dict(
                color=viz.COLORS['primary'],
                opacity=0.6
            ),
            hovertemplate=(
                f"{x_label}: %{x:.2f}<br>"
                f"{y_label}: %{y:.2f}<br>"
                "<extra></extra>"
            )
        ))
        
        # Add regression line
        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)
        x_sorted = np.sort(x)
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=poly1d_fn(x_sorted),
            mode='lines',
            name='Regression Line',
            line=dict(color=viz.COLORS['secondary']),
            hovertemplate=(
                f"{x_label}: %{x:.2f}<br>"
                f"Predicted {y_label}: %{y:.2f}<br>"
                "<extra></extra>"
            )
        ))
        
        # Update layout
        fig.update_layout(
            title='Scatter Plot with Regression Line',
            xaxis_title=x_label,
            yaxis_title=y_label,
            template='plotly_white'
        )
        
        plots.append(fig)
        
        # Add significance visualization
        plots.append(viz.create_significance_plot(
            p_val,
            alpha,
            "Pearson Correlation"
        ))
        
        # Add effect size visualization
        plots.append(viz.create_effect_size_plot(
            abs(stat),
            "Correlation",
            ci_lower,
            ci_upper
        ))
        
        # Calculate additional metrics
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = stat ** 2
        
        additional_metrics = {
            'correlation_coefficient': stat,
            'r_squared': r_squared,
            'confidence_interval': (ci_lower, ci_upper),
            'regression_line': {
                'slope': slope,
                'intercept': intercept
            },
            'sample_size': len(x),
            'removed_pairs': len(data['x']) - len(x),
            'descriptive_stats': {
                x_label: {
                    'mean': np.mean(x),
                    'std': np.std(x, ddof=1),
                    'min': np.min(x),
                    'max': np.max(x)
                },
                y_label: {
                    'mean': np.mean(y),
                    'std': np.std(y, ddof=1),
                    'min': np.min(y),
                    'max': np.max(y)
                }
            }
        }
        
        # Generate interpretation
        if p_val < alpha:
            direction = "positive" if stat > 0 else "negative"
            interpretation = (
                f"There is a significant {direction} correlation between {x_label} and "
                f"{y_label} (r = {stat:.3f}, p = {p_val:.4f}). The correlation strength is "
                f"{self._get_effect_size_interpretation(abs(stat))}. The coefficient of "
                f"determination (R²) indicates that {(r_squared*100):.1f}% of the variance "
                f"in {y_label} can be explained by {x_label}. We can be {(1-alpha)*100:.0f}% "
                f"confident that the true correlation lies between {ci_lower:.3f} and {ci_upper:.3f}."
            )
        else:
            interpretation = (
                f"There is no significant correlation between {x_label} and {y_label} "
                f"(r = {stat:.3f}, p = {p_val:.4f}). The observed correlation strength is "
                f"{self._get_effect_size_interpretation(abs(stat))}, but this could be due to "
                f"random chance."
            )
        
        return TestResult(
            test_statistic=stat,
            p_value=p_val,
            effect_size=abs(stat),
            interpretation=interpretation,
            additional_metrics=additional_metrics,
            plots=plots
        )
    
    def check_assumptions(self, data: Dict[str, Any]) -> AssumptionCheck:
        """
        Check Pearson correlation assumptions.
        
        Args:
            data: Dictionary containing variable data
            
        Returns:
            AssumptionCheck object with results and recommendations
        """
        x = data['x']
        y = data['y']
        x_label = data.get('x_label', 'Variable 1')
        y_label = data.get('y_label', 'Variable 2')
        
        checker = AssumptionChecker()
        
        # Initialize lists for results
        checks = []
        warnings = []
        recommendations = []
        
        # Check for missing values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        missing_pairs = len(x) - np.sum(valid_mask)
        if missing_pairs > 0:
            warnings.append(f"Found {missing_pairs} pairs with missing values")
            recommendations.append("Consider imputation or pair-wise deletion")
        
        # Check normality for both variables
        for var, label in [(x[valid_mask], x_label), (y[valid_mask], y_label)]:
            norm_check = checker.check_normality(var, label)
            checks.append(norm_check)
            
            if norm_check.result != "Pass":
                warnings.append(f"Non-normal distribution in {label}")
                recommendations.append("Consider Spearman correlation")
        
        # Check linearity
        slope, intercept = np.polyfit(x[valid_mask], y[valid_mask], 1)
        y_pred = slope * x[valid_mask] + intercept
        residuals = y[valid_mask] - y_pred
        r_squared = 1 - (np.sum(residuals**2) / np.sum((y[valid_mask] - np.mean(y[valid_mask]))**2))
        
        linearity_check = {
            'test_name': 'Linearity',
            'result': 'Pass' if r_squared > 0.5 else 'Warning',
            'details': f'R² = {r_squared:.3f}'
        }
        checks.append(linearity_check)
        
        if linearity_check['result'] != "Pass":
            warnings.append("Weak linear relationship")
            recommendations.append("Consider non-linear relationship measures")
        
        # Check for outliers
        z_scores_x = np.abs(stats.zscore(x[valid_mask]))
        z_scores_y = np.abs(stats.zscore(y[valid_mask]))
        outliers_mask = (z_scores_x > 3) | (z_scores_y > 3)
        
        if np.any(outliers_mask):
            warnings.append(f"Found {np.sum(outliers_mask)} potential outliers")
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
    
    def _get_effect_size_interpretation(self, correlation: float) -> str:
        """Get interpretation of correlation strength."""
        if correlation < 0.1:
            return "negligible"
        elif correlation < 0.3:
            return "weak"
        elif correlation < 0.5:
            return "moderate"
        elif correlation < 0.7:
            return "strong"
        else:
            return "very strong"

class ChiSquareTest(StatisticalTestBase):
    """Implementation of Chi-square test for independence with comprehensive analysis."""
    
    def run_test(self, data: Dict[str, Any]) -> TestResult:
        """
        Run Chi-square test with visualization and comprehensive interpretation.
        
        Args:
            data: Dictionary containing:
                - table: Contingency table (pandas DataFrame or numpy array)
                - row_labels: Optional row category labels
                - col_labels: Optional column category labels
                - alpha: Optional significance level
        
        Returns:
            TestResult object containing test results and visualizations
        """
        # Extract and validate data
        table = data['table']
        if isinstance(table, pd.DataFrame):
            contingency_table = table.copy()
        else:
            row_labels = data.get('row_labels', [f"Row {i+1}" for i in range(table.shape[0])])
            col_labels = data.get('col_labels', [f"Col {i+1}" for i in range(table.shape[1])])
            contingency_table = pd.DataFrame(table, index=row_labels, columns=col_labels)
        
        alpha = data.get('alpha', 0.05)
        
        # Perform chi-square test
        chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramer's V effect size
        n = contingency_table.values.sum()
        min_dim = min(contingency_table.shape) - 1
        cramer_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0
        
        # For 2x2 tables, also perform Fisher's exact test
        fisher_results = None
        odds_ratio = None
        if contingency_table.shape == (2, 2):
            odds_ratio, fisher_p = fisher_exact(contingency_table)
            fisher_results = {
                'p_value': fisher_p,
                'odds_ratio': odds_ratio,
                'odds_ratio_interpretation': self._interpret_odds_ratio(odds_ratio)
            }
        
        # Generate visualizations
        viz = TestVisualizations()
        plots = []
        
        # Create heatmap of observed frequencies
        fig_obs = go.Figure(data=go.Heatmap(
            z=contingency_table.values,
            x=contingency_table.columns,
            y=contingency_table.index,
            text=contingency_table.values,
            texttemplate="%{text:.0f}",
            colorscale='YlOrRd',
            showscale=True,
            hoverongaps=False,
            hovertemplate=(
                "Row: %{y}<br>"
                "Column: %{x}<br>"
                "Count: %{z}<br>"
                "<extra></extra>"
            )
        ))
        
        fig_obs.update_layout(
            title='Observed Frequencies',
            xaxis_title="Column Variable",
            yaxis_title="Row Variable",
            template='plotly_white'
        )
        
        plots.append(fig_obs)
        
        # Create heatmap of standardized residuals
        expected_df = pd.DataFrame(
            expected,
            index=contingency_table.index,
            columns=contingency_table.columns
        )
        
        residuals = (contingency_table - expected_df) / np.sqrt(expected_df)
        
        fig_res = go.Figure(data=go.Heatmap(
            z=residuals.values,
            x=residuals.columns,
            y=residuals.index,
            text=residuals.values,
            texttemplate="%{text:.2f}",
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            hoverongaps=False,
            hovertemplate=(
                "Row: %{y}<br>"
                "Column: %{x}<br>"
                "Standardized Residual: %{z:.2f}<br>"
                "<extra></extra>"
            )
        ))
        
        fig_res.update_layout(
            title='Standardized Residuals',
            xaxis_title="Column Variable",
            yaxis_title="Row Variable",
            template='plotly_white'
        )
        
        plots.append(fig_res)
        
        # Create mosaic plot
        if len(contingency_table.index) * len(contingency_table.columns) <= 25:  # Limit for readability
            props = contingency_table / contingency_table.values.sum()
            
            fig_mosaic = go.Figure()
            
            y_start = 0
            for i, row_label in enumerate(props.index):
                x_start = 0
                row_sum = props.loc[row_label].sum()
                
                for j, col_label in enumerate(props.columns):
                    cell_prop = props.loc[row_label, col_label]
                    
                    fig_mosaic.add_shape(
                        type="rect",
                        x0=x_start,
                        y0=y_start,
                        x1=x_start + props.loc[row_label, col_label] / row_sum,
                        y1=y_start + row_sum,
                        fillcolor=f'hsl({360 * i / len(props.index)}, 70%, 50%)',
                        opacity=0.6,
                        line=dict(color="white", width=2),
                    )
                    
                    # Add text labels
                    fig_mosaic.add_annotation(
                        x=(x_start + x_start + cell_prop / row_sum) / 2,
                        y=(y_start + y_start + row_sum) / 2,
                        text=f'{contingency_table.loc[row_label, col_label]}\n({cell_prop:.1%})',
                        showarrow=False,
                        font=dict(size=10, color="white")
                    )
                    
                    x_start += cell_prop / row_sum
                y_start += row_sum
            
            fig_mosaic.update_layout(
                title="Mosaic Plot of Observed Frequencies",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, title="Column Proportions"),
                yaxis=dict(showgrid=False, zeroline=False, title="Row Proportions"),
                template='plotly_white'
            )
            
            plots.append(fig_mosaic)
        
        # Add significance visualization
        plots.append(viz.create_significance_plot(
            p_val,
            alpha,
            "Chi-square Test"
        ))
        
        # Add effect size visualization
        plots.append(viz.create_effect_size_plot(
            cramer_v,
            "Cramer's V"
        ))
        
        # Calculate additional metrics
        row_totals = contingency_table.sum(axis=1)
        col_totals = contingency_table.sum(axis=0)
        total = contingency_table.values.sum()
        
        # Calculate contributions to chi-square
        contributions = (contingency_table - expected_df) ** 2 / expected_df
        
        additional_metrics = {
            'degrees_of_freedom': dof,
            'expected_frequencies': expected_df.to_dict(),
            'row_percentages': (contingency_table.div(row_totals, axis=0) * 100).to_dict(),
            'column_percentages': (contingency_table.div(col_totals, axis=1) * 100).to_dict(),
            'total_percentage': (contingency_table / total * 100).to_dict(),
            'contributions': contributions.to_dict(),
            'minimum_expected': np.min(expected),
            'fisher_results': fisher_results
        }
        
        # Generate interpretation
        if p_val < alpha:
            interpretation = (
                f"There is a significant association between the variables "
                f"(χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.4f}). The effect size "
                f"(Cramer's V = {cramer_v:.3f}) indicates a "
                f"{self._get_effect_size_interpretation(cramer_v)} association. "
            )
            
            # Add Fisher's exact test results for 2x2 tables
            if fisher_results:
                interpretation += (
                    f"Fisher's exact test confirms this association (p = {fisher_p:.4f}). "
                    f"The odds ratio of {odds_ratio:.2f} indicates that {fisher_results['odds_ratio_interpretation']}. "
                )
            
            # Add information about strongest associations
            strong_residuals = np.abs(residuals) > 2
            if np.any(strong_residuals):
                strong_pairs = []
                for i, row in enumerate(contingency_table.index):
                    for j, col in enumerate(contingency_table.columns):
                        if abs(residuals.iloc[i, j]) > 2:
                            relation = "more" if residuals.iloc[i, j] > 0 else "fewer"
                            strong_pairs.append(f"{row} had {relation} {col} than expected")
                
                interpretation += (
                    f"Notable associations were found where: {'; '.join(strong_pairs)}."
                )
        else:
            interpretation = (
                f"No significant association was found between the variables "
                f"(χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.4f}). The effect size "
                f"(Cramer's V = {cramer_v:.3f}) indicates a "
                f"{self._get_effect_size_interpretation(cramer_v)} association. "
            )
            
            if fisher_results:
                interpretation += (
                    f"Fisher's exact test confirms this lack of association (p = {fisher_p:.4f})."
                )
        
        return TestResult(
            test_statistic=chi2_stat,
            p_value=p_val,
            effect_size=cramer_v,
            interpretation=interpretation,
            additional_metrics=additional_metrics,
            plots=plots
        )

    def check_assumptions(self, data: Dict[str, Any]) -> AssumptionCheck:
        """
        Check Chi-square test assumptions.
        
        Args:
            data: Dictionary containing contingency table
            
        Returns:
            AssumptionCheck object with results and recommendations
        """
        table = data['table']
        if isinstance(table, pd.DataFrame):
            contingency_table = table
        else:
            contingency_table = pd.DataFrame(table)
        
        checks = []
        warnings = []
        recommendations = []
        
        # Check independence assumption
        checks.append({
            'test_name': 'Independence',
            'result': 'Requires Review',
            'details': 'Independence must be verified by study design'
        })
        recommendations.append(
            "Verify that observations are independent based on data collection method"
        )
        
        # Check for adequate expected frequencies
        _, _, _, expected = chi2_contingency(contingency_table)
        min_expected = np.min(expected)
        expected_check = {
            'test_name': 'Expected Frequencies',
            'result': 'Pass' if min_expected >= 5 else 'Warning',
            'details': f'Minimum expected frequency: {min_expected:.2f}'
        }
        checks.append(expected_check)
        
        if min_expected < 5:
            warnings.append("Some expected frequencies are less than 5")
            if contingency_table.shape == (2, 2):
                recommendations.append("Use Fisher's exact test")
            else:
                recommendations.extend([
                    "Consider combining categories if meaningful",
                    "Use exact tests or simulation-based methods"
                ])
        
        # Check sample size
        total_n = contingency_table.values.sum()
        size_check = {
            'test_name': 'Sample Size',
            'result': 'Pass' if total_n >= 30 else 'Warning',
            'details': f'Total sample size: {total_n}'
        }
        checks.append(size_check)
        
        if total_n < 30:
            warnings.append("Small total sample size")
            recommendations.append("Consider exact tests for small samples")
        
        # Check for zero cells
        zero_cells = (contingency_table == 0).sum().sum()
        if zero_cells > 0:
            warnings.append(f"Found {zero_cells} cells with zero frequency")
            recommendations.append(
                "Consider combining categories or using exact methods"
            )
        
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
    
    def _get_effect_size_interpretation(self, cramer_v: float) -> str:
        """Get interpretation of Cramer's V effect size."""
        if cramer_v < 0.1:
            return "negligible"
        elif cramer_v < 0.3:
            return "weak"
        elif cramer_v < 0.5:
            return "moderate"
        else:
            return "strong"
    
    def _interpret_odds_ratio(self, odds_ratio: float) -> str:
        """Get interpretation of odds ratio for 2x2 tables."""
        if odds_ratio == 1:
            return "the odds are equal between groups"
        elif odds_ratio < 1:
            return f"the odds are {1/odds_ratio:.2f} times higher in the reference group"
        else:
            return f"the odds are {odds_ratio:.2f} times higher in the comparison group"