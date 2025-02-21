"""
Comprehensive assumption checking functionality for statistical tests.
Provides tools for checking and visualizing statistical assumptions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, levene
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
from core import AssumptionCheck

@dataclass
class AssumptionResult:
    """
    Stores results of a single assumption check.
    
    Attributes:
        name: Name of the assumption
        result: Pass/Fail/Warning status
        statistic: Test statistic (if applicable)
        p_value: P-value (if applicable)
        details: Additional details about the check
        recommendation: What to do if assumption is violated
        visualization: Optional plotly figure
    """
    name: str
    result: str  # "Pass", "Fail", or "Warning"
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    details: str = ""
    recommendation: str = ""
    visualization: Optional[go.Figure] = None

class AssumptionChecker:
    """Class for checking statistical test assumptions."""
    
    @staticmethod
    def check_normality(data: np.ndarray, 
                       group_name: str = "",
                       alpha: float = 0.05) -> AssumptionResult:
        """
        Check normality assumption using Shapiro-Wilk test and Q-Q plot.
        
        Args:
            data: Array of numeric values
            group_name: Optional name for the group being checked
            alpha: Significance level for the test
            
        Returns:
            AssumptionResult with normality check details
        """
        if len(data) < 3:
            return AssumptionResult(
                name=f"Normality ({group_name})" if group_name else "Normality",
                result="Skip",
                details="Not enough data points for normality test (minimum 3 required)",
                recommendation="Collect more data or consider non-parametric alternatives"
            )
        
        # Remove any missing values
        data = data[~np.isnan(data)]
        
        # Perform Shapiro-Wilk test
        statistic, p_value = shapiro(data)
        
        # Create Q-Q plot
        sorted_data = np.sort(data)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
        
        fig = go.Figure()
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
            y=[min_val * np.std(data) + np.mean(data), 
               max_val * np.std(data) + np.mean(data)],
            mode='lines',
            name='Reference Line',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Q-Q Plot: {group_name}' if group_name else 'Q-Q Plot',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            template='plotly_white'
        )
        
        # Determine result and recommendation
        result = "Pass" if p_value > alpha else "Warning"
        
        if result == "Warning":
            recommendation = """
            Consider the following options:
            1. Transform the data (e.g., log, square root)
            2. Use non-parametric alternatives
            3. For large samples (n > 30), you might proceed due to Central Limit Theorem
            """
        else:
            recommendation = "Normality assumption appears to be met"
            
        details = (f"Shapiro-Wilk test: W = {statistic:.4f}, p = {p_value:.4f}\n"
                  f"{'✓' if result == 'Pass' else '⚠'} ")
        
        if p_value < alpha:
            details += "Data appears to deviate from normality"
        else:
            details += "No significant deviation from normality detected"
        
        return AssumptionResult(
            name=f"Normality ({group_name})" if group_name else "Normality",
            result=result,
            statistic=statistic,
            p_value=p_value,
            details=details,
            recommendation=recommendation,
            visualization=fig
        )

    @staticmethod
    def check_homogeneity(*groups: np.ndarray, 
                         group_names: Optional[List[str]] = None,
                         alpha: float = 0.05) -> AssumptionResult:
        """
        Check homogeneity of variances using Levene's test and box plots.
        
        Args:
            *groups: Arrays of numeric values to compare
            group_names: Optional names for the groups
            alpha: Significance level for the test
            
        Returns:
            AssumptionResult with homogeneity check details
        """
        if group_names is None:
            group_names = [f"Group {i+1}" for i in range(len(groups))]
            
        # Remove missing values
        groups = [g[~np.isnan(g)] for g in groups]
        
        # Perform Levene's test
        statistic, p_value = levene(*groups)
        
        # Create box plot
        fig = go.Figure()
        for group, name in zip(groups, group_names):
            fig.add_trace(go.Box(
                y=group,
                name=name,
                boxpoints='outliers'
            ))
            
        fig.update_layout(
            title='Distribution Comparison',
            yaxis_title='Values',
            template='plotly_white'
        )
        
        # Calculate variance ratios
        variances = [np.var(g, ddof=1) for g in groups]
        max_ratio = max(variances) / min(variances)
        
        # Determine result and recommendation
        result = "Pass" if p_value > alpha and max_ratio < 4 else "Warning"
        
        if result == "Warning":
            recommendation = """
            Consider the following options:
            1. Use Welch's test (for t-test) or Welch's ANOVA
            2. Transform the data to stabilize variances
            3. Use non-parametric alternatives
            """
        else:
            recommendation = "Homogeneity of variance assumption appears to be met"
            
        details = (f"Levene's test: W = {statistic:.4f}, p = {p_value:.4f}\n"
                  f"Maximum variance ratio: {max_ratio:.2f}\n"
                  f"{'✓' if result == 'Pass' else '⚠'} ")
        
        if p_value < alpha:
            details += "Significant difference in variances detected"
        else:
            details += "No significant difference in variances detected"
            
        return AssumptionResult(
            name="Homogeneity of Variance",
            result=result,
            statistic=statistic,
            p_value=p_value,
            details=details,
            recommendation=recommendation,
            visualization=fig
        )

    @staticmethod
    def check_independence(data: np.ndarray, lag: int = 1) -> AssumptionResult:
        """
        Check independence assumption using Durbin-Watson test.
        
        Args:
            data: Array of numeric values
            lag: Lag order for autocorrelation check
            
        Returns:
            AssumptionResult with independence check details
        """
        if len(data) < 3:
            return AssumptionResult(
                name="Independence",
                result="Skip",
                details="Not enough data points for independence test",
                recommendation="Collect more data points"
            )
            
        # Calculate Durbin-Watson statistic
        residuals = np.diff(data)
        dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
        
        # Create lag plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data[:-lag],
            y=data[lag:],
            mode='markers',
            name='Lag Plot'
        ))
        
        # Add reference line
        min_val = min(data)
        max_val = max(data)
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Reference Line',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Lag Plot (lag={lag})',
            xaxis_title='Value(t)',
            yaxis_title=f'Value(t+{lag})',
            template='plotly_white'
        )
        
        # Determine result and recommendation
        result = "Pass" if 1.5 < dw_stat < 2.5 else "Warning"
        
        if result == "Warning":
            recommendation = """
            Consider the following:
            1. Check for time-based patterns
            2. Use methods that account for dependence
            3. Consider hierarchical/multilevel models
            """
        else:
            recommendation = "Independence assumption appears to be met"
            
        details = (f"Durbin-Watson statistic: {dw_stat:.4f}\n"
                  f"{'✓' if result == 'Pass' else '⚠'} ")
        
        if result == "Warning":
            details += "Potential autocorrelation detected"
        else:
            details += "No significant autocorrelation detected"
            
        return AssumptionResult(
            name="Independence",
            result=result,
            statistic=dw_stat,
            details=details,
            recommendation=recommendation,
            visualization=fig
        )

    @staticmethod
    def check_sample_size(data: np.ndarray, 
                         threshold: int = 30,
                         group_name: str = "") -> AssumptionResult:
        """
        Check sample size adequacy.
        
        Args:
            data: Array of values
            threshold: Minimum recommended sample size
            group_name: Optional name for the group being checked
            
        Returns:
            AssumptionResult with sample size check details
        """
        n = len(data[~np.isnan(data)])  # Count non-missing values
        
        if n >= threshold:
            result = "Pass"
            recommendation = "Sample size is adequate"
            details = f"✓ Sample size (n={n}) meets minimum requirement"
        elif n >= threshold // 2:
            result = "Warning"
            recommendation = """
            Consider:
            1. Collecting more data if possible
            2. Using methods suitable for smaller samples
            3. Being cautious in interpretation
            """
            details = f"⚠ Sample size (n={n}) is smaller than recommended"
        else:
            result = "Warning"
            recommendation = """
            Strongly consider:
            1. Collecting more data
            2. Using exact tests or non-parametric methods
            3. Being very cautious in interpretation
            """
            details = f"⚠ Sample size (n={n}) is much smaller than recommended"
            
        return AssumptionResult(
            name=f"Sample Size ({group_name})" if group_name else "Sample Size",
            result=result,
            details=details,
            recommendation=recommendation
        )

    @staticmethod
    def check_outliers(data: np.ndarray,
                      threshold: float = 3.0,
                      group_name: str = "") -> AssumptionResult:
        """
        Check for outliers using z-scores and box plot.
        
        Args:
            data: Array of numeric values
            threshold: Z-score threshold for outlier detection
            group_name: Optional name for the group being checked
            
        Returns:
            AssumptionResult with outlier check details
        """
        clean_data = data[~np.isnan(data)]
        z_scores = np.abs(stats.zscore(clean_data))
        outliers = np.where(z_scores > threshold)[0]
        
        # Create box plot with outliers highlighted
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=clean_data,
            name=group_name if group_name else "Data",
            boxpoints='outliers'
        ))
        
        fig.update_layout(
            title='Box Plot with Outliers',
            yaxis_title='Values',
            template='plotly_white'
        )
        
        if len(outliers) == 0:
            result = "Pass"
            recommendation = "No concerning outliers detected"
            details = "✓ No outliers detected beyond threshold"
        else:
            result = "Warning"
            recommendation = """
            Consider the following steps:
            1. Verify outlier values are correct
            2. Investigate reason for outliers
            3. Consider robust statistical methods
            4. Document treatment of outliers
            """
            details = (f"⚠ Found {len(outliers)} potential outliers\n"
                      f"Outlier values: {clean_data[outliers].tolist()}")
            
        return AssumptionResult(
            name=f"Outliers ({group_name})" if group_name else "Outliers",
            result=result,
            details=details,
            recommendation=recommendation,
            visualization=fig
        )
