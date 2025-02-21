"""
Visualization components for statistical testing results.
Provides comprehensive plotting utilities with accessibility and interactivity.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from scipy import stats

class TestVisualizations:
    """Class for creating statistical test visualizations with consistent styling."""
    
    # Color scheme designed for accessibility (colorblind-friendly)
    COLORS = {
        'primary': '#4C72B0',       # Main blue
        'secondary': '#DD8452',     # Orange
        'tertiary': '#55A868',      # Green
        'quaternary': '#C44E52',    # Red
        'reference': '#8172B3',     # Purple
        'background': '#FFFFFF',    # White
        'grid': '#E5E5E5'          # Light gray
    }
    
    # Common layout settings
    LAYOUT_DEFAULTS = {
        'template': 'plotly_white',
        'font': dict(size=12),
        'showlegend': True,
        'margin': dict(l=60, r=30, t=50, b=50)
    }

    @classmethod
    def create_box_plot(cls, 
                       data: Dict[str, np.ndarray], 
                       title: str,
                       y_label: str = "Values") -> go.Figure:
        """
        Create an enhanced box plot for group comparisons.
        
        Args:
            data: Dictionary mapping group names to data arrays
            title: Plot title
            y_label: Label for y-axis
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for i, (group_name, values) in enumerate(data.items()):
            # Add box plot
            fig.add_trace(go.Box(
                y=values,
                name=group_name,
                boxpoints='outliers',
                marker_color=list(cls.COLORS.values())[i % len(cls.COLORS)],
                hovertemplate=(
                    f"{group_name}<br>"
                    "Value: %{y}<br>"
                    "<extra></extra>"
                )
            ))
            
            # Add individual points with jitter for better visualization
            jitter = np.random.normal(0, 0.04, size=len(values))
            fig.add_trace(go.Scatter(
                x=np.full_like(values, i) + jitter,
                y=values,
                mode='markers',
                marker=dict(
                    color=list(cls.COLORS.values())[i % len(cls.COLORS)],
                    opacity=0.5,
                    size=4
                ),
                name=f"{group_name} Points",
                showlegend=False,
                hovertemplate=(
                    f"{group_name}<br>"
                    "Value: %{y}<br>"
                    "<extra></extra>"
                )
            ))
        
        # Add summary statistics annotations
        for i, (group_name, values) in enumerate(data.items()):
            stats_text = (
                f"Mean: {np.mean(values):.2f}<br>"
                f"Median: {np.median(values):.2f}<br>"
                f"SD: {np.std(values):.2f}<br>"
                f"N: {len(values)}"
            )
            fig.add_annotation(
                x=i,
                y=max(values),
                text=stats_text,
                showarrow=False,
                yshift=20,
                align='center',
                bgcolor='rgba(255,255,255,0.8)'
            )
        
        # Update layout with common settings
        fig.update_layout(
            **cls.LAYOUT_DEFAULTS,
            title=title,
            yaxis_title=y_label,
            boxmode='group'
        )
        
        return fig

    @classmethod
    def create_qq_plot(cls, 
                      data: np.ndarray, 
                      title: str,
                      sample_label: str = "Sample Quantiles") -> go.Figure:
        """
        Create an enhanced Q-Q plot for normality check.
        
        Args:
            data: Array of numeric values
            title: Plot title
            sample_label: Label for sample quantiles
            
        Returns:
            Plotly figure object
        """
        sorted_data = np.sort(data)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
        
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_data,
            mode='markers',
            name='Data Points',
            marker=dict(
                color=cls.COLORS['primary'],
                size=6
            ),
            hovertemplate=(
                "Theoretical: %{x:.2f}<br>"
                f"{sample_label}: %{y:.2f}<br>"
                "<extra></extra>"
            )
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
            line=dict(
                color=cls.COLORS['reference'],
                dash='dash'
            ),
            showlegend=True
        ))
        
        # Add annotations with distribution statistics
        stats_text = (
            f"Mean: {np.mean(data):.2f}<br>"
            f"SD: {np.std(data):.2f}<br>"
            f"Skewness: {stats.skew(data):.2f}<br>"
            f"Kurtosis: {stats.kurtosis(data):.2f}"
        )
        fig.add_annotation(
            x=min_val,
            y=max(sorted_data),
            text=stats_text,
            showarrow=False,
            yshift=20,
            align='left',
            bgcolor='rgba(255,255,255,0.8)'
        )
        
        fig.update_layout(
            **cls.LAYOUT_DEFAULTS,
            title=title,
            xaxis_title='Theoretical Quantiles',
            yaxis_title=sample_label
        )
        
        return fig

    @classmethod
    def create_residual_plot(cls,
                            residuals: np.ndarray,
                            fitted_values: np.ndarray,
                            title: str) -> go.Figure:
        """
        Create an enhanced residual plot for regression diagnostics.
        
        Args:
            residuals: Array of residual values
            fitted_values: Array of fitted/predicted values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add scatter plot of residuals
        fig.add_trace(go.Scatter(
            x=fitted_values,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                color=cls.COLORS['primary'],
                size=6,
                opacity=0.7
            ),
            hovertemplate=(
                "Fitted Value: %{x:.2f}<br>"
                "Residual: %{y:.2f}<br>"
                "<extra></extra>"
            )
        ))
        
        # Add reference line at y=0
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=cls.COLORS['reference'],
            name='Zero Line'
        )
        
        # Add loess smoothed line
        sorted_idx = np.argsort(fitted_values)
        x_smooth = fitted_values[sorted_idx]
        y_smooth = residuals[sorted_idx]
        
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=pd.Series(y_smooth).rolling(
                window=min(50, len(y_smooth)//5),
                center=True
            ).mean(),
            mode='lines',
            name='Trend',
            line=dict(
                color=cls.COLORS['secondary'],
                width=2
            )
        ))
        
        # Add annotations with residual statistics
        stats_text = (
            f"Mean: {np.mean(residuals):.2f}<br>"
            f"SD: {np.std(residuals):.2f}<br>"
            f"Range: [{np.min(residuals):.2f}, {np.max(residuals):.2f}]"
        )
        fig.add_annotation(
            x=min(fitted_values),
            y=max(residuals),
            text=stats_text,
            showarrow=False,
            yshift=20,
            align='left',
            bgcolor='rgba(255,255,255,0.8)'
        )
        
        fig.update_layout(
            **cls.LAYOUT_DEFAULTS,
            title=title,
            xaxis_title='Fitted Values',
            yaxis_title='Residuals'
        )
        
        return fig

    @classmethod
    def create_correlation_heatmap(cls,
                                 corr_matrix: np.ndarray,
                                 labels: List[str],
                                 title: str = "Correlation Matrix") -> go.Figure:
        """
        Create an enhanced correlation heatmap.
        
        Args:
            corr_matrix: Correlation matrix
            labels: Variable labels
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=corr_matrix,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate=(
                "X: %{x}<br>"
                "Y: %{y}<br>"
                "Correlation: %{z:.3f}<br>"
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            **cls.LAYOUT_DEFAULTS,
            title=title,
            xaxis_title='Variables',
            yaxis_title='Variables',
            width=max(400, len(labels) * 50),
            height=max(400, len(labels) * 50)
        )
        
        return fig

    @classmethod
    def create_effect_size_plot(cls,
                              effect_size: float,
                              effect_type: str,
                              ci_lower: Optional[float] = None,
                              ci_upper: Optional[float] = None) -> go.Figure:
        """
        Create a visualization of effect size with interpretation.
        
        Args:
            effect_size: Calculated effect size
            effect_type: Type of effect size (e.g., "Cohen's d")
            ci_lower: Lower confidence interval (optional)
            ci_upper: Upper confidence interval (optional)
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Define thresholds based on effect size type
        thresholds = {
            "Cohen's d": {'small': 0.2, 'medium': 0.5, 'large': 0.8},
            'Correlation': {'small': 0.1, 'medium': 0.3, 'large': 0.5},
            'Eta-squared': {'small': 0.01, 'medium': 0.06, 'large': 0.14}
        }[effect_type]
        
        # Create gauge chart
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=abs(effect_size),
            gauge={
                'axis': {'range': [0, max(1, abs(effect_size) * 1.2)]},
                'bar': {'color': cls.COLORS['primary']},
                'steps': [
                    {'range': [0, thresholds['small']], 
                     'color': 'rgba(200,200,200,0.2)'},
                    {'range': [thresholds['small'], thresholds['medium']], 
                     'color': 'rgba(200,200,200,0.4)'},
                    {'range': [thresholds['medium'], thresholds['large']], 
                     'color': 'rgba(200,200,200,0.6)'},
                    {'range': [thresholds['large'], max(1, abs(effect_size) * 1.2)], 
                     'color': 'rgba(200,200,200,0.8)'}
                ],
                'threshold': {
                    'line': {'color': cls.COLORS['quaternary'], 'width': 2},
                    'thickness': 0.75,
                    'value': abs(effect_size)
                }
            },
            title={'text': f"{effect_type} Effect Size"}
        ))
        
        # Add interpretation annotation
        interpretation = (
            "Small" if abs(effect_size) < thresholds['medium']
            else "Medium" if abs(effect_size) < thresholds['large']
            else "Large"
        )
        
        fig.add_annotation(
            x=0.5,
            y=0.2,
            text=f"Interpretation: {interpretation} effect",
            showarrow=False,
            font=dict(size=14)
        )
        
        # Add confidence interval if provided
        if ci_lower is not None and ci_upper is not None:
            ci_text = f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]"
            fig.add_annotation(
                x=0.5,
                y=0.1,
                text=ci_text,
                showarrow=False,
                font=dict(size=12)
            )
        
        fig.update_layout(
            **cls.LAYOUT_DEFAULTS,
            height=300,
            showlegend=False
        )
        
        return fig

    @classmethod
    def create_distribution_comparison(cls,
                                    groups: Dict[str, np.ndarray],
                                    title: str = "Distribution Comparison") -> go.Figure:
        """
        Create overlaid density plots for comparing distributions.
        
        Args:
            groups: Dictionary mapping group names to data arrays
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for i, (group_name, values) in enumerate(groups.items()):
            # Calculate kernel density estimation
            kernel = stats.gaussian_kde(values)
            x_range = np.linspace(min(values), max(values), 200)
            density = kernel(x_range)
            
            # Add density plot
            fig.add_trace(go.Scatter(
                x=x_range,
                y=density,
                name=group_name,
                mode='lines',
                fill='tonexty',
                line=dict(
                    color=list(cls.COLORS.values())[i % len(cls.COLORS)],
                    width=2
                ),
                hovertemplate=(
                    f"{group_name}<br>"
                    "Value: %{x:.2f}<br>"
                    "Density: %{y:.3f}<br>"
                    "<extra></extra>"
                )
            ))
            
            # Add mean line
            mean_val = np.mean(values)
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color=list(cls.COLORS.values())[i % len(cls.COLORS)],
                annotation_text=f"{group_name} Mean: {mean_val:.2f}",
                annotation_position="top"
            )
        
        fig.update_layout(
            **cls.LAYOUT_DEFAULTS,
            title=title,
            xaxis_title='Values',
            yaxis_title='Density',
            showlegend=True
        )
        
        return fig

    @classmethod
    def create_significance_plot(cls,
                               p_value: float,
                               alpha: float = 0.05,
                               test_name: str = "Test") -> go.Figure:
        """
        Create a visual representation of statistical significance.
        
        Args:
            p_value: Test p-value
            alpha: Significance level
            test_name: Name of the test
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Create gauge chart for p-value
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=p_value,
            number={'valueformat': '.3f'},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1},
                'bar': {'color': cls.COLORS['primary'] if p_value > alpha else cls.COLORS['quaternary']},
                'steps': [
                    {'range': [0, alpha], 'color': 'rgba(196, 78, 82, 0.2)'},  # Light red
                    {'range': [alpha, 1], 'color': 'rgba(255, 255, 255, 0.1)'}  # Light white
                ],
                'threshold': {
                    'line': {'color': cls.COLORS['reference'], 'width': 2},
                    'thickness': 0.75,
                    'value': alpha
                }
            },
            title={'text': f"{test_name}<br>p-value"}
        ))
        
        # Add interpretation
        significance = "Significant" if p_value < alpha else "Not Significant"
        color = cls.COLORS['quaternary'] if p_value < alpha else cls.COLORS['primary']
        
        fig.add_annotation(
            x=0.5,
            y=0.2,
            text=f"Result: {significance} at α = {alpha}",
            showarrow=False,
            font=dict(
                size=14,
                color=color
            )
        )
        
        fig.update_layout(
            **cls.LAYOUT_DEFAULTS,
            height=300,
            showlegend=False
        )
        
        return fig

    @classmethod
    def create_power_analysis_plot(cls,
                                 effect_sizes: np.ndarray,
                                 powers: np.ndarray,
                                 observed_power: Optional[float] = None,
                                 title: str = "Power Analysis") -> go.Figure:
        """
        Create a power analysis curve plot.
        
        Args:
            effect_sizes: Array of effect sizes
            powers: Array of corresponding power values
            observed_power: Observed power value (optional)
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add power curve
        fig.add_trace(go.Scatter(
            x=effect_sizes,
            y=powers,
            mode='lines',
            name='Power Curve',
            line=dict(
                color=cls.COLORS['primary'],
                width=2
            ),
            hovertemplate=(
                "Effect Size: %{x:.2f}<br>"
                "Power: %{y:.3f}<br>"
                "<extra></extra>"
            )
        ))
        
        # Add reference lines
        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color=cls.COLORS['reference'],
            annotation_text="Recommended Power (0.8)",
            annotation_position="right"
        )
        
        if observed_power is not None:
            fig.add_hline(
                y=observed_power,
                line_dash="dash",
                line_color=cls.COLORS['secondary'],
                annotation_text=f"Observed Power ({observed_power:.2f})",
                annotation_position="left"
            )
        
        fig.update_layout(
            **cls.LAYOUT_DEFAULTS,
            title=title,
            xaxis_title='Effect Size',
            yaxis_title='Statistical Power',
            yaxis_range=[0, 1]
        )
        
        return fig

    @classmethod
    def create_assumption_summary(cls,
                                assumption_results: List[Dict[str, Any]],
                                title: str = "Assumption Check Summary") -> go.Figure:
        """
        Create a visual summary of assumption check results.
        
        Args:
            assumption_results: List of assumption check results
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Prepare data
        assumptions = [result['name'] for result in assumption_results]
        statuses = [result['result'] for result in assumption_results]
        
        # Define colors for different statuses
        status_colors = {
            'Pass': cls.COLORS['tertiary'],
            'Warning': cls.COLORS['secondary'],
            'Fail': cls.COLORS['quaternary'],
            'Skip': cls.COLORS['reference']
        }
        
        colors = [status_colors[status] for status in statuses]
        
        fig = go.Figure()
        
        # Add horizontal bars
        fig.add_trace(go.Bar(
            y=assumptions,
            x=[1] * len(assumptions),
            orientation='h',
            marker_color=colors,
            text=statuses,
            textposition='inside',
            hovertemplate=(
                "Assumption: %{y}<br>"
                "Status: %{text}<br>"
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            **cls.LAYOUT_DEFAULTS,
            title=title,
            xaxis_visible=False,
            yaxis_title='Assumptions',
            showlegend=False,
            height=max(300, len(assumptions) * 40)
        )
        
        # Add status counts annotation
        status_counts = {status: statuses.count(status) for status in set(statuses)}
        status_text = "<br>".join([f"{status}: {count}" for status, count in status_counts.items()])
        
        fig.add_annotation(
            x=1,
            y=len(assumptions) - 1,
            text=status_text,
            showarrow=False,
            align='left',
            bgcolor='rgba(255,255,255,0.8)'
        )
        
        return fig