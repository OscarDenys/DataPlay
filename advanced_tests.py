"""
Implementation of advanced statistical tests including Wald test and other sophisticated analyses.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

@dataclass
class AdvancedTestResult:
    """Results from advanced statistical tests"""
    test_name: str
    test_statistic: Union[float, np.ndarray]
    p_value: Union[float, np.ndarray]
    effect_size: Optional[Union[float, np.ndarray]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    additional_metrics: Dict[str, Any] = None
    figures: List[go.Figure] = None
    interpretation: str = ""
    assumptions_summary: Dict[str, Any] = None
    warnings: List[str] = None

class AdvancedTestBase:
    """Base class for advanced statistical tests"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def check_sample_size(self, n_samples: int, n_parameters: int) -> Dict[str, Any]:
        """Check if sample size is adequate for model complexity"""
        min_samples = max(30, 10 * n_parameters)  # Rule of thumb
        
        return {
            'passed': n_samples >= min_samples,
            'warning': (f"Sample size ({n_samples}) may be too small for "
                       f"{n_parameters} parameters") if n_samples < min_samples else None,
            'details': (f"Sample size: {n_samples}, Parameters: {n_parameters}, "
                       f"Recommended minimum: {min_samples}")
        }
    
    def check_multicollinearity(self, X: np.ndarray) -> Dict[str, Any]:
        """Check for multicollinearity using VIF"""
        if X.ndim == 1:
            return {'passed': True, 'warning': None, 'details': "Single predictor"}
        
        try:
            X_with_const = add_constant(X)
            vif_data = pd.DataFrame()
            vif_data["Variable"] = range(X_with_const.shape[1])
            vif_data["VIF"] = [variance_inflation_factor(X_with_const, i) 
                              for i in range(X_with_const.shape[1])]
            
            max_vif = vif_data["VIF"].max()
            has_high_vif = max_vif > 5
            
            return {
                'passed': not has_high_vif,
                'warning': f"High multicollinearity detected (VIF > 5)" if has_high_vif else None,
                'details': f"Maximum VIF: {max_vif:.2f}"
            }
        except Exception as e:
            return {
                'passed': False,
                'warning': f"Unable to check multicollinearity: {str(e)}",
                'details': "VIF calculation failed"
            }
    
    def create_comparison_plot(self, 
                             observed: np.ndarray,
                             predicted: np.ndarray,
                             ci: Optional[np.ndarray] = None) -> go.Figure:
        """Create plot comparing observed vs predicted values with confidence intervals"""
        fig = go.Figure()
        
        # Add scatter plot of observed vs predicted
        fig.add_trace(go.Scatter(
            x=predicted,
            y=observed,
            mode='markers',
            name='Observations',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        # Add diagonal reference line
        min_val = min(observed.min(), predicted.min())
        max_val = max(observed.max(), predicted.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        # Add confidence intervals if provided
        if ci is not None:
            fig.add_trace(go.Scatter(
                x=predicted,
                y=ci[:, 0],
                mode='lines',
                name='Lower CI',
                line=dict(color='gray', dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=predicted,
                y=ci[:, 1],
                mode='lines',
                name='Upper CI',
                line=dict(color='gray', dash='dot'),
                fill='tonexty'
            ))
        
        fig.update_layout(
            title='Observed vs Predicted Values',
            xaxis_title='Predicted Values',
            yaxis_title='Observed Values',
            showlegend=True
        )
        
        return fig
    
    def create_residual_plots(self, 
                            residuals: np.ndarray,
                            predicted: np.ndarray) -> go.Figure:
        """Create comprehensive residual diagnostic plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Residuals vs Fitted',
                'Normal Q-Q Plot',
                'Scale-Location Plot',
                'Residuals Distribution'
            )
        )
        
        # Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='blue', opacity=0.6)
            ),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Q-Q Plot
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(residuals))
        )
        sorted_residuals = np.sort(residuals)
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='blue', opacity=0.6)
            ),
            row=1, col=2
        )
        
        # Add reference line for Q-Q plot
        slope = np.std(sorted_residuals)
        intercept = np.mean(sorted_residuals)
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=slope * theoretical_quantiles + intercept,
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        
        # Scale-Location Plot
        standardized_residuals = residuals / np.std(residuals)
        abs_sqrt_std_resid = np.sqrt(np.abs(standardized_residuals))
        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=abs_sqrt_std_resid,
                mode='markers',
                name='Scale-Location',
                marker=dict(color='blue', opacity=0.6)
            ),
            row=2, col=1
        )
        
        # Residuals Distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residuals Dist',
                nbinsx=30,
                marker_color='blue',
                opacity=0.6
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Residual Diagnostic Plots",
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_xaxes(title_text="Fitted Values", row=2, col=1)
        fig.update_xaxes(title_text="Residuals", row=2, col=2)
        
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="√|Standardized Residuals|", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        return fig

class WaldTest(AdvancedTestBase):
    """Implementation of Wald test for parameter significance"""
    
    def check_assumptions(self, 
                         X: np.ndarray, 
                         y: np.ndarray,
                         standard_errors: np.ndarray) -> Dict[str, Any]:
        """Check assumptions for Wald test"""
        n_samples = len(y)
        n_parameters = X.shape[1] if X.ndim > 1 else 1
        
        assumptions = {
            'sample_size': self.check_sample_size(n_samples, n_parameters),
            'multicollinearity': self.check_multicollinearity(X),
            'standard_errors': {
                'passed': np.all(standard_errors > 0),
                'warning': "Some standard errors are zero or negative" 
                          if not np.all(standard_errors > 0) else None,
                'details': f"Range of SEs: [{min(standard_errors):.4f}, "
                          f"{max(standard_errors):.4f}]"
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
                 X: np.ndarray,
                 y: np.ndarray,
                 parameter_constraints: Optional[Dict[str, float]] = None,
                 parameter_names: Optional[List[str]] = None) -> AdvancedTestResult:
        """
        Run Wald test for parameter significance or constraints.
        
        Parameters:
        -----------
        X : np.ndarray
            Design matrix
        y : np.ndarray
            Response variable
        parameter_constraints : Optional[Dict[str, float]]
            Dictionary of parameter constraints to test
        parameter_names : Optional[List[str]]
            Names of parameters for interpretation
        
        Returns:
        --------
        AdvancedTestResult
            Complete test results including statistics and visualizations
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Fit OLS model
        model = OLS(y, add_constant(X))
        results = model.fit()
        
        # Get parameter estimates and covariance matrix
        beta_hat = results.params
        cov_beta = results.cov_params()
        standard_errors = np.sqrt(np.diag(cov_beta))
        
        # Check assumptions
        assumptions = self.check_assumptions(X, y, standard_errors)
        
        # Handle parameter names
        if parameter_names is None:
            parameter_names = [f'β{i}' for i in range(len(beta_hat))]
        
        # Calculate Wald statistics and p-values
        if parameter_constraints is not None:
            # Test specific constraints
            wald_stats = []
            p_values = []
            confidence_intervals = {}
            
            for param, constraint in parameter_constraints.items():
                idx = parameter_names.index(param)
                diff = beta_hat[idx] - constraint
                wald_stat = diff ** 2 / cov_beta[idx, idx]
                p_value = 1 - stats.chi2.cdf(wald_stat, df=1)
                
                wald_stats.append(wald_stat)
                p_values.append(p_value)
                
                # Calculate confidence interval
                critical_value = stats.norm.ppf(1 - self.alpha/2)
                ci = (
                    beta_hat[idx] - critical_value * standard_errors[idx],
                    beta_hat[idx] + critical_value * standard_errors[idx]
                )
                confidence_intervals[param] = ci
        else:
            # Test all parameters against zero
            wald_stats = (beta_hat ** 2) / np.diag(cov_beta)
            p_values = 1 - stats.chi2.cdf(wald_stats, df=1)
            
            # Calculate confidence intervals
            critical_value = stats.norm.ppf(1 - self.alpha/2)
            confidence_intervals = {
                name: (
                    beta_hat[i] - critical_value * standard_errors[i],
                    beta_hat[i] + critical_value * standard_errors[i]
                )
                for i, name in enumerate(parameter_names)
            }
        
        # Calculate effect sizes (standardized coefficients)
        if X.shape[1] > 0:  # Skip intercept
            x_std = np.std(X, axis=0)
            y_std = np.std(y)
            effect_sizes = beta_hat[1:] * (x_std / y_std)
        else:
            effect_sizes = None
        
        # Prepare additional metrics
        additional_metrics = {
            'coefficients': dict(zip(parameter_names, beta_hat)),
            'standard_errors': dict(zip(parameter_names, standard_errors)),
            'r_squared': results.rsquared,
            'adjusted_r_squared': results.rsquared_adj,
            'f_statistic': results.fvalue,
            'f_pvalue': results.f_pvalue,
            'aic': results.aic,
            'bic': results.bic
        }
        
        # Create visualizations
        figures = [
            self.create_comparison_plot(
                y, results.fittedvalues,
                results.conf_int().values
            ),
            self.create_residual_plots(
                results.resid,
                results.fittedvalues
            ),
            self._create_coefficient_plot(
                parameter_names, beta_hat,
                standard_errors, confidence_intervals
            )
        ]
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            parameter_names, beta_hat, p_values,
            confidence_intervals, effect_sizes,
            additional_metrics
        )
        
        return AdvancedTestResult(
            test_name="Wald Test",
            test_statistic=wald_stats,
            p_value=p_values,
            effect_size=effect_sizes,
            confidence_intervals=confidence_intervals,
            additional_metrics=additional_metrics,
            figures=figures,
            interpretation=interpretation,
            assumptions_summary=assumptions,
            warnings=assumptions['warnings']
        )
    
    def _create_coefficient_plot(self,
                               parameter_names: List[str],
                               coefficients: np.ndarray,
                               standard_errors: np.ndarray,
                               confidence_intervals: Dict[str, Tuple[float, float]]) -> go.Figure:
        """Create forest plot of coefficient estimates with confidence intervals"""
        fig = go.Figure()
        
        # Add coefficients as points
        fig.add_trace(go.Scatter(
            x=coefficients,
            y=parameter_names,
            mode='markers',
            name='Coefficient',
            marker=dict(color='blue', size=10)
        ))
        
        # Add error bars for confidence intervals
        fig.add_trace(go.Scatter(
            x=[ci[0] for ci in confidence_intervals.values()],
            y=parameter_names,
            mode='lines',
            name='95% CI',
            line=dict(color='gray', width=1),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[ci[1] for ci in confidence_intervals.values()],
            y=parameter_names,
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title='Parameter Estimates with 95% Confidence Intervals',
            xaxis_title='Estimate',
            yaxis_title='Parameter',
            showlegend=True
        )
        
        return fig
    
    def _generate_interpretation(self,
                               parameter_names: List[str],
                               coefficients: np.ndarray,
                               p_values: np.ndarray,
                               confidence_intervals: Dict[str, Tuple[float, float]],
                               effect_sizes: Optional[np.ndarray],
                               metrics: Dict[str, Any]) -> str:
        """Generate detailed interpretation of Wald test results"""
        parts = []
        
        # Overall model fit
        if isinstance(metrics.get('r_squared'), (int, float)):
            parts.append(
                f"The model explains {metrics['r_squared']:.1%} of the variance "
                f"(adjusted R² = {metrics['adjusted_r_squared']:.1%})."
            )
        
        # F-test interpretation if available
        if isinstance(metrics.get('f_pvalue'), (int, float)):
            if metrics['f_pvalue'] < self.alpha:
                parts.append(
                    f"The model is statistically significant "
                    f"(F = {metrics['f_statistic']:.2f}, p = {metrics['f_pvalue']:.4f})."
                )
            else:
                parts.append(
                    f"The model is not statistically significant "
                    f"(F = {metrics['f_statistic']:.2f}, p = {metrics['f_pvalue']:.4f})."
                )
        
        # Individual parameter tests
        significant_params = []
        nonsig_params = []
        
        for i, (name, coef) in enumerate(zip(parameter_names, coefficients)):
            ci = confidence_intervals[name]
            p_val = p_values[i] if isinstance(p_values, np.ndarray) else p_values
            
            if p_val < self.alpha:
                effect_desc = ""
                if effect_sizes is not None and i > 0:  # Skip intercept
                    effect_size = effect_sizes[i-1]
                    effect_desc = (
                        "negligible" if abs(effect_size) < 0.1 else
                        "small" if abs(effect_size) < 0.3 else
                        "medium" if abs(effect_size) < 0.5 else
                        "large"
                    )
                    effect_desc = f" with {effect_desc} effect size"
                
                significant_params.append(
                    f"{name} (β = {coef:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}])"
                    f"{effect_desc}"
                )
            else:
                nonsig_params.append(f"{name} (p = {p_val:.4f})")
        
        if significant_params:
            parts.append(
                "Significant parameters: " + "; ".join(significant_params) + "."
            )
        if nonsig_params:
            parts.append(
                "Non-significant parameters: " + "; ".join(nonsig_params) + "."
            )
        
        # Model selection metrics
        parts.append(
            f"Model selection metrics: AIC = {metrics['aic']:.2f}, "
            f"BIC = {metrics['bic']:.2f}"
        )
        
        return " ".join(parts)

class GrangerCausalityTest(AdvancedTestBase):
    """Implementation of Granger Causality test for time series data"""
    
    def check_assumptions(self,
                         x: np.ndarray,
                         y: np.ndarray,
                         max_lag: int) -> Dict[str, Any]:
        """Check assumptions for Granger causality test"""
        n_samples = len(x)
        
        assumptions = {
            'sample_size': {
                'passed': n_samples >= 30 + max_lag,
                'warning': "Sample size may be too small for specified lag order"
                          if n_samples < 30 + max_lag else None,
                'details': f"Sample size: {n_samples}, Maximum lag: {max_lag}"
            },
            'stationarity_x': self._check_stationarity(x),
            'stationarity_y': self._check_stationarity(y),
            'equal_length': {
                'passed': len(x) == len(y),
                'warning': "Series must have equal length" if len(x) != len(y) else None,
                'details': f"Length x: {len(x)}, Length y: {len(y)}"
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
    
    def _check_stationarity(self, series: np.ndarray) -> Dict[str, Any]:
        """Check stationarity using Augmented Dickey-Fuller test"""
        from statsmodels.tsa.stattools import adfuller
        
        try:
            adf_stat, p_value, _, _, _, _ = adfuller(series)
            return {
                'passed': p_value < 0.05,
                'warning': "Series may not be stationary" if p_value >= 0.05 else None,
                'details': f"ADF statistic: {adf_stat:.4f}, p-value: {p_value:.4f}"
            }
        except Exception as e:
            return {
                'passed': False,
                'warning': f"Unable to check stationarity: {str(e)}",
                'details': "Stationarity test failed"
            }
    
    def run_test(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 max_lag: int,
                 labels: Tuple[str, str] = ('X', 'Y')) -> AdvancedTestResult:
        """
        Run Granger causality test.
        
        Parameters:
        -----------
        x : np.ndarray
            First time series
        y : np.ndarray
            Second time series
        max_lag : int
            Maximum number of lags to test
        labels : tuple of str
            Labels for the two series
        
        Returns:
        --------
        AdvancedTestResult
            Complete test results including statistics and visualizations
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Check assumptions
        assumptions = self.check_assumptions(x, y, max_lag)
        
        # Run Granger causality tests for each lag
        gc_results = grangercausalitytests(
            pd.DataFrame({labels[0]: x, labels[1]: y}),
            maxlag=max_lag,
            verbose=False
        )
        
        # Extract test statistics and p-values
        test_stats = []
        p_values = []
        for lag in range(1, max_lag + 1):
            # Use F-test results
            test_stats.append(gc_results[lag][0]['ssr_ftest'][0])
            p_values.append(gc_results[lag][0]['ssr_ftest'][1])
        
        # Determine optimal lag using AIC
        aic_values = []
        for lag in range(1, max_lag + 1):
            model = self._fit_var_model(x, y, lag)
            aic_values.append(model.aic)
        optimal_lag = np.argmin(aic_values) + 1
        
        # Calculate additional metrics
        additional_metrics = {
            'optimal_lag': optimal_lag,
            'aic_values': dict(zip(range(1, max_lag + 1), aic_values)),
            'test_statistics': dict(zip(range(1, max_lag + 1), test_stats)),
            'p_values': dict(zip(range(1, max_lag + 1), p_values))
        }
        
        # Create visualizations
        figures = [
            self._create_causality_plot(x, y, labels),
            self._create_lag_selection_plot(aic_values, test_stats, p_values)
        ]
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            test_stats, p_values, optimal_lag, labels
        )
        
        return AdvancedTestResult(
            test_name="Granger Causality Test",
            test_statistic=test_stats[optimal_lag - 1],
            p_value=p_values[optimal_lag - 1],
            effect_size=None,  # No standard effect size for Granger causality
            confidence_intervals=None,
            additional_metrics=additional_metrics,
            figures=figures,
            interpretation=interpretation,
            assumptions_summary=assumptions,
            warnings=assumptions['warnings']
        )
    
    def _fit_var_model(self, x: np.ndarray, y: np.ndarray, lag: int):
        """Fit VAR model for given lag order"""
        from statsmodels.tsa.vector_ar.var_model import VAR
        data = pd.DataFrame(np.column_stack([x, y]))
        model = VAR(data)
        return model.fit(lag)
    
    def _create_causality_plot(self,
                             x: np.ndarray,
                             y: np.ndarray,
                             labels: Tuple[str, str]) -> go.Figure:
        """Create visualization of the two time series"""
        fig = go.Figure()
        
        # Add both series
        fig.add_trace(go.Scatter(
            x=np.arange(len(x)),
            y=x,
            name=labels[0],
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=np.arange(len(y)),
            y=y,
            name=labels[1],
            mode='lines'
        ))
        
        fig.update_layout(
            title='Time Series Comparison',
            xaxis_title='Time',
            yaxis_title='Value',
            showlegend=True
        )
        
        return fig
    
    def _create_lag_selection_plot(self,
                                 aic_values: List[float],
                                 test_stats: List[float],
                                 p_values: List[float]) -> go.Figure:
        """Create plot showing lag selection criteria"""
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('AIC by Lag Order',
                                         'Test Statistics and P-values'))
        
        # AIC plot
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(aic_values) + 1)),
                y=aic_values,
                mode='lines+markers',
                name='AIC'
            ),
            row=1, col=1
        )
        
        # Test statistics and p-values
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(test_stats) + 1)),
                y=test_stats,
                mode='lines+markers',
                name='F-statistic'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(p_values) + 1)),
                y=p_values,
                mode='lines+markers',
                name='P-value'
            ),
            row=2, col=1
        )
        
        # Add significance level line
        fig.add_hline(
            y=self.alpha,
            line_dash="dash",
            line_color="red",
            row=2, col=1,
            annotation_text="Significance Level"
        )
        
        fig.update_layout(
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _generate_interpretation(self,
                               test_stats: List[float],
                               p_values: List[float],
                               optimal_lag: int,
                               labels: Tuple[str, str]) -> str:
        """Generate detailed interpretation of Granger causality test results"""
        parts = []
        
        # Optimal lag interpretation
        parts.append(
            f"The optimal lag order based on AIC is {optimal_lag} time period(s)."
        )
        
        # Causality interpretation at optimal lag
        p_value = p_values[optimal_lag - 1]
        if p_value < self.alpha:
            parts.append(
                f"At the optimal lag order, there is evidence that {labels[0]} "
                f"Granger-causes {labels[1]} (F = {test_stats[optimal_lag - 1]:.3f}, "
                f"p = {p_value:.4f})."
            )
        else:
            parts.append(
                f"At the optimal lag order, there is no evidence that {labels[0]} "
                f"Granger-causes {labels[1]} (F = {test_stats[optimal_lag - 1]:.3f}, "
                f"p = {p_value:.4f})."
            )
        
        # Additional lag information
        significant_lags = [i + 1 for i, p in enumerate(p_values) if p < self.alpha]
        if len(significant_lags) > 1:
            parts.append(
                f"Significant Granger causality was also found at lags: "
                f"{', '.join(map(str, significant_lags))}."
            )
        
        # Testing approach
        parts.append(
            f"The test examined lags 1 through {len(test_stats)}, using F-tests "
            f"to compare restricted and unrestricted models."
        )
        
        return " ".join(parts)

class MultivariateTest(AdvancedTestBase):
    """Implementation of multivariate statistical tests"""
    
    def check_assumptions(self,
                         data: np.ndarray,
                         groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Check assumptions for multivariate tests"""
        n_samples, n_variables = data.shape
        
        assumptions = {
            'sample_size': self.check_sample_size(n_samples, n_variables),
            'missing_values': {
                'passed': not np.any(np.isnan(data)),
                'warning': "Missing values detected" if np.any(np.isnan(data)) else None,
                'details': f"Missing values: {np.isnan(data).sum()}"
            },
            'multicollinearity': self.check_multicollinearity(data)
        }
        
        if groups is not None:
            # Add group-specific checks
            unique_groups = np.unique(groups)
            min_group_size = min(np.sum(groups == g) for g in unique_groups)
            
            assumptions['group_sizes'] = {
                'passed': min_group_size >= n_variables + 2,
                'warning': "Some groups may be too small for reliable analysis"
                          if min_group_size < n_variables + 2 else None,
                'details': f"Minimum group size: {min_group_size}"
            }
        
        all_passed = all(check['passed'] for check in assumptions.values())
        warnings = [check['warning'] for check in assumptions.values() 
                   if check['warning'] is not None]
        
        return {
            'assumptions': assumptions,
            'all_passed': all_passed,
            'warnings': warnings
        }
    
    def run_manova(self,
                   data: np.ndarray,
                   groups: np.ndarray,
                   variable_names: Optional[List[str]] = None) -> AdvancedTestResult:
        """
        Run Multivariate Analysis of Variance (MANOVA).
        
        Parameters:
        -----------
        data : np.ndarray
            Matrix of dependent variables (n_samples x n_variables)
        groups : np.ndarray
            Group labels for samples
        variable_names : Optional[List[str]]
            Names of dependent variables
        
        Returns:
        --------
        AdvancedTestResult
            Complete test results including statistics and visualizations
        """
        from scipy.stats import f
        
        # Check assumptions
        assumptions = self.check_assumptions(data, groups)
        
        # Generate variable names if not provided
        if variable_names is None:
            variable_names = [f"Variable_{i+1}" for i in range(data.shape[1])]
        
        # Calculate MANOVA statistics
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        n_variables = data.shape[1]
        
        # Calculate means and covariance matrices
        group_means = []
        within_cov = np.zeros((n_variables, n_variables))
        total_mean = np.mean(data, axis=0)
        
        for group in unique_groups:
            group_data = data[groups == group]
            group_mean = np.mean(group_data, axis=0)
            group_means.append(group_mean)
            
            # Update within-groups covariance
            centered_data = group_data - group_mean
            within_cov += centered_data.T @ centered_data
        
        # Calculate between-groups covariance
        between_cov = np.zeros((n_variables, n_variables))
        total_n = len(data)
        
        for group, mean in zip(unique_groups, group_means):
            n_group = np.sum(groups == group)
            diff = mean - total_mean
            between_cov += n_group * np.outer(diff, diff)
        
        # Calculate Wilks' Lambda
        try:
            eigenvalues = np.linalg.eigvals(
                np.linalg.inv(within_cov) @ between_cov
            )
            wilks_lambda = np.prod(1 / (1 + eigenvalues))
            
            # Calculate test statistic and p-value
            n = total_n - n_groups
            p = n_variables
            t = n_groups - 1
            
            # Approximate F-statistic
            w = n - (p + t + 1)/2
            u = (p * t - 2)/4
            if p**2 + t**2 - 5 > 0:
                v = np.sqrt((p**2 * t**2 - 4)/(p**2 + t**2 - 5))
            else:
                v = 1
                
            df1 = p * t
            df2 = w * v - u
            
            f_stat = ((1 - wilks_lambda**(1/v)) / (wilks_lambda**(1/v))) * (df2/df1)
            p_value = 1 - f.cdf(f_stat, df1, df2)
            
            # Calculate effect size (partial η²)
            effect_size = 1 - wilks_lambda
            
            # Calculate univariate ANOVAs for each variable
            univariate_results = []
            for i, var_name in enumerate(variable_names):
                f_val, p_val = f_oneway(*[data[groups == g, i] 
                                        for g in unique_groups])
                univariate_results.append({
                    'variable': var_name,
                    'f_statistic': f_val,
                    'p_value': p_val
                })
            
            additional_metrics = {
                'wilks_lambda': wilks_lambda,
                'n_groups': n_groups,
                'n_variables': n_variables,
                'degrees_of_freedom': (df1, df2),
                'univariate_results': univariate_results,
                'group_means': dict(zip(unique_groups, group_means)),
                'total_mean': total_mean
            }
            
            # Create visualizations
            figures = [
                self._create_group_comparison_plot(data, groups, variable_names),
                self._create_canonical_plot(data, groups, eigenvalues),
                self._create_univariate_results_plot(univariate_results)
            ]
            
            # Generate interpretation
            interpretation = self._generate_manova_interpretation(
                p_value, effect_size, additional_metrics, variable_names
            )
            
            return AdvancedTestResult(
                test_name="MANOVA",
                test_statistic=f_stat,
                p_value=p_value,
                effect_size=effect_size,
                confidence_intervals=None,  # MANOVA doesn't provide simple CIs
                additional_metrics=additional_metrics,
                figures=figures,
                interpretation=interpretation,
                assumptions_summary=assumptions,
                warnings=assumptions['warnings']
            )
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error calculating MANOVA statistics: {str(e)}")
    
    def _create_group_comparison_plot(self,
                                    data: np.ndarray,
                                    groups: np.ndarray,
                                    variable_names: List[str]) -> go.Figure:
        """Create visualization comparing groups across variables"""
        fig = make_subplots(rows=1, cols=data.shape[1],
                           subplot_titles=variable_names)
        
        unique_groups = np.unique(groups)
        colors = px.colors.qualitative.Set3[:len(unique_groups)]
        
        for i, var_name in enumerate(variable_names):
            for group, color in zip(unique_groups, colors):
                group_data = data[groups == group, i]
                
                fig.add_trace(
                    go.Box(
                        y=group_data,
                        name=str(group),
                        marker_color=color,
                        showlegend=(i == 0)
                    ),
                    row=1, col=i+1
                )
        
        fig.update_layout(
            title='Group Comparisons by Variable',
            height=400
        )
        
        return fig
    
    def _create_canonical_plot(self,
                             data: np.ndarray,
                             groups: np.ndarray,
                             eigenvalues: np.ndarray) -> go.Figure:
        """Create canonical correlation plot"""
        # Perform canonical discriminant analysis
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis(n_components=2)
        transformed = lda.fit_transform(data, groups)
        
        fig = go.Figure()
        
        unique_groups = np.unique(groups)
        colors = px.colors.qualitative.Set3[:len(unique_groups)]
        
        for group, color in zip(unique_groups, colors):
            mask = groups == group
            fig.add_trace(go.Scatter(
                x=transformed[mask, 0],
                y=transformed[mask, 1] if transformed.shape[1] > 1 else np.zeros(sum(mask)),
                mode='markers',
                name=str(group),
                marker=dict(color=color)
            ))
        
        fig.update_layout(
            title='Canonical Discriminant Analysis',
            xaxis_title='First Canonical Variable',
            yaxis_title='Second Canonical Variable' if transformed.shape[1] > 1 else '',
            showlegend=True
        )
        
        return fig
    
    def _create_univariate_results_plot(self,
                                      univariate_results: List[Dict[str, Any]]) -> go.Figure:
        """Create plot of univariate test results"""
        fig = go.Figure()
        
        variables = [r['variable'] for r in univariate_results]
        f_stats = [r['f_statistic'] for r in univariate_results]
        p_values = [r['p_value'] for r in univariate_results]
        
        # Add F-statistics
        fig.add_trace(go.Bar(
            x=variables,
            y=f_stats,
            name='F-statistic'
        ))
        
        # Add p-values on secondary y-axis
        fig.add_trace(go.Scatter(
            x=variables,
            y=p_values,
            name='p-value',
            yaxis='y2',
            mode='lines+markers'
        ))
        
        # Add significance level line
        fig.add_hline(
            y=self.alpha,
            line_dash="dash",
            line_color="red",
            yaxis='y2',
            annotation_text="Significance Level"
        )
        
        fig.update_layout(
            title='Univariate Test Results',
            xaxis_title='Variable',
            yaxis_title='F-statistic',
            yaxis2=dict(
                title='p-value',
                overlaying='y',
                side='right'
            ),
            showlegend=True
        )
        
        return fig
    
    def _generate_manova_interpretation(self,
                                      p_value: float,
                                      effect_size: float,
                                      metrics: Dict[str, Any],
                                      variable_names: List[str]) -> str:
        """Generate detailed interpretation of MANOVA results"""
        parts = []
        
        # Overall test result
        if p_value < self.alpha:
            parts.append(
                f"The MANOVA revealed significant differences among the groups "
                f"(Wilks' Λ = {metrics['wilks_lambda']:.3f}, F({metrics['degrees_of_freedom'][0]}, "
                f"{metrics['degrees_of_freedom'][1]}) = {metrics['test_statistic']:.3f}, "
                f"p = {p_value:.4f})."
            )
        else:
            parts.append(
                f"The MANOVA did not reveal significant differences among the groups "
                f"(Wilks' Λ = {metrics['wilks_lambda']:.3f}, F({metrics['degrees_of_freedom'][0]}, "
                f"{metrics['degrees_of_freedom'][1]}) = {metrics['test_statistic']:.3f}, "
                f"p = {p_value:.4f})."
            )
        
        # Effect size interpretation
        effect_size_desc = (
            "small" if effect_size < 0.06 else
            "medium" if effect_size < 0.14 else
            "large"
        )
        parts.append(
            f"The effect size was {effect_size_desc} "
            f"(partial η² = {effect_size:.3f})."
        )
        
        # Univariate results
        significant_vars = []
        nonsig_vars = []
        
        for result in metrics['univariate_results']:
            if result['p_value'] < self.alpha:
                significant_vars.append(
                    f"{result['variable']} (F = {result['f_statistic']:.3f}, "
                    f"p = {result['p_value']:.4f})"
                )
            else:
                nonsig_vars.append(result['variable'])
        
        if significant_vars:
            parts.append(
                "Follow-up univariate ANOVAs showed significant group differences "
                f"for: {', '.join(significant_vars)}."
            )
        if nonsig_vars:
            parts.append(
                f"No significant group differences were found for: "
                f"{', '.join(nonsig_vars)}."
            )
        
        return " ".join(parts)

def get_test_description(test_type: str) -> Dict[str, Any]:
    """Get comprehensive description of advanced statistical tests"""
    descriptions = {
        "Wald Test": {
            "description": "Test hypotheses about parameter estimates in statistical models",
            "assumptions": [
                "Parameter estimates approximately normally distributed",
                "Accurately estimated standard errors",
                "Adequate sample size for asymptotic properties",
                "Independent observations"
            ],
            "use_cases": [
                "Testing regression coefficients",
                "Parameter constraints in statistical models",
                "Hypothesis testing in maximum likelihood estimation",
                "Model comparison and selection"
            ],
            "example": (
                "Testing whether regression coefficients are significantly "
                "different from zero or testing specific parameter constraints"
            ),
            "interpretation_guide": {
                "test_statistic": "Larger values indicate stronger evidence against null",
                "p_value": "Probability of observing such extreme values under null",
                "effect_size": "Standardized coefficient size indicates practical significance",
                "confidence_intervals": "Range of plausible parameter values"
            },
            "alternatives": [
                "Likelihood ratio test for nested models",
                "Score test for model constraints",
                "Bootstrap tests for small samples"
            ]
        },
        "Granger Causality Test": {
            "description": "Test whether one time series helps predict another",
            "assumptions": [
                "Stationary time series",
                "No serial correlation in residuals",
                "Adequate sample size for lag order",
                "Linear relationships between variables"
            ],
            "use_cases": [
                "Economic forecasting",
                "Financial market analysis",
                "Temporal relationship analysis",
                "Predictive modeling validation"
            ],
            "example": (
                "Testing whether changes in interest rates help predict "
                "changes in housing prices"
            ),
            "interpretation_guide": {
                "test_statistic": "F-statistic comparing restricted and unrestricted models",
                "p_value": "Probability of observing such predictive power by chance",
                "lag_selection": "Optimal lag determined by information criteria"
            },
            "alternatives": [
                "Transfer entropy for nonlinear relationships",
                "Cross-correlation analysis",
                "Vector autoregression (VAR) modeling"
            ]
        },
        "MANOVA": {
            "description": "Compare groups on multiple dependent variables simultaneously",
            "assumptions": [
                "Multivariate normality within groups",
                "Homogeneity of covariance matrices",
                "Independent observations",
                "Adequate sample size relative to variables"
            ],
            "use_cases": [
                "Comparing groups on multiple outcomes",
                "Analyzing multivariate treatment effects",
                "Profile analysis across multiple measures",
                "Group discrimination studies"
            ],
            "example": (
                "Testing whether different teaching methods affect both student "
                "performance and satisfaction simultaneously"
            ),
            "interpretation_guide": {
                "test_statistic": "Wilks' Lambda or equivalent test statistic",
                "p_value": "Overall significance of group differences",
                "effect_size": "Partial η² indicates variance explained",
                "univariate_results": "Follow-up tests for individual variables"
            },
            "alternatives": [
                "Separate univariate ANOVAs",
                "Discriminant function analysis",
                "Profile analysis"
            ]
        }
    }
    
    return descriptions.get(test_type, {})

def recommend_advanced_test(data_properties: Dict[str, Any]) -> str:
    """
    Recommend appropriate advanced test based on data properties.
    
    Parameters:
    -----------
    data_properties : Dict[str, Any]
        Dictionary containing data characteristics:
        - time_series: bool
        - multiple_groups: bool
        - multiple_dvs: bool
        - sample_size: int
        - parameter_test: bool
    
    Returns:
    --------
    str
        Recommended test type with explanation
    """
    recommendations = []
    
    if data_properties.get('time_series'):
        recommendations.append(
            "Granger Causality Test - Suitable for analyzing temporal relationships "
            "and predictive patterns in time series data."
        )
    
    if data_properties.get('multiple_groups') and data_properties.get('multiple_dvs'):
        recommendations.append(
            "MANOVA - Appropriate for comparing groups on multiple dependent "
            "variables while controlling family-wise error rate."
        )
    
    if data_properties.get('parameter_test'):
        recommendations.append(
            "Wald Test - Suitable for testing hypotheses about model parameters "
            "and coefficient constraints."
        )
    
    if len(recommendations) == 0:
        return "Unable to make specific recommendation. Please provide more information about your data and analysis goals."
    
    return "\n\n".join(recommendations)

def validate_advanced_data(data: Union[np.ndarray, pd.DataFrame],
                         test_type: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Validate and preprocess data for advanced statistical tests.
    
    Parameters:
    -----------
    data : Union[np.ndarray, pd.DataFrame]
        Input data
    test_type : str
        Type of test to be performed
    
    Returns:
    --------
    Tuple[np.ndarray, Dict[str, Any]]
        Processed data and validation info
    """
    validation_info = {
        'warnings': [],
        'modifications': [],
        'is_valid': True
    }
    
    try:
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            validation_info['modifications'].append(
                "Converted DataFrame to numpy array"
            )
            data = data.values
        
        # Check data type
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be numpy array or pandas DataFrame")
        
        # Handle missing values
        if np.any(np.isnan(data)):
            validation_info['warnings'].append(
                f"Found {np.isnan(data).sum()} missing values"
            )
            validation_info['modifications'].append(
                "Removed rows with missing values"
            )
            data = data[~np.any(np.isnan(data), axis=1)]
        
        # Test-specific validation
        if test_type == "MANOVA":
            if data.ndim != 2:
                raise ValueError("MANOVA requires 2D data array")
            if data.shape[1] < 2:
                raise ValueError("MANOVA requires at least two dependent variables")
                
        elif test_type == "Granger Causality Test":
            if data.ndim != 1 and data.shape[1] != 2:
                raise ValueError("Granger causality test requires two time series")
                
        elif test_type == "Wald Test":
            if data.ndim != 2:
                raise ValueError("Wald test requires 2D data array")
        
        return data, validation_info
        
    except Exception as e:
        validation_info['is_valid'] = False
        validation_info['warnings'].append(str(e))
        return data, validation_info

def initialize_advanced_test(test_type: str,
                           alpha: float = 0.05) -> Union[WaldTest, GrangerCausalityTest, MultivariateTest]:
    """
    Initialize appropriate advanced test class.
    
    Parameters:
    -----------
    test_type : str
        Type of test to initialize
    alpha : float
        Significance level
    
    Returns:
    --------
    Union[WaldTest, GrangerCausalityTest, MultivariateTest]
        Initialized test object
    """
    test_map = {
        "Wald Test": WaldTest,
        "Granger Causality Test": GrangerCausalityTest,
        "MANOVA": MultivariateTest
    }
    
    test_class = test_map.get(test_type)
    if test_class is None:
        raise ValueError(f"Unsupported test type: {test_type}")
    
    return test_class(alpha=alpha)