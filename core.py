"""
Core components for statistical testing functionality.
Provides base classes and data structures used across all statistical tests.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import plotly.graph_objects as go

@dataclass
class TestResult:
    """
    Stores and organizes statistical test results.
    
    Attributes:
        test_statistic: The calculated test statistic
        p_value: The test's p-value
        effect_size: Standardized measure of effect size
        interpretation: Plain-language interpretation of results
        additional_metrics: Any additional test-specific metrics
        plots: List of generated visualizations
        diagnostics: List of diagnostic messages
    """
    test_statistic: float
    p_value: float
    effect_size: Optional[float] = None
    interpretation: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    plots: List[go.Figure] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)

    def add_diagnostic(self, message: str) -> None:
        """Add a diagnostic message."""
        self.diagnostics.append(message)

    def add_plot(self, plot: go.Figure) -> None:
        """Add a visualization plot."""
        self.plots.append(plot)

    def add_metric(self, name: str, value: Any) -> None:
        """Add an additional metric."""
        self.additional_metrics[name] = value

@dataclass
class AssumptionCheck:
    """
    Stores results of statistical assumption checks.
    
    Attributes:
        checks: List of individual assumption check results
        recommendations: List of suggested actions
        warnings: List of warning messages
        proceed: Whether to proceed with the test
    """
    checks: List[Dict[str, Any]]
    recommendations: List[str]
    warnings: List[str]
    proceed: bool

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation."""
        self.recommendations.append(recommendation)

    def add_check(self, check: Dict[str, Any]) -> None:
        """Add an assumption check result."""
        self.checks.append(check)

class StatisticalTestBase(ABC):
    """
    Abstract base class for all statistical tests.
    Defines the interface that all statistical tests must implement.
    """
    
    @abstractmethod
    def run_test(self, data: Dict[str, Any]) -> TestResult:
        """
        Run the statistical test.
        
        Args:
            data: Dictionary containing test data and parameters
            
        Returns:
            TestResult object containing test results
        """
        pass
    
    @abstractmethod
    def check_assumptions(self, data: Dict[str, Any]) -> AssumptionCheck:
        """
        Check test assumptions.
        
        Args:
            data: Dictionary containing test data
            
        Returns:
            AssumptionCheck object containing assumption check results
        """
        pass
    
    @abstractmethod
    def create_visualizations(self, data: Dict[str, Any], result: TestResult) -> List[go.Figure]:
        """
        Create test-specific visualizations.
        
        Args:
            data: Dictionary containing test data
            result: TestResult object containing test results
            
        Returns:
            List of plotly figures
        """
        pass

def validate_numeric_data(data: np.ndarray, name: str = "data") -> Tuple[bool, List[str]]:
    """
    Validate numeric data for statistical testing.
    
    Args:
        data: Array to validate
        name: Name of the data for error messages
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not isinstance(data, (np.ndarray, pd.Series)):
        errors.append(f"{name} must be a numpy array or pandas series")
        return False, errors
        
    if not np.issubdtype(data.dtype, np.number):
        errors.append(f"{name} must contain numeric values")
        return False, errors
        
    if len(data) == 0:
        errors.append(f"{name} cannot be empty")
        return False, errors
        
    if np.all(np.isnan(data)):
        errors.append(f"{name} cannot contain all NaN values")
        return False, errors
        
    return True, errors

def validate_categorical_data(data: np.ndarray, name: str = "data") -> Tuple[bool, List[str]]:
    """
    Validate categorical data for statistical testing.
    
    Args:
        data: Array to validate
        name: Name of the data for error messages
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not isinstance(data, (np.ndarray, pd.Series)):
        errors.append(f"{name} must be a numpy array or pandas series")
        return False, errors
        
    if len(data) == 0:
        errors.append(f"{name} cannot be empty")
        return False, errors
        
    if len(np.unique(data[~pd.isna(data)])) < 2:
        errors.append(f"{name} must have at least two unique values")
        return False, errors
        
    return True, errors

class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass

def ensure_valid_data(data: Dict[str, Any], requirements: Dict[str, str]) -> None:
    """
    Validate data against requirements and raise DataValidationError if invalid.
    
    Args:
        data: Dictionary of data to validate
        requirements: Dictionary mapping data keys to required types ('numeric' or 'categorical')
        
    Raises:
        DataValidationError: If data doesn't meet requirements
    """
    errors = []
    
    for key, req_type in requirements.items():
        if key not in data:
            errors.append(f"Missing required data: {key}")
            continue
            
        validate_func = (validate_numeric_data if req_type == 'numeric' 
                        else validate_categorical_data)
        is_valid, field_errors = validate_func(data[key], key)
        
        if not is_valid:
            errors.extend(field_errors)
    
    if errors:
        raise DataValidationError("\n".join(errors))
