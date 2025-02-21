import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Import our custom components
from core import (
    TestResult,
    StatisticalTestBase,
    DataValidationError,
    validate_numeric_data,
    validate_categorical_data,
    ensure_valid_data
)

from assumption_checking import (
    AssumptionChecker,
    AssumptionResult,
    AssumptionCheck
)

from visualizations import TestVisualizations

# Import test implementations
from group_tests import StudentTTest, AnovaTest
from relation_tests import PearsonCorrelation, ChiSquareTest
from paired_tests import WilcoxonTest, PairedTTest, SignTest
from advanced_tests import WaldTest

# Import test descriptions
from descriptions import (
    get_test_description,
    get_test_categories,
    get_test_selection_guide,
    get_assumption_guide,
    get_effect_size_guide
)

# Initialize visualization utilities
viz = TestVisualizations()

def show():
    """Main function to render the statistical testing page."""
    st.title("Statistical Testing")
    
    if not st.session_state.dataframes:
        st.warning("Please upload data in the Data Upload section first.")
        return
    
    # Create tabs for different stages of analysis
    intro_tab, analysis_tab = st.tabs(["📚 Introduction & Guidance", "🔍 Analysis"])
    
    with intro_tab:
        _render_introduction()
    
    with analysis_tab:
        try:
            _render_analysis_workflow()
        except Exception as e:
            st.error("An error occurred during analysis")
            st.exception(e)

def _render_introduction():
    """Render the introduction and guidance tab."""
    st.markdown("""
    ## Statistical Testing Guide
    
    Statistical testing helps you make data-driven decisions by systematically analyzing your data.
    Follow these steps for a successful analysis:
    
    1. **Choose Your Test**: Select based on your data type and research question
    2. **Check Assumptions**: Verify your data meets test requirements
    3. **Run Analysis**: Perform the test and interpret results
    4. **Draw Conclusions**: Make informed decisions based on results
    """)
    
    # Get comprehensive guides from descriptions module
    test_categories = get_test_categories()
    test_selection = get_test_selection_guide()
    assumption_guide = get_assumption_guide()
    effect_size_guide = get_effect_size_guide()
    
    # Display guides in expandable sections
    with st.expander("🎯 How to Choose the Right Test"):
        for category, info in test_categories.items():
            st.markdown(f"### {category}")
            st.markdown(info["description"])
            st.markdown("**When to use:**")
            for use_case in info["when_to_use"]:
                st.markdown(f"- {use_case}")
    
    with st.expander("📊 Test Selection Guide"):
        for scenario, guide in test_selection.items():
            st.markdown(f"### {scenario}")
            st.markdown(guide["help"])
    
    with st.expander("✅ Understanding Assumptions"):
        for assumption, info in assumption_guide.items():
            st.markdown(f"### {assumption}")
            st.markdown(info["description"])
            st.markdown("**How to check:**")
            for check in info["how_to_check"]:
                st.markdown(f"- {check}")

def _render_analysis_workflow():
    """Render the main analysis workflow."""
    # Data Selection
    st.header("1️⃣ Data Selection")
    selected_file = st.selectbox(
        "Choose your dataset",
        list(st.session_state.dataframes.keys())
    )
    df = st.session_state.dataframes[selected_file].copy()
    
    # Data Overview
    _render_data_overview(df)
    
    # Test Selection
    st.header("2️⃣ Test Selection")
    test_name = _render_test_selection(df)
    
    if not test_name:
        return
    
    # Test Configuration
    st.header("3️⃣ Test Configuration")
    test_instance = _create_test_instance(test_name)
    
    if test_instance is None:
        st.error("Invalid test selection")
        return
    
    # Render test form
    test_data = _render_test_form(test_name, df, test_instance)
    
    if test_data is None:
        return
    
    # Run test and display results
    _run_and_display_test(test_instance, test_data)

def _render_data_overview(df: pd.DataFrame):
    """Render data overview section."""
    cols = st.columns(4)
    
    # Basic metrics
    with cols[0]:
        st.metric("Rows", df.shape[0])
    with cols[1]:
        st.metric("Columns", df.shape[1])
    with cols[2]:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        st.metric("Numeric Columns", len(numeric_cols))
    with cols[3]:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        st.metric("Categorical Columns", len(cat_cols))
    
    # Detailed data overview
    with st.expander("📊 Data Overview", expanded=False):
        st.dataframe(df.head())
        
        st.markdown("### Column Information")
        col_info = pd.DataFrame({
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info)

def _render_test_selection(df: pd.DataFrame) -> Optional[str]:
    """Render test selection section and return selected test name."""
    # Help guide test selection based on data characteristics
    test_category = _recommend_test_category(df)
    test_type = st.radio(
        "Select Test Category",
        ["Parametric Tests", "Non-Parametric Tests"],
        index=0 if test_category == "parametric" else 1,
        help="""
        Parametric Tests: Assume normal distribution, better for larger samples
        Non-Parametric Tests: No distribution assumptions, better for small samples
        """
    )
    
    # Available tests based on data characteristics
    test_options = _get_available_tests(df, test_type)
    
    if not test_options:
        st.warning("No suitable tests found for the current data structure.")
        return None
    
    test_name = st.selectbox(
        "Select Statistical Test",
        options=test_options,
        help="Choose the appropriate test based on your hypothesis and data type"
    )
    
    # Show test description and guidance
    if test_name:
        with st.expander("ℹ️ About This Test", expanded=True):
            description = get_test_description(test_name)
            if description:
                st.markdown(f"**Purpose:** {description['purpose']}")
                st.markdown("**When to use this test:**")
                for use_case in description['when_to_use']:
                    st.markdown(f"- {use_case}")
                
                st.markdown("**Key Assumptions:**")
                for assumption in description['assumptions']:
                    st.markdown(f"✓ {assumption}")
    
    return test_name

# [... continue with remaining helper functions from previous version, 
# but updated to use our custom components for assumption checking,
# visualization, and core functionality ...]

def _validate_test_data(test_data: Dict[str, Any], test_name: str) -> bool:
    """Validate test data using our core validation functions."""
    try:
        if test_name == "Student's t-test":
            ensure_valid_data(
                test_data,
                {'group1': 'numeric', 'group2': 'numeric'}
            )
        elif test_name == "Chi-Square Test":
            ensure_valid_data(
                test_data,
                {'table': 'categorical'}
            )
        # Add validation for other tests...
        
        return True
    
    except DataValidationError as e:
        st.error(f"Data validation error: {str(e)}")
        return False
    
    except Exception as e:
        st.error(f"Unexpected error during data validation: {str(e)}")
        return False

def _run_and_display_test(
    test_instance: StatisticalTestBase,
    test_data: Dict[str, Any]
):
    """Run test and display results using our custom components."""
    try:
        # Validate test data
        if not _validate_test_data(test_data, test_instance.__class__.__name__):
            return
        
        # Check assumptions using our AssumptionChecker
        with st.spinner("Checking assumptions..."):
            assumptions = test_instance.check_assumptions(test_data)
        
        # Display assumption results
        _display_assumption_results(assumptions)
        
        # Run test if assumptions allow
        if assumptions.proceed:
            with st.spinner("Running statistical test..."):
                result = test_instance.run_test(test_data)
            
            # Display results using our visualization components
            _display_test_results(result)
            
        else:
            st.error("Cannot proceed with test due to assumption violations.")
            st.markdown("""
            **Recommended Actions:**
            1. Review the warnings and recommendations above
            2. Consider using a different statistical test
            3. Check your data for potential issues
            """)
    
    except Exception as e:
        st.error(f"Error running test: {str(e)}")
        st.exception(e)

def _display_assumption_results(assumptions: AssumptionCheck):
    """Display assumption check results using our custom components."""
    st.header("4️⃣ Assumption Check Results")
    
    # Display each assumption check
    for check in assumptions.checks:
        result_color = {
            'Pass': 'green',
            'Warning': 'orange',
            'Fail': 'red',
            'Skip': 'grey'
        }.get(check['result'], 'black')
        
        st.markdown(f"""
        **{check['test_name']}**  
        :{result_color}[{check['result']}]  
        _{check['details']}_
        """)
    
    # Display warnings and recommendations
    if assumptions.warnings or assumptions.recommendations:
        col1, col2 = st.columns(2)
        
        with col1:
            if assumptions.warnings:
                st.warning("⚠️ Warnings")
                for warning in assumptions.warnings:
                    st.markdown(f"- {warning}")
        
        with col2:
            if assumptions.recommendations:
                st.info("💡 Recommendations")
                for recommendation in assumptions.recommendations:
                    st.markdown(f"- {recommendation}")

def _display_test_results(result: TestResult):
    """Display test results using our visualization components."""
    st.header("5️⃣ Test Results")
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Statistic", f"{result.test_statistic:.4f}")
    with col2:
        st.metric("p-value", f"{result.p_value:.4f}")
    with col3:
        if result.effect_size is not None:
            st.metric("Effect Size", f"{result.effect_size:.4f}")
    
    # Interpretation
    st.subheader("Interpretation")
    st.write(result.interpretation)
    
    # Additional metrics
    if result.additional_metrics:
        with st.expander("📊 Additional Metrics", expanded=True):
            for metric_name, value in result.additional_metrics.items():
                if isinstance(value, dict):
                    st.write(f"**{metric_name}:**")
                    st.json(value)
                elif isinstance(value, pd.DataFrame):
                    st.write(f"**{metric_name}:**")
                    st.dataframe(value)
                else:
                    st.write(f"**{metric_name}:** {value}")
    
    # Display visualizations using our plotting utilities
    if result.plots:
        st.subheader("Visualizations")
        for plot in result.plots:
            st.plotly_chart(plot, use_container_width=True)
    
    # Save results if requested
    if st.checkbox("Save results to session"):
        _save_results_to_session(result)

def _save_results_to_session(result: TestResult):
    """Save test results to session state."""
    if 'test_results' not in st.session_state:
        st.session_state.test_results = []
    
    # Store summary of results
    summary = {
        'test_type': result.__class__.__name__,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_statistic': result.test_statistic,
        'p_value': result.p_value,
        'effect_size': result.effect_size,
        'significant': result.p_value < 0.05  # Default significance level
    }
    
    st.session_state.test_results.append(summary)
    st.success("Results saved to session!")