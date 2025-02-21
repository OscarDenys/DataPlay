import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import datetime as dt
import psutil
import gc

def estimate_memory_usage(df):
    """
    Estimate memory usage of DataFrame in MB.
    """
    return df.memory_usage(deep=True).sum() / 1024 / 1024

def get_recommended_mode(df):
    """
    Recommend profile mode based on dataset size.
    """
    rows = len(df)
    cols = len(df.columns)
    memory_usage = estimate_memory_usage(df)
    
    if rows * cols > 1000000 or memory_usage > 1000:  # 1M cells or 1GB
        return "Minimal"
    return "Complete"

def preprocess_dataframe(df):
    """
    Preprocess dataframe for profiling.
    
    Parameters:
    df: pd.DataFrame, DataFrame to process
    
    Returns:
    pd.DataFrame: Processed DataFrame
    str: Any warnings generated during preprocessing
    """
    warnings = []
    processed_df = df.copy()
    
    # Convert object columns to categorical
    object_cols = processed_df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        processed_df[object_cols] = processed_df[object_cols].astype('category')
        warnings.append(f"Converted {len(object_cols)} text columns to categorical")
    
    # Handle datetime columns
    datetime_cols = processed_df.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        for col in datetime_cols:
            processed_df[col] = processed_df[col].astype(str)
        warnings.append(f"Converted {len(datetime_cols)} datetime columns to string")
    
    # Handle infinite values
    inf_cols = processed_df.isin([np.inf, -np.inf]).any()
    inf_cols = inf_cols[inf_cols].index
    if len(inf_cols) > 0:
        processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
        warnings.append(f"Replaced infinite values with NaN in {len(inf_cols)} columns")
    
    return processed_df, warnings

def get_profile_config(title, is_minimal=True):
    """
    Get configuration for profile report.
    
    Parameters:
    title: str, Report title
    is_minimal: bool, Whether to use minimal configuration
    
    Returns:
    dict: Profile configuration
    """
    config = {
        "title": title,
        "minimal": is_minimal,
        "explorative": True
    }
    
    if not is_minimal:
        config.update({
            "correlations": {
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": True},
                "phi_k": {"calculate": True},
                "cramers": {"calculate": True},
            },
            "interactions": {"value": True},
            "samples": {"head": 10},
            "variables": {
                "descriptions": {}
            }
        })
    
    return config

def show():
    """
    Render the data profiling page with enhanced user guidance and error handling.
    """
    st.title("Data Profiling")

    # Introduction and guidance
    with st.expander("ℹ️ About Data Profiling", expanded=True):
        st.markdown("""
        Data profiling provides a comprehensive analysis of your dataset, including:
        
        - 📊 Basic Statistics
        - 🔍 Data Quality Metrics
        - 🔗 Correlations
        - 📈 Distributions
        - ❌ Missing Values
        
        **Available Modes:**
        - **Minimal**: Faster, uses less memory, good for large datasets
        - **Complete**: More detailed analysis, including correlations and interactions
        
        **Note:** Large datasets may require significant processing time and memory.
        Consider using minimal mode for datasets over 100,000 rows.
        """)

    if not st.session_state.dataframes:
        st.warning("Please upload data in the Data Upload section first.")
        return

    # File selection with preview
    selected_file = st.selectbox(
        "Choose a dataset",
        list(st.session_state.dataframes.keys()),
        help="Select the dataset you want to analyze"
    )
    
    try:
        df = st.session_state.dataframes[selected_file].copy()
        
        # Show dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{estimate_memory_usage(df):.1f} MB")
        with col4:
            recommended_mode = get_recommended_mode(df)
            st.metric("Recommended Mode", recommended_mode)

        # Configuration options
        st.markdown("### 1️⃣ Profile Configuration")
        
        with st.expander("⚙️ Configuration Options", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                profile_type = st.radio(
                    "Variable Selection",
                    ["All Variables", "Subset of Variables"],
                    help="Choose whether to profile all variables or a subset"
                )
                
                if profile_type == "Subset of Variables":
                    selected_columns = st.multiselect(
                        "Select columns to profile",
                        df.columns,
                        help="Choose specific columns for analysis"
                    )
                    if selected_columns:
                        df = df[selected_columns]

            with col2:
                report_type = st.radio(
                    "Report Type",
                    ["Minimal", "Complete"],
                    index=0 if recommended_mode == "Minimal" else 1,
                    help="""
                    Minimal: Faster, basic analysis
                    Complete: Detailed analysis with correlations
                    """
                )
                
                if report_type != recommended_mode:
                    st.warning(f"Based on your dataset size, {recommended_mode} mode is recommended.")

        # Split dataset option
        st.markdown("### 2️⃣ Comparison Options")
        split_option = st.checkbox(
            "Compare two segments of data",
            help="Generate comparison report based on a categorical variable"
        )
        
        if split_option:
            # Filter to show only categorical columns with 2 unique values
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            valid_split_cols = [col for col in categorical_cols 
                              if df[col].nunique() == 2]
            
            if valid_split_cols:
                split_column = st.selectbox(
                    "Select column for comparison",
                    valid_split_cols,
                    help="Choose a categorical column with exactly two unique values"
                )
                
                unique_values = df[split_column].unique()
                df1 = df[df[split_column] == unique_values[0]].copy()
                df2 = df[df[split_column] == unique_values[1]].copy()
                
                st.info(f"""
                Data will be split into two groups:
                - Group 1 ({unique_values[0]}): {len(df1)} rows
                - Group 2 ({unique_values[1]}): {len(df2)} rows
                """)
            else:
                st.warning("""
                No suitable columns found for comparison. You need a categorical column with exactly two values.
                You can create one using the English to Python or Data Manipulation page.
                """)
                split_option = False

        if st.button("🚀 Generate Profile Report", type="primary"):
            try:
                # Create a status container
                status_container = st.empty()
                progress_bar = st.progress(0)
                
                def update_status(message, progress):
                    status_container.text(message)
                    progress_bar.progress(progress)
                
                if split_option and valid_split_cols:
                    update_status("Preprocessing first dataset...", 10)
                    df1, warnings1 = preprocess_dataframe(df1)
                    
                    update_status("Preprocessing second dataset...", 20)
                    df2, warnings2 = preprocess_dataframe(df2)
                    
                    update_status(f"Generating profile for {unique_values[0]}...", 30)
                    profile1 = ProfileReport(
                        df1,
                        **get_profile_config(
                            f"Group: {unique_values[0]}", 
                            is_minimal=(report_type == "Minimal")
                        )
                    )
                    
                    # Clear memory
                    gc.collect()
                    
                    update_status(f"Generating profile for {unique_values[1]}...", 60)
                    profile2 = ProfileReport(
                        df2,
                        **get_profile_config(
                            f"Group: {unique_values[1]}", 
                            is_minimal=(report_type == "Minimal")
                        )
                    )
                    
                    update_status("Creating comparison report...", 80)
                    comparison_report = profile1.compare(profile2)
                    report_html = comparison_report.to_html()
                    
                    # Show preprocessing warnings
                    if warnings1 or warnings2:
                        with st.expander("⚠️ Preprocessing Notices", expanded=False):
                            if warnings1:
                                st.write(f"Group {unique_values[0]}:")
                                for warning in warnings1:
                                    st.info(warning)
                            if warnings2:
                                st.write(f"Group {unique_values[1]}:")
                                for warning in warnings2:
                                    st.info(warning)
                    
                else:
                    update_status("Preprocessing dataset...", 20)
                    processed_df, warnings = preprocess_dataframe(df)
                    
                    if warnings:
                        with st.expander("⚠️ Preprocessing Notices", expanded=False):
                            for warning in warnings:
                                st.info(warning)
                    
                    update_status("Generating profile report...", 40)
                    profile = ProfileReport(
                        processed_df,
                        **get_profile_config(
                            f"Data Profile Report - {selected_file}",
                            is_minimal=(report_type == "Minimal")
                        )
                    )
                    
                    update_status("Preparing HTML report...", 80)
                    report_html = profile.to_html()
                
                # Display the report
                update_status("Displaying report...", 90)
                st.components.v1.html(
                    report_html,
                    width=1100,
                    height=800,
                    scrolling=True
                )
                
                # Add download button
                st.download_button(
                    "📥 Download Report",
                    report_html,
                    file_name=f"profile_report_{selected_file}.html",
                    mime="text/html",
                )
                
                # Clear status and show success
                status_container.empty()
                progress_bar.empty()
                st.success("✅ Profile report generated successfully!")
                
            except Exception as e:
                st.error(f"""
                Error generating profile report. This might happen due to:
                - Insufficient memory
                - Invalid data types
                - Too many unique categories
                
                Try using minimal mode or selecting fewer columns.
                
                Technical details: {str(e)}
                """)
                
                # Memory usage info
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                st.warning(f"""
                Current memory usage: {memory_usage:.1f} MB
                
                **Suggestions:**
                1. Use minimal mode for large datasets
                2. Select fewer columns
                3. Filter rows if possible
                4. Close other browser tabs
                """)
                
    except Exception as e:
        st.error(f"""
        Error loading the dataset. Please ensure:
        - The data is properly formatted
        - Column names are valid
        - Data types are supported
        
        Technical details: {str(e)}
        """)