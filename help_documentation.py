import streamlit as st

def show():
    """
    Render the Help and Documentation page with comprehensive guide and instructions.
    """
    st.title("Help and Documentation")

    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "Getting Started & Core Features", 
        "Analysis Features", 
        "Best Practices & Tips", 
        "Support"
    ])

    with tab1:
        st.header("🚀 Getting Started")
        st.write("""
        1. Start by uploading your data in the Data Upload page
        2. Preview and verify your data in the interactive grid view
        3. Navigate through different features using the sidebar menu
        4. Use the Export/Download page to save your work at any time
        """)

        st.header("📤 Data Upload & Manipulation")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Supported Formats")
            st.write("""
            - Recommended: CSV files (UTF-8 encoding) 
            - Excel files (.xlsx)
            """)
        with col2:
            st.subheader("Features")
            st.write("""
            - Interactive preview with sorting and filtering
            - Direct data editing in preview grid
            - Automatic code generation for reproducibility 
            - Changes can be saved directly
            """)
        
        with st.expander("Important Considerations for Data Upload"):
            st.write("""
            - Ensure your CSV files are properly formatted with consistent delimiters
            - Check column headers for special characters that might cause issues
            - Verify data types are correctly interpreted after upload
            - Maximum file size limit: Check Streamlit Cloud documentation for current limits
            - All data is stored in session state and will be cleared when the session ends
            """)

        st.header("📋 Data Profiling")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Analysis Options")
            st.write("""
            - **Minimal Mode**: Faster, suitable for large datasets
            - **Complete Mode**: More detailed analysis, better for smaller datasets
            """)
        with col4:
            st.subheader("Key Features")
            st.write("""
            - Comprehensive data quality reports
            - Column-specific analysis
            - Correlation detection
            - Missing value analysis
            - Distribution visualizations
            """)
        
        with st.expander("Dataset Comparison Guide"):
            st.write("""
            Prerequisites for comparison reports:
            - Need a categorical column with exactly two unique values
            - Can create such columns using the English to Python (recommended) or the Data Manipulation page
            - Example: "Create a new column 'age_group' with 'young' for age < 30 and 'senior' for age >= 30"
            """)

    with tab2:
        st.header("💬 English to Python")
        st.subheader("Supported Operations")
        ops_col1, ops_col2 = st.columns(2)
        with ops_col1:
            st.write("Data Analysis")
            st.write("""
            - Basic statistics
            - Group operations
            - Complex calculations
            - Feature engineering
            """)
        with ops_col2:
            st.write("Data Visualization")
            st.write("""
            - Various chart types
            - Customizable plots
            - Data manipulation
            - State management
            """)

        with st.expander("Query Best Practices"):
            st.write("""
            - Use exact column names
            - Be specific about operations
            - Break complex queries into steps
            - Specify chart types for visualizations
            - Use clear metrics for calculations
            """)

        st.header("🔑 Key Driver Analysis")
        kda_col1, kda_col2 = st.columns(2)
        with kda_col1:
            st.subheader("Features")
            st.write("""
            - Relative importance analysis
            - Johnson's Relative Weights method
            - Interactive visualizations
            - Detailed statistical reporting
            """)
        with kda_col2:
            st.subheader("Prerequisites")
            st.write("""
            - Numeric data for target and features
            - No missing values
            - Sufficient sample size
            - Review of multicollinearity
            """)

        st.header("📊 Statistical Testing")
        test_col1, test_col2 = st.columns(2)
        with test_col1:
            st.subheader("Parametric Tests")
            st.write("""
            - Student's t-test
            - ANOVA
            - Pearson Correlation
            - Linear Regression
            """)
        with test_col2:
            st.subheader("Non-Parametric Tests")
            st.write("""
            - Mann-Whitney U Test
            - Chi-Square Test
            - Wilcoxon Signed-Rank Test
            - Kruskal-Wallis H Test
            """)

        st.header("🤖 ML Prototype")
        ml_col1, ml_col2 = st.columns(2)
        with ml_col1:
            st.subheader("Supported Problems")
            st.write("""
            Classification:
            - Logistic Regression
            - Random Forest
            - XGBoost
            
            Regression:
            - Linear Regression
            - Random Forest
            - XGBoost
            """)
        with ml_col2:
            st.subheader("Features")
            st.write("""
            - Automatic train-test split
            - Model performance metrics
            - Interactive prediction interface
            - Model export functionality
            """)

    with tab3:
        st.header("💡 Best Practices & Tips")
        
        st.subheader("Data Preparation")
        st.write("""
        1. Clean your data before upload
        2. Handle missing values appropriately
        3. Check for data quality issues
        4. Document any data transformations
        """)

        st.subheader("Analysis Flow")
        st.write("""
        1. Start with data profiling
        2. Investigate relationships
        3. Perform necessary statistical tests
        4. Build and validate models
        5. Export results regularly
        """)

        st.subheader("Performance Optimization")
        st.write("""
        1. Use minimal profiling for large datasets
        2. Select relevant feature subsets
        3. Consider data sampling for large datasets
        4. Clear unused variables when possible
        """)

        st.header("🔧 Troubleshooting")
        with st.expander("Common Issues and Solutions"):
            st.write("""
            Upload Issues:
            - Check file format: use CSV UTF-8 
            - Verify encoding
            - Reduce file size if needed
            - Check column names

            Performance Issues:
            - Use minimal profiling
            - Reduce dataset size
            - Clear browser cache
            - Refresh session

            Analysis Issues:
            - Verify data types
            - Check for missing values
            - Validate assumptions
            - Review error messages

            Export Issues:
            - Check file sizes
            - Use supported formats
            - Download before session end
            - Verify data has been processed
            """)

    with tab4:
        st.header("🤝 Support & Resources")
        
        st.subheader("Documentation")
        st.write("For more information, visit:")
        st.markdown("[Streamlit Documentation](https://docs.streamlit.io)")

        st.divider()
        
        st.header("🔒 Privacy & Security")
        st.write("""
        - Data is stored only in session state
        - Session data is cleared when browser is closed
        - No data persistence between sessions
        - No external storage used
        
        Important: Don't upload sensitive data as this is a public app.
        """)