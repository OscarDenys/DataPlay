import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler

def calculate_relative_importance(df, target, features):
    """
    Calculate relative importance using Johnson's Relative Weights method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    target (str): Target variable name
    features (list): List of feature names
    
    Returns:
    pd.DataFrame: Results dataframe with raw and normalized relative importance
    """
    # Prepare the data
    X = df[features].astype(float)
    y = df[target].astype(float)
    
    # Standardize features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Calculate correlation matrix
    corr_xx = np.corrcoef(X_scaled, rowvar=False)
    corr_xy = np.array([np.corrcoef(X_scaled[:, i], y_scaled)[0, 1] for i in range(X_scaled.shape[1])])
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr_xx)
    
    # Ensure numerical stability
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    # Calculate relative weights
    delta = np.sqrt(np.diag(eigenvalues))
    lambda_matrix = eigenvectors @ delta @ eigenvectors.T
    
    # Calculate raw importance
    raw_importance = np.square(lambda_matrix) @ np.square(corr_xy)
    
    # Calculate normalized importance
    total_importance = np.sum(raw_importance)
    if total_importance > 0:
        normalized_importance = (raw_importance / total_importance) * 100
    else:
        normalized_importance = np.zeros_like(raw_importance)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'feature': features,
        'raw_rel_imp': raw_importance,
        'norm_rel_imp': normalized_importance
    })
    
    # Sort by importance
    results_df = results_df.sort_values('raw_rel_imp', ascending=False)
    
    return results_df

def show():
    """
    Render the key driver analysis page with enhanced user guidance and visualization.
    """
    st.title("Key Driver Analysis")

    # Introduction and explanation
    with st.expander("ℹ️ About Key Driver Analysis", expanded=True):
        st.markdown("""
        Key Driver Analysis helps identify which factors (drivers) have the strongest influence 
        on your target variable. This analysis uses Johnson's Relative Weights method to 
        quantify the importance of each feature.
        
        **When to use:**
        - Identify factors influencing customer satisfaction
        - Determine key predictors of sales performance
        - Understand variables affecting product quality
        - Find important features for any numerical outcome
        
        **Requirements:**
        - Numeric target variable (what you want to explain)
        - Numeric feature variables (potential drivers)
        - No missing values in selected variables
        """)

    # Check if data is available
    if not st.session_state.dataframes:
        st.warning("Please upload data in the Data Upload section first.")
        return

    # File selection with preview
    selected_file = st.selectbox(
        "Choose a dataset",
        list(st.session_state.dataframes.keys()),
        help="Select the dataset containing your target and feature variables"
    )
    
    try:
        df = st.session_state.dataframes[selected_file].copy()
        
        # Show dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            st.metric("Numeric Columns", len(numeric_cols))
            
        if len(numeric_cols) < 2:
            st.error("This analysis requires at least 2 numeric columns (1 target + 1 feature).")
            return

        # Data preparation guidance
        st.markdown("### 1️⃣ Data Preparation")
        
        # Select target variable
        st.markdown("#### Select Target Variable")
        st.info("This is the outcome variable you want to explain (e.g., sales, satisfaction score).")
        target_variable = st.selectbox(
            "Target variable",
            numeric_cols,
            help="Choose the numeric variable you want to analyze"
        )

        # Select features for analysis
        st.markdown("#### Select Feature Variables")
        st.info("These are the potential drivers that might influence your target variable.")
        available_features = [col for col in numeric_cols if col != target_variable]
        features = st.multiselect(
            "Feature variables",
            available_features,
            default=available_features,
            help="Choose the variables you think might influence your target"
        )

        # Analysis options
        st.markdown("### 2️⃣ Analysis Options")
        with st.expander("⚙️ Advanced Options", expanded=False):
            handle_missing = st.checkbox(
                "Remove rows with missing values",
                value=True,
                help="Automatically remove rows with missing values in selected variables"
            )
            handle_infinite = st.checkbox(
                "Replace infinite values with NaN",
                value=True,
                help="Convert infinite values to missing values before analysis"
            )

        if st.button("🚀 Run Analysis", type="primary"):
            if not features:
                st.error("Please select at least one feature variable.")
                return
                
            try:
                with st.spinner("Performing Key Driver Analysis..."):
                    # Data preprocessing
                    analysis_df = df.copy()
                    
                    if handle_infinite:
                        analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan)
                    
                    if handle_missing:
                        analysis_df = analysis_df.dropna(subset=[target_variable] + features)
                    
                    # Check remaining sample size
                    if len(analysis_df) < 10:
                        st.error("Insufficient data after preprocessing. Ensure you have at least 10 complete cases.")
                        return
                    
                    # Perform key driver analysis
                    results_df = calculate_relative_importance(
                        analysis_df,
                        target=target_variable,
                        features=features
                    )

                    # Display results
                    st.markdown("### 3️⃣ Analysis Results")
                    
                    # Summary metrics
                    total_importance = results_df['raw_rel_imp'].sum()
                    top_driver = results_df.iloc[0]['feature']
                    top_importance = results_df.iloc[0]['norm_rel_imp']
                    
                    # Key metrics
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric("Top Driver", top_driver)
                    with metric_cols[1]:
                        st.metric("Top Driver Importance", f"{top_importance:.1f}%")
                    with metric_cols[2]:
                        st.metric("Sample Size", len(analysis_df))

                    # Results visualization
                    st.markdown("#### Relative Importance of Drivers")
                    fig = px.bar(
                        results_df,
                        y='feature',
                        x='norm_rel_imp',
                        orientation='h',
                        title=f'Relative Importance of Drivers for {target_variable}',
                        labels={
                            'feature': 'Driver',
                            'norm_rel_imp': 'Relative Importance (%)'
                        }
                    )
                    
                    fig.update_layout(
                        showlegend=False,
                        height=max(400, len(features) * 30),
                        xaxis_title="Relative Importance (%)",
                        yaxis_title="Driver",
                        yaxis={'categoryorder': 'total ascending'}
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed results table
                    with st.expander("📊 Detailed Results", expanded=True):
                        st.dataframe(
                            results_df.style.format({
                                'raw_rel_imp': '{:.4f}',
                                'norm_rel_imp': '{:.2f}%'
                            })
                        )
                        
                        # Export results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="key_driver_analysis_results.csv",
                            mime="text/csv"
                        )

                    # Results interpretation
                    st.markdown("### 4️⃣ Interpretation Guide")
                    with st.expander("📖 How to Interpret Results", expanded=True):
                        st.markdown(f"""
                        **Key Findings:**
                        - **{top_driver}** is the strongest driver, explaining {top_importance:.1f}% of the variance
                        - The top 3 drivers account for {results_df.head(3)['norm_rel_imp'].sum():.1f}% of the total influence
                        
                        **What This Means:**
                        - Higher percentages indicate stronger influence on {target_variable}
                        - Focus on drivers with higher importance scores for maximum impact
                        - Consider the practical significance of each driver
                        
                        **Limitations:**
                        - Results assume linear relationships
                        - Does not imply causation
                        - Based on historical data patterns
                        - May not capture complex interactions
                        """)

            except Exception as e:
                st.error(f"""
                Error during analysis. This might happen if:
                - Variables contain non-numeric values
                - There are too many missing values
                - Variables have zero variance
                
                Technical details: {str(e)}
                """)
                
                st.info("""
                **Suggestions:**
                1. Check your data for non-numeric values
                2. Verify selected variables have sufficient variation
                3. Handle missing values before analysis
                4. Consider using a subset of your data
                """)
                
    except Exception as e:
        st.error(f"""
        Error loading the dataset. Please ensure:
        - The data is properly formatted
        - Variables are properly typed
        - The data is not corrupted
        
        Technical details: {str(e)}
        """)