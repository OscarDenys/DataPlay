import streamlit as st
import pandas as pd
import json
from io import BytesIO
import datetime as dt

def show():
    """
    Render the Export / Download page with options to download:
    - Processed datasets
    - Generated code
    - Statistical test results
    - Visualizations
    - Machine learning models
    """
    st.title("Export / Download")
    
    if not st.session_state.dataframes:
        st.warning("No data available for download. Please upload data first.")
        return
    
    # Create tabs for different types of exports
    data_tab, code_tab, stats_tab, viz_tab, model_tab = st.tabs([
        "Data Export", 
        "Generated Code", 
        "Statistical Results",
        "Visualizations", 
        "ML Models"
    ])
    
    with data_tab:
        _render_data_export()
    
    with code_tab:
        _render_code_export()
    
    with stats_tab:
        _render_stats_export()
    
    with viz_tab:
        _render_visualization_export()
    
    with model_tab:
        _render_model_export()

def _render_data_export():
    """Handle data export functionality."""
    st.subheader("Export Processed Data")
    
    selected_file = st.selectbox(
        "Choose a dataset to download", 
        list(st.session_state.dataframes.keys())
    )
    
    if selected_file:
        df = st.session_state.dataframes[selected_file]
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{selected_file}_processed.csv",
            mime="text/csv",
        )
        
        # Show data preview
        with st.expander("Preview Data"):
            st.dataframe(df.head())
            
            # Add data info
            st.markdown("### Dataset Information")
            info = {
                "Rows": df.shape[0],
                "Columns": df.shape[1],
                "Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                "Column Types": df.dtypes.value_counts().to_dict()
            }
            st.json(info)

def _render_code_export():
    """Handle code export functionality."""
    st.subheader("Download Generated Code")
    
    if st.session_state.generated_code and st.session_state.generated_code != "No code has been generated":
        st.code(st.session_state.generated_code, language='python')
        
        st.download_button(
            label="Download Python Code",
            data=st.session_state.generated_code,
            file_name=f"generated_code_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
            mime="text/plain",
        )
    else:
        st.info("No code has been generated yet. Use the English to Python page to generate code.")

def _render_stats_export():
    """Handle statistical test results export."""
    st.subheader("Export Statistical Test Results")
    
    if 'test_results' in st.session_state and st.session_state.test_results:
        # Create a more readable format for the results
        formatted_results = []
        for result in st.session_state.test_results:
            formatted_result = {
                'Test Type': result['test_type'],
                'Timestamp': result['timestamp'],
                'Test Statistic': f"{result['test_statistic']:.4f}",
                'P-value': f"{result['p_value']:.4f}",
                'Effect Size': f"{result['effect_size']:.4f}" if result.get('effect_size') is not None else 'N/A',
                'Significant': 'Yes' if result['significant'] else 'No'
            }
            formatted_results.append(formatted_result)
        
        # Display results table
        st.dataframe(pd.DataFrame(formatted_results))
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            json_str = json.dumps(formatted_results, indent=2)
            st.download_button(
                label="Download Results (JSON)",
                data=json_str,
                file_name=f"statistical_results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # CSV export
            csv = pd.DataFrame(formatted_results).to_csv(index=False)
            st.download_button(
                label="Download Results (CSV)",
                data=csv,
                file_name=f"statistical_results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No statistical test results available. Use the Statistical Testing page to perform analyses.")

def _render_visualization_export():
    """Handle visualization export functionality."""
    st.subheader("Export Visualizations")
    
    if st.session_state.images:
        st.write(f"Number of visualizations available: {len(st.session_state.images)}")
        
        # Option to download latest visualization
        st.download_button(
            label="Download Latest Visualization",
            data=st.session_state.images[-1],
            file_name=f"visualization_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
        )
        
        # Preview latest visualization
        st.image(st.session_state.images[-1], caption="Latest visualization")
        
        # Option to download all visualizations as ZIP
        if len(st.session_state.images) > 1:
            try:
                import zipfile
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for i, img in enumerate(st.session_state.images):
                        zip_file.writestr(
                            f"visualization_{i+1}.png",
                            img
                        )
                
                zip_buffer.seek(0)
                st.download_button(
                    label="Download All Visualizations (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"visualizations_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"Error creating ZIP file: {str(e)}")
    else:
        st.info("No visualizations have been generated yet.")

def _render_model_export():
    """Handle machine learning model export functionality."""
    st.subheader("Export Trained Models")
    
    if 'trained_model' in st.session_state:
        st.success("A trained model is available for download")
        
        # Get model info
        model_type = type(st.session_state.trained_model).__name__
        problem_type = st.session_state.get('problem_type', 'Unknown')
        
        # Display model information
        st.write(f"Model Type: {model_type}")
        st.write(f"Problem Type: {problem_type}")
        
        if 'preprocessing' in st.session_state:
            st.write("Preprocessing Configuration:")
            st.json(st.session_state.preprocessing)
        
        # Create model export
        try:
            import pickle
            
            # Prepare model data including metadata
            model_data = {
                'model': st.session_state.trained_model,
                'preprocessing': st.session_state.get('preprocessing', {}),
                'metadata': {
                    'timestamp': dt.datetime.now().isoformat(),
                    'model_type': model_type,
                    'problem_type': problem_type
                }
            }
            
            # Serialize model data
            model_bytes = pickle.dumps(model_data)
            
            # Add download button
            st.download_button(
                label="Download Trained Model",
                data=model_bytes,
                file_name=f"trained_model_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                mime="application/octet-stream"
            )
            
            st.info("""
                The downloaded model includes:
                - Trained model object
                - Preprocessing configuration
                - Feature names
                - Target encoder (for classification)
            """)
            
        except Exception as e:
            st.error(f"Error preparing model for download: {str(e)}")
    else:
        st.info("No trained models available. Use the Machine Learning page to train a model.")