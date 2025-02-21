import streamlit as st
import pandas as pd
from io import StringIO, BytesIO

def read_file(file):
    """
    Read file content into a pandas DataFrame.
    
    Parameters:
    file: StreamlitUploadedFile object
    
    Returns:
    pd.DataFrame: Loaded data
    """
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        raise ValueError('Unsupported file format')

def show():
    """Render the data upload page."""
    st.title("Data Upload")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your CSV files",
        accept_multiple_files=True,
        type=['csv', 'xlsx']
    )
    
    if uploaded_files:
        with st.spinner("Loading preview..."):
            for file in uploaded_files:
                try:
                    # Read the data
                    df = read_file(file)

                    # Store in session state
                    st.session_state.dataframes[file.name] = df
                    st.session_state.dataframe_state_queries[file.name] = ["Initial State"]

                except Exception as e:
                    st.error(f"Error processing file {file.name}: {str(e)}")
                    continue
        
        # Show success message
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
        
        # Display data preview
        if uploaded_files:
            st.subheader("Data Preview")
            # Show the last uploaded file
            last_file = uploaded_files[-1].name
            df = st.session_state.dataframes[last_file]
            st.dataframe(df)

            # Add file info
            st.info(f"""
                File Information:
                - Rows: {df.shape[0]}
                - Columns: {df.shape[1]}
                - Memory Usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB
            """)

            # Display column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info)