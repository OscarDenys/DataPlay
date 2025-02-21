import streamlit as st
import pandas as pd
import pygwalker as pyg

def show():
    """
    Render the interactive visualization page with PyGWalker integration.
    """
    st.title("Data Visualization")

    # Check if data is available
    if not st.session_state.dataframes:
        st.warning("Please upload data in the Data Upload section first.")
        return

    # Add introduction and guidance
    with st.expander("📊 About Interactive Visualization", expanded=True):
        st.markdown("""
        Create powerful visualizations using a drag-and-drop interface similar to Tableau.
        
        **Quick Start:**
        1. Select your dataset below
        2. Drag columns from the left panel to the canvas
        3. Choose chart types and customize appearance
        4. Export or save your visualizations
        
        **Tips:**
        - Use the 'Marks' card to adjust colors, sizes, and labels
        - Switch between chart types using the top menu
        - Double-click elements to customize them
        - Right-click for additional options
        """)

    # File selection with preview
    selected_file = st.selectbox(
        "Choose a dataset",
        list(st.session_state.dataframes.keys()),
        help="Select the dataset you want to visualize"
    )
    
    try:
        df = st.session_state.dataframes[selected_file]
        
        # Show dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Numeric Columns", 
                     len(df.select_dtypes(include=['int64', 'float64']).columns))
        
        # Data preview
        with st.expander("👀 Preview Data", expanded=False):
            st.dataframe(df.head(), use_container_width=True)
            
            # Show column info
            st.write("Column Information:")
            col_info = pd.DataFrame({
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)

        # Generate PyGWalker interface
        st.markdown("""
        ### Interactive Visualization Canvas
        Use the drag-and-drop interface below to create your visualizations.
        """)
        
        # Add visualization tips
        with st.expander("💡 Visualization Tips", expanded=False):
            st.markdown("""
            **Best Practices:**
            - Start with a clear goal for your visualization
            - Choose appropriate chart types for your data
            - Use color meaningfully
            - Keep it simple and focused
            
            **Recommended Charts:**
            - Bar Charts: Compare categories
            - Line Charts: Show trends over time
            - Scatter Plots: Show relationships
            - Pie Charts: Show composition (use sparingly)
            """)
        
        try:
            # Generate PyGWalker HTML
            pyg_html = pyg.to_html(df)
            
            # Display PyGWalker interface
            st.components.v1.html(pyg_html, width=1100, height=800)
            
            # Add note about data persistence
            st.info("""
            ℹ️ **Note:** Visualizations created here are session-based and will not persist 
            after closing your browser. Use the browser's screenshot functionality or the 
            built-in export options to save your visualizations.
            """)
            
        except Exception as e:
            st.error(f"""
            Error creating visualization interface. This might happen if:
            - The dataset is too large
            - There are incompatible data types
            - Browser memory is constrained
            
            Try refreshing the page or using a smaller dataset.
            
            Technical details: {str(e)}
            """)
            
            # Provide constructive suggestions
            st.warning("""
            **Suggestions:**
            1. Try reducing the number of rows or columns
            2. Check for and handle any missing values
            3. Ensure column data types are appropriate
            4. Close other browser tabs to free up memory
            """)
            
    except Exception as e:
        st.error(f"""
        Error loading the dataset. Please ensure:
        - The data is properly formatted
        - All columns have valid names
        - The data types are supported
        
        Technical details: {str(e)}
        """)
        
        # Provide helpful guidance
        st.info("""
        **Need help?**
        - Check the data in the Data Upload section
        - Verify the data format and encoding
        - Try processing the data first using the Data Manipulation page
        """)