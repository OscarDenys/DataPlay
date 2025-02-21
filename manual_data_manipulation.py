import streamlit as st
import pandas as pd
from mitosheet.streamlit.v1 import spreadsheet

def show():
    """
    Render the manual data manipulation page with Mito spreadsheet integration.
    """
    st.title("Manual Data Manipulation")

    # Check if any data is available
    if not st.session_state.dataframes:
        st.warning("Please upload data in the Data Upload section first.")
        return

    # File selection
    selected_file = st.selectbox(
        "Choose a dataset", 
        list(st.session_state.dataframes.keys())
    )
    current_df = st.session_state.dataframes[selected_file]

    # Display information about data persistence
    with st.expander("ℹ️ About Data Changes", expanded=False):
        st.info("""
        - Changes made here are stored in your session
        - Data persists while browser tab remains open
        - Download modified data before closing browser
        - Changes are reflected across all pages
        """)

    # Display Mito sheet and capture changes
    mito_result = spreadsheet(current_df)
    
    if mito_result is not None:
        try:
            # Get the edited DataFrame from the Mito result
            # Mito returns a tuple where first element is an OrderedDict with key 'df1'
            edited_data = mito_result[0]
            if isinstance(edited_data, dict) and 'df1' in edited_data:
                edited_df = edited_data['df1']
                
                if isinstance(edited_df, pd.DataFrame):
                    # Check for actual changes
                    has_column_changes = set(edited_df.columns) != set(current_df.columns)
                    
                    # Update session state with the new dataframe
                    st.session_state.dataframes[selected_file] = edited_df
                    
                    # Update state tracking
                    if selected_file not in st.session_state.dataframe_state_queries:
                        st.session_state.dataframe_state_queries[selected_file] = ["Initial State"]
                    
                    # Create meaningful change message
                    if has_column_changes:
                        added_cols = set(edited_df.columns) - set(current_df.columns)
                        removed_cols = set(current_df.columns) - set(edited_df.columns)
                        change_msg = "Manual Edit via Mito: "
                        if removed_cols:
                            change_msg += f"Removed columns {removed_cols} "
                        if added_cols:
                            change_msg += f"Added columns {added_cols}"
                    else:
                        change_msg = "Manual Edit via Mito: Modified values"
                            
                    st.session_state.dataframe_state_queries[selected_file].append(change_msg)
                    
                    # Show success message
                    st.success("Changes saved successfully!")
                    
        except Exception as e:
            st.error(f"Error processing changes: {str(e)}")
            st.exception(e)

    # State Management Section
    if selected_file in st.session_state.dataframe_state_queries:
        with st.expander("📋 Data State History"):
            past_states = st.session_state.dataframe_state_queries[selected_file]
            st.write("Past Changes:")
            for state in past_states:
                st.write(f"- {state}")
            
            # Current Data State Display
            if st.checkbox("Show current data"):
                st.write("Current Data State:")
                display_df = st.session_state.dataframes[selected_file]
                st.write(f"Shape: {display_df.shape}")
                st.dataframe(display_df)
                
                # Add download button for current state
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download Current Data",
                    data=csv,
                    file_name=f"{selected_file}_current_state.csv",
                    mime="text/csv",
                )

    # Add data validation section
    with st.expander("🔍 Data Validation", expanded=False):
        if st.session_state.dataframes[selected_file] is not None:
            df = st.session_state.dataframes[selected_file]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isna().sum().sum())
            
            if st.checkbox("Show Data Types"):
                st.write("Column Data Types:")
                st.dataframe(pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isna().sum()
                }))