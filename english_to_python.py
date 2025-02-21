import streamlit as st
import pandas as pd
import numpy as np
from pandasai import SmartDataframe
import os
from tempfile import mkdtemp
import shutil
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Tuple, Union
import time

class QueryEngine:
    """Handles natural language query processing and result management"""
    
    def __init__(self, temp_dir: str):
        """
        Initialize query engine with temporary directory for visualizations
        
        Args:
            temp_dir: Path to temporary directory
        """
        self.temp_dir = temp_dir
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
    
    def process_query(
        self, 
        df: pd.DataFrame, 
        query: str, 
        llm: Any
    ) -> Tuple[Any, str, str]:
        """
        Process natural language query and return results
        
        Args:
            df: Input DataFrame
            query: Natural language query
            llm: Language model instance
            
        Returns:
            Tuple of (result, generated_code, result_type)
        """
        try:
            # Configure SmartDataframe
            sdf = SmartDataframe(
                df,
                config={
                    "llm": llm,
                    "save_charts": True,
                    "save_charts_path": self.temp_dir,
                    "enable_cache": True,
                    "verbose": False
                }
            )
            
            # Process query and get result
            result = sdf.chat(query)
            code = sdf.last_code_generated if hasattr(sdf, 'last_code_generated') else None
            
            # Determine result type
            if isinstance(result, pd.DataFrame):
                result_type = 'dataframe'
            elif isinstance(result, str) and result.endswith(('.png', '.jpg', '.jpeg')):
                result_type = 'visualization'
            elif isinstance(result, (dict, list)):
                result_type = 'data_structure'
            else:
                result_type = 'text'
                
            return result, code, result_type
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return None, None, None

class StateManager:
    """Manages application state and history"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize all required session state variables"""
        defaults = {
            'query_history': [],
            'results_history': [],
            'result_types_history': [],
            'code_history': [],
            'temp_dir': mkdtemp(),
            'current_df': None,
            'current_query': None,
            'current_result': None,
            'current_result_type': None,
            'current_code': None,
            'show_code': False,
            'error': None
        }
        
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default
    
    @staticmethod
    def update_state(
        query: str, 
        result: Any, 
        result_type: str,
        code: str, 
        df: Optional[pd.DataFrame] = None
    ):
        """
        Update session state with new query results
        
        Args:
            query: User query
            result: Query result
            result_type: Type of result (dataframe, visualization, etc)
            code: Generated code
            df: Optional updated DataFrame
        """
        st.session_state.query_history.append(query)
        st.session_state.results_history.append(result)
        st.session_state.result_types_history.append(result_type)
        st.session_state.code_history.append(code)
        
        st.session_state.current_query = query
        st.session_state.current_result = result
        st.session_state.current_result_type = result_type
        st.session_state.current_code = code
        
        if isinstance(result, pd.DataFrame):
            st.session_state.current_df = result

    @staticmethod
    def undo():
        """Revert to previous state"""
        if len(st.session_state.query_history) > 1:
            # Remove current state
            st.session_state.query_history.pop()
            st.session_state.results_history.pop()
            st.session_state.result_types_history.pop()
            st.session_state.code_history.pop()
            
            # Set current state to previous
            st.session_state.current_query = st.session_state.query_history[-1]
            st.session_state.current_result = st.session_state.results_history[-1]
            st.session_state.current_result_type = st.session_state.result_types_history[-1]
            st.session_state.current_code = st.session_state.code_history[-1]
            
            if isinstance(st.session_state.current_result, pd.DataFrame):
                st.session_state.current_df = st.session_state.current_result

    @staticmethod
    def reset(initial_df: pd.DataFrame):
        """
        Reset state to initial condition
        
        Args:
            initial_df: Initial DataFrame to reset to
        """
        st.session_state.query_history = []
        st.session_state.results_history = []
        st.session_state.result_types_history = []
        st.session_state.code_history = []
        
        st.session_state.current_query = None
        st.session_state.current_result = None
        st.session_state.current_result_type = None
        st.session_state.current_code = None
        st.session_state.current_df = initial_df.copy()
        
        st.session_state.show_code = False
        st.session_state.error = None

def render_result(result: Any, result_type: str):
    """
    Render query result based on type
    
    Args:
        result: Query result to render
        result_type: Type of result to render
    """
    if result is None:
        return
        
    if result_type == 'dataframe':
        st.dataframe(result, use_container_width=True)
        
        # Show quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", result.shape[0])
        with col2:
            st.metric("Columns", result.shape[1])
        with col3:
            memory = result.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory", f"{memory:.1f} MB")
            
    elif result_type == 'visualization':
        try:
            st.image(result, use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying visualization: {str(e)}")
            
    elif result_type == 'data_structure':
        if isinstance(result, dict):
            # Create bar chart for dictionaries
            fig = px.bar(
                x=list(result.keys()),
                y=list(result.values()),
                labels={'x': 'Category', 'y': 'Value'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif isinstance(result, list):
            # Create line plot for lists
            fig = px.line(
                y=result,
                labels={'index': 'Index', 'value': 'Value'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Also show raw data
        with st.expander("View Raw Data"):
            st.json(result)
            
    else:  # Text or other
        st.write(result)

def show():
    """Main function to render the English to Python interface"""
    st.title("English to Python")
    
    if not st.session_state.dataframes:
        st.warning("Please upload data in the Data Upload section first.")
        return
        
    # Initialize state
    StateManager.initialize_session_state()
    
    # Setup query engine
    query_engine = QueryEngine(st.session_state.temp_dir)
    
    # Data selection
    selected_file = st.selectbox(
        "Choose your dataset",
        list(st.session_state.dataframes.keys())
    )
    
    if selected_file:
        df = st.session_state.dataframes[selected_file]
        
        # Initialize current_df if needed
        if st.session_state.current_df is None:
            st.session_state.current_df = df.copy()
            
        # Current data preview
        with st.expander("📊 Current Data Preview", expanded=True):
            st.dataframe(
                st.session_state.current_df,
                use_container_width=True,
                height=300
            )
                    
        # Query interface
        st.markdown("### 🤖 Ask me anything about your data")
        query = st.text_area(
            "",
            placeholder="""Examples:
- Show a histogram of sales
- Calculate average revenue by product
- Create a correlation matrix heatmap
- Add a new column with total price
- Find the top 5 customers by revenue""",
            help="Enter your request in plain English"
        )
        
        # Action buttons
        cols = st.columns([1, 1, 1, 1, 1])
        
        # Run button
        if cols[0].button("🚀 Run", type="primary", use_container_width=True):
            if query:
                with st.spinner("Processing query..."):
                    result, code, result_type = query_engine.process_query(
                        st.session_state.current_df,
                        query,
                        st.session_state.llm
                    )
                    
                    if result is not None:
                        StateManager.update_state(query, result, result_type, code)
                        time.sleep(0.1)  # Small delay to ensure state updates
                        st.rerun()
                        
        # Undo button
        if cols[1].button("↩️ Undo", use_container_width=True):
            StateManager.undo()
            time.sleep(0.1)
            st.rerun()
            
        # Reset button
        if cols[2].button("🔄 Reset", use_container_width=True):
            StateManager.reset(df)
            time.sleep(0.1)
            st.rerun()
            
        # Code toggle
        if cols[3].button("💻 Code", use_container_width=True):
            st.session_state.show_code = not st.session_state.show_code
            
        # Download button
        if cols[4].button("⬇️ Save", use_container_width=True):
            if st.session_state.current_result is not None:
                if isinstance(st.session_state.current_result, pd.DataFrame):
                    csv = st.session_state.current_result.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        data=csv,
                        file_name="result.csv",
                        mime="text/csv"
                    )
                elif isinstance(st.session_state.current_result, str) and \
                     st.session_state.current_result.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        with open(st.session_state.current_result, "rb") as f:
                            st.download_button(
                                "Download Image",
                                data=f.read(),
                                file_name="visualization.png",
                                mime="image/png"
                            )
                    except Exception as e:
                        st.error(f"Error preparing download: {str(e)}")
        
        # Display generated code if toggled
        if st.session_state.show_code and st.session_state.current_code:
            st.markdown("### 💻 Generated Code")
            st.code(st.session_state.current_code, language="python")
        
        # Query history
        if st.session_state.query_history:
            st.markdown("### 📜 Query History")
            selected_query = st.selectbox(
                "Previous queries:",
                st.session_state.query_history,
                index=len(st.session_state.query_history) - 1
            )
            
            if selected_query:
                idx = st.session_state.query_history.index(selected_query)
                result = st.session_state.results_history[idx]
                result_type = st.session_state.result_types_history[idx]
                code = st.session_state.code_history[idx]
                
                st.session_state.current_result = result
                st.session_state.current_result_type = result_type
                st.session_state.current_code = code
                
                if isinstance(result, pd.DataFrame):
                    st.session_state.current_df = result
                
                time.sleep(0.1)
                st.rerun()
        
        # Display current result
        if st.session_state.current_result is not None:
            st.markdown("### 📊 Result")
            render_result(
                st.session_state.current_result,
                st.session_state.current_result_type
            )

# Register cleanup handler
def cleanup():
    """Clean up temporary files on exit"""
    if hasattr(st.session_state, 'temp_dir') and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except Exception:
            pass

import atexit
atexit.register(cleanup)