import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import home
import data_upload
import manual_data_manipulation
import data_profiling
import english_to_python
import interactive_visualization
import key_driver_analysis
import statistical_testing
import machine_learning
import export_download
import help_documentation

# Set up page config first, before any other st calls
st.set_page_config(
    layout='wide',
    page_title="DataPlay",
    page_icon="assets/TableTalk-Tab-Icon.png"
)

def initialize_session_state():
    """Initialize all session state variables"""
    if 'dataframes' not in st.session_state:
        st.session_state.dataframes = {}
    
    if 'dataframe_state_queries' not in st.session_state:
        st.session_state.dataframe_state_queries = {}
    
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = "No code has been generated"
    
    if 'images' not in st.session_state:
        st.session_state.images = []
        
    # Add statistical testing specific session state variables
    if 'test_results' not in st.session_state:
        st.session_state.test_results = []

# Initialize session state
initialize_session_state()

# Sidebar navigation
with st.sidebar:
    st.sidebar.image("assets/TT_Main_Logo.png")
    choose = option_menu(
        menu_title=None,
        options=[
            "Home", "Data Upload", "Manual Data Manipulation", "Data Profiling",
            "English to Python", "Interactive Visualization", "Key Driver Analysis",
            "Statistical Testing", "Machine Learning", "Export / Download",
            "Help and Documentation"
        ],
        icons=[
            'house', 'cloud-arrow-up', 'table', 'search', 'chat-dots',
            'bar-chart-line', 'key', 'calculator', 'gear', 'download',
            'question-circle'
        ],
        default_index=0
    )

# Route to appropriate page based on selection
if choose == "Home":
    home.show()
elif choose == "Data Upload":
    data_upload.show()
elif choose == "Manual Data Manipulation":
    manual_data_manipulation.show()
elif choose == "Data Profiling":
    data_profiling.show()
elif choose == "English to Python":
    english_to_python.show()
elif choose == "Interactive Visualization":
    interactive_visualization.show()
elif choose == "Key Driver Analysis":
    key_driver_analysis.show()
elif choose == "Statistical Testing":
    statistical_testing.show()
elif choose == "Machine Learning":
    machine_learning.show()
elif choose == "Export / Download":
    export_download.show()
elif choose == "Help and Documentation":
    help_documentation.show()