import streamlit as st

def show():
    """Render the home page"""
    st.title("Welcome to DataPlay")
    
    # Hero Section
    st.markdown("""
    Your end-to-end data analysis companion that combines the power of conversational AI with advanced analytics. Ask questions in plain English, create stunning visualizations, run statistical tests, and build ML models - no coding required. From rapid data profiling to finding key drivers, DataPlay streamlines your workflow and helps you make data-driven decisions with confidence.""")

    # Feature Overview
    st.header("Overview")

    # Data Upload
    with st.expander("📤 Data Upload", expanded=False):
        st.markdown("""
        Upload and preview CSV (UTF-8) files with interactive grid view for data verification and editing.
        
        **Capabilities:**
        - Support for CSV files
        - Data preview 
        """)
    
    # Data Manipulation
    with st.expander("✂️ Data Manipulation", expanded=False):
        st.markdown("""
        Easily manipulate and transform your data.
        
        **Capabilities:**
        - Interactive editing
        - Clean and filter your data
        - Automatic code generation for reproducible analysis
        """)

    # Data Profiling
    with st.expander("🔍 Data Profiling", expanded=False):
        st.markdown("""
        Generate comprehensive data quality reports with statistical analysis and visualizations.
        
        **Capabilities:**
        - Complete & minimal profiling modes
        - Column-specific analysis
        - Comparison reports for data segments
        - Missing value analysis
        - Correlation detection
        """)

    # Natural Language Query
    with st.expander("💬 English to Python", expanded=False):
        st.markdown("""
        Ask questions about your data in plain English for analysis, visualization, and manipulation.
        
        **Capabilities:**
        - Create new columns
        - Generate visualizations
        - Query your data
        - Clean and filter your data
        - State management with undo/redo
        """)

    # Interactive Visualization
    with st.expander("📊 Interactive Visualization", expanded=False):
        st.markdown("""
        Create sophisticated visualizations with a Tableau-like drag-and-drop interface.
        
        **Capabilities:**
        - Multiple chart types
        - Interactive dashboards
        - Customizable aesthetics
        """)

    # Key Driver Analysis
    with st.expander("🔑 Key Driver Analysis", expanded=False):
        st.markdown("""
        Identify and quantify the most important factors influencing your target variables.
        
        **Capabilities:**
        - Relative importance analysis
        - Advanced statistical modeling
        - Automatic visualization
        """)

    # Statistical Testing
    with st.expander("🧮 Statistical Testing", expanded=False):
        st.markdown("""
        Perform various statistical tests with comprehensive results and visualizations.
        
        **Capabilities:**
        - Parametric & non-parametric tests
        - Automated assumption checking
        - Effect size calculations
        - Visual result interpretation
        """)

    # ML Prototype
    with st.expander("⚙️ ML Prototype", expanded=False):
        st.markdown("""
        Build and evaluate machine learning models with an intuitive interface.
        
        **Capabilities:**
        - Classification & regression support
        - Multiple model options
        - Automated model evaluation
        - Interactive prediction interface
        """)

    # Export & Download
    with st.expander("⬇️ Export & Download", expanded=False):
        st.markdown("""
        Export your analysis results, visualizations, and trained models.
        
        **Capabilities:**
        - Processed data export
        - Generated code download
        - Visualization export
        - Model export
        """)
        
    # Help and Documentation
    with st.expander("❓ Help and Documentation", expanded=False):
        st.markdown("""
        The Help and Documentation page is your one-stop-shop for getting the most out of TableTalk. 
                    
        **What you'll find:**
        - Getting started and core feature guides
        - Detailed explanations of analysis features
        - Best practices and tips for effective data analysis
        - Troubleshooting common issues and solutions
        - Information on privacy and security measures
        """)

    st.divider()
    # Quick Start Guide
    st.header("Quick Start Guide")
    
    # Step 1
    st.markdown("#### 1️. Upload Data 📤")
    with st.container(border=True):
        st.markdown("Upload your CSV (UTF-8) file, preview data in the interactive grid, and verify data types and content")
    st.divider()
    
    # Step 2
    st.markdown("#### 2️. Understand Data 🔍")
    with st.container(border=True):
        st.markdown("Generate quality report, check for missing values, identify correlations, and review data distributions")
    st.divider()
    
    # Step 3
    st.markdown("#### 3️. Analyze & Visualize 💬")
    with st.container(border=True):
        st.markdown("Ask questions in plain English, create interactive charts, perform statistical tests, and identify key drivers")
    st.divider()
    
    # Step 4
    st.markdown("#### 4️. Export Results ⬇️")
    with st.container(border=True):
        st.markdown("Download processed data, export visualizations, save trained models, and download analysis code")
    st.divider()