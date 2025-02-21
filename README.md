# TableTalk

A streamlit application for interactive data analysis and visualization.

## Local Setup and Installation

### Prerequisites

1. Python 3.10 or higher
2. Git

### First Time Setup

1. Clone the repository
```bash
git clone [your-repository-url]
```

2. Navigate to the project directory
```bash
cd tabletalk/deployment/streamlit_app
```

3. Create a virtual environment
```bash
python3.10 -m venv tt
```

4. Activate the virtual environment
```bash
source tt/bin/activate  # On Unix/macOS
tt\Scripts\activate     # On Windows
```

5. Install required packages
```bash
pip install -r requirements.txt
```

6. Run the application
```bash
streamlit run main.py --browser.gatherUsageStats False
```

### Subsequent Runs

1. Navigate to the project directory
```bash
cd tabletalk/deployment/streamlit_app
```

2. Activate the virtual environment
```bash
source tt/bin/activate  # On Unix/macOS
tt\Scripts\activate     # On Windows
```

3. Run the application
```bash
streamlit run main.py --browser.gatherUsageStats False
```

## Features

- Data Upload: Support for CSV and Excel files
- Data Profiling: Comprehensive data quality analysis
- Interactive Visualization: Drag-and-drop interface for creating charts
- English to Python: Natural language interface for data analysis
- Statistical Testing: Various statistical tests with visualizations
- Machine Learning: Simple interface for training and evaluating models
- Export/Download: Save your analysis results and processed data

## Support

For issues or questions, please create an issue in the repository.