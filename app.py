import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processing import DataProcessor
from utils.calculations import GeochemicalCalculator
from utils.visualizations import GeochemicalPlotter

# Page configuration
st.set_page_config(
    page_title="Geochemical Data Analysis",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'calculator' not in st.session_state:
    st.session_state.calculator = GeochemicalCalculator()
if 'plotter' not in st.session_state:
    st.session_state.plotter = GeochemicalPlotter()
if 'main_data' not in st.session_state:
    st.session_state.main_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Main page
st.title("🧪 Geochemical Data Analysis Platform")
st.markdown("""
This application provides comprehensive tools for geochemical data analysis including:
- **Data Upload & Validation**: Load CSV/Excel files with automatic validation
- **Data Processing**: Clean, transform, and calculate derived parameters
- **Interactive Visualizations**: Create publication-ready plots and charts
- **Statistical Analysis**: Perform statistical tests and generate summary tables
""")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the pages in the sidebar to navigate through different analysis steps.")

# Data status indicator
if st.session_state.main_data is not None:
    st.sidebar.success(f"✅ Data loaded: {len(st.session_state.main_data)} samples")
    
    # Show basic data info
    st.sidebar.markdown("### Data Summary")
    st.sidebar.write(f"**Samples**: {len(st.session_state.main_data)}")
    
    if 'Sample' in st.session_state.main_data.columns:
        unique_samples = st.session_state.main_data['Sample'].nunique()
        st.sidebar.write(f"**Unique Samples**: {unique_samples}")
    
    if 'Lithology' in st.session_state.main_data.columns:
        unique_lithologies = st.session_state.main_data['Lithology'].nunique()
        st.sidebar.write(f"**Lithologies**: {unique_lithologies}")
    
    if 'Zone' in st.session_state.main_data.columns:
        unique_zones = st.session_state.main_data['Zone'].nunique()
        st.sidebar.write(f"**Zones**: {unique_zones}")
else:
    st.sidebar.info("📁 No data loaded yet. Go to Data Upload to get started.")

# Main content area
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📁 Data Upload")
    st.markdown("Upload your geochemical data files (CSV/Excel) and validate the data structure.")
    if st.button("Go to Data Upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    st.subheader("⚙️ Data Processing")
    st.markdown("Clean, transform, and calculate derived geochemical parameters.")
    if st.button("Go to Data Processing", use_container_width=True):
        st.switch_page("pages/2_Data_Processing.py")

with col3:
    st.subheader("📊 Visualizations")
    st.markdown("Create interactive plots and charts for data exploration and publication.")
    if st.button("Go to Visualizations", use_container_width=True):
        st.switch_page("pages/3_Visualizations.py")

# Quick start section
st.markdown("---")
st.subheader("🚀 Quick Start Guide")

with st.expander("How to use this application"):
    st.markdown("""
    1. **Upload Data**: Start by uploading your geochemical data files in the Data Upload section
    2. **Validate Structure**: Ensure your data has the required columns (Sample, major elements, trace elements)
    3. **Process Data**: Clean your data and calculate derived parameters like Mg#, εNd, εHf
    4. **Visualize**: Create interactive plots to explore your data
    5. **Analyze**: Perform statistical analysis and generate summary tables
    6. **Export**: Download processed data and visualizations
    """)

with st.expander("Supported Data Formats"):
    st.markdown("""
    **File Types**: CSV, Excel (.xlsx, .xls)
    
    **Required Columns**:
    - Sample: Sample identifier
    - Major elements: SiO2, TiO2, Al2O3, FeO, MnO, MgO, CaO, Na2O, K2O, P2O5
    - Trace elements: REE, transition metals, HFSE
    - Isotope ratios: 87Sr/86Sr, 143Nd/144Nd, 176Hf/177Hf (optional)
    
    **Optional Columns**:
    - Lithology: Rock type classification
    - Zone: Geological zone or unit
    - Coordinates: Lat, Long for spatial analysis
    """)

# Footer
st.markdown("---")
st.markdown("**Geochemical Data Analysis Platform** | Built with Streamlit")
