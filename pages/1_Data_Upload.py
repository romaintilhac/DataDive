import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import plotly.express as px
from utils.data_processing import DataProcessor

st.set_page_config(page_title="Data Upload", page_icon="📁", layout="wide")

st.title("📁 Data Upload and Validation")

# Initialize data processor
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

data_processor = st.session_state.data_processor

# File upload section
st.header("Upload Your Data Files")

uploaded_files = st.file_uploader(
    "Choose CSV or Excel files",
    accept_multiple_files=True,
    type=['csv', 'xlsx', 'xls'],
    help="Upload one or more geochemical data files. The first file will be treated as the main dataset."
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
    
    # Process files
    datasets = []
    file_info = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner(f"Loading {uploaded_file.name}..."):
            df = data_processor.load_file(uploaded_file, file_type)
            
            if df is not None:
                datasets.append(df)
                file_info.append({
                    'filename': uploaded_file.name,
                    'type': file_type,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'main': i == 0
                })
                
                if i == 0:  # Main dataset
                    st.session_state.main_data = df
    
    # Display file information
    if file_info:
        st.header("📋 File Information")
        info_df = pd.DataFrame(file_info)
        st.dataframe(info_df, use_container_width=True)
    
    # Data validation section
    if datasets:
        st.header("🔍 Data Validation")
        
        main_df = datasets[0]
        
        # Validate main dataset
        issues = data_processor.validate_data(main_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Validation Results")
            
            # Missing required columns
            if issues['missing_required']:
                st.error("❌ Missing Required Columns:")
                for col in issues['missing_required']:
                    st.write(f"- {col}")
            else:
                st.success("✅ All required columns present")
            
            # Empty columns
            if issues['empty_columns']:
                st.warning("⚠️ Empty Columns Found:")
                for col in issues['empty_columns']:
                    st.write(f"- {col}")
            
            # Invalid values
            if issues['invalid_values']:
                st.warning("⚠️ Invalid Values Found:")
                for issue in issues['invalid_values']:
                    st.write(f"- {issue}")
        
        with col2:
            st.subheader("Data Summary")
            summary = data_processor.get_data_summary(main_df)
            
            st.metric("Total Samples", summary['total_samples'])
            st.metric("Total Columns", summary['total_columns'])
            st.metric("Missing Values", summary['missing_values'])
            
            if 'unique_samples' in summary:
                st.metric("Unique Samples", summary['unique_samples'])
            
            if 'lithologies' in summary:
                st.metric("Lithologies", summary['lithologies'])
            
            if 'zones' in summary:
                st.metric("Zones", summary['zones'])
        
        # Display sample data
        st.header("📊 Data Preview")
        
        # Show first few rows
        st.subheader("First 10 Rows")
        st.dataframe(main_df.head(10), use_container_width=True, height=300)
        
        # Column information
        st.subheader("Column Information")
        col_info = []
        for col in main_df.columns:
            col_info.append({
                'Column': col,
                'Type': str(main_df[col].dtype),
                'Non-null Count': main_df[col].count(),
                'Null Count': main_df[col].isna().sum(),
                'Unique Values': main_df[col].nunique() if main_df[col].dtype == 'object' else 'N/A'
            })
        
        col_info_df = pd.DataFrame(col_info)
        st.dataframe(col_info_df, use_container_width=True)
        
        # Data cleaning options
        st.header("🧹 Data Cleaning Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            remove_duplicates = st.checkbox("Remove duplicate samples", value=True)
            fill_missing = st.checkbox("Fill missing values", value=True)
        
        with col2:
            if st.button("Clean Data", type="primary"):
                with st.spinner("Cleaning data..."):
                    cleaned_df = data_processor.clean_data(
                        main_df,
                        remove_duplicates=remove_duplicates,
                        fill_missing=fill_missing
                    )
                    
                    # Update session state
                    st.session_state.main_data = cleaned_df
                    st.session_state.processed_data = cleaned_df
                    
                    st.success("✅ Data cleaned successfully!")
                    st.rerun()
        
        # Merge additional datasets
        if len(datasets) > 1:
            st.header("🔗 Merge Additional Datasets")
            
            st.write("Additional datasets will be merged with the main dataset based on Sample ID.")
            
            if st.button("Merge Datasets", type="secondary"):
                with st.spinner("Merging datasets..."):
                    merged_df = data_processor.merge_datasets(
                        main_df,
                        datasets[1:],
                        merge_on='Sample'
                    )
                    
                    # Update session state
                    st.session_state.main_data = merged_df
                    st.session_state.processed_data = merged_df
                    
                    st.success(f"✅ Merged {len(datasets)} datasets successfully!")
                    st.rerun()
        
        # Export options
        st.header("💾 Export Processed Data")
        
        if st.session_state.main_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv_data = data_processor.export_data(st.session_state.main_data, 'processed_data.csv', 'csv')
                st.download_button(
                    label="Download as CSV",
                    data=csv_data,
                    file_name="processed_geochemical_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel export
                excel_data = data_processor.export_data(st.session_state.main_data, 'processed_data.xlsx', 'excel')
                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name="processed_geochemical_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# Sample data section
else:
    st.info("👆 Please upload your geochemical data files to get started.")
    
    with st.expander("📋 Data Format Requirements"):
        st.markdown("""
        ### Required Columns:
        - **Sample**: Unique sample identifier
        - **Major Elements**: SiO2, TiO2, Al2O3, FeO, MnO, MgO, CaO, Na2O, K2O, P2O5 (in wt%)
        
        ### Optional Columns:
        - **Metadata**: Lithology, Zone, Unit, Lat, Long, Distance
        - **Trace Elements**: REE, transition metals, HFSE (in ppm)
        - **Isotope Ratios**: 87Sr/86Sr, 143Nd/144Nd, 176Hf/177Hf
        
        ### File Formats:
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        
        ### Data Quality:
        - Use consistent decimal notation (periods, not commas)
        - Missing values should be empty cells or 'NaN'
        - Avoid special characters in column names
        - Ensure numeric data is properly formatted
        """)
    
    with st.expander("📊 Example Data Structure"):
        # Create example data
        example_data = {
            'Sample': ['ABC-001', 'ABC-002', 'ABC-003'],
            'Lithology': ['Basalt', 'Andesite', 'Dacite'],
            'Zone': ['Zone A', 'Zone B', 'Zone A'],
            'SiO2': [50.2, 58.7, 65.3],
            'TiO2': [1.2, 0.8, 0.6],
            'Al2O3': [15.5, 16.2, 14.8],
            'FeO': [10.2, 8.5, 6.2],
            'MgO': [8.5, 4.2, 2.1],
            'CaO': [10.8, 7.2, 4.5],
            'Na2O': [2.2, 3.1, 3.8],
            'K2O': [0.8, 1.5, 2.9],
            'La': [15.2, 22.5, 35.8],
            'Ce': [32.1, 48.2, 72.5],
            'Nd': [18.5, 25.3, 31.2]
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
