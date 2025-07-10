import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import plotly.express as px
from utils.data_processing import DataProcessor
from utils.multi_file_loader import MultiFileLoader
from utils.global_database import GlobalDatabase

st.set_page_config(page_title="Data Upload", page_icon="📁", layout="wide")

st.title("📁 Data Upload and Validation")

# Initialize processors
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

if 'multi_file_loader' not in st.session_state:
    st.session_state.multi_file_loader = MultiFileLoader()

if 'global_database' not in st.session_state:
    st.session_state.global_database = GlobalDatabase()

data_processor = st.session_state.data_processor
multi_file_loader = st.session_state.multi_file_loader
global_database = st.session_state.global_database

# Upload mode selection
st.header("Data Loading Options")

upload_mode = st.radio(
    "Choose data loading mode:",
    ["Single File Upload", "Multi-File Combination", "Global Database Comparison"],
    horizontal=True
)

if upload_mode == "Single File Upload":
    st.subheader("📁 Single File Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your main geochemical data file."
    )
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner(f"Loading {uploaded_file.name}..."):
            df = data_processor.load_file(uploaded_file, file_type)
            
            if df is not None:
                st.session_state.main_data = df
                st.success(f"✅ File loaded successfully! {len(df)} samples, {len(df.columns)} columns")

elif upload_mode == "Multi-File Combination":
    st.subheader("📁 Multi-File Combination")
    st.info("Upload multiple files to combine them based on sample matching logic")
    
    uploaded_files = st.file_uploader(
        "Choose CSV or Excel files",
        accept_multiple_files=True,
        type=['csv', 'xlsx', 'xls'],
        help="Upload multiple geochemical data files. They will be combined based on sample names.",
        key="multi_files"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        # Load all files
        files_dict = {}
        file_info = []
        
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            with st.spinner(f"Loading {uploaded_file.name}..."):
                try:
                    df = multi_file_loader.load_file(uploaded_file, file_type, file_label=uploaded_file.name)
                    files_dict[uploaded_file.name] = df
                    file_info.append({
                        'filename': uploaded_file.name,
                        'type': file_type,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'sample_column': 'Sample' if 'Sample' in df.columns else 'Unknown'
                    })
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        
        # Display file information
        if file_info:
            st.subheader("📋 Loaded Files")
            info_df = pd.DataFrame(file_info)
            st.dataframe(info_df, use_container_width=True)
        
        # File combination settings
        if len(files_dict) > 1:
            st.subheader("🔧 Combination Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sample column selection
                sample_column = st.selectbox(
                    "Sample matching column:",
                    options=['Sample', 'sample', 'SAMPLE', 'Sample_ID', 'SampleID'],
                    help="Column name used for matching samples across files"
                )
                
                # Catalogue file selection
                catalogue_options = ['Auto-detect'] + list(files_dict.keys())
                catalogue_file = st.selectbox(
                    "Catalogue file:",
                    options=catalogue_options,
                    help="File containing sample metadata (auto-detected if not specified)"
                )
                
                if catalogue_file == 'Auto-detect':
                    catalogue_file = None
            
            with col2:
                # Conflict resolution
                conflict_resolution = st.selectbox(
                    "Conflict resolution:",
                    options=['keep_first', 'keep_last', 'average'],
                    help="How to handle conflicting values in overlapping columns"
                )
                
                # Priority order
                priority_order = st.multiselect(
                    "File priority order:",
                    options=list(files_dict.keys()),
                    default=list(files_dict.keys()),
                    help="Priority order for conflict resolution"
                )
            
            # Combine files button
            if st.button("🔄 Combine Files", type="primary"):
                try:
                    with st.spinner("Combining files..."):
                        # Validate sample consistency
                        validation_results = multi_file_loader.validate_sample_consistency(sample_column)
                        
                        if 'error' not in validation_results:
                            # Combine files
                            combined_df = multi_file_loader.combine_files_by_sample(
                                files_dict, 
                                sample_column=sample_column,
                                catalogue_file=catalogue_file,
                                priority_order=priority_order
                            )
                            
                            # Resolve conflicts
                            if conflict_resolution != 'keep_first':
                                combined_df = multi_file_loader.resolve_conflicts(
                                    combined_df, 
                                    conflict_resolution=conflict_resolution,
                                    priority_files=priority_order
                                )
                            
                            # Store combined data
                            st.session_state.main_data = combined_df
                            st.session_state.combined_data = combined_df
                            st.session_state.multi_file_summary = multi_file_loader.get_merge_summary()
                            
                            st.success(f"✅ Files combined successfully! {len(combined_df)} samples, {len(combined_df.columns)} columns")
                            
                        else:
                            st.error(f"Validation error: {validation_results['error']}")
                            
                except Exception as e:
                    st.error(f"Error combining files: {str(e)}")
            
            # Display validation results
            if st.button("🔍 Validate Sample Consistency"):
                validation_results = multi_file_loader.validate_sample_consistency(sample_column)
                
                if 'error' not in validation_results:
                    st.subheader("📊 Sample Validation Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Files", validation_results['total_files'])
                    with col2:
                        st.metric("Common Samples", validation_results['common_samples'])
                    with col3:
                        st.metric("Unique Samples", validation_results['unique_samples'])
                    
                    # File statistics
                    st.subheader("📈 File Statistics")
                    file_stats_df = pd.DataFrame(validation_results['file_stats']).T
                    st.dataframe(file_stats_df, use_container_width=True)
                    
                    # Overlap matrix
                    if 'overlap_matrix' in validation_results:
                        st.subheader("🔗 Sample Overlap Matrix")
                        overlap_df = pd.DataFrame(validation_results['overlap_matrix'])
                        st.dataframe(overlap_df, use_container_width=True)
                
                else:
                    st.error(validation_results['error'])
        
        elif len(files_dict) == 1:
            # Single file loaded
            df = list(files_dict.values())[0]
            st.session_state.main_data = df
            st.success("✅ Single file loaded successfully!")

elif upload_mode == "Global Database Comparison":
    st.subheader("🌍 Global Database Comparison")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Available Reference Databases:**")
        
        # List available databases
        databases = global_database.list_databases()
        
        for db_name in databases:
            db_info = global_database.get_database_info(db_name)
            with st.expander(f"📊 {db_name}"):
                st.write(f"**Samples:** {db_info['samples']}")
                st.write(f"**Elements:** {len(db_info['elements'])}")
                st.write(f"**Lithologies:** {', '.join(db_info['lithologies'][:3])}..." if len(db_info['lithologies']) > 3 else f"**Lithologies:** {', '.join(db_info['lithologies'])}")
                
                # Show preview
                if st.button(f"Preview {db_name}", key=f"preview_{db_name}"):
                    preview_df = global_database.get_database(db_name).head()
                    st.dataframe(preview_df, use_container_width=True)
    
    with col2:
        st.write("**Upload Your Data for Comparison:**")
        
        comparison_file = st.file_uploader(
            "Choose your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your geochemical data to compare with global databases",
            key="comparison_file"
        )
        
        if comparison_file:
            file_type = comparison_file.name.split('.')[-1].lower()
            
            with st.spinner(f"Loading {comparison_file.name}..."):
                df = data_processor.load_file(comparison_file, file_type)
                
                if df is not None:
                    st.session_state.main_data = df
                    st.success(f"✅ Data loaded for comparison!")
                    
                    # Database comparison
                    selected_database = st.selectbox(
                        "Select database for comparison:",
                        options=databases,
                        help="Choose which reference database to compare your data against"
                    )
                    
                    if st.button("🔍 Compare with Database", type="primary"):
                        try:
                            with st.spinner("Comparing with database..."):
                                comparison_results = global_database.compare_with_database(
                                    df, selected_database
                                )
                                
                                if 'error' not in comparison_results:
                                    st.session_state.comparison_results = comparison_results
                                    st.success("✅ Comparison completed!")
                                    
                                    # Display comparison results
                                    st.subheader("📊 Comparison Results")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Your Samples", comparison_results['user_samples'])
                                    with col2:
                                        st.metric("Reference Samples", comparison_results['reference_samples'])
                                    with col3:
                                        st.metric("Compared Elements", len(comparison_results['comparison_elements']))
                                    
                                    # Similarity metrics
                                    similarity_data = []
                                    for element, metrics in comparison_results['similarity_metrics'].items():
                                        similarity_data.append({
                                            'Element': element,
                                            'Similarity Score': f"{metrics['similarity_score']:.1f}%",
                                            'Relative Difference': f"{metrics['relative_difference_percent']:.1f}%",
                                            'Range Overlap': f"{metrics['range_overlap_percent']:.1f}%"
                                        })
                                    
                                    if similarity_data:
                                        st.subheader("📈 Element Similarity Analysis")
                                        similarity_df = pd.DataFrame(similarity_data)
                                        st.dataframe(similarity_df, use_container_width=True)
                                
                                else:
                                    st.error(comparison_results['error'])
                                    
                        except Exception as e:
                            st.error(f"Error during comparison: {str(e)}")

# Display combined data summary if available
if 'multi_file_summary' in st.session_state:
    st.header("📋 Multi-File Combination Summary")
    
    summary = st.session_state.multi_file_summary
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Files Combined", summary['files_loaded'])
    with col2:
        if summary['combined_shape']:
            st.metric("Final Samples", summary['combined_shape'][0])
    with col3:
        if summary['combined_shape']:
            st.metric("Final Columns", summary['combined_shape'][1])
    
    # Merge log
    if summary['merge_log']:
        with st.expander("🔍 Detailed Merge Log"):
            for log_entry in summary['merge_log']:
                st.write(f"• {log_entry}")

# Display current data status
if st.session_state.main_data is not None:
    st.header("📊 Current Data Status")
    
    df = st.session_state.main_data
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        if 'Sample' in df.columns:
            st.metric("Unique Samples", df['Sample'].nunique())
    with col4:
        if 'Lithology' in df.columns:
            st.metric("Lithologies", df['Lithology'].nunique())
# Data validation section
if st.session_state.main_data is not None:
    st.header("🔍 Data Validation")
    
    main_df = st.session_state.main_data
    
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
                st.session_state.main_data = cleaned_df
                st.success(f"✅ Data cleaned successfully! {len(cleaned_df)} samples remaining.")
                st.rerun()
        
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
