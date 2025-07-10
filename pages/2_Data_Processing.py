import streamlit as st
import pandas as pd
import numpy as np
from utils.calculations import GeochemicalCalculator
from utils.data_processing import DataProcessor

st.set_page_config(page_title="Data Processing", page_icon="⚙️", layout="wide")

st.title("⚙️ Data Processing and Calculations")

# Check if data is loaded
if st.session_state.main_data is None:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

# Initialize calculators
if 'calculator' not in st.session_state:
    st.session_state.calculator = GeochemicalCalculator()

if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

calculator = st.session_state.calculator
data_processor = st.session_state.data_processor

# Current data
df = st.session_state.main_data.copy()

st.header("📊 Current Dataset Overview")

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

# Data processing options
st.header("🔧 Processing Options")

tab1, tab2, tab3, tab4 = st.tabs(["Basic Calculations", "Isotope Calculations", "Normalization", "Data Filtering"])

with tab1:
    st.subheader("Basic Geochemical Calculations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Available Calculations:**")
        
        calc_mg_number = st.checkbox("Calculate Mg# (Mg-number)", value=True)
        calc_sum_ree = st.checkbox("Calculate ΣREE (Sum of REE)", value=True)
        calc_eu_anomaly = st.checkbox("Calculate Eu/Eu* (Eu anomaly)", value=True)
        calc_ratios = st.checkbox("Calculate elemental ratios", value=True)
        
        if st.button("Perform Basic Calculations", type="primary"):
            with st.spinner("Calculating..."):
                # Mg number
                if calc_mg_number:
                    df['Mg#'] = calculator.calculate_mg_number(df)
                
                # Sum of REE
                if calc_sum_ree:
                    df['ΣREE'] = calculator.calculate_sum_ree(df)
                
                # Eu anomaly
                if calc_eu_anomaly:
                    df['Eu/Eu*'] = calculator.calculate_eu_anomaly(df)
                
                # Element ratios
                if calc_ratios:
                    df = calculator.calculate_ratios(df)
                
                # Update session state
                st.session_state.processed_data = df
                st.success("✅ Basic calculations completed!")
    
    with col2:
        st.write("**Calculation Results Preview:**")
        
        # Show calculated columns
        calc_columns = ['Mg#', 'ΣREE', 'Eu/Eu*', 'La/Yb', 'Th/U', 'Nb/Ta']
        available_calc_cols = [col for col in calc_columns if col in df.columns]
        
        if available_calc_cols:
            st.dataframe(df[['Sample'] + available_calc_cols].head(5), use_container_width=True)
        else:
            st.info("No calculations performed yet")

with tab2:
    st.subheader("Isotope Calculations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Isotope Parameters:**")
        
        # Age input for initial ratio calculations
        age_ma = st.number_input("Age (Ma) for initial ratio calculations", 
                                min_value=0.0, max_value=4500.0, value=100.0, step=10.0)
        
        calc_isotope_ratios = st.checkbox("Calculate isotope ratios (147Sm/144Nd, 176Lu/177Hf)", value=True)
        calc_epsilon_present = st.checkbox("Calculate present-day epsilon values", value=True)
        calc_epsilon_initial = st.checkbox("Calculate initial epsilon values", value=True)
        calc_delta_epsilon = st.checkbox("Calculate ΔεHf", value=True)
        
        if st.button("Perform Isotope Calculations", type="primary"):
            with st.spinner("Calculating isotope parameters..."):
                # Isotope ratios
                if calc_isotope_ratios:
                    df['147Sm/144Nd'] = calculator.calculate_sm_nd_ratio(df)
                    df['176Lu/177Hf'] = calculator.calculate_lu_hf_ratio(df)
                
                # Present-day epsilon values
                if calc_epsilon_present:
                    df['εNd'] = calculator.calculate_epsilon_nd(df, initial=False)
                    df['εHf'] = calculator.calculate_epsilon_hf(df, initial=False)
                
                # Initial epsilon values
                if calc_epsilon_initial:
                    df['εNd(i)'] = calculator.calculate_epsilon_nd(df, initial=True, age_ma=age_ma)
                    df['εHf(i)'] = calculator.calculate_epsilon_hf(df, initial=True, age_ma=age_ma)
                
                # Delta epsilon
                if calc_delta_epsilon:
                    df['ΔεHf'] = calculator.calculate_delta_epsilon_hf(df, initial=False)
                    df['ΔεHf(i)'] = calculator.calculate_delta_epsilon_hf(df, initial=True, age_ma=age_ma)
                
                # Update session state
                st.session_state.processed_data = df
                st.success("✅ Isotope calculations completed!")
    
    with col2:
        st.write("**Isotope Results Preview:**")
        
        isotope_columns = ['147Sm/144Nd', '176Lu/177Hf', 'εNd', 'εHf', 'εNd(i)', 'εHf(i)', 'ΔεHf', 'ΔεHf(i)']
        available_iso_cols = [col for col in isotope_columns if col in df.columns]
        
        if available_iso_cols:
            st.dataframe(df[['Sample'] + available_iso_cols].head(5), use_container_width=True)
        else:
            st.info("No isotope calculations performed yet")

with tab3:
    st.subheader("Element Normalization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Normalization Options:**")
        
        # REE normalization
        st.write("**REE Normalization:**")
        norm_ree_chondrite = st.checkbox("Normalize REE to Chondrite", value=True)
        norm_ree_pm = st.checkbox("Normalize REE to Primitive Mantle", value=False)
        
        # Trace element normalization
        st.write("**Trace Element Normalization:**")
        norm_trace_pm = st.checkbox("Normalize trace elements to Primitive Mantle", value=True)
        
        if st.button("Perform Normalization", type="primary"):
            with st.spinner("Normalizing elements..."):
                ree_elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
                trace_elements = ['Ba', 'Th', 'U', 'Nb', 'Ta', 'La', 'Ce', 'Pr', 'Nd', 'Sr', 'Sm', 'Zr', 'Hf']
                
                # Chondrite normalization
                if norm_ree_chondrite:
                    df = calculator.normalize_to_chondrite(df, ree_elements)
                
                # Primitive mantle normalization
                if norm_ree_pm:
                    df = calculator.normalize_to_primitive_mantle(df, ree_elements)
                
                if norm_trace_pm:
                    df = calculator.normalize_to_primitive_mantle(df, trace_elements)
                
                # Update session state
                st.session_state.processed_data = df
                st.success("✅ Element normalization completed!")
    
    with col2:
        st.write("**Normalization Results Preview:**")
        
        norm_columns = [col for col in df.columns if col.endswith('_N') or col.endswith('_PM')]
        
        if norm_columns:
            preview_cols = ['Sample'] + norm_columns[:5]  # Show first 5 normalized columns
            st.dataframe(df[preview_cols].head(5), use_container_width=True)
        else:
            st.info("No normalization performed yet")

with tab4:
    st.subheader("Data Filtering and Grouping")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Filter Options:**")
        
        # Sample filtering
        if 'Sample' in df.columns:
            all_samples = df['Sample'].unique().tolist()
            selected_samples = st.multiselect("Select Samples", all_samples, default=all_samples)
        else:
            selected_samples = None
        
        # Lithology filtering
        if 'Lithology' in df.columns:
            all_lithologies = df['Lithology'].unique().tolist()
            selected_lithologies = st.multiselect("Select Lithologies", all_lithologies, default=all_lithologies)
        else:
            selected_lithologies = None
        
        # Zone filtering
        if 'Zone' in df.columns:
            all_zones = df['Zone'].unique().tolist()
            selected_zones = st.multiselect("Select Zones", all_zones, default=all_zones)
        else:
            selected_zones = None
        
        # Numeric filtering
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            filter_column = st.selectbox("Numeric Filter Column", ['None'] + numeric_columns)
            
            if filter_column != 'None':
                min_val = float(df[filter_column].min())
                max_val = float(df[filter_column].max())
                
                filter_range = st.slider(
                    f"Filter {filter_column} Range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
            else:
                filter_range = None
        else:
            filter_column = None
            filter_range = None
        
        if st.button("Apply Filters", type="primary"):
            filtered_df = df.copy()
            
            # Apply filters
            if selected_samples:
                filtered_df = filtered_df[filtered_df['Sample'].isin(selected_samples)]
            
            if selected_lithologies and 'Lithology' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Lithology'].isin(selected_lithologies)]
            
            if selected_zones and 'Zone' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Zone'].isin(selected_zones)]
            
            if filter_column and filter_column != 'None' and filter_range:
                filtered_df = filtered_df[
                    (filtered_df[filter_column] >= filter_range[0]) & 
                    (filtered_df[filter_column] <= filter_range[1])
                ]
            
            # Update session state
            st.session_state.processed_data = filtered_df
            st.success(f"✅ Filters applied! {len(filtered_df)} samples remaining.")
    
    with col2:
        st.write("**Current Data Summary:**")
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            filtered_df = st.session_state.processed_data
            
            st.metric("Filtered Samples", len(filtered_df))
            
            if 'Lithology' in filtered_df.columns:
                st.write("**Lithology Distribution:**")
                lithology_counts = filtered_df['Lithology'].value_counts()
                st.dataframe(lithology_counts.reset_index(), use_container_width=True)
            
            if 'Zone' in filtered_df.columns:
                st.write("**Zone Distribution:**")
                zone_counts = filtered_df['Zone'].value_counts()
                st.dataframe(zone_counts.reset_index(), use_container_width=True)
        else:
            st.info("No processed data available")

# Complete processing
st.header("🎯 Complete Processing")

if st.button("Run All Calculations", type="primary", use_container_width=True):
    with st.spinner("Running complete geochemical analysis..."):
        # Run all calculations
        processed_df = calculator.calculate_all_parameters(df, age_ma=100.0)
        
        # Reorder columns for better presentation
        processed_df = data_processor.reorder_columns(processed_df)
        
        # Update session state
        st.session_state.processed_data = processed_df
        
        st.success("✅ Complete processing finished!")
        
        # Show summary
        st.subheader("📋 Processing Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Columns", len(processed_df.columns))
        
        with col2:
            calc_columns = [col for col in processed_df.columns if any(x in col for x in ['_calc', '_N', '_PM', 'ε', 'Δε', 'Mg#', 'ΣREE'])]
            st.metric("Calculated Columns", len(calc_columns))
        
        with col3:
            st.metric("Final Samples", len(processed_df))

# Export processed data
st.header("💾 Export Processed Data")

if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    processed_df = st.session_state.processed_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv_data = data_processor.export_data(processed_df, 'processed_data.csv', 'csv')
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="processed_geochemical_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel export
        excel_data = data_processor.export_data(processed_df, 'processed_data.xlsx', 'excel')
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name="processed_geochemical_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Show final data preview
    st.subheader("📊 Final Data Preview")
    st.dataframe(processed_df.head(10), use_container_width=True, height=300)

else:
    st.info("No processed data available for export")
