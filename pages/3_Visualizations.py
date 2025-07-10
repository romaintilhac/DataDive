import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.visualizations import GeochemicalPlotter
from utils.pyrolite_integration import PyroliteAnalyzer

st.set_page_config(page_title="Visualizations", page_icon="📊", layout="wide")

st.title("📊 Interactive Visualizations")

# Check if data is loaded
if st.session_state.main_data is None:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

# Use processed data if available, otherwise use main data
if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    df = st.session_state.processed_data.copy()
    st.info("📈 Using processed data with calculated parameters")
else:
    df = st.session_state.main_data.copy()
    st.info("📊 Using raw data - consider processing data first for additional plot options")

# Initialize plotter and pyrolite analyzer
if 'plotter' not in st.session_state:
    st.session_state.plotter = GeochemicalPlotter()

if 'pyrolite_analyzer' not in st.session_state:
    st.session_state.pyrolite_analyzer = PyroliteAnalyzer()

plotter = st.session_state.plotter
pyrolite_analyzer = st.session_state.pyrolite_analyzer

# Check pyrolite availability
pyrolite_available = pyrolite_analyzer.check_availability()
if pyrolite_available:
    st.success("🔬 Pyrolite integration active - Enhanced geochemical analysis available!")
else:
    st.warning("⚠️ Pyrolite not available - Using standard analysis functions")

# Data overview
st.header("📋 Data Overview")

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

# Visualization tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Scatter Plots", "Geochemical Plots", "Statistical Plots", "Classification", "Pyrolite Enhanced", "Custom Plots"])

with tab1:
    st.subheader("🔍 Scatter Plots and Correlations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Plot Settings:**")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # X and Y axis selection
        x_axis = st.selectbox("X-axis", numeric_cols, index=0 if numeric_cols else 0)
        y_axis = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        
        # Color and size options
        color_by = st.selectbox("Color by", ['None'] + categorical_cols + numeric_cols)
        size_by = st.selectbox("Size by", ['None'] + numeric_cols)
        
        # Log scales
        log_x = st.checkbox("Log X-axis")
        log_y = st.checkbox("Log Y-axis")
        
        # Create plot button
        if st.button("Create Scatter Plot", type="primary"):
            color_col = None if color_by == 'None' else color_by
            size_col = None if size_by == 'None' else size_by
            
            fig = plotter.create_scatter_plot(
                df, x_axis, y_axis, 
                color_col=color_col, size_col=size_col,
                log_x=log_x, log_y=log_y
            )
            
            st.session_state.current_plot = fig
    
    with col2:
        if 'current_plot' in st.session_state:
            st.plotly_chart(st.session_state.current_plot, use_container_width=True)
        else:
            st.info("Configure plot settings and click 'Create Scatter Plot'")
    
    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Correlation Settings:**")
        
        # Select elements for correlation
        all_elements = numeric_cols
        selected_elements = st.multiselect(
            "Select elements for correlation",
            all_elements,
            default=all_elements[:10] if len(all_elements) > 10 else all_elements
        )
        
        if st.button("Create Correlation Heatmap", type="primary"):
            if len(selected_elements) >= 2:
                fig = plotter.create_correlation_heatmap(df, selected_elements)
                st.session_state.correlation_plot = fig
            else:
                st.error("Please select at least 2 elements")
    
    with col2:
        if 'correlation_plot' in st.session_state:
            st.plotly_chart(st.session_state.correlation_plot, use_container_width=True)
        else:
            st.info("Configure correlation settings and click 'Create Correlation Heatmap'")

with tab2:
    st.subheader("🧪 Geochemical Plots")
    
    # Harker diagrams
    st.subheader("📈 Harker Diagrams")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Harker Diagram Settings:**")
        
        # Element selection
        major_elements = ['TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']
        trace_elements = ['La', 'Ce', 'Nd', 'Sm', 'Eu', 'Yb', 'Lu', 'Ba', 'Sr', 'Zr', 'Hf', 'Nb', 'Ta', 'Th', 'U']
        
        available_major = [elem for elem in major_elements if elem in df.columns]
        available_trace = [elem for elem in trace_elements if elem in df.columns]
        
        element_type = st.selectbox("Element Type", ["Major Elements", "Trace Elements"])
        
        if element_type == "Major Elements":
            element = st.selectbox("Element", available_major)
        else:
            element = st.selectbox("Element", available_trace)
        
        color_by = st.selectbox("Color by", ['None'] + categorical_cols, key="harker_color")
        
        if st.button("Create Harker Diagram", type="primary"):
            if 'SiO2' in df.columns and element in df.columns:
                color_col = None if color_by == 'None' else color_by
                fig = plotter.create_harker_diagram(df, element, color_col)
                st.session_state.harker_plot = fig
            else:
                st.error("SiO2 or selected element not found in data")
    
    with col2:
        if 'harker_plot' in st.session_state:
            st.plotly_chart(st.session_state.harker_plot, use_container_width=True)
        else:
            st.info("Configure Harker diagram settings and click 'Create Harker Diagram'")
    
    # REE Spider Plot
    st.subheader("🕷️ REE Spider Diagram")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**REE Spider Settings:**")
        
        # Normalization
        normalize_to = st.selectbox("Normalize to", ["chondrite", "primitive_mantle"])
        color_by = st.selectbox("Color by", ['None'] + categorical_cols, key="ree_color")
        
        if st.button("Create REE Spider Plot", type="primary"):
            color_col = None if color_by == 'None' else color_by
            fig = plotter.create_ree_spider_plot(df, normalize_to, color_col)
            st.session_state.ree_plot = fig
    
    with col2:
        if 'ree_plot' in st.session_state:
            st.plotly_chart(st.session_state.ree_plot, use_container_width=True)
        else:
            st.info("Configure REE spider settings and click 'Create REE Spider Plot'")
    
    # Multi-element Plot
    st.subheader("🌐 Multi-element Plot")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Multi-element Settings:**")
        
        # Element selection
        trace_elements = ['Ba', 'Th', 'U', 'Nb', 'Ta', 'La', 'Ce', 'Pr', 'Nd', 'Sr', 'Sm', 'Zr', 'Hf', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        available_trace = [elem for elem in trace_elements if elem in df.columns]
        
        selected_elements = st.multiselect(
            "Select elements",
            available_trace,
            default=available_trace[:10] if len(available_trace) > 10 else available_trace
        )
        
        color_by = st.selectbox("Color by", ['None'] + categorical_cols, key="multi_color")
        
        if st.button("Create Multi-element Plot", type="primary"):
            if selected_elements:
                color_col = None if color_by == 'None' else color_by
                fig = plotter.create_multi_element_plot(df, selected_elements, color_col)
                st.session_state.multi_plot = fig
            else:
                st.error("Please select at least one element")
    
    with col2:
        if 'multi_plot' in st.session_state:
            st.plotly_chart(st.session_state.multi_plot, use_container_width=True)
        else:
            st.info("Configure multi-element settings and click 'Create Multi-element Plot'")

with tab3:
    st.subheader("📊 Statistical Plots")
    
    # Box plots
    st.subheader("📦 Box Plots")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Box Plot Settings:**")
        
        y_variable = st.selectbox("Y variable", numeric_cols, key="box_y")
        x_variable = st.selectbox("Group by", ['None'] + categorical_cols, key="box_x")
        
        if st.button("Create Box Plot", type="primary"):
            x_col = None if x_variable == 'None' else x_variable
            fig = plotter.create_box_plot(df, y_variable, x_col)
            st.session_state.box_plot = fig
    
    with col2:
        if 'box_plot' in st.session_state:
            st.plotly_chart(st.session_state.box_plot, use_container_width=True)
        else:
            st.info("Configure box plot settings and click 'Create Box Plot'")
    
    # Histograms
    st.subheader("📊 Histograms")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Histogram Settings:**")
        
        hist_variable = st.selectbox("Variable", numeric_cols, key="hist_var")
        hist_color = st.selectbox("Color by", ['None'] + categorical_cols, key="hist_color")
        bins = st.slider("Number of bins", 5, 100, 30)
        
        if st.button("Create Histogram", type="primary"):
            color_col = None if hist_color == 'None' else hist_color
            fig = plotter.create_histogram(df, hist_variable, color_col, bins)
            st.session_state.hist_plot = fig
    
    with col2:
        if 'hist_plot' in st.session_state:
            st.plotly_chart(st.session_state.hist_plot, use_container_width=True)
        else:
            st.info("Configure histogram settings and click 'Create Histogram'")

with tab4:
    st.subheader("🎯 Classification Diagrams")
    
    # TAS diagram
    st.subheader("🌋 TAS Diagram")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**TAS Diagram Settings:**")
        
        if all(col in df.columns for col in ['SiO2', 'Na2O', 'K2O']):
            st.success("✅ Required columns available: SiO2, Na2O, K2O")
            
            if st.button("Create TAS Diagram", type="primary"):
                fig = plotter.create_classification_plot(df, 'TAS')
                st.session_state.tas_plot = fig
        else:
            st.error("❌ Required columns missing: SiO2, Na2O, K2O")
    
    with col2:
        if 'tas_plot' in st.session_state:
            st.plotly_chart(st.session_state.tas_plot, use_container_width=True)
        else:
            st.info("Configure TAS diagram settings and click 'Create TAS Diagram'")
    
    # AFM diagram
    st.subheader("🔺 AFM Diagram")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**AFM Diagram Settings:**")
        
        if all(col in df.columns for col in ['Na2O', 'K2O', 'FeO', 'MgO']):
            st.success("✅ Required columns available: Na2O, K2O, FeO, MgO")
            
            if st.button("Create AFM Diagram", type="primary"):
                fig = plotter.create_classification_plot(df, 'AFM')
                st.session_state.afm_plot = fig
        else:
            st.error("❌ Required columns missing: Na2O, K2O, FeO, MgO")
    
    with col2:
        if 'afm_plot' in st.session_state:
            st.plotly_chart(st.session_state.afm_plot, use_container_width=True)
        else:
            st.info("Configure AFM diagram settings and click 'Create AFM Diagram'")

with tab5:
    st.subheader("🔬 Pyrolite Enhanced Analysis")
    
    if not pyrolite_available:
        st.error("🚫 Pyrolite is not available. Please install pyrolite to use these advanced features.")
        st.stop()
    
    # Enhanced REE Spider Plot
    st.subheader("🕷️ Enhanced REE Spider Plot")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Enhanced REE Settings:**")
        
        # Reference selection
        reference_options = ['Chondrite_PON', 'PrimitiveMantle_PM', 'MORB_SM89', 'OIB_SM89']
        reference = st.selectbox("Reference for normalization", reference_options, key="pyro_ref")
        
        # Sample identification
        sample_col = st.selectbox("Sample column", df.columns, 
                                 index=df.columns.get_loc('Sample') if 'Sample' in df.columns else 0,
                                 key="pyro_sample")
        
        # Color coding
        color_by = st.selectbox("Color by", ['None'] + categorical_cols, key="pyro_color")
        
        if st.button("Create Enhanced REE Plot", type="primary"):
            try:
                color_col = None if color_by == 'None' else color_by
                fig = pyrolite_analyzer.create_ree_spider_plot(df, reference, sample_col, color_col)
                st.session_state.pyro_ree_plot = fig
            except Exception as e:
                st.error(f"Error creating enhanced REE plot: {str(e)}")
    
    with col2:
        if 'pyro_ree_plot' in st.session_state:
            st.plotly_chart(st.session_state.pyro_ree_plot, use_container_width=True)
        else:
            st.info("Configure enhanced REE settings and click 'Create Enhanced REE Plot'")
    
    # Enhanced TAS Diagram
    st.subheader("🌋 Enhanced TAS Diagram")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Enhanced TAS Settings:**")
        
        if all(col in df.columns for col in ['SiO2', 'Na2O', 'K2O']):
            st.success("✅ Required columns available: SiO2, Na2O, K2O")
            
            color_by = st.selectbox("Color by", ['None'] + categorical_cols, key="pyro_tas_color")
            
            if st.button("Create Enhanced TAS Diagram", type="primary"):
                try:
                    color_col = None if color_by == 'None' else color_by
                    fig = pyrolite_analyzer.create_tas_diagram(df, color_col)
                    st.session_state.pyro_tas_plot = fig
                except Exception as e:
                    st.error(f"Error creating enhanced TAS diagram: {str(e)}")
        else:
            st.error("❌ Required columns missing: SiO2, Na2O, K2O")
    
    with col2:
        if 'pyro_tas_plot' in st.session_state:
            st.plotly_chart(st.session_state.pyro_tas_plot, use_container_width=True)
        else:
            st.info("Configure enhanced TAS settings and click 'Create Enhanced TAS Diagram'")
    
    # Mineral Chemistry Analysis
    st.subheader("💎 Mineral Chemistry Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Mineral Chemistry Settings:**")
        
        mineral_types = ['olivine', 'plagioclase', 'orthopyroxene', 'clinopyroxene']
        selected_mineral = st.selectbox("Mineral type", mineral_types, key="pyro_mineral")
        
        if st.button("Calculate Mineral Chemistry", type="primary"):
            try:
                df_mineral = pyrolite_analyzer.calculate_mineral_chemistry(df, selected_mineral)
                st.session_state.mineral_data = df_mineral
                st.session_state.selected_mineral = selected_mineral
                st.success(f"✅ {selected_mineral.title()} chemistry calculated!")
            except Exception as e:
                st.error(f"Error calculating mineral chemistry: {str(e)}")
    
    with col2:
        if 'mineral_data' in st.session_state:
            mineral_data = st.session_state.mineral_data
            selected_mineral = st.session_state.selected_mineral
            
            # Display mineral-specific parameters
            if selected_mineral == 'olivine' and 'Fo' in mineral_data.columns:
                st.write("**Olivine Forsterite Content (Fo):**")
                st.dataframe(mineral_data[['Sample', 'Fo']].head(10))
            elif selected_mineral == 'plagioclase' and 'An' in mineral_data.columns:
                st.write("**Plagioclase Composition:**")
                plag_cols = ['Sample', 'An', 'Ab', 'Or']
                available_plag_cols = [col for col in plag_cols if col in mineral_data.columns]
                st.dataframe(mineral_data[available_plag_cols].head(10))
            else:
                st.info("No mineral chemistry data available for display")
        else:
            st.info("Calculate mineral chemistry to see results")
    
    # Advanced Normalization
    st.subheader("📊 Advanced Normalization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Advanced Normalization Settings:**")
        
        # Reference compositions
        ref_options = ['Chondrite_PON', 'PrimitiveMantle_PM', 'MORB_SM89', 'OIB_SM89', 'UCC_RR', 'BCC_RR']
        norm_reference = st.selectbox("Reference composition", ref_options, key="pyro_norm_ref")
        
        # Element selection
        all_elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                       'Ba', 'Th', 'U', 'Nb', 'Ta', 'Sr', 'Zr', 'Hf', 'Y']
        available_elements = [elem for elem in all_elements if elem in df.columns]
        
        selected_elements = st.multiselect(
            "Select elements to normalize",
            available_elements,
            default=available_elements[:10] if len(available_elements) > 10 else available_elements,
            key="pyro_norm_elements"
        )
        
        if st.button("Apply Advanced Normalization", type="primary"):
            try:
                df_normalized = pyrolite_analyzer.normalize_to_reference(df, norm_reference, selected_elements)
                st.session_state.normalized_data = df_normalized
                st.session_state.norm_reference = norm_reference
                st.success(f"✅ Normalization to {norm_reference} completed!")
            except Exception as e:
                st.error(f"Error applying normalization: {str(e)}")
    
    with col2:
        if 'normalized_data' in st.session_state:
            normalized_data = st.session_state.normalized_data
            norm_reference = st.session_state.norm_reference
            
            # Show normalized columns
            norm_cols = [col for col in normalized_data.columns if norm_reference.split('_')[-1] in col]
            
            if norm_cols:
                st.write(f"**Normalized Data ({norm_reference}):**")
                display_cols = ['Sample'] + norm_cols[:5]  # Show first 5 normalized columns
                st.dataframe(normalized_data[display_cols].head(10))
            else:
                st.info("No normalized data available for display")
        else:
            st.info("Apply advanced normalization to see results")

with tab6:
    st.subheader("🎨 Custom Plots")
    
    # Epsilon plots
    st.subheader("📐 Epsilon Plots")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Epsilon Plot Settings:**")
        
        # Find epsilon columns
        epsilon_cols = [col for col in df.columns if col.startswith('ε')]
        
        if len(epsilon_cols) >= 2:
            epsilon_x = st.selectbox("X-axis epsilon", epsilon_cols, key="eps_x")
            epsilon_y = st.selectbox("Y-axis epsilon", epsilon_cols, index=1, key="eps_y")
            color_by = st.selectbox("Color by", ['None'] + categorical_cols, key="eps_color")
            
            if st.button("Create Epsilon Plot", type="primary"):
                color_col = None if color_by == 'None' else color_by
                fig = plotter.create_epsilon_plot(df, epsilon_x, epsilon_y, color_col)
                st.session_state.epsilon_plot = fig
        else:
            st.warning("⚠️ Epsilon values not found. Process data first to calculate epsilon values.")
    
    with col2:
        if 'epsilon_plot' in st.session_state:
            st.plotly_chart(st.session_state.epsilon_plot, use_container_width=True)
        else:
            st.info("Configure epsilon plot settings and click 'Create Epsilon Plot'")
    
    # Custom ratio plots
    st.subheader("📊 Custom Ratio Plots")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Custom Ratio Settings:**")
        
        # Create custom ratios
        numerator = st.selectbox("Numerator", numeric_cols, key="ratio_num")
        denominator = st.selectbox("Denominator", numeric_cols, key="ratio_den")
        
        y_axis_custom = st.selectbox("Y-axis", numeric_cols, key="custom_y")
        color_by = st.selectbox("Color by", ['None'] + categorical_cols, key="custom_color")
        
        if st.button("Create Custom Ratio Plot", type="primary"):
            if numerator != denominator:
                # Calculate ratio
                df_custom = df.copy()
                ratio_name = f"{numerator}/{denominator}"
                df_custom[ratio_name] = df_custom[numerator] / df_custom[denominator]
                
                color_col = None if color_by == 'None' else color_by
                fig = plotter.create_scatter_plot(
                    df_custom, ratio_name, y_axis_custom, color_col=color_col
                )
                st.session_state.custom_plot = fig
            else:
                st.error("Numerator and denominator must be different")
    
    with col2:
        if 'custom_plot' in st.session_state:
            st.plotly_chart(st.session_state.custom_plot, use_container_width=True)
        else:
            st.info("Configure custom ratio settings and click 'Create Custom Ratio Plot'")

# Export plots
st.header("💾 Export Plots")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export Current Plot as PNG", use_container_width=True):
        if 'current_plot' in st.session_state:
            # Note: In a real deployment, you would implement plot export functionality
            st.info("Plot export functionality would be implemented here")
        else:
            st.error("No current plot to export")

with col2:
    if st.button("Export Current Plot as PDF", use_container_width=True):
        if 'current_plot' in st.session_state:
            st.info("PDF export functionality would be implemented here")
        else:
            st.error("No current plot to export")

with col3:
    if st.button("Export Current Plot as SVG", use_container_width=True):
        if 'current_plot' in st.session_state:
            st.info("SVG export functionality would be implemented here")
        else:
            st.error("No current plot to export")

# Plot gallery
st.header("🖼️ Plot Gallery")

plot_names = []
plots = []

for key in st.session_state.keys():
    if key.endswith('_plot'):
        plot_names.append(key.replace('_plot', '').title())
        plots.append(st.session_state[key])

if plots:
    selected_plot = st.selectbox("Select plot to view", plot_names)
    
    if selected_plot:
        plot_key = selected_plot.lower().replace(' ', '_') + '_plot'
        if plot_key in st.session_state:
            st.plotly_chart(st.session_state[plot_key], use_container_width=True)
else:
    st.info("No plots created yet. Create some plots using the tabs above!")
