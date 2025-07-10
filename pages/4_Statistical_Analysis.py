import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Statistical Analysis", page_icon="📈", layout="wide")

st.title("📈 Statistical Analysis")

# Check if data is loaded
if st.session_state.main_data is None:
    st.warning("⚠️ No data loaded. Please upload data first.")
    if st.button("Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

# Use processed data if available, otherwise use main data
if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    df = st.session_state.processed_data.copy()
    st.info("📊 Using processed data with calculated parameters")
else:
    df = st.session_state.main_data.copy()
    st.info("📋 Using raw data - consider processing data first for additional analysis options")

# Data overview
st.header("📋 Data Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Samples", len(df))
with col2:
    st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
with col3:
    if 'Sample' in df.columns:
        st.metric("Unique Samples", df['Sample'].nunique())
with col4:
    if 'Lithology' in df.columns:
        st.metric("Lithologies", df['Lithology'].nunique())

# Get numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Statistical analysis tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary Statistics", "Correlation Analysis", "Hypothesis Testing", "Multivariate Analysis", "Advanced Analytics"])

with tab1:
    st.subheader("📊 Summary Statistics")
    
    # Basic statistics
    st.subheader("🔢 Descriptive Statistics")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Statistics Settings:**")
        
        # Column selection
        selected_columns = st.multiselect(
            "Select columns for analysis",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        )
        
        # Group by option
        group_by = st.selectbox("Group by", ['None'] + categorical_cols)
        
        # Statistics to include
        include_stats = st.multiselect(
            "Include statistics",
            ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis'],
            default=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        )
        
        if st.button("Calculate Statistics", type="primary"):
            if selected_columns:
                if group_by != 'None':
                    # Grouped statistics
                    stats_df = df.groupby(group_by)[selected_columns].describe()
                    
                    # Add skewness and kurtosis if requested
                    if 'skew' in include_stats:
                        skew_df = df.groupby(group_by)[selected_columns].skew()
                        for col in selected_columns:
                            stats_df[(col, 'skew')] = skew_df[col]
                    
                    if 'kurtosis' in include_stats:
                        kurt_df = df.groupby(group_by)[selected_columns].apply(lambda x: x.kurtosis())
                        for col in selected_columns:
                            stats_df[(col, 'kurtosis')] = kurt_df[col]
                    
                    st.session_state.summary_stats = stats_df
                else:
                    # Overall statistics
                    stats_df = df[selected_columns].describe()
                    
                    # Add skewness and kurtosis if requested
                    if 'skew' in include_stats:
                        skew_row = df[selected_columns].skew()
                        stats_df.loc['skew'] = skew_row
                    
                    if 'kurtosis' in include_stats:
                        kurt_row = df[selected_columns].kurtosis()
                        stats_df.loc['kurtosis'] = kurt_row
                    
                    st.session_state.summary_stats = stats_df
                
                st.success("✅ Statistics calculated!")
            else:
                st.error("Please select at least one column")
    
    with col2:
        if 'summary_stats' in st.session_state:
            st.write("**Summary Statistics:**")
            st.dataframe(st.session_state.summary_stats, use_container_width=True)
        else:
            st.info("Configure settings and click 'Calculate Statistics'")
    
    # Distribution analysis
    st.subheader("📊 Distribution Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Distribution Settings:**")
        
        dist_column = st.selectbox("Select column", numeric_cols, key="dist_col")
        group_by_dist = st.selectbox("Group by", ['None'] + categorical_cols, key="dist_group")
        
        # Normality tests
        test_normality = st.checkbox("Test for normality", value=True)
        
        if st.button("Analyze Distribution", type="primary"):
            if dist_column in df.columns:
                # Create distribution plot
                if group_by_dist != 'None':
                    fig = px.histogram(
                        df, 
                        x=dist_column, 
                        color=group_by_dist,
                        marginal="box",
                        title=f"Distribution of {dist_column} by {group_by_dist}"
                    )
                else:
                    fig = px.histogram(
                        df, 
                        x=dist_column, 
                        marginal="box",
                        title=f"Distribution of {dist_column}"
                    )
                
                fig.update_layout(template='plotly_white')
                st.session_state.dist_plot = fig
                
                # Normality tests
                if test_normality:
                    # Shapiro-Wilk test (for sample size < 5000)
                    if len(df[dist_column].dropna()) < 5000:
                        stat_sw, p_sw = stats.shapiro(df[dist_column].dropna())
                        st.session_state.normality_results = {
                            'shapiro_stat': stat_sw,
                            'shapiro_p': p_sw,
                            'shapiro_normal': p_sw > 0.05
                        }
                    
                    # D'Agostino's test
                    stat_da, p_da = stats.normaltest(df[dist_column].dropna())
                    st.session_state.normality_results.update({
                        'dagostino_stat': stat_da,
                        'dagostino_p': p_da,
                        'dagostino_normal': p_da > 0.05
                    })
                
                st.success("✅ Distribution analysis completed!")
            else:
                st.error("Selected column not found")
    
    with col2:
        if 'dist_plot' in st.session_state:
            st.plotly_chart(st.session_state.dist_plot, use_container_width=True)
            
            # Show normality test results
            if 'normality_results' in st.session_state:
                st.write("**Normality Test Results:**")
                results = st.session_state.normality_results
                
                if 'shapiro_stat' in results:
                    st.write(f"**Shapiro-Wilk Test:**")
                    st.write(f"- Statistic: {results['shapiro_stat']:.4f}")
                    st.write(f"- p-value: {results['shapiro_p']:.4f}")
                    st.write(f"- Normal distribution: {'Yes' if results['shapiro_normal'] else 'No'}")
                
                st.write(f"**D'Agostino's Test:**")
                st.write(f"- Statistic: {results['dagostino_stat']:.4f}")
                st.write(f"- p-value: {results['dagostino_p']:.4f}")
                st.write(f"- Normal distribution: {'Yes' if results['dagostino_normal'] else 'No'}")
        else:
            st.info("Configure distribution settings and click 'Analyze Distribution'")

with tab2:
    st.subheader("🔗 Correlation Analysis")
    
    # Correlation matrix
    st.subheader("📊 Correlation Matrix")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Correlation Settings:**")
        
        # Column selection
        corr_columns = st.multiselect(
            "Select columns for correlation",
            numeric_cols,
            default=numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols
        )
        
        # Correlation method
        corr_method = st.selectbox("Correlation method", ['pearson', 'spearman', 'kendall'])
        
        # Significance level
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
        
        if st.button("Calculate Correlations", type="primary"):
            if len(corr_columns) >= 2:
                # Calculate correlation matrix
                corr_matrix = df[corr_columns].corr(method=corr_method)
                
                # Calculate p-values
                p_values = np.zeros((len(corr_columns), len(corr_columns)))
                for i, col1 in enumerate(corr_columns):
                    for j, col2 in enumerate(corr_columns):
                        if i != j:
                            if corr_method == 'pearson':
                                _, p_val = pearsonr(df[col1].dropna(), df[col2].dropna())
                            elif corr_method == 'spearman':
                                _, p_val = spearmanr(df[col1].dropna(), df[col2].dropna())
                            else:  # kendall
                                _, p_val = kendalltau(df[col1].dropna(), df[col2].dropna())
                            p_values[i, j] = p_val
                
                # Create significance mask
                significant_mask = p_values < alpha
                
                # Store results
                st.session_state.corr_matrix = corr_matrix
                st.session_state.p_values = p_values
                st.session_state.significant_mask = significant_mask
                
                # Create correlation heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title=f"{corr_method.title()} Correlation Matrix",
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                
                fig.update_layout(width=600, height=600, template='plotly_white')
                st.session_state.corr_heatmap = fig
                
                st.success("✅ Correlation analysis completed!")
            else:
                st.error("Please select at least 2 columns")
    
    with col2:
        if 'corr_heatmap' in st.session_state:
            st.plotly_chart(st.session_state.corr_heatmap, use_container_width=True)
            
            # Show significant correlations
            if 'corr_matrix' in st.session_state:
                st.write("**Significant Correlations:**")
                
                corr_matrix = st.session_state.corr_matrix
                p_values = st.session_state.p_values
                significant_mask = st.session_state.significant_mask
                
                significant_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if significant_mask[i, j]:
                            significant_corrs.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j],
                                'p-value': p_values[i, j]
                            })
                
                if significant_corrs:
                    sig_df = pd.DataFrame(significant_corrs)
                    sig_df = sig_df.sort_values('Correlation', key=abs, ascending=False)
                    st.dataframe(sig_df, use_container_width=True)
                else:
                    st.info("No significant correlations found at the selected significance level")
        else:
            st.info("Configure correlation settings and click 'Calculate Correlations'")
    
    # Pairwise analysis
    st.subheader("🔍 Pairwise Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Pairwise Settings:**")
        
        var1 = st.selectbox("Variable 1", numeric_cols, key="pair_var1")
        var2 = st.selectbox("Variable 2", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="pair_var2")
        color_by = st.selectbox("Color by", ['None'] + categorical_cols, key="pair_color")
        
        if st.button("Analyze Pair", type="primary"):
            if var1 != var2:
                # Create scatter plot
                color_col = None if color_by == 'None' else color_by
                fig = px.scatter(
                    df, 
                    x=var1, 
                    y=var2, 
                    color=color_col,
                    trendline="ols",
                    title=f"{var2} vs {var1}"
                )
                
                fig.update_layout(template='plotly_white')
                st.session_state.pair_plot = fig
                
                # Calculate correlation and regression statistics
                correlation, p_value = pearsonr(df[var1].dropna(), df[var2].dropna())
                
                # Linear regression
                slope, intercept, r_value, p_value_reg, std_err = stats.linregress(df[var1].dropna(), df[var2].dropna())
                
                st.session_state.pair_stats = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'r_squared': r_value**2,
                    'slope': slope,
                    'intercept': intercept,
                    'std_error': std_err
                }
                
                st.success("✅ Pairwise analysis completed!")
            else:
                st.error("Please select different variables")
    
    with col2:
        if 'pair_plot' in st.session_state:
            st.plotly_chart(st.session_state.pair_plot, use_container_width=True)
            
            # Show regression statistics
            if 'pair_stats' in st.session_state:
                st.write("**Regression Statistics:**")
                stats_data = st.session_state.pair_stats
                
                st.write(f"**Correlation coefficient (r):** {stats_data['correlation']:.4f}")
                st.write(f"**p-value:** {stats_data['p_value']:.4f}")
                st.write(f"**R-squared:** {stats_data['r_squared']:.4f}")
                st.write(f"**Slope:** {stats_data['slope']:.4f}")
                st.write(f"**Intercept:** {stats_data['intercept']:.4f}")
                st.write(f"**Standard error:** {stats_data['std_error']:.4f}")
        else:
            st.info("Configure pairwise settings and click 'Analyze Pair'")

with tab3:
    st.subheader("🧪 Hypothesis Testing")
    
    # t-tests
    st.subheader("📊 t-Tests")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**t-Test Settings:**")
        
        test_type = st.selectbox("Test type", ["One-sample t-test", "Two-sample t-test", "Paired t-test"])
        test_variable = st.selectbox("Test variable", numeric_cols, key="ttest_var")
        
        if test_type == "One-sample t-test":
            test_value = st.number_input("Test value (μ₀)", value=0.0)
        elif test_type == "Two-sample t-test":
            group_variable = st.selectbox("Group variable", categorical_cols, key="ttest_group")
            if group_variable in df.columns:
                unique_groups = df[group_variable].unique()
                if len(unique_groups) >= 2:
                    group1 = st.selectbox("Group 1", unique_groups, key="ttest_group1")
                    group2 = st.selectbox("Group 2", unique_groups, index=1 if len(unique_groups) > 1 else 0, key="ttest_group2")
                else:
                    st.warning("Selected group variable has fewer than 2 unique values")
        else:  # Paired t-test
            variable2 = st.selectbox("Second variable", numeric_cols, key="ttest_var2")
        
        alpha_test = st.slider("Significance level", 0.01, 0.10, 0.05, 0.01, key="ttest_alpha")
        
        if st.button("Perform t-Test", type="primary"):
            try:
                if test_type == "One-sample t-test":
                    data = df[test_variable].dropna()
                    t_stat, p_value = stats.ttest_1samp(data, test_value)
                    
                    result = {
                        'test_type': test_type,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < alpha_test,
                        'mean': data.mean(),
                        'test_value': test_value,
                        'n': len(data)
                    }
                
                elif test_type == "Two-sample t-test":
                    group1_data = df[df[group_variable] == group1][test_variable].dropna()
                    group2_data = df[df[group_variable] == group2][test_variable].dropna()
                    
                    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                    
                    result = {
                        'test_type': test_type,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < alpha_test,
                        'group1_mean': group1_data.mean(),
                        'group2_mean': group2_data.mean(),
                        'group1_n': len(group1_data),
                        'group2_n': len(group2_data),
                        'group1_name': group1,
                        'group2_name': group2
                    }
                
                else:  # Paired t-test
                    data1 = df[test_variable].dropna()
                    data2 = df[variable2].dropna()
                    
                    # Align data by index
                    common_index = data1.index.intersection(data2.index)
                    data1_aligned = data1.loc[common_index]
                    data2_aligned = data2.loc[common_index]
                    
                    t_stat, p_value = stats.ttest_rel(data1_aligned, data2_aligned)
                    
                    result = {
                        'test_type': test_type,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < alpha_test,
                        'mean_diff': (data1_aligned - data2_aligned).mean(),
                        'n_pairs': len(common_index),
                        'var1_name': test_variable,
                        'var2_name': variable2
                    }
                
                st.session_state.ttest_result = result
                st.success("✅ t-Test completed!")
                
            except Exception as e:
                st.error(f"Error performing t-test: {str(e)}")
    
    with col2:
        if 'ttest_result' in st.session_state:
            st.write("**t-Test Results:**")
            result = st.session_state.ttest_result
            
            st.write(f"**Test type:** {result['test_type']}")
            st.write(f"**t-statistic:** {result['t_statistic']:.4f}")
            st.write(f"**p-value:** {result['p_value']:.4f}")
            st.write(f"**Significant:** {'Yes' if result['significant'] else 'No'}")
            
            if result['test_type'] == "One-sample t-test":
                st.write(f"**Sample mean:** {result['mean']:.4f}")
                st.write(f"**Test value:** {result['test_value']:.4f}")
                st.write(f"**Sample size:** {result['n']}")
            elif result['test_type'] == "Two-sample t-test":
                st.write(f"**{result['group1_name']} mean:** {result['group1_mean']:.4f}")
                st.write(f"**{result['group2_name']} mean:** {result['group2_mean']:.4f}")
                st.write(f"**{result['group1_name']} n:** {result['group1_n']}")
                st.write(f"**{result['group2_name']} n:** {result['group2_n']}")
            else:  # Paired t-test
                st.write(f"**Mean difference:** {result['mean_diff']:.4f}")
                st.write(f"**Number of pairs:** {result['n_pairs']}")
                st.write(f"**Variable 1:** {result['var1_name']}")
                st.write(f"**Variable 2:** {result['var2_name']}")
        else:
            st.info("Configure t-test settings and click 'Perform t-Test'")
    
    # ANOVA
    st.subheader("📊 Analysis of Variance (ANOVA)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**ANOVA Settings:**")
        
        anova_variable = st.selectbox("Dependent variable", numeric_cols, key="anova_var")
        anova_group = st.selectbox("Group variable", categorical_cols, key="anova_group")
        
        if st.button("Perform ANOVA", type="primary"):
            try:
                if anova_group in df.columns:
                    groups = []
                    group_names = []
                    
                    for group_name in df[anova_group].unique():
                        if pd.notna(group_name):
                            group_data = df[df[anova_group] == group_name][anova_variable].dropna()
                            if len(group_data) > 0:
                                groups.append(group_data)
                                group_names.append(group_name)
                    
                    if len(groups) >= 2:
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        # Calculate group statistics
                        group_stats = []
                        for i, group_data in enumerate(groups):
                            group_stats.append({
                                'Group': group_names[i],
                                'N': len(group_data),
                                'Mean': group_data.mean(),
                                'Std Dev': group_data.std(),
                                'Min': group_data.min(),
                                'Max': group_data.max()
                            })
                        
                        result = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'group_stats': pd.DataFrame(group_stats),
                            'n_groups': len(groups)
                        }
                        
                        st.session_state.anova_result = result
                        st.success("✅ ANOVA completed!")
                    else:
                        st.error("Need at least 2 groups with data for ANOVA")
                else:
                    st.error("Group variable not found")
            except Exception as e:
                st.error(f"Error performing ANOVA: {str(e)}")
    
    with col2:
        if 'anova_result' in st.session_state:
            st.write("**ANOVA Results:**")
            result = st.session_state.anova_result
            
            st.write(f"**F-statistic:** {result['f_statistic']:.4f}")
            st.write(f"**p-value:** {result['p_value']:.4f}")
            st.write(f"**Significant:** {'Yes' if result['significant'] else 'No'}")
            st.write(f"**Number of groups:** {result['n_groups']}")
            
            st.write("**Group Statistics:**")
            st.dataframe(result['group_stats'], use_container_width=True)
        else:
            st.info("Configure ANOVA settings and click 'Perform ANOVA'")

with tab4:
    st.subheader("🎯 Multivariate Analysis")
    
    # Principal Component Analysis
    st.subheader("🔍 Principal Component Analysis (PCA)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**PCA Settings:**")
        
        pca_columns = st.multiselect(
            "Select variables for PCA",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        )
        
        standardize = st.checkbox("Standardize variables", value=True)
        color_by_pca = st.selectbox("Color by", ['None'] + categorical_cols, key="pca_color")
        
        if st.button("Perform PCA", type="primary"):
            if len(pca_columns) >= 2:
                try:
                    # Prepare data
                    pca_data = df[pca_columns].dropna()
                    
                    if standardize:
                        scaler = StandardScaler()
                        pca_data_scaled = scaler.fit_transform(pca_data)
                    else:
                        pca_data_scaled = pca_data.values
                    
                    # Perform PCA
                    pca = PCA()
                    pca_result = pca.fit_transform(pca_data_scaled)
                    
                    # Create results dataframe
                    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
                    pca_df.index = pca_data.index
                    
                    # Add original categorical columns
                    for col in categorical_cols:
                        if col in df.columns:
                            pca_df[col] = df.loc[pca_data.index, col]
                    
                    # Store results
                    st.session_state.pca_result = pca_df
                    st.session_state.pca_explained_variance = pca.explained_variance_ratio_
                    st.session_state.pca_components = pd.DataFrame(
                        pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
                        index=pca_columns
                    )
                    
                    # Create PCA plot
                    color_col = None if color_by_pca == 'None' else color_by_pca
                    fig = px.scatter(
                        pca_df, 
                        x='PC1', 
                        y='PC2',
                        color=color_col,
                        title='PCA Plot (PC1 vs PC2)'
                    )
                    
                    fig.update_layout(template='plotly_white')
                    st.session_state.pca_plot = fig
                    
                    st.success("✅ PCA completed!")
                    
                except Exception as e:
                    st.error(f"Error performing PCA: {str(e)}")
            else:
                st.error("Please select at least 2 variables")
    
    with col2:
        if 'pca_plot' in st.session_state:
            st.plotly_chart(st.session_state.pca_plot, use_container_width=True)
            
            # Show explained variance
            if 'pca_explained_variance' in st.session_state:
                st.write("**Explained Variance:**")
                explained_var = st.session_state.pca_explained_variance
                
                var_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(len(explained_var))],
                    'Explained Variance': explained_var,
                    'Cumulative Variance': np.cumsum(explained_var)
                })
                
                st.dataframe(var_df.head(10), use_container_width=True)
        else:
            st.info("Configure PCA settings and click 'Perform PCA'")
    
    # Show component loadings
    if 'pca_components' in st.session_state:
        st.subheader("📊 Component Loadings")
        
        components_df = st.session_state.pca_components
        
        # Create loadings heatmap
        fig = px.imshow(
            components_df.T,
            text_auto=True,
            aspect="auto",
            title="PCA Component Loadings",
            color_continuous_scale='RdBu_r'
        )
        
        fig.update_layout(width=800, height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show loadings table
        st.dataframe(components_df, use_container_width=True)
    
    # Cluster Analysis
    st.subheader("🎯 Cluster Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Clustering Settings:**")
        
        cluster_columns = st.multiselect(
            "Select variables for clustering",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols
        )
        
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        standardize_cluster = st.checkbox("Standardize variables", value=True, key="cluster_std")
        
        if st.button("Perform Clustering", type="primary"):
            if len(cluster_columns) >= 2:
                try:
                    # Prepare data
                    cluster_data = df[cluster_columns].dropna()
                    
                    if standardize_cluster:
                        scaler = StandardScaler()
                        cluster_data_scaled = scaler.fit_transform(cluster_data)
                    else:
                        cluster_data_scaled = cluster_data.values
                    
                    # Perform K-means clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(cluster_data_scaled)
                    
                    # Calculate silhouette score
                    silhouette = silhouette_score(cluster_data_scaled, cluster_labels)
                    
                    # Create results dataframe
                    cluster_df = cluster_data.copy()
                    cluster_df['Cluster'] = cluster_labels
                    cluster_df['Cluster'] = cluster_df['Cluster'].astype(str)
                    
                    # Store results
                    st.session_state.cluster_result = cluster_df
                    st.session_state.silhouette_score = silhouette
                    
                    # Create cluster plot
                    if len(cluster_columns) >= 2:
                        fig = px.scatter(
                            cluster_df, 
                            x=cluster_columns[0], 
                            y=cluster_columns[1],
                            color='Cluster',
                            title=f'K-means Clustering (k={n_clusters})'
                        )
                        
                        fig.update_layout(template='plotly_white')
                        st.session_state.cluster_plot = fig
                    
                    st.success("✅ Clustering completed!")
                    
                except Exception as e:
                    st.error(f"Error performing clustering: {str(e)}")
            else:
                st.error("Please select at least 2 variables")
    
    with col2:
        if 'cluster_plot' in st.session_state:
            st.plotly_chart(st.session_state.cluster_plot, use_container_width=True)
            
            # Show clustering statistics
            if 'silhouette_score' in st.session_state:
                st.write("**Clustering Statistics:**")
                st.write(f"**Silhouette Score:** {st.session_state.silhouette_score:.4f}")
                
                # Show cluster summary
                if 'cluster_result' in st.session_state:
                    cluster_summary = st.session_state.cluster_result.groupby('Cluster').agg({
                        cluster_columns[0]: ['count', 'mean', 'std'],
                        cluster_columns[1]: ['mean', 'std'] if len(cluster_columns) > 1 else ['mean']
                    }).round(4)
                    
                    st.write("**Cluster Summary:**")
                    st.dataframe(cluster_summary, use_container_width=True)
        else:
            st.info("Configure clustering settings and click 'Perform Clustering'")

with tab5:
    st.subheader("🔬 Advanced Analytics")
    
    # Outlier Detection
    st.subheader("🎯 Outlier Detection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Outlier Detection Settings:**")
        
        outlier_column = st.selectbox("Select variable", numeric_cols, key="outlier_var")
        outlier_method = st.selectbox("Detection method", ["Z-score", "IQR", "Modified Z-score"])
        
        if outlier_method == "Z-score":
            z_threshold = st.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.1)
        elif outlier_method == "IQR":
            iqr_multiplier = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
        else:  # Modified Z-score
            modified_z_threshold = st.slider("Modified Z-score threshold", 2.0, 5.0, 3.5, 0.1)
        
        if st.button("Detect Outliers", type="primary"):
            try:
                data = df[outlier_column].dropna()
                
                if outlier_method == "Z-score":
                    z_scores = np.abs(stats.zscore(data))
                    outliers = data[z_scores > z_threshold]
                    outlier_indices = data.index[z_scores > z_threshold]
                    
                elif outlier_method == "IQR":
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                    outlier_indices = outliers.index
                    
                else:  # Modified Z-score
                    median = data.median()
                    mad = np.median(np.abs(data - median))
                    modified_z_scores = 0.6745 * (data - median) / mad
                    outliers = data[np.abs(modified_z_scores) > modified_z_threshold]
                    outlier_indices = outliers.index
                
                # Create outlier plot
                outlier_status = ['Outlier' if idx in outlier_indices else 'Normal' for idx in data.index]
                
                plot_df = pd.DataFrame({
                    outlier_column: data,
                    'Status': outlier_status,
                    'Index': data.index
                })
                
                fig = px.scatter(
                    plot_df,
                    x='Index',
                    y=outlier_column,
                    color='Status',
                    title=f'Outlier Detection - {outlier_column} ({outlier_method})'
                )
                
                fig.update_layout(template='plotly_white')
                st.session_state.outlier_plot = fig
                
                # Store outlier information
                st.session_state.outlier_info = {
                    'n_outliers': len(outliers),
                    'outlier_percentage': (len(outliers) / len(data)) * 100,
                    'outlier_values': outliers.tolist(),
                    'outlier_indices': outlier_indices.tolist()
                }
                
                st.success("✅ Outlier detection completed!")
                
            except Exception as e:
                st.error(f"Error detecting outliers: {str(e)}")
    
    with col2:
        if 'outlier_plot' in st.session_state:
            st.plotly_chart(st.session_state.outlier_plot, use_container_width=True)
            
            # Show outlier statistics
            if 'outlier_info' in st.session_state:
                info = st.session_state.outlier_info
                
                st.write("**Outlier Statistics:**")
                st.write(f"**Number of outliers:** {info['n_outliers']}")
                st.write(f"**Percentage of outliers:** {info['outlier_percentage']:.2f}%")
                
                if info['n_outliers'] > 0:
                    st.write("**Outlier values:**")
                    outlier_df = pd.DataFrame({
                        'Index': info['outlier_indices'],
                        'Value': info['outlier_values']
                    })
                    st.dataframe(outlier_df, use_container_width=True)
        else:
            st.info("Configure outlier detection settings and click 'Detect Outliers'")
    
    # Data Quality Assessment
    st.subheader("📊 Data Quality Assessment")
    
    if st.button("Assess Data Quality", type="primary"):
        try:
            quality_report = {}
            
            # Missing values
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100
            
            quality_report['missing_data'] = pd.DataFrame({
                'Column': missing_counts.index,
                'Missing Count': missing_counts.values,
                'Missing Percentage': missing_percentages.values
            })
            
            # Duplicate rows
            duplicate_count = df.duplicated().sum()
            quality_report['duplicates'] = duplicate_count
            
            # Data types
            quality_report['data_types'] = df.dtypes.value_counts()
            
            # Unique values for categorical columns
            categorical_info = []
            for col in categorical_cols:
                if col in df.columns:
                    unique_count = df[col].nunique()
                    categorical_info.append({
                        'Column': col,
                        'Unique Values': unique_count,
                        'Most Common': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
                    })
            
            quality_report['categorical_info'] = pd.DataFrame(categorical_info)
            
            # Numeric data ranges
            numeric_info = []
            for col in numeric_cols:
                if col in df.columns:
                    numeric_info.append({
                        'Column': col,
                        'Min': df[col].min(),
                        'Max': df[col].max(),
                        'Mean': df[col].mean(),
                        'Std Dev': df[col].std(),
                        'Zeros': (df[col] == 0).sum(),
                        'Negative': (df[col] < 0).sum()
                    })
            
            quality_report['numeric_info'] = pd.DataFrame(numeric_info)
            
            st.session_state.quality_report = quality_report
            st.success("✅ Data quality assessment completed!")
            
        except Exception as e:
            st.error(f"Error assessing data quality: {str(e)}")
    
    # Display quality report
    if 'quality_report' in st.session_state:
        report = st.session_state.quality_report
        
        st.write("**Data Quality Report:**")
        
        # Missing data
        st.write("**Missing Data:**")
        missing_data = report['missing_data']
        missing_data = missing_data[missing_data['Missing Count'] > 0]
        if not missing_data.empty:
            st.dataframe(missing_data, use_container_width=True)
        else:
            st.success("No missing data found!")
        
        # Duplicates
        st.write(f"**Duplicate Rows:** {report['duplicates']}")
        
        # Data types
        st.write("**Data Types Distribution:**")
        st.dataframe(report['data_types'].reset_index(), use_container_width=True)
        
        # Categorical information
        if not report['categorical_info'].empty:
            st.write("**Categorical Variables:**")
            st.dataframe(report['categorical_info'], use_container_width=True)
        
        # Numeric information
        if not report['numeric_info'].empty:
            st.write("**Numeric Variables:**")
            st.dataframe(report['numeric_info'], use_container_width=True)

# Export statistical results
st.header("💾 Export Statistical Results")

if st.button("Generate Statistical Report", type="primary"):
    report_data = {}
    
    # Collect all analysis results
    for key in st.session_state.keys():
        if any(x in key for x in ['_result', '_stats', '_report', 'corr_matrix']):
            try:
                value = st.session_state[key]
                if isinstance(value, pd.DataFrame):
                    report_data[key] = value
                elif isinstance(value, dict):
                    # Convert dict to DataFrame if possible
                    try:
                        report_data[key] = pd.DataFrame([value])
                    except:
                        report_data[key] = str(value)
            except:
                pass
    
    if report_data:
        # Create Excel file with multiple sheets
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, data in report_data.items():
                if isinstance(data, pd.DataFrame):
                    sheet_name_clean = sheet_name.replace('_', ' ').title()[:31]  # Excel sheet name limit
                    data.to_excel(writer, sheet_name=sheet_name_clean, index=False)
        
        excel_data = output.getvalue()
        
        st.download_button(
            label="Download Statistical Report (Excel)",
            data=excel_data,
            file_name="statistical_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.success("✅ Statistical report generated!")
    else:
        st.info("No statistical results to export. Perform some analyses first!")

