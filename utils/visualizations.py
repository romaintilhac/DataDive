import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from utils.constants import CHONDRITE_VALUES, PRIMITIVE_MANTLE_VALUES

class GeochemicalPlotter:
    """Class for creating geochemical visualizations"""
    
    def __init__(self):
        self.color_schemes = {
            'Lithology': px.colors.qualitative.Set1,
            'Zone': px.colors.qualitative.Set2,
            'Unit': px.colors.qualitative.Dark2
        }
    
    def create_scatter_plot(self, df: pd.DataFrame, 
                           x_col: str, y_col: str, 
                           color_col: str = None, 
                           size_col: str = None,
                           title: str = None,
                           log_x: bool = False,
                           log_y: bool = False) -> go.Figure:
        """Create scatter plot with customizable options"""
        
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            size=size_col,
            hover_data=['Sample'] if 'Sample' in df.columns else None,
            title=title or f"{y_col} vs {x_col}",
            log_x=log_x,
            log_y=log_y,
            color_discrete_sequence=self.color_schemes.get(color_col, px.colors.qualitative.Plotly)
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode='closest',
            template='plotly_white',
            width=700,
            height=500
        )
        
        return fig
    
    def create_harker_diagram(self, df: pd.DataFrame, 
                             element: str, 
                             color_col: str = None) -> go.Figure:
        """Create Harker diagram (element vs SiO2)"""
        
        if 'SiO2' not in df.columns or element not in df.columns:
            st.error(f"Required columns not found: SiO2 or {element}")
            return go.Figure()
        
        fig = px.scatter(
            df,
            x='SiO2',
            y=element,
            color=color_col,
            hover_data=['Sample'] if 'Sample' in df.columns else None,
            title=f"Harker Diagram: {element} vs SiO2",
            color_discrete_sequence=self.color_schemes.get(color_col, px.colors.qualitative.Plotly)
        )
        
        fig.update_layout(
            xaxis_title='SiO2 (wt%)',
            yaxis_title=f'{element} (wt%)' if element in ['TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5'] else f'{element} (ppm)',
            template='plotly_white',
            width=700,
            height=500
        )
        
        return fig
    
    def create_ree_spider_plot(self, df: pd.DataFrame, 
                              normalize: str = 'chondrite',
                              color_col: str = None) -> go.Figure:
        """Create REE spider diagram"""
        
        ree_elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        
        # Check which REE elements are available
        available_ree = [elem for elem in ree_elements if elem in df.columns]
        
        if not available_ree:
            st.error("No REE elements found in the dataset")
            return go.Figure()
        
        # Normalize values
        if normalize == 'chondrite':
            normalization_values = CHONDRITE_VALUES
            norm_suffix = '_N'
            y_title = 'Sample / Chondrite'
        else:
            normalization_values = PRIMITIVE_MANTLE_VALUES
            norm_suffix = '_PM'
            y_title = 'Sample / Primitive Mantle'
        
        fig = go.Figure()
        
        # Create color mapping
        if color_col and color_col in df.columns:
            unique_values = df[color_col].unique()
            color_map = {val: self.color_schemes.get(color_col, px.colors.qualitative.Plotly)[i % len(self.color_schemes.get(color_col, px.colors.qualitative.Plotly))] 
                        for i, val in enumerate(unique_values)}
        
        # Plot each sample
        for idx, row in df.iterrows():
            y_values = []
            x_labels = []
            
            for elem in available_ree:
                if pd.notna(row[elem]) and elem in normalization_values:
                    normalized_value = row[elem] / normalization_values[elem]
                    y_values.append(normalized_value)
                    x_labels.append(elem)
            
            if y_values:
                line_color = color_map.get(row[color_col], 'blue') if color_col and color_col in df.columns else 'blue'
                sample_name = row['Sample'] if 'Sample' in df.columns else f'Sample {idx}'
                group_name = f"{row[color_col]} - {sample_name}" if color_col and color_col in df.columns else sample_name
                
                fig.add_trace(go.Scatter(
                    x=x_labels,
                    y=y_values,
                    mode='lines+markers',
                    name=group_name,
                    line=dict(color=line_color),
                    marker=dict(color=line_color, size=6)
                ))
        
        fig.update_layout(
            title="REE Spider Diagram",
            xaxis_title="REE Elements",
            yaxis_title=y_title,
            yaxis_type='log',
            template='plotly_white',
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_multi_element_plot(self, df: pd.DataFrame, 
                                 elements: List[str] = None,
                                 color_col: str = None) -> go.Figure:
        """Create multi-element spider plot normalized to primitive mantle"""
        
        default_elements = ['Ba', 'Th', 'U', 'Nb', 'Ta', 'La', 'Ce', 'Pr', 'Nd', 'Sr', 'Sm', 'Zr', 'Hf', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        
        if elements is None:
            elements = default_elements
        
        # Check which elements are available
        available_elements = [elem for elem in elements if elem in df.columns]
        
        if not available_elements:
            st.error("No trace elements found in the dataset")
            return go.Figure()
        
        fig = go.Figure()
        
        # Create color mapping
        if color_col and color_col in df.columns:
            unique_values = df[color_col].unique()
            color_map = {val: self.color_schemes.get(color_col, px.colors.qualitative.Plotly)[i % len(self.color_schemes.get(color_col, px.colors.qualitative.Plotly))] 
                        for i, val in enumerate(unique_values)}
        
        # Plot each sample
        for idx, row in df.iterrows():
            y_values = []
            x_labels = []
            
            for elem in available_elements:
                if pd.notna(row[elem]) and elem in PRIMITIVE_MANTLE_VALUES:
                    normalized_value = row[elem] / PRIMITIVE_MANTLE_VALUES[elem]
                    y_values.append(normalized_value)
                    x_labels.append(elem)
            
            if y_values:
                line_color = color_map.get(row[color_col], 'blue') if color_col and color_col in df.columns else 'blue'
                sample_name = row['Sample'] if 'Sample' in df.columns else f'Sample {idx}'
                group_name = f"{row[color_col]} - {sample_name}" if color_col and color_col in df.columns else sample_name
                
                fig.add_trace(go.Scatter(
                    x=x_labels,
                    y=y_values,
                    mode='lines+markers',
                    name=group_name,
                    line=dict(color=line_color),
                    marker=dict(color=line_color, size=6)
                ))
        
        fig.update_layout(
            title="Multi-element Plot (Primitive Mantle Normalized)",
            xaxis_title="Elements",
            yaxis_title="Sample / Primitive Mantle",
            yaxis_type='log',
            template='plotly_white',
            width=900,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_box_plot(self, df: pd.DataFrame, 
                       y_col: str, 
                       x_col: str = None, 
                       title: str = None) -> go.Figure:
        """Create box plot for comparing groups"""
        
        if x_col and x_col in df.columns:
            fig = px.box(
                df, 
                x=x_col, 
                y=y_col,
                title=title or f"{y_col} by {x_col}",
                color=x_col,
                color_discrete_sequence=self.color_schemes.get(x_col, px.colors.qualitative.Plotly)
            )
        else:
            fig = px.box(
                df, 
                y=y_col,
                title=title or f"Distribution of {y_col}"
            )
        
        fig.update_layout(
            template='plotly_white',
            width=600,
            height=500
        )
        
        return fig
    
    def create_histogram(self, df: pd.DataFrame, 
                        col: str, 
                        color_col: str = None,
                        bins: int = 30) -> go.Figure:
        """Create histogram"""
        
        if color_col and color_col in df.columns:
            fig = px.histogram(
                df, 
                x=col, 
                color=color_col,
                nbins=bins,
                title=f"Distribution of {col}",
                color_discrete_sequence=self.color_schemes.get(color_col, px.colors.qualitative.Plotly)
            )
        else:
            fig = px.histogram(
                df, 
                x=col, 
                nbins=bins,
                title=f"Distribution of {col}"
            )
        
        fig.update_layout(
            template='plotly_white',
            width=600,
            height=500
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame, 
                                  elements: List[str] = None) -> go.Figure:
        """Create correlation heatmap"""
        
        if elements is None:
            # Use numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            elements = [col for col in numeric_cols if not col.endswith('_err')]
        
        # Filter available elements
        available_elements = [elem for elem in elements if elem in df.columns]
        
        if len(available_elements) < 2:
            st.error("Need at least 2 numeric columns for correlation analysis")
            return go.Figure()
        
        # Calculate correlation matrix
        corr_matrix = df[available_elements].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(
            width=800,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_epsilon_plot(self, df: pd.DataFrame, 
                           epsilon_x: str = 'εNd', 
                           epsilon_y: str = 'εHf',
                           color_col: str = None) -> go.Figure:
        """Create epsilon-epsilon plot"""
        
        if epsilon_x not in df.columns or epsilon_y not in df.columns:
            st.error(f"Required columns not found: {epsilon_x} or {epsilon_y}")
            return go.Figure()
        
        fig = px.scatter(
            df,
            x=epsilon_x,
            y=epsilon_y,
            color=color_col,
            hover_data=['Sample'] if 'Sample' in df.columns else None,
            title=f"{epsilon_y} vs {epsilon_x}",
            color_discrete_sequence=self.color_schemes.get(color_col, px.colors.qualitative.Plotly)
        )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="CHUR")
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="CHUR")
        
        fig.update_layout(
            xaxis_title=epsilon_x,
            yaxis_title=epsilon_y,
            template='plotly_white',
            width=700,
            height=500
        )
        
        return fig
    
    def create_classification_plot(self, df: pd.DataFrame, 
                                  plot_type: str = 'TAS') -> go.Figure:
        """Create classification diagrams"""
        
        if plot_type == 'TAS':
            # Total Alkali vs Silica
            if 'SiO2' not in df.columns or 'Na2O' not in df.columns or 'K2O' not in df.columns:
                st.error("Required columns not found for TAS diagram: SiO2, Na2O, K2O")
                return go.Figure()
            
            df['Total_Alkali'] = df['Na2O'] + df['K2O']
            
            fig = px.scatter(
                df,
                x='SiO2',
                y='Total_Alkali',
                color='Lithology' if 'Lithology' in df.columns else None,
                hover_data=['Sample'] if 'Sample' in df.columns else None,
                title="Total Alkali vs Silica (TAS) Diagram"
            )
            
            fig.update_layout(
                xaxis_title='SiO2 (wt%)',
                yaxis_title='Na2O + K2O (wt%)',
                template='plotly_white',
                width=700,
                height=500
            )
            
        elif plot_type == 'AFM':
            # Alkali-FeO-MgO
            if not all(col in df.columns for col in ['Na2O', 'K2O', 'FeO', 'MgO']):
                st.error("Required columns not found for AFM diagram: Na2O, K2O, FeO, MgO")
                return go.Figure()
            
            # Calculate ternary coordinates
            A = df['Na2O'] + df['K2O']
            F = df['FeO']
            M = df['MgO']
            
            total = A + F + M
            A_norm = A / total
            F_norm = F / total
            M_norm = M / total
            
            fig = go.Figure()
            
            # This is a simplified AFM plot - full ternary would require more complex plotting
            fig.add_trace(go.Scatter(
                x=F_norm,
                y=A_norm,
                mode='markers',
                marker=dict(size=8),
                text=df['Sample'] if 'Sample' in df.columns else None,
                name='Samples'
            ))
            
            fig.update_layout(
                title="AFM Diagram (Simplified)",
                xaxis_title="FeO (normalized)",
                yaxis_title="Na2O + K2O (normalized)",
                template='plotly_white',
                width=700,
                height=500
            )
        
        return fig
