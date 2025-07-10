"""
Pyrolite integration for advanced geochemical analysis.
This module provides enhanced normalization, classification, and geochemical plotting capabilities.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Union

try:
    import pyrolite.geochem as geochem
    import pyrolite.plot as pyroplot
    import pyrolite.comp as comp
    from pyrolite.geochem import REE, get_reference_composition
    from pyrolite.plot.spider import REE_v_N
    from pyrolite.plot.binary import TAS, QAP
    from pyrolite.plot.ternary import Ternary
    from pyrolite.mineral import olivine, orthopyroxene, clinopyroxene, plagioclase
    PYROLITE_AVAILABLE = True
except ImportError:
    PYROLITE_AVAILABLE = False
    print("Pyrolite not available - some advanced features will be disabled")

class PyroliteAnalyzer:
    """Advanced geochemical analysis using pyrolite"""
    
    def __init__(self):
        self.pyrolite_available = PYROLITE_AVAILABLE
        
    def check_availability(self):
        """Check if pyrolite is available"""
        return self.pyrolite_available
    
    def normalize_to_reference(self, df: pd.DataFrame, 
                              reference: str = 'Chondrite_PON',
                              elements: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize elements to reference composition using pyrolite
        
        Parameters:
        - df: DataFrame with geochemical data
        - reference: Reference composition ('Chondrite_PON', 'PrimitiveMantle_PM', etc.)
        - elements: List of elements to normalize (default: REE)
        
        Returns:
        - DataFrame with normalized values
        """
        if not self.pyrolite_available:
            raise ImportError("Pyrolite not available")
            
        df_norm = df.copy()
        
        # Get reference composition
        ref_comp = get_reference_composition(reference)
        
        # Default to REE elements if not specified
        if elements is None:
            elements = REE()
        
        # Normalize available elements
        for element in elements:
            if element in df.columns and element in ref_comp.index:
                norm_col = f"{element}_{reference.split('_')[-1]}"
                df_norm[norm_col] = df[element] / ref_comp[element]
        
        return df_norm
    
    def create_ree_spider_plot(self, df: pd.DataFrame, 
                              reference: str = 'Chondrite_PON',
                              sample_col: str = 'Sample',
                              color_col: Optional[str] = None) -> go.Figure:
        """
        Create REE spider plot using pyrolite normalization
        
        Parameters:
        - df: DataFrame with geochemical data
        - reference: Reference for normalization
        - sample_col: Column name for sample identification
        - color_col: Column for color coding
        
        Returns:
        - Plotly figure
        """
        if not self.pyrolite_available:
            raise ImportError("Pyrolite not available")
        
        # Normalize REE elements
        df_norm = self.normalize_to_reference(df, reference)
        
        # REE elements in order
        ree_elements = REE()
        norm_suffix = reference.split('_')[-1]
        ree_norm_cols = [f"{elem}_{norm_suffix}" for elem in ree_elements]
        
        # Filter available columns
        available_cols = [col for col in ree_norm_cols if col in df_norm.columns]
        
        if not available_cols:
            raise ValueError("No REE data available for spider plot")
        
        fig = go.Figure()
        
        # Create traces
        if color_col and color_col in df.columns:
            unique_groups = df[color_col].unique()
            colors = px.colors.qualitative.Set1
            
            for i, group in enumerate(unique_groups):
                group_data = df_norm[df_norm[color_col] == group]
                
                for idx, row in group_data.iterrows():
                    y_values = [row[col] for col in available_cols if pd.notna(row[col])]
                    x_values = [col.replace(f'_{norm_suffix}', '') for col in available_cols if pd.notna(row[col])]
                    
                    if y_values:
                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode='lines+markers',
                            name=f"{row[sample_col]} ({group})",
                            line=dict(color=colors[i % len(colors)]),
                            marker=dict(size=6),
                            showlegend=(idx == group_data.index[0])
                        ))
        else:
            colors = px.colors.qualitative.Set1
            for idx, row in df_norm.iterrows():
                y_values = [row[col] for col in available_cols if pd.notna(row[col])]
                x_values = [col.replace(f'_{norm_suffix}', '') for col in available_cols if pd.notna(row[col])]
                
                if y_values:
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode='lines+markers',
                        name=row[sample_col],
                        line=dict(color=colors[idx % len(colors)]),
                        marker=dict(size=6)
                    ))
        
        fig.update_layout(
            title=f'REE Spider Diagram ({reference} Normalized)',
            xaxis_title='REE Elements',
            yaxis_title=f'Sample / {reference}',
            yaxis_type='log',
            template='plotly_white',
            showlegend=True,
            height=600
        )
        
        return fig
    
    def create_tas_diagram(self, df: pd.DataFrame, 
                          color_col: Optional[str] = None) -> go.Figure:
        """
        Create TAS (Total Alkali-Silica) classification diagram
        
        Parameters:
        - df: DataFrame with SiO2, Na2O, K2O data
        - color_col: Column for color coding
        
        Returns:
        - Plotly figure
        """
        if not self.pyrolite_available:
            raise ImportError("Pyrolite not available")
        
        # Check required columns
        required_cols = ['SiO2', 'Na2O', 'K2O']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate total alkalis
        df_plot = df.copy()
        df_plot['Total_Alkalis'] = df_plot['Na2O'] + df_plot['K2O']
        
        # Create scatter plot
        fig = go.Figure()
        
        if color_col and color_col in df.columns:
            unique_groups = df[color_col].unique()
            colors = px.colors.qualitative.Set1
            
            for i, group in enumerate(unique_groups):
                group_data = df_plot[df_plot[color_col] == group]
                
                fig.add_trace(go.Scatter(
                    x=group_data['SiO2'],
                    y=group_data['Total_Alkalis'],
                    mode='markers',
                    name=group,
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=8,
                        line=dict(width=1, color='black')
                    )
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df_plot['SiO2'],
                y=df_plot['Total_Alkalis'],
                mode='markers',
                name='Samples',
                marker=dict(
                    color='blue',
                    size=8,
                    line=dict(width=1, color='black')
                )
            ))
        
        # Add TAS classification fields (simplified)
        # This is a basic implementation - pyrolite provides more detailed fields
        fig.add_shape(
            type="line",
            x0=45, y0=5, x1=61.32, y1=13.5,
            line=dict(color="black", width=1, dash="dash"),
        )
        
        fig.update_layout(
            title='TAS Classification Diagram',
            xaxis_title='SiO₂ (wt%)',
            yaxis_title='Na₂O + K₂O (wt%)',
            template='plotly_white',
            showlegend=True,
            height=600,
            xaxis=dict(range=[35, 80]),
            yaxis=dict(range=[0, 20])
        )
        
        return fig
    
    def calculate_mineral_chemistry(self, df: pd.DataFrame, 
                                   mineral: str = 'olivine') -> pd.DataFrame:
        """
        Calculate mineral chemistry parameters
        
        Parameters:
        - df: DataFrame with major element data
        - mineral: Mineral type ('olivine', 'orthopyroxene', 'clinopyroxene', 'plagioclase')
        
        Returns:
        - DataFrame with mineral chemistry parameters
        """
        if not self.pyrolite_available:
            raise ImportError("Pyrolite not available")
        
        df_mineral = df.copy()
        
        if mineral == 'olivine':
            # Calculate forsterite content (Fo)
            if 'MgO' in df.columns and 'FeO' in df.columns:
                mg_mol = df['MgO'] / 40.30  # MgO molecular weight
                fe_mol = df['FeO'] / 71.85  # FeO molecular weight
                df_mineral['Fo'] = 100 * mg_mol / (mg_mol + fe_mol)
        
        elif mineral == 'plagioclase':
            # Calculate anorthite content (An)
            if 'CaO' in df.columns and 'Na2O' in df.columns and 'K2O' in df.columns:
                ca_mol = df['CaO'] / 56.08
                na_mol = df['Na2O'] / 61.98
                k_mol = df['K2O'] / 94.20
                total_mol = ca_mol + na_mol + k_mol
                df_mineral['An'] = 100 * ca_mol / total_mol
                df_mineral['Ab'] = 100 * na_mol / total_mol
                df_mineral['Or'] = 100 * k_mol / total_mol
        
        return df_mineral
    
    def get_compositional_data(self, df: pd.DataFrame, 
                              elements: List[str]) -> pd.DataFrame:
        """
        Prepare compositional data for analysis
        
        Parameters:
        - df: DataFrame with geochemical data
        - elements: List of elements to include
        
        Returns:
        - DataFrame with compositional data
        """
        if not self.pyrolite_available:
            raise ImportError("Pyrolite not available")
        
        # Filter available elements
        available_elements = [elem for elem in elements if elem in df.columns]
        
        if not available_elements:
            raise ValueError("No specified elements found in data")
        
        # Get compositional data
        comp_data = df[available_elements].copy()
        
        # Remove rows with all NaN values
        comp_data = comp_data.dropna(how='all')
        
        # Fill remaining NaN with 0 (for compositional analysis)
        comp_data = comp_data.fillna(0)
        
        return comp_data
    
    def calculate_mg_number(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Mg# using pyrolite conventions
        
        Parameters:
        - df: DataFrame with MgO and FeO data
        
        Returns:
        - Series with Mg# values
        """
        if 'MgO' in df.columns and 'FeO' in df.columns:
            # Convert to molecular proportions
            mg_mol = df['MgO'] / 40.30  # MgO molecular weight
            fe_mol = df['FeO'] / 71.85  # FeO molecular weight
            
            # Calculate Mg# = 100 * Mg/(Mg + Fe)
            mg_number = 100 * mg_mol / (mg_mol + fe_mol)
            
            return mg_number
        else:
            return pd.Series(dtype=float)
    
    def analyze_trace_element_patterns(self, df: pd.DataFrame,
                                      elements: Optional[List[str]] = None) -> Dict:
        """
        Analyze trace element patterns and anomalies
        
        Parameters:
        - df: DataFrame with trace element data
        - elements: List of elements to analyze (default: common trace elements)
        
        Returns:
        - Dictionary with pattern analysis results
        """
        if not self.pyrolite_available:
            raise ImportError("Pyrolite not available")
        
        if elements is None:
            elements = ['Ba', 'Th', 'U', 'Nb', 'Ta', 'La', 'Ce', 'Pr', 'Nd', 'Sr', 'Sm', 'Zr', 'Hf']
        
        # Filter available elements
        available_elements = [elem for elem in elements if elem in df.columns]
        
        if not available_elements:
            return {"error": "No trace elements available for analysis"}
        
        results = {
            "available_elements": available_elements,
            "element_statistics": {}
        }
        
        # Calculate statistics for each element
        for elem in available_elements:
            elem_data = df[elem].dropna()
            if len(elem_data) > 0:
                results["element_statistics"][elem] = {
                    "mean": float(elem_data.mean()),
                    "median": float(elem_data.median()),
                    "std": float(elem_data.std()),
                    "min": float(elem_data.min()),
                    "max": float(elem_data.max()),
                    "n_samples": len(elem_data)
                }
        
        return results