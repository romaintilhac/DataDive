"""
Global database functionality for geochemical data comparison.
Provides access to reference datasets and comparison tools.

Reference Data Sources:
- MORB: Based on Gale et al. (2013) and PetDB average compositions
- OIB: Based on Hawaiian and Azores basalt compositions (Hofmann, 1997)
- Arc Basalts: Based on Kelemen et al. (2003) and global arc compilations
- Continental Crust: Based on Taylor & McLennan (1985) and Rudnick & Gao (2003)

Note: Current implementation uses representative synthetic values. 
For production use, these should be replaced with actual database connections 
to GEOROC, PetDB, or other geochemical databases.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st
import requests
import io
import json
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
import xmltodict

class GlobalDatabase:
    """Manage global reference datasets for geochemical comparison"""
    
    def __init__(self):
        self.databases = {}
        self.load_default_databases()
    
    def load_default_databases(self):
        """Load default reference databases"""
        
        # Representative reference compositions based on published literature
        # MORB: Mid-Ocean Ridge Basalts (Gale et al., 2013; PetDB)
        # OIB: Ocean Island Basalts (Hofmann, 1997; Hawaiian studies)
        # Arc Basalts: Island/Continental Arc basalts (Kelemen et al., 2003)
        # Continental Crust: Average continental crust (Taylor & McLennan, 1985)
        
        self.databases['MORB'] = self._create_morb_reference()
        self.databases['OIB'] = self._create_oib_reference()
        self.databases['Arc_Basalts'] = self._create_arc_basalt_reference()
        self.databases['Continental_Crust'] = self._create_continental_crust_reference()
        
    def _create_morb_reference(self) -> pd.DataFrame:
        """Create MORB reference dataset based on Gale et al. (2013) and PetDB"""
        # Representative MORB compositions with typical ranges
        morb_data = {
            'Sample': ['MORB_001', 'MORB_002', 'MORB_003', 'MORB_004', 'MORB_005'],
            'SiO2': [50.5, 49.8, 51.2, 50.1, 50.9],
            'TiO2': [1.45, 1.52, 1.38, 1.48, 1.41],
            'Al2O3': [15.2, 15.8, 14.9, 15.4, 15.1],
            'FeO': [10.8, 11.2, 10.5, 10.9, 10.7],
            'MnO': [0.18, 0.19, 0.17, 0.18, 0.17],
            'MgO': [7.8, 8.2, 7.5, 7.9, 7.7],
            'CaO': [11.2, 11.8, 10.9, 11.4, 11.0],
            'Na2O': [2.8, 2.9, 2.7, 2.8, 2.9],
            'K2O': [0.15, 0.18, 0.12, 0.16, 0.14],
            'P2O5': [0.12, 0.13, 0.11, 0.12, 0.11],
            'Lithology': ['MORB', 'MORB', 'MORB', 'MORB', 'MORB'],
            'Tectonic_Setting': ['Mid-Ocean Ridge', 'Mid-Ocean Ridge', 'Mid-Ocean Ridge', 'Mid-Ocean Ridge', 'Mid-Ocean Ridge']
        }
        
        # Add trace elements
        trace_elements = {
            'Rb': [0.56, 0.68, 0.45, 0.62, 0.51],
            'Sr': [90, 95, 85, 92, 88],
            'Y': [28, 32, 26, 30, 27],
            'Zr': [74, 82, 68, 78, 71],
            'Nb': [2.3, 2.8, 2.1, 2.5, 2.2],
            'Ba': [6.3, 7.8, 5.9, 7.1, 6.5],
            'La': [2.5, 2.8, 2.2, 2.6, 2.4],
            'Ce': [7.5, 8.2, 6.8, 7.8, 7.2],
            'Nd': [7.3, 8.1, 6.9, 7.6, 7.0],
            'Sm': [2.6, 2.9, 2.4, 2.7, 2.5],
            'Eu': [1.0, 1.1, 0.9, 1.0, 0.9],
            'Yb': [2.4, 2.7, 2.2, 2.5, 2.3]
        }
        
        morb_data.update(trace_elements)
        return pd.DataFrame(morb_data)
    
    def _create_oib_reference(self) -> pd.DataFrame:
        """Create OIB reference dataset based on Hawaiian and Azores studies"""
        oib_data = {
            'Sample': ['OIB_001', 'OIB_002', 'OIB_003', 'OIB_004', 'OIB_005'],
            'SiO2': [45.2, 44.8, 46.1, 45.5, 45.9],
            'TiO2': [2.85, 3.12, 2.68, 2.95, 2.74],
            'Al2O3': [13.8, 14.2, 13.5, 13.9, 13.7],
            'FeO': [12.2, 12.8, 11.9, 12.4, 12.1],
            'MnO': [0.19, 0.21, 0.18, 0.20, 0.18],
            'MgO': [9.5, 10.2, 9.1, 9.8, 9.3],
            'CaO': [10.8, 11.4, 10.5, 11.0, 10.6],
            'Na2O': [3.2, 3.4, 3.0, 3.3, 3.1],
            'K2O': [0.85, 0.95, 0.78, 0.88, 0.82],
            'P2O5': [0.35, 0.38, 0.32, 0.36, 0.34],
            'Lithology': ['OIB', 'OIB', 'OIB', 'OIB', 'OIB'],
            'Tectonic_Setting': ['Ocean Island', 'Ocean Island', 'Ocean Island', 'Ocean Island', 'Ocean Island']
        }
        
        # Add trace elements
        trace_elements = {
            'Rb': [15.2, 18.5, 13.8, 16.9, 14.7],
            'Sr': [350, 385, 325, 368, 342],
            'Y': [22, 25, 20, 23, 21],
            'Zr': [185, 210, 165, 195, 178],
            'Nb': [28, 35, 24, 31, 26],
            'Ba': [285, 320, 265, 305, 275],
            'La': [28, 32, 25, 30, 27],
            'Ce': [65, 72, 58, 68, 62],
            'Nd': [35, 38, 32, 36, 34],
            'Sm': [7.8, 8.5, 7.2, 8.0, 7.6],
            'Eu': [2.4, 2.6, 2.2, 2.5, 2.3],
            'Yb': [1.8, 2.0, 1.6, 1.9, 1.7]
        }
        
        oib_data.update(trace_elements)
        return pd.DataFrame(oib_data)
    
    def _create_arc_basalt_reference(self) -> pd.DataFrame:
        """Create arc basalt reference dataset based on Kelemen et al. (2003)"""
        arc_data = {
            'Sample': ['ARC_001', 'ARC_002', 'ARC_003', 'ARC_004', 'ARC_005'],
            'SiO2': [52.5, 51.8, 53.2, 52.1, 52.9],
            'TiO2': [0.85, 0.92, 0.78, 0.88, 0.81],
            'Al2O3': [18.2, 18.8, 17.9, 18.4, 18.1],
            'FeO': [8.5, 9.2, 8.1, 8.8, 8.3],
            'MnO': [0.15, 0.16, 0.14, 0.15, 0.14],
            'MgO': [5.8, 6.2, 5.5, 5.9, 5.7],
            'CaO': [9.2, 9.8, 8.9, 9.4, 9.0],
            'Na2O': [3.5, 3.7, 3.3, 3.6, 3.4],
            'K2O': [1.2, 1.4, 1.0, 1.3, 1.1],
            'P2O5': [0.18, 0.20, 0.16, 0.19, 0.17],
            'Lithology': ['Arc_Basalt', 'Arc_Basalt', 'Arc_Basalt', 'Arc_Basalt', 'Arc_Basalt'],
            'Tectonic_Setting': ['Island Arc', 'Island Arc', 'Island Arc', 'Island Arc', 'Island Arc']
        }
        
        # Add trace elements
        trace_elements = {
            'Rb': [25, 35, 18, 28, 22],
            'Sr': [485, 520, 450, 502, 468],
            'Y': [18, 22, 15, 20, 17],
            'Zr': [95, 110, 82, 105, 88],
            'Nb': [3.2, 4.1, 2.8, 3.5, 3.0],
            'Ba': [185, 220, 158, 205, 172],
            'La': [8.5, 10.2, 7.8, 9.1, 8.0],
            'Ce': [22, 26, 19, 24, 21],
            'Nd': [15, 18, 13, 16, 14],
            'Sm': [4.2, 4.8, 3.8, 4.5, 4.0],
            'Eu': [1.3, 1.5, 1.2, 1.4, 1.2],
            'Yb': [1.9, 2.2, 1.7, 2.0, 1.8]
        }
        
        arc_data.update(trace_elements)
        return pd.DataFrame(arc_data)
    
    def _create_continental_crust_reference(self) -> pd.DataFrame:
        """Create continental crust reference dataset"""
        cc_data = {
            'Sample': ['CC_001', 'CC_002', 'CC_003', 'CC_004', 'CC_005'],
            'SiO2': [66.0, 65.2, 67.1, 65.8, 66.5],
            'TiO2': [0.64, 0.71, 0.58, 0.67, 0.61],
            'Al2O3': [15.4, 15.8, 15.1, 15.6, 15.2],
            'FeO': [5.0, 5.5, 4.6, 5.2, 4.8],
            'MnO': [0.09, 0.10, 0.08, 0.09, 0.08],
            'MgO': [2.8, 3.2, 2.5, 3.0, 2.7],
            'CaO': [4.2, 4.6, 3.9, 4.4, 4.0],
            'Na2O': [3.9, 4.1, 3.7, 4.0, 3.8],
            'K2O': [2.8, 3.2, 2.5, 3.0, 2.7],
            'P2O5': [0.15, 0.17, 0.13, 0.16, 0.14],
            'Lithology': ['Continental_Crust', 'Continental_Crust', 'Continental_Crust', 'Continental_Crust', 'Continental_Crust'],
            'Tectonic_Setting': ['Continental', 'Continental', 'Continental', 'Continental', 'Continental']
        }
        
        # Add trace elements
        trace_elements = {
            'Rb': [112, 125, 98, 118, 105],
            'Sr': [350, 385, 325, 368, 342],
            'Y': [22, 25, 20, 23, 21],
            'Zr': [190, 210, 175, 200, 185],
            'Nb': [12, 15, 10, 13, 11],
            'Ba': [550, 620, 485, 580, 520],
            'La': [30, 35, 26, 32, 28],
            'Ce': [64, 70, 58, 67, 61],
            'Nd': [26, 30, 23, 28, 25],
            'Sm': [4.5, 5.1, 4.0, 4.8, 4.3],
            'Eu': [0.88, 0.95, 0.82, 0.91, 0.85],
            'Yb': [2.2, 2.5, 2.0, 2.3, 2.1]
        }
        
        cc_data.update(trace_elements)
        return pd.DataFrame(cc_data)
    
    def add_custom_database(self, name: str, data: pd.DataFrame):
        """Add a custom reference database"""
        self.databases[name] = data.copy()
        return True
    
    def load_from_file(self, name: str, uploaded_file, file_type: str = 'auto'):
        """Load reference database from uploaded file"""
        try:
            if file_type == 'auto':
                file_type = uploaded_file.name.split('.')[-1].lower()
            
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                return {'error': f'Unsupported file type: {file_type}'}
            
            # Validate required columns
            if 'Sample' not in df.columns:
                return {'error': 'Reference database must contain a "Sample" column'}
            
            self.databases[name] = df
            return {'success': f'Loaded {len(df)} samples from {uploaded_file.name}'}
            
        except Exception as e:
            return {'error': f'Error loading file: {str(e)}'}
    
    def connect_to_georoc(self, query_params: Dict = None):
        """Connect to GEOROC database and fetch data"""
        try:
            # GEOROC uses a different API structure
            # For now, we'll implement a placeholder that simulates the connection
            # In production, you would need API keys or different endpoints
            
            # Create enhanced synthetic data based on GEOROC-style compositions
            georoc_data = self._create_enhanced_georoc_data(query_params)
            
            # Add to databases
            setting = query_params.get('setting', 'UNKNOWN') if query_params else 'UNKNOWN'
            db_name = f"GEOROC_{setting.replace(' ', '_')}"
            self.databases[db_name] = georoc_data
            
            return {'success': f'Loaded {len(georoc_data)} samples from GEOROC (simulated)', 'database_name': db_name}
                
        except Exception as e:
            return {'error': f'Error processing GEOROC data: {str(e)}'}
    
    def connect_to_petdb(self, query_params: Dict = None):
        """Connect to PetDB database and fetch data"""
        try:
            # PetDB/EarthChem APIs require specific authentication
            # For now, we'll create enhanced synthetic data representative of PetDB
            
            # Create enhanced synthetic data based on PetDB-style compositions
            petdb_data = self._create_enhanced_petdb_data(query_params)
            
            # Add to databases
            rocktype = query_params.get('rocktype', 'UNKNOWN') if query_params else 'UNKNOWN'
            db_name = f"PetDB_{rocktype.replace(' ', '_')}"
            self.databases[db_name] = petdb_data
            
            return {'success': f'Loaded {len(petdb_data)} samples from PetDB (simulated)', 'database_name': db_name}
                
        except Exception as e:
            return {'error': f'Error processing PetDB data: {str(e)}'}
    
    def connect_to_earthchem(self, query_params: Dict = None):
        """Connect to EarthChem database and fetch data"""
        try:
            # EarthChem APIs require specific authentication and endpoints
            # For now, we'll create enhanced synthetic data representative of EarthChem
            
            # Create enhanced synthetic data based on EarthChem-style compositions
            earthchem_data = self._create_enhanced_earthchem_data(query_params)
            
            # Add to databases
            rocktype = query_params.get('rocktype', 'UNKNOWN') if query_params else 'UNKNOWN'
            db_name = f"EarthChem_{rocktype.replace(' ', '_')}"
            self.databases[db_name] = earthchem_data
            
            return {'success': f'Loaded {len(earthchem_data)} samples from EarthChem (simulated)', 'database_name': db_name}
                
        except Exception as e:
            return {'error': f'Error processing EarthChem data: {str(e)}'}
    
    def search_online_databases(self, rock_type: str = None, setting: str = None, 
                               elements: List[str] = None, limit: int = 100):
        """Search multiple online databases with unified parameters"""
        results = {}
        
        # Try GEOROC
        if setting:
            georoc_params = {
                'setting': setting.upper(),
                'limit': limit
            }
            if rock_type:
                georoc_params['rocktype'] = rock_type.upper()
            
            georoc_result = self.connect_to_georoc(georoc_params)
            results['GEOROC'] = georoc_result
        
        # Try PetDB
        if rock_type:
            petdb_params = {
                'rocktype': rock_type,
                'limit': limit
            }
            petdb_result = self.connect_to_petdb(petdb_params)
            results['PetDB'] = petdb_result
        
        # Try EarthChem
        earthchem_params = {
            'rocktype': rock_type if rock_type else 'igneous',
            'limit': limit
        }
        earthchem_result = self.connect_to_earthchem(earthchem_params)
        results['EarthChem'] = earthchem_result
        
        return results
    
    def _create_enhanced_georoc_data(self, query_params: Dict = None) -> pd.DataFrame:
        """Create enhanced GEOROC-style synthetic data with more samples"""
        np.random.seed(42)  # For reproducible results
        
        setting = query_params.get('setting', 'VOLCANIC ARC') if query_params else 'VOLCANIC ARC'
        n_samples = min(query_params.get('limit', 50), 200) if query_params else 50
        
        # Generate sample names
        samples = [f"GEOROC_{setting.replace(' ', '_')}_{i:03d}" for i in range(1, n_samples + 1)]
        
        # Base compositions vary by setting
        if 'ARC' in setting.upper():
            base_data = self._create_arc_basalt_reference().iloc[0].to_dict()
        elif 'RIDGE' in setting.upper():
            base_data = self._create_morb_reference().iloc[0].to_dict()
        elif 'ISLAND' in setting.upper():
            base_data = self._create_oib_reference().iloc[0].to_dict()
        else:
            base_data = self._create_arc_basalt_reference().iloc[0].to_dict()
        
        # Generate variations
        data = {'Sample': samples}
        for element, base_value in base_data.items():
            if element not in ['Sample', 'Lithology', 'Tectonic_Setting']:
                if isinstance(base_value, (int, float)):
                    # Add realistic geological variation (±20%)
                    variations = np.random.normal(base_value, base_value * 0.1, n_samples)
                    variations = np.maximum(variations, 0)  # No negative values
                    data[element] = variations
        
        data['Lithology'] = [setting.title() for _ in range(n_samples)]
        data['Tectonic_Setting'] = [setting for _ in range(n_samples)]
        data['Database'] = ['GEOROC' for _ in range(n_samples)]
        
        return pd.DataFrame(data)
    
    def _create_enhanced_petdb_data(self, query_params: Dict = None) -> pd.DataFrame:
        """Create enhanced PetDB-style synthetic data with more samples"""
        np.random.seed(43)  # For reproducible results
        
        rocktype = query_params.get('rocktype', 'MORB') if query_params else 'MORB'
        n_samples = min(query_params.get('limit', 50), 200) if query_params else 50
        
        # Generate sample names
        samples = [f"PetDB_{rocktype}_{i:03d}" for i in range(1, n_samples + 1)]
        
        # Base compositions vary by rock type
        if 'MORB' in rocktype.upper():
            base_data = self._create_morb_reference().iloc[0].to_dict()
        elif 'OIB' in rocktype.upper():
            base_data = self._create_oib_reference().iloc[0].to_dict()
        else:
            base_data = self._create_morb_reference().iloc[0].to_dict()
        
        # Generate variations
        data = {'Sample': samples}
        for element, base_value in base_data.items():
            if element not in ['Sample', 'Lithology', 'Tectonic_Setting']:
                if isinstance(base_value, (int, float)):
                    # Add realistic geological variation (±15%)
                    variations = np.random.normal(base_value, base_value * 0.08, n_samples)
                    variations = np.maximum(variations, 0)  # No negative values
                    data[element] = variations
        
        data['Lithology'] = [rocktype for _ in range(n_samples)]
        data['Tectonic_Setting'] = ['Marine' for _ in range(n_samples)]
        data['Database'] = ['PetDB' for _ in range(n_samples)]
        
        return pd.DataFrame(data)
    
    def _create_enhanced_earthchem_data(self, query_params: Dict = None) -> pd.DataFrame:
        """Create enhanced EarthChem-style synthetic data with more samples"""
        np.random.seed(44)  # For reproducible results
        
        rocktype = query_params.get('rocktype', 'igneous') if query_params else 'igneous'
        n_samples = min(query_params.get('limit', 50), 200) if query_params else 50
        
        # Generate sample names
        samples = [f"EarthChem_{rocktype}_{i:03d}" for i in range(1, n_samples + 1)]
        
        # Use mixed compositions for general igneous
        base_data = self._create_continental_crust_reference().iloc[0].to_dict()
        
        # Generate variations
        data = {'Sample': samples}
        for element, base_value in base_data.items():
            if element not in ['Sample', 'Lithology', 'Tectonic_Setting']:
                if isinstance(base_value, (int, float)):
                    # Add realistic geological variation (±25%)
                    variations = np.random.normal(base_value, base_value * 0.12, n_samples)
                    variations = np.maximum(variations, 0)  # No negative values
                    data[element] = variations
        
        data['Lithology'] = [rocktype.title() for _ in range(n_samples)]
        data['Tectonic_Setting'] = ['Continental' for _ in range(n_samples)]
        data['Database'] = ['EarthChem' for _ in range(n_samples)]
        
        return pd.DataFrame(data)
    
    def test_online_connectivity(self):
        """Test connectivity to online databases"""
        connectivity_results = {}
        
        # Since the actual APIs require authentication or different endpoints,
        # we'll provide information about the simulated connectivity
        connectivity_results['GEOROC'] = {
            'status': 'simulated',
            'message': 'Enhanced synthetic data based on GEOROC compositions',
            'note': 'Production requires API keys and proper authentication'
        }
        
        connectivity_results['PetDB'] = {
            'status': 'simulated',
            'message': 'Enhanced synthetic data based on PetDB compositions',
            'note': 'Production requires API keys and proper authentication'
        }
        
        connectivity_results['EarthChem'] = {
            'status': 'simulated',
            'message': 'Enhanced synthetic data based on EarthChem compositions',
            'note': 'Production requires API keys and proper authentication'
        }
        
        return connectivity_results
    
    def get_database(self, name: str) -> Optional[pd.DataFrame]:
        """Get a reference database by name"""
        return self.databases.get(name)
    
    def list_databases(self) -> List[str]:
        """List available databases"""
        return list(self.databases.keys())
    
    def get_database_info(self, name: str) -> Dict:
        """Get information about a database"""
        if name not in self.databases:
            return {'error': f'Database {name} not found'}
        
        df = self.databases[name]
        return {
            'name': name,
            'samples': len(df),
            'columns': len(df.columns),
            'elements': [col for col in df.columns if col not in ['Sample', 'Lithology', 'Tectonic_Setting']],
            'lithologies': df['Lithology'].unique().tolist() if 'Lithology' in df.columns else [],
            'tectonic_settings': df['Tectonic_Setting'].unique().tolist() if 'Tectonic_Setting' in df.columns else []
        }
    
    def compare_with_database(self, user_data: pd.DataFrame, 
                             database_name: str,
                             comparison_elements: List[str] = None) -> Dict:
        """
        Compare user data with a reference database
        
        Parameters:
        - user_data: User's geochemical data
        - database_name: Name of reference database
        - comparison_elements: Elements to compare (default: all common elements)
        
        Returns:
        - Dictionary with comparison results
        """
        if database_name not in self.databases:
            return {'error': f'Database {database_name} not found'}
        
        ref_data = self.databases[database_name]
        
        # Find common elements
        if comparison_elements is None:
            user_elements = set(user_data.columns)
            ref_elements = set(ref_data.columns)
            comparison_elements = list(user_elements.intersection(ref_elements))
            comparison_elements = [col for col in comparison_elements 
                                 if col not in ['Sample', 'Lithology', 'Tectonic_Setting']]
        
        if not comparison_elements:
            return {'error': 'No common elements found for comparison'}
        
        # Calculate statistics
        user_stats = {}
        ref_stats = {}
        
        for element in comparison_elements:
            if element in user_data.columns and element in ref_data.columns:
                user_values = user_data[element].dropna()
                ref_values = ref_data[element].dropna()
                
                if len(user_values) > 0 and len(ref_values) > 0:
                    user_stats[element] = {
                        'mean': float(user_values.mean()),
                        'median': float(user_values.median()),
                        'std': float(user_values.std()),
                        'min': float(user_values.min()),
                        'max': float(user_values.max()),
                        'n': len(user_values)
                    }
                    
                    ref_stats[element] = {
                        'mean': float(ref_values.mean()),
                        'median': float(ref_values.median()),
                        'std': float(ref_values.std()),
                        'min': float(ref_values.min()),
                        'max': float(ref_values.max()),
                        'n': len(ref_values)
                    }
        
        # Calculate similarity metrics
        similarity_metrics = self._calculate_similarity_metrics(user_stats, ref_stats)
        
        return {
            'database': database_name,
            'comparison_elements': comparison_elements,
            'user_stats': user_stats,
            'reference_stats': ref_stats,
            'similarity_metrics': similarity_metrics,
            'user_samples': len(user_data),
            'reference_samples': len(ref_data)
        }
    
    def _calculate_similarity_metrics(self, user_stats: Dict, ref_stats: Dict) -> Dict:
        """Calculate similarity metrics between user data and reference"""
        similarity_metrics = {}
        
        for element in user_stats:
            if element in ref_stats:
                user_mean = user_stats[element]['mean']
                ref_mean = ref_stats[element]['mean']
                
                # Calculate relative difference
                rel_diff = abs(user_mean - ref_mean) / ref_mean * 100 if ref_mean != 0 else 0
                
                # Calculate overlap (simplified)
                user_range = (user_stats[element]['min'], user_stats[element]['max'])
                ref_range = (ref_stats[element]['min'], ref_stats[element]['max'])
                
                overlap_min = max(user_range[0], ref_range[0])
                overlap_max = min(user_range[1], ref_range[1])
                overlap = max(0, overlap_max - overlap_min)
                
                total_range = max(user_range[1], ref_range[1]) - min(user_range[0], ref_range[0])
                overlap_percent = (overlap / total_range * 100) if total_range > 0 else 0
                
                similarity_metrics[element] = {
                    'relative_difference_percent': rel_diff,
                    'range_overlap_percent': overlap_percent,
                    'similarity_score': max(0, 100 - rel_diff)  # Simple similarity score
                }
        
        return similarity_metrics
    
    def get_combined_reference_data(self, database_names: List[str] = None) -> pd.DataFrame:
        """
        Get combined reference data from multiple databases
        
        Parameters:
        - database_names: List of database names (default: all databases)
        
        Returns:
        - Combined DataFrame
        """
        if database_names is None:
            database_names = list(self.databases.keys())
        
        combined_data = []
        
        for db_name in database_names:
            if db_name in self.databases:
                df = self.databases[db_name].copy()
                df['Database'] = db_name
                combined_data.append(df)
        
        if combined_data:
            return pd.concat(combined_data, ignore_index=True, sort=False)
        else:
            return pd.DataFrame()
    
    def search_database(self, query_filters: Dict) -> pd.DataFrame:
        """
        Search across all databases with filters
        
        Parameters:
        - query_filters: Dictionary of column -> value filters
        
        Returns:
        - Filtered DataFrame
        """
        all_data = self.get_combined_reference_data()
        
        if all_data.empty:
            return pd.DataFrame()
        
        # Apply filters
        filtered_data = all_data.copy()
        
        for column, value in query_filters.items():
            if column in filtered_data.columns:
                if isinstance(value, (list, tuple)):
                    filtered_data = filtered_data[filtered_data[column].isin(value)]
                else:
                    filtered_data = filtered_data[filtered_data[column] == value]
        
        return filtered_data