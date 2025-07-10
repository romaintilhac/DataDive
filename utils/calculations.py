import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional
from utils.constants import (
    CHONDRITE_VALUES, PRIMITIVE_MANTLE_VALUES, MOLECULAR_WEIGHTS,
    LAMBDA_147SM, LAMBDA_176LU, CHUR_143ND_144ND, CHUR_176HF_177HF,
    CHUR_147SM_144ND, CHUR_176LU_177HF, HFND_ARRAY_A, HFND_ARRAY_B,
    DM_143ND_144ND, DM_176HF_177HF, ABUNDANCE_147SM, ABUNDANCE_144ND,
    ABUNDANCE_176LU, ABUNDANCE_177HF
)

class GeochemicalCalculator:
    """Class for geochemical calculations and derived parameters"""
    
    def __init__(self):
        self.molecular_weights = MOLECULAR_WEIGHTS
        
        # Constants for isotope calculations
        self.lambda_147Sm = LAMBDA_147SM
        self.lambda_176Lu = LAMBDA_176LU
        self.age_Ma = 100  # Default age in Ma
        
        # Present-day values
        self.CHUR_143Nd_144Nd = CHUR_143ND_144ND
        self.CHUR_176Hf_177Hf = CHUR_176HF_177HF
        self.CHUR_147Sm_144Nd = CHUR_147SM_144ND
        self.CHUR_176Lu_177Hf = CHUR_176LU_177HF
        self.DM_143Nd_144Nd = DM_143ND_144ND
        self.DM_176Hf_177Hf = DM_176HF_177HF
        
        # Mantle array parameters
        self.HfNd_array_a = HFND_ARRAY_A
        self.HfNd_array_b = HFND_ARRAY_B
    
    def calculate_mg_number(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Mg# = 100 * Mg/(Mg + Fe)"""
        if 'MgO' in df.columns and 'FeO' in df.columns:
            # Convert to molar ratios
            mg_mol = df['MgO'] / self.molecular_weights['MgO']
            fe_mol = df['FeO'] / self.molecular_weights['FeO']
            
            mg_number = 100 * mg_mol / (mg_mol + fe_mol)
            return mg_number
        else:
            return pd.Series(np.nan, index=df.index)
    
    def calculate_epsilon_nd(self, df: pd.DataFrame, 
                           initial: bool = False, 
                           age_ma: float = None) -> pd.Series:
        """Calculate εNd or εNd(i)"""
        if '143Nd/144Nd' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        if initial and age_ma is not None and '147Sm/144Nd' in df.columns:
            # Calculate initial ratios
            age_seconds = age_ma * 1e6 * 365.25 * 24 * 3600
            initial_ratio = df['143Nd/144Nd'] - df['147Sm/144Nd'] * (np.exp(self.lambda_147Sm * age_seconds) - 1)
            epsilon_nd = ((initial_ratio / self.CHUR_143Nd_144Nd) - 1) * 10000
        else:
            # Present-day values
            epsilon_nd = ((df['143Nd/144Nd'] / self.CHUR_143Nd_144Nd) - 1) * 10000
        
        return epsilon_nd
    
    def calculate_epsilon_hf(self, df: pd.DataFrame, 
                           initial: bool = False, 
                           age_ma: float = None) -> pd.Series:
        """Calculate εHf or εHf(i)"""
        if '176Hf/177Hf' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        if initial and age_ma is not None and '176Lu/177Hf' in df.columns:
            # Calculate initial ratios
            age_seconds = age_ma * 1e6 * 365.25 * 24 * 3600
            initial_ratio = df['176Hf/177Hf'] - df['176Lu/177Hf'] * (np.exp(self.lambda_176Lu * age_seconds) - 1)
            epsilon_hf = ((initial_ratio / self.CHUR_176Hf_177Hf) - 1) * 10000
        else:
            # Present-day values
            epsilon_hf = ((df['176Hf/177Hf'] / self.CHUR_176Hf_177Hf) - 1) * 10000
        
        return epsilon_hf
    
    def calculate_delta_epsilon_hf(self, df: pd.DataFrame, 
                                  initial: bool = False, 
                                  age_ma: float = None) -> pd.Series:
        """Calculate ΔεHf = εHf - (a*εNd + b) using mantle array"""
        epsilon_hf = self.calculate_epsilon_hf(df, initial, age_ma)
        epsilon_nd = self.calculate_epsilon_nd(df, initial, age_ma)
        
        # Use mantle array correction
        mantle_array_hf = self.HfNd_array_a * epsilon_nd + self.HfNd_array_b
        return epsilon_hf - mantle_array_hf
    
    def calculate_lu_hf_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 176Lu/177Hf ratio"""
        if 'Lu' in df.columns and 'Hf' in df.columns:
            # Use improved calculation with proper atomic masses
            lu_hf_ratio = (df['Lu'] / df['Hf']) * (ABUNDANCE_176LU / ABUNDANCE_177HF)
            return lu_hf_ratio
        else:
            return pd.Series(np.nan, index=df.index)
    
    def calculate_sm_nd_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate 147Sm/144Nd ratio"""
        if 'Sm' in df.columns and 'Nd' in df.columns:
            # Use improved calculation with proper atomic abundances
            sm_nd_ratio = (df['Sm'] / df['Nd']) * (ABUNDANCE_147SM / ABUNDANCE_144ND)
            return sm_nd_ratio
        else:
            return pd.Series(np.nan, index=df.index)
    
    def normalize_to_chondrite(self, df: pd.DataFrame, elements: List[str]) -> pd.DataFrame:
        """Normalize elements to chondrite values"""
        normalized_df = df.copy()
        
        for element in elements:
            if element in df.columns and element in CHONDRITE_VALUES:
                col_name = f"{element}_N"
                normalized_df[col_name] = df[element] / CHONDRITE_VALUES[element]
        
        return normalized_df
    
    def normalize_to_primitive_mantle(self, df: pd.DataFrame, elements: List[str]) -> pd.DataFrame:
        """Normalize elements to primitive mantle values"""
        normalized_df = df.copy()
        
        for element in elements:
            if element in df.columns and element in PRIMITIVE_MANTLE_VALUES:
                col_name = f"{element}_PM"
                normalized_df[col_name] = df[element] / PRIMITIVE_MANTLE_VALUES[element]
        
        return normalized_df
    
    def calculate_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various elemental ratios"""
        df_ratios = df.copy()
        
        # REE ratios
        if 'La' in df.columns and 'Yb' in df.columns:
            df_ratios['La/Yb'] = df['La'] / df['Yb']
        
        if 'La' in df.columns and 'Sm' in df.columns:
            df_ratios['La/Sm'] = df['La'] / df['Sm']
        
        if 'Gd' in df.columns and 'Yb' in df.columns:
            df_ratios['Gd/Yb'] = df['Gd'] / df['Yb']
        
        # Trace element ratios
        if 'Th' in df.columns and 'U' in df.columns:
            df_ratios['Th/U'] = df['Th'] / df['U']
        
        if 'Nb' in df.columns and 'Ta' in df.columns:
            df_ratios['Nb/Ta'] = df['Nb'] / df['Ta']
        
        if 'Zr' in df.columns and 'Hf' in df.columns:
            df_ratios['Zr/Hf'] = df['Zr'] / df['Hf']
        
        # Primitive mantle normalized ratios
        if 'Sr' in df.columns and 'Nd' in df.columns:
            if 'Sr' in PRIMITIVE_MANTLE_VALUES and 'Nd' in PRIMITIVE_MANTLE_VALUES:
                df_ratios['(Sr/Nd)PM'] = (df['Sr'] / PRIMITIVE_MANTLE_VALUES['Sr']) / (df['Nd'] / PRIMITIVE_MANTLE_VALUES['Nd'])
        
        if 'Ba' in df.columns and 'La' in df.columns:
            if 'Ba' in PRIMITIVE_MANTLE_VALUES and 'La' in PRIMITIVE_MANTLE_VALUES:
                df_ratios['(Ba/La)PM'] = (df['Ba'] / PRIMITIVE_MANTLE_VALUES['Ba']) / (df['La'] / PRIMITIVE_MANTLE_VALUES['La'])
        
        return df_ratios
    
    def calculate_eu_anomaly(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Eu anomaly: Eu/Eu* = EuN / sqrt(SmN * GdN)"""
        if all(col in df.columns for col in ['Eu', 'Sm', 'Gd']):
            eu_n = df['Eu'] / CHONDRITE_VALUES.get('Eu', 1)
            sm_n = df['Sm'] / CHONDRITE_VALUES.get('Sm', 1)
            gd_n = df['Gd'] / CHONDRITE_VALUES.get('Gd', 1)
            
            eu_anomaly = eu_n / np.sqrt(sm_n * gd_n)
            return eu_anomaly
        else:
            return pd.Series(np.nan, index=df.index)
    
    def calculate_sum_ree(self, df: pd.DataFrame) -> pd.Series:
        """Calculate sum of REE elements"""
        ree_elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        
        ree_columns = [col for col in ree_elements if col in df.columns]
        if ree_columns:
            return df[ree_columns].sum(axis=1)
        else:
            return pd.Series(np.nan, index=df.index)
    
    def calculate_all_parameters(self, df: pd.DataFrame, age_ma: float = None) -> pd.DataFrame:
        """Calculate all derived parameters"""
        df_calc = df.copy()
        
        # Basic calculations
        df_calc['Mg#'] = self.calculate_mg_number(df)
        df_calc['SREE'] = self.calculate_sum_ree(df)
        df_calc['Eu/Eu*'] = self.calculate_eu_anomaly(df)
        
        # Isotope ratios
        df_calc['147Sm/144Nd'] = self.calculate_sm_nd_ratio(df)
        df_calc['176Lu/177Hf'] = self.calculate_lu_hf_ratio(df)
        
        # Epsilon values
        df_calc['εNd'] = self.calculate_epsilon_nd(df, initial=False)
        df_calc['εHf'] = self.calculate_epsilon_hf(df, initial=False)
        
        if age_ma is not None:
            df_calc['εNd(i)'] = self.calculate_epsilon_nd(df, initial=True, age_ma=age_ma)
            df_calc['εHf(i)'] = self.calculate_epsilon_hf(df, initial=True, age_ma=age_ma)
            df_calc['ΔεHf'] = self.calculate_delta_epsilon_hf(df, initial=False)
            df_calc['ΔεHf(i)'] = self.calculate_delta_epsilon_hf(df, initial=True, age_ma=age_ma)
        
        # Element ratios
        df_calc = self.calculate_ratios(df_calc)
        
        # Normalization
        ree_elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        df_calc = self.normalize_to_chondrite(df_calc, ree_elements)
        
        trace_elements = ['Ba', 'Th', 'U', 'Nb', 'Ta', 'La', 'Ce', 'Pr', 'Nd', 'Sr', 'Sm', 'Zr', 'Hf']
        df_calc = self.normalize_to_primitive_mantle(df_calc, trace_elements)
        
        return df_calc
