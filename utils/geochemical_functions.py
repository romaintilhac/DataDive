"""
Advanced geochemical calculation functions incorporating user's improved tools.
This module contains the safe assignment functionality and comprehensive calculation functions.
"""

import numpy as np
import pandas as pd
import inspect
import sys
from utils.constants import (
    LAMBDA_147SM, LAMBDA_176LU, CHUR_143ND_144ND, CHUR_176HF_177HF,
    CHUR_147SM_144ND, CHUR_176LU_177HF, HFND_ARRAY_A, HFND_ARRAY_B,
    MOLECULAR_WEIGHTS, ABUNDANCE_147SM, ABUNDANCE_144ND, ABUNDANCE_176LU, ABUNDANCE_177HF
)

def safe_assign(df, col_name, series):
    """
    Assigns a new column to the DataFrame.
    If a column with the same name already exists:
    - Adds '_calc' to the new column unless it's an error column derived from a duplicate parent.
    """
    # If the column already exists
    if col_name in df.columns:
        # Handle _err columns
        if col_name.endswith('_err'):
            # Check if parent column is duplicated
            base = col_name[:-4]  # remove '_err'
            if f'{base}_calc' in df.columns:
                dup_col_name = f'{col_name}_calc'
                df[dup_col_name] = series
                return df
            else:
                # Overwrite safely if the parent is not duplicated
                df[col_name] = series
                return df
        else:
            # If not an _err column, assign with _calc suffix
            dup_col_name = f'{col_name}_calc'
            df[dup_col_name] = series
            return df
    else:
        # Column doesn't exist — just assign
        df[col_name] = series
        return df

def get_decay(const_lambda, age):
    """Calculate decay constant for age correction"""
    if age is None:
        raise ValueError("No age provided")
    return np.exp(age * const_lambda * 1e6) - 1

# --- Individual calculation functions ---

## Trace element ratios
def calc_LaSm(df):
    """Calculate La/Sm ratio"""
    if 'La' in df.columns and 'Sm' in df.columns:
        return safe_assign(df, 'La/Sm', df['La']/df['Sm'])
    return df

def calc_GdLu(df):
    """Calculate Gd/Lu ratio"""
    if 'Gd' in df.columns and 'Lu' in df.columns:
        return safe_assign(df, 'Gd/Lu', df['Gd']/df['Lu'])
    return df

def calc_SmNd(df):
    """Calculate Sm/Nd ratio"""
    if 'Sm' in df.columns and 'Nd' in df.columns:
        return safe_assign(df, 'Sm/Nd', df['Sm']/df['Nd'])
    return df

def calc_LuHf(df):
    """Calculate Lu/Hf ratio"""
    if 'Lu' in df.columns and 'Hf' in df.columns:
        return safe_assign(df, 'Lu/Hf', df['Lu']/df['Hf'])
    return df

def calc_ZrHf(df):
    """Calculate Zr/Hf ratio"""
    if 'Zr' in df.columns and 'Hf' in df.columns:
        return safe_assign(df, 'Zr/Hf', df['Zr']/df['Hf'])
    return df

def calc_NbTa(df):
    """Calculate Nb/Ta ratio"""
    if 'Nb' in df.columns and 'Ta' in df.columns:
        return safe_assign(df, 'Nb/Ta', df['Nb']/df['Ta'])
    return df

## Major element ratios
def calc_Mg_number(df):
    """Calculate Mg# = 100 * Mg/(Mg + Fe)"""
    if 'MgO' in df.columns and 'FeO' in df.columns:
        mg_mol = df['MgO'] / MOLECULAR_WEIGHTS['MgO']
        fe_mol = df['FeO'] / MOLECULAR_WEIGHTS['FeO']
        mg_number = 100 * mg_mol / (mg_mol + fe_mol)
        return safe_assign(df, 'Mg#', mg_number)
    return df

## Isotope ratios and corrections
def calc_147Sm_144Nd(df):
    """Calculate 147Sm/144Nd isotope ratio"""
    if 'Sm' in df.columns and 'Nd' in df.columns:
        ratio = (df['Sm'] / df['Nd']) * (ABUNDANCE_147SM / ABUNDANCE_144ND)
        return safe_assign(df, '147Sm/144Nd', ratio)
    return df

def calc_176Lu_177Hf(df):
    """Calculate 176Lu/177Hf isotope ratio"""
    if 'Lu' in df.columns and 'Hf' in df.columns:
        ratio = (df['Lu'] / df['Hf']) * (ABUNDANCE_176LU / ABUNDANCE_177HF)
        return safe_assign(df, '176Lu/177Hf', ratio)
    return df

def calc_143Nd_144Nd_i(df, age):
    """Calculate initial 143Nd/144Nd ratio"""
    if '143Nd/144Nd' in df.columns and '147Sm/144Nd' in df.columns:
        decay_147 = get_decay(LAMBDA_147SM, age)
        initial_ratio = df['143Nd/144Nd'] - df['147Sm/144Nd'] * decay_147
        return safe_assign(df, '143Nd/144Nd(i)', initial_ratio)
    return df

def calc_176Hf_177Hf_i(df, age):
    """Calculate initial 176Hf/177Hf ratio"""
    if '176Hf/177Hf' in df.columns and '176Lu/177Hf' in df.columns:
        decay_176 = get_decay(LAMBDA_176LU, age)
        initial_ratio = df['176Hf/177Hf'] - df['176Lu/177Hf'] * decay_176
        return safe_assign(df, '176Hf/177Hf(i)', initial_ratio)
    return df

def calc_epsilon_Nd(df):
    """Calculate εNd"""
    if '143Nd/144Nd' in df.columns:
        epsilon_nd = ((df['143Nd/144Nd'] / CHUR_143ND_144ND) - 1) * 1e4
        return safe_assign(df, 'εNd', epsilon_nd)
    return df

def calc_epsilon_Hf(df):
    """Calculate εHf"""
    if '176Hf/177Hf' in df.columns:
        epsilon_hf = ((df['176Hf/177Hf'] / CHUR_176HF_177HF) - 1) * 1e4
        return safe_assign(df, 'εHf', epsilon_hf)
    return df

def calc_epsilon_Nd_i(df, age):
    """Calculate initial εNd(i)"""
    if '143Nd/144Nd(i)' in df.columns:
        epsilon_nd_i = ((df['143Nd/144Nd(i)'] / CHUR_143ND_144ND) - 1) * 1e4
        return safe_assign(df, 'εNd(i)', epsilon_nd_i)
    return df

def calc_epsilon_Hf_i(df, age):
    """Calculate initial εHf(i)"""
    if '176Hf/177Hf(i)' in df.columns:
        epsilon_hf_i = ((df['176Hf/177Hf(i)'] / CHUR_176HF_177HF) - 1) * 1e4
        return safe_assign(df, 'εHf(i)', epsilon_hf_i)
    return df

def calc_delta_epsilon_Hf(df):
    """Calculate ΔεHf using mantle array"""
    if 'εHf' in df.columns and 'εNd' in df.columns:
        delta_epsilon_hf = df['εHf'] - (HFND_ARRAY_A * df['εNd'] + HFND_ARRAY_B)
        return safe_assign(df, 'ΔεHf', delta_epsilon_hf)
    return df

def calc_delta_epsilon_Hf_i(df):
    """Calculate initial ΔεHf(i) using mantle array"""
    if 'εHf(i)' in df.columns and 'εNd(i)' in df.columns:
        delta_epsilon_hf_i = df['εHf(i)'] - (HFND_ARRAY_A * df['εNd(i)'] + HFND_ARRAY_B)
        return safe_assign(df, 'ΔεHf(i)', delta_epsilon_hf_i)
    return df

def calc_inverse_Hf(df):
    """Calculate 1/Hf"""
    if 'Hf' in df.columns:
        inverse_hf = 1 / df['Hf']
        return safe_assign(df, '1/Hf', inverse_hf)
    return df

def calc_HfNd(df):
    """Calculate Hf/Nd ratio"""
    if 'Hf' in df.columns and 'Nd' in df.columns:
        hf_nd = df['Hf'] / df['Nd']
        return safe_assign(df, 'Hf/Nd', hf_nd)
    return df

## Error propagation functions
def calc_error_143Nd_144Nd_i(df, age):
    """Calculate error for initial 143Nd/144Nd"""
    if '143Nd/144Nd_err' in df.columns and '147Sm/144Nd_err' in df.columns:
        decay_147 = get_decay(LAMBDA_147SM, age)
        err = np.sqrt(df['143Nd/144Nd_err']**2 + (decay_147 * df['147Sm/144Nd_err'])**2)
        return safe_assign(df, '143Nd/144Nd(i)_err', err)
    return df

def calc_error_176Hf_177Hf_i(df, age):
    """Calculate error for initial 176Hf/177Hf"""
    if '176Hf/177Hf_err' in df.columns and '176Lu/177Hf_err' in df.columns:
        decay_176 = get_decay(LAMBDA_176LU, age)
        err = np.sqrt(df['176Hf/177Hf_err']**2 + (decay_176 * df['176Lu/177Hf_err'])**2)
        return safe_assign(df, '176Hf/177Hf(i)_err', err)
    return df

def calc_error_epsilon_Nd(df):
    """Calculate error for εNd"""
    if '143Nd/144Nd_err' in df.columns:
        err = (1e4 / CHUR_143ND_144ND) * df['143Nd/144Nd_err']
        return safe_assign(df, 'εNd_err', err)
    return df

def calc_error_epsilon_Hf(df):
    """Calculate error for εHf"""
    if '176Hf/177Hf_err' in df.columns:
        err = (1e4 / CHUR_176HF_177HF) * df['176Hf/177Hf_err']
        return safe_assign(df, 'εHf_err', err)
    return df

def calc_error_epsilon_Nd_i(df, age):
    """Calculate error for εNd(i)"""
    if '143Nd/144Nd(i)_err' in df.columns:
        decay_147 = get_decay(LAMBDA_147SM, age)
        denom = CHUR_143ND_144ND - CHUR_147SM_144ND * decay_147
        err = (1e4 / denom) * df['143Nd/144Nd(i)_err']
        return safe_assign(df, 'εNd(i)_err', err)
    return df

def calc_error_epsilon_Hf_i(df, age):
    """Calculate error for εHf(i)"""
    if '176Hf/177Hf(i)_err' in df.columns:
        decay_176 = get_decay(LAMBDA_176LU, age)
        denom = CHUR_176HF_177HF - CHUR_176LU_177HF * decay_176
        err = (1e4 / denom) * df['176Hf/177Hf(i)_err']
        return safe_assign(df, 'εHf(i)_err', err)
    return df

def calc_error_delta_epsilon_Hf(df):
    """Calculate error for ΔεHf"""
    if 'εHf_err' in df.columns and 'εNd_err' in df.columns:
        err = np.sqrt(df['εHf_err']**2 + (HFND_ARRAY_A**2) * df['εNd_err']**2)
        return safe_assign(df, 'ΔεHf_err', err)
    return df

def calc_error_delta_epsilon_Hf_i(df):
    """Calculate error for ΔεHf(i)"""
    if 'εHf(i)_err' in df.columns and 'εNd(i)_err' in df.columns:
        err = np.sqrt(df['εHf(i)_err']**2 + (HFND_ARRAY_A**2) * df['εNd(i)_err']**2)
        return safe_assign(df, 'ΔεHf(i)_err', err)
    return df

def calc_error_inverse_Hf(df):
    """Calculate error for 1/Hf"""
    if 'Hf_err' in df.columns and 'Hf' in df.columns:
        err = df['Hf_err'] / (df['Hf']**2)
        return safe_assign(df, '1/Hf_err', err)
    return df

def calc_relative_error(df, missing_error):
    """Calculate relative errors for missing error columns"""
    for col in ['147Sm/144Nd', '176Lu/177Hf']:
        err_col = col + '_err'
        if col in df.columns and err_col not in df.columns:
            df[err_col] = missing_error * df[col]
    return df

def calc_all(df, age=None, missing_error=None):
    """
    Calculate all geochemical parameters automatically.
    
    Parameters:
    - df: DataFrame with geochemical data
    - age: Age in Ma for initial ratio calculations
    - missing_error: Relative error for missing error columns
    
    Returns:
    - DataFrame with all calculated parameters
    """
    df = df.copy()
    
    # Discover all calc_ functions in this module
    current_module = sys.modules[__name__]
    calc_funcs = [
        func for name, func in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("calc_") and name != "calc_all"
    ]
    
    for func in calc_funcs:
        try:
            params = inspect.signature(func).parameters
            if 'df' not in params:
                continue  # skip any malformed function
            kwargs = {}
            if 'age' in params and age is not None:
                kwargs['age'] = age
            if 'missing_error' in params and missing_error is not None:
                kwargs['missing_error'] = missing_error
            df = func(df, **kwargs)
        except Exception as e:
            # Continue if calculation fails for any reason
            continue
    
    return df