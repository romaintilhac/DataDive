import numpy as np
import pandas as pd
import inspect
import sys
from utils.constants import *


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
    if age is None:
        raise ValueError("No age provided")
    return np.exp(age * const_lambda * 1e6) - 1

# --- Individual calc_ functions ---

## TE ratios
def calc_LaSm(df):
    return safe_assign(df, 'La/Sm', df['La']/df['Sm'])
def calc_GdL(df):
    return safe_assign(df, 'Gd/Lu', df['Gd']/df['Lu'])
def calc_SmNd(df):
    return safe_assign(df, 'Sm/Nd', df['Sm']/df['Nd'])
def calc_LuHf(df):
    return safe_assign(df, 'Lu/Hf', df['Lu']/df['Hf'])
def calc_ZrHf(df):
    return safe_assign(df, 'Zr/Hf', df['Zr']/df['Hf'])
def calc_NbTa(df):
    return safe_assign(df, 'Nb/Ta', df['Nb']/df['Ta'])

## ME ratios
def calc_Mg_number(df):
    return safe_assign(df, 'Mg#', 100 * (df['MgO'] / MgO_mass) / ((df['FeO'] / FeO_mass) + (df['MgO'] / MgO_mass)))

## Isotopes
def calc_143Nd_144Nd_i(df, age):
    decay_147 = get_decay(lambda_147Sm, age)
    return safe_assign(df, '143Nd/144Nd(i)', df['143Nd/144Nd'] - df['147Sm/144Nd'] * decay_147)

def calc_176Hf_177Hf_i(df, age):
    decay_176 = get_decay(lambda_176Lu, age)
    return safe_assign(df, '176Hf/177Hf(i)', df['176Hf/177Hf'] - df['176Lu/177Hf'] * decay_176)

def calc_epsilon_Nd(df):
    return safe_assign(df, 'εNd', ((df['143Nd/144Nd'] / CHUR_143Nd144Nd) - 1) * 1e4)

def calc_epsilon_Hf(df):
    return safe_assign(df, 'εHf', ((df['176Hf/177Hf'] / CHUR_176Hf177Hf) - 1) * 1e4)

def calc_epsilon_Nd_i(df, age):
    return safe_assign(df, 'εNd(i)', ((df['143Nd/144Nd(i)'] / CHUR_143Nd144Nd) - 1) * 1e4)

def calc_epsilon_Hf_i(df, age):
    return safe_assign(df, 'εHf(i)', ((df['176Hf/177Hf(i)'] / CHUR_176Hf177Hf) - 1) * 1e4)

def calc_delta_epsilon_Hf(df):
    return safe_assign(df, 'ΔεHf', df['εHf'] - (HfNd_array_a * df['εNd'] + HfNd_array_b))

def calc_delta_epsilon_Hf_i(df):
    return safe_assign(df, 'ΔεHf(i)', df['εHf(i)'] - (HfNd_array_a * df['εNd(i)'] + HfNd_array_b))

def calc_inverse_Hf(df):
    return safe_assign(df, '1/Hf', 1 / df['Hf'])

def calc_HfNd(df):
    return safe_assign(df, 'Hf/Nd',  df['Hf'] / df['Nd'])

def calc_error_143Nd_144Nd_i(df, age):
    decay_147 = get_decay(lambda_147Sm, age)
    err = np.sqrt(df['143Nd/144Nd_err']**2 + (decay_147 * df['147Sm/144Nd_err'])**2)
    return safe_assign(df, '143Nd/144Nd(i)_err', err)

def calc_error_176Hf_177Hf_i(df, age):
    decay_176 = get_decay(lambda_176Lu, age)
    err = np.sqrt(df['176Hf/177Hf_err']**2 + (decay_176 * df['176Lu/177Hf_err'])**2)
    return safe_assign(df, '176Hf/177Hf(i)_err', err)

def calc_error_epsilon_Nd(df):
    return safe_assign(df, 'εNd_err', (1e4 / CHUR_143Nd144Nd) * df['143Nd/144Nd_err'])

def calc_error_epsilon_Hf(df):
    return safe_assign(df, 'εHf_err', (1e4 / CHUR_176Hf177Hf) * df['176Hf/177Hf_err'])

def calc_error_epsilon_Nd_i(df, age):
    decay_147 = get_decay(lambda_147Sm, age)
    denom = CHUR_143Nd144Nd - CHUR_147Sm144Nd * decay_147
    err = (1e4 / denom) * df['143Nd/144Nd(i)_err']
    return safe_assign(df, 'εNd(i)_err', err)

def calc_error_epsilon_Hf_i(df, age):
    decay_176 = get_decay(lambda_176Lu, age)
    denom = CHUR_176Hf177Hf - CHUR_176Lu177Hf * decay_176
    err = (1e4 / denom) * df['176Hf/177Hf(i)_err']
    return safe_assign(df, 'εHf(i)_err', err)

def calc_error_delta_epsilon_Hf(df):
    err = np.sqrt(df['εHf_err']**2 + (HfNd_array_a**2) * df['εNd_err']**2)
    return safe_assign(df, 'ΔεHf_err', err)

def calc_error_delta_epsilon_Hf_i(df):
    err = np.sqrt(df['εHf(i)_err']**2 + (HfNd_array_a**2) * df['εNd(i)_err']**2)
    return safe_assign(df, 'ΔεHf(i)_err', err)

def calc_error_inverse_Hf(df):
    err = df['Hf_err'] / (df['Hf']**2)
    return safe_assign(df, '1/Hf_err', err)

def calc_abs_relative_error(df, missing_error):
    return safe_assign(df, '', abs(df[''] * missing_error)) # WARNING not currently used

def calc_relative_error(df, missing_error):
    for col in ['147Sm/144Nd', '176Lu/177Hf']:
        err_col = col + '_err'
        if col in df.columns and err_col not in df.columns:
            df[err_col] = missing_error * df[col]
    return df


def calc_all(df, age=None, missing_error=None):
    df = df.copy()

    # Discover all calc_ functions in this module
    current_module = sys.modules[__name__]
    calc_funcs = [
        func for name, func in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("calc_")
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
        except Exception:
            pass

    return df
