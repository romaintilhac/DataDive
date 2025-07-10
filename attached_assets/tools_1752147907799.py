import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re

# for multiple-line assignments, a trailing space is required
meta_order = "Sample Lithology Unit Zone Distance Lat(N) Long(E) dup# run#".split()
ME_order = "SiO2 TiO2 Al2O3 Cr2O3 Fe2O3 Fe2O3T FeO FeOT NiO MnO MgO CaO Na2O K2O P2O5 LOI Total".split()
ME_ratio_order =  "Mg#".split()
TE_order = (
    "La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "F Cl H Ag As Au B Ba Be Bi Br C Ca Cd Co Cr Cs Cu Ga Ge Hf In "
    "Ir K Li Mn Mo N Na Nb Ni Os P Pb Pd Pt Rb Re Rh Ru Sb Sc Se Sn Sr Ta Te Th Ti Tl U V W Y Zn Zr"
    ).split()
iso_order = (
    "87Rb/86Sr 87Sr/86Sr 87Sr/86Sr(i) "
    "147Sm/144Nd 143Nd/144Nd 143Nd/144Nd(i) εNd εNd(i) "
    "176Lu/177Hf 176Hf/177Hf 176Hf/177Hf(i) εHf εHf(i) ΔεHf ΔεHf(i) "
    "208Pb/204Pb 207Pb/204Pb 206Pb/204Pb"
    ).split()

import re

manual_labels = {
    'Distance': r"Distance from Moho [m]",
    "εNd": r"$\mathrm{\varepsilon}_{\mathrm{Nd}}$",
    "εHf": r"$\mathrm{\varepsilon}_{\mathrm{Hf}}$",
    "εNd(i)": r"$\mathrm{\varepsilon}_{\mathrm{Nd}}(i)$",
    "εHf(i)": r"$\mathrm{\varepsilon}_{\mathrm{Hf}}(i)$",
    "ΔεHf": r"$\Delta \mathrm{\varepsilon}_{\mathrm{Hf}}$",
    "ΔεHf(i)": r"$\Delta \mathrm{\varepsilon}_{\mathrm{Hf}}(i)$",
}

def oxide_to_subscript(formula: str) -> str:
    # Mapping digits to unicode subscripts
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return formula.translate(subscript_map)

def format_subscripts(label):
    # Insert LaTeX-style subscripts for numbers
    def repl(match):
        elem = match.group(1)
        num = match.group(2)
        return f"{elem}$_{{{num}}}$"
    return re.sub(r'([A-Za-z]+)(\d+)', repl, label)

def plot_labels(label):
    # Manual labels
    if label in manual_labels:
        return manual_labels[label]
    
    # Trace elements: just [label] + " [ppm]"
    if label in TE_order:
        return f"{label} [μg/g]"
    
    # Major elements: format subscripts + add " [wt. %]"
    if label in ME_order:
        return format_subscripts(label) + " [wt. %]"
    
    # Otherwise just format subscripts if any
    return format_subscripts(label)

def plot_columns(df, x_col, y_col, xlabel=None, ylabel=None, title=None):
    plt.figure(figsize=(8, 5))
    plt.scatter(df[x_col], df[y_col], alpha=0.7)
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.title(title if title else f'{y_col} vs {x_col}')
    plt.grid(True)
    plt.show()

def reorder_columns(df):

    base_order = meta_order + ME_order + ME_ratio_order + TE_order + iso_order
    suffixes = ['_err', '_meta', '_conflict']

    df_cols_set = set(df.columns)

    ordered_cols = []

    for col in base_order:
        if col in df_cols_set:
            ordered_cols.append(col)
            # Add known suffixes immediately after base col
            for suf in suffixes:
                suff_col = col + suf
                if suff_col in df_cols_set:
                    ordered_cols.append(suff_col)
            # Add *any other* suffix columns starting with '_' but not in known suffixes
            other_suf_cols = [c for c in df_cols_set if c.startswith(col + '_') and all(not c.endswith(suf) for suf in suffixes)]
            # Sort these other suffix cols alphabetically for consistency
            ordered_cols.extend(sorted(other_suf_cols))

    # Add remaining columns not already ordered
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    ordered_cols.extend(remaining_cols)

    return df[ordered_cols]

def merge_catalogue(
    catalogue_df,
    data_df,
    catalogue_columns_to_add=None,
    data_columns_to_add=None,
    key='Sample',
    suffix='_meta'
):

    # Make copies to avoid modifying originals
    data_df = data_df.copy()
    catalogue_df = catalogue_df.copy()

    # Clean key columns for merge
    data_df.loc[:, key] = data_df[key].astype(str).str.strip()
    catalogue_df.loc[:, key] = catalogue_df[key].astype(str).str.strip()

    # If no catalogue columns specified, add all except key
    if catalogue_columns_to_add is None:
        catalogue_columns_to_add = [col for col in catalogue_df.columns if col != key]

    # If no data columns specified, keep all columns in data_df
    if data_columns_to_add is None:
        data_columns_to_add = list(data_df.columns)
    else:
        # Make sure key is included
        if key not in data_columns_to_add:
            data_columns_to_add = [key] + data_columns_to_add

    # Subset DataFrames accordingly
    data_sub = data_df[data_columns_to_add]
    catalogue_sub = catalogue_df[[key] + catalogue_columns_to_add]

    # Merge with suffixes for overlapping columns
    merged_df = data_sub.merge(
        catalogue_sub,
        on=key,
        how='left',
        suffixes=('', suffix)
    )

    # Reconcile overlapping columns
    original_cols = list(data_sub.columns)
    all_cols = set(merged_df.columns)

    for col in original_cols:
        suffixed_col = col + suffix
        if suffixed_col in all_cols:
            base = merged_df[col]
            meta = merged_df[suffixed_col]

            conflict_mask = (~base.isna()) & (~meta.isna()) & (base != meta)
            if conflict_mask.any():
                merged_df[col + '_conflict'] = conflict_mask
            else:
                merged_df[col] = base.combine_first(meta)
                merged_df.drop(columns=[suffixed_col], inplace=True)

    return merged_df

def remove_duplicate_rows(df):
    """
    Remove duplicate rows from a DataFrame and print the indices of removed duplicates.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Cleaned DataFrame without duplicate rows.
    """
    import pandas as pd  # in case not already imported

    duplicate_mask = df.duplicated()
    duplicate_indices = df.index[duplicate_mask].tolist()

    if duplicate_indices:
        print(f"Removed {len(duplicate_indices)} duplicate rows at indices: {duplicate_indices}")
    else:
        print("No duplicate rows found.")

    return df.drop_duplicates()

def prioritize_rows(df, group_cols, priority_col, preferred_values):
    """
    Prioritize rows within each group based on preferred values in a specific column.

    Parameters:
    - df: pandas DataFrame
    - group_cols: list of column names to group by
    - priority_col: column name where priority values are checked
    - preferred_values: list (or set) of preferred values (in priority order)

    Returns:
    - Filtered DataFrame keeping only rows with preferred values if they exist, 
      else fallback to other values within the group.
    """
    preferred_values = list(preferred_values)  # ensure list for order

    def filter_group(group):
        for val in preferred_values:
            preferred_rows = group[group[priority_col] == val]
            if not preferred_rows.empty:
                return preferred_rows
        # fallback: if none of the preferred values are present, return all rows
        return group

    return df.groupby(group_cols, group_keys=False).apply(filter_group)

def compare_datafiles(file1, file2, key=None, sheet_name1=0, sheet_name2=0):
    """
    Compare two data files (CSV or Excel).

    Parameters:
    - file1, file2: str or Path, paths to the data files (csv, xls, xlsx)
    - key: str or list, optional column(s) to align rows by before comparison
    - sheet_name1: int or str, sheet for file1 if Excel (default first sheet)
    - sheet_name2: int or str, sheet for file2 if Excel (default first sheet)

    Prints differences or confirms if identical.
    """

    def load_file(filepath, sheet_name):
        filepath = Path(filepath)
        ext = filepath.suffix.lower()
        if ext == '.csv':
            return pd.read_csv(filepath)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    df1 = load_file(file1, sheet_name1)
    df2 = load_file(file2, sheet_name2)

    common_cols = df1.columns.intersection(df2.columns)
    if len(common_cols) == 0:
        print("No common columns found to compare.")
        return

    df1 = df1[common_cols].copy()
    df2 = df2[common_cols].copy()

    if key:
        df1 = df1.sort_values(by=key).reset_index(drop=True)
        df2 = df2.sort_values(by=key).reset_index(drop=True)
    else:
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

    if df1.shape != df2.shape:
        print(f"Warning: shape mismatch between files: {df1.shape} vs {df2.shape}")

    try:
        diff = df1.compare(df2)
        if diff.empty:
            print("Files are identical (in shared columns and compared rows).")
        else:
            print("Differences found:")
            print(diff)
    except Exception as e:
        print(f"Error comparing dataframes: {e}")

