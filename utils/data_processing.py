import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import re

class DataProcessor:
    """Class for processing geochemical data files"""
    
    def __init__(self):
        self.required_columns = [
            'Sample', 'SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5'
        ]
        self.optional_columns = [
            'Lithology', 'Zone', 'Unit', 'Lat', 'Long', 'Distance',
            'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Ba', 'Cr', 'Cs', 'Hf', 'Nb', 'Ni', 'Pb', 'Rb', 'Sc', 'Sr', 'Ta', 'Th', 'U', 'V', 'Y', 'Zr',
            '87Sr/86Sr', '143Nd/144Nd', '176Hf/177Hf'
        ]
    
    def load_file(self, file, file_type: str) -> pd.DataFrame:
        """Load data from uploaded file"""
        try:
            if file_type == 'csv':
                df = pd.read_csv(file)
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(file)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate data structure and return issues"""
        issues = {
            'missing_required': [],
            'empty_columns': [],
            'invalid_values': [],
            'warnings': []
        }
        
        # Check required columns
        for col in self.required_columns:
            if col not in df.columns:
                issues['missing_required'].append(col)
        
        # Check for empty columns
        for col in df.columns:
            if df[col].isna().all():
                issues['empty_columns'].append(col)
        
        # Check for invalid values in numeric columns
        numeric_columns = [col for col in df.columns if col not in ['Sample', 'Lithology', 'Zone', 'Unit']]
        for col in numeric_columns:
            if col in df.columns:
                non_numeric = df[col].apply(lambda x: not pd.isna(x) and not isinstance(x, (int, float)))
                if non_numeric.any():
                    issues['invalid_values'].append(f"{col}: {non_numeric.sum()} non-numeric values")
        
        # Check for duplicate samples
        if 'Sample' in df.columns:
            duplicates = df['Sample'].duplicated().sum()
            if duplicates > 0:
                issues['warnings'].append(f"Found {duplicates} duplicate samples")
        
        return issues
    
    def clean_data(self, df: pd.DataFrame, 
                   remove_duplicates: bool = True,
                   fill_missing: bool = True) -> pd.DataFrame:
        """Clean and preprocess data"""
        df_clean = df.copy()
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Remove completely empty columns
        df_clean = df_clean.dropna(axis=1, how='all')
        
        # Handle duplicates
        if remove_duplicates and 'Sample' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['Sample'], keep='first')
        
        # Convert numeric columns
        numeric_columns = [col for col in df_clean.columns if col not in ['Sample', 'Lithology', 'Zone', 'Unit']]
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Fill missing values if requested
        if fill_missing:
            # For numeric columns, fill with median
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
            # For categorical columns, fill with 'Unknown'
            categorical_columns = ['Lithology', 'Zone', 'Unit']
            for col in categorical_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna('Unknown')
        
        return df_clean
    
    def merge_datasets(self, main_df: pd.DataFrame, 
                      additional_dfs: List[pd.DataFrame],
                      merge_on: str = 'Sample') -> pd.DataFrame:
        """Enhanced merge with conflict detection and metadata preservation"""
        merged_df = main_df.copy()
        
        for i, df in enumerate(additional_dfs):
            if merge_on in df.columns:
                # Clean merge keys
                merged_df[merge_on] = merged_df[merge_on].astype(str).str.strip()
                df[merge_on] = df[merge_on].astype(str).str.strip()
                
                # Merge with metadata suffix
                suffix = f'_meta_{i}' if i > 0 else '_meta'
                temp_merged = merged_df.merge(df, on=merge_on, how='left', suffixes=('', suffix))
                
                # Handle overlapping columns with conflict detection
                original_cols = list(merged_df.columns)
                all_cols = set(temp_merged.columns)
                
                for col in original_cols:
                    suffixed_col = col + suffix
                    if suffixed_col in all_cols:
                        base = temp_merged[col]
                        meta = temp_merged[suffixed_col]
                        
                        # Check for conflicts
                        conflict_mask = (~base.isna()) & (~meta.isna()) & (base != meta)
                        if conflict_mask.any():
                            temp_merged[col + '_conflict'] = conflict_mask
                        else:
                            # Use combine_first to fill missing values
                            temp_merged[col] = base.combine_first(meta)
                            temp_merged.drop(columns=[suffixed_col], inplace=True)
                
                merged_df = temp_merged
        
        return merged_df
    
    def prioritize_rows(self, df: pd.DataFrame, 
                       group_cols: List[str], 
                       priority_col: str, 
                       preferred_values: List[str]) -> pd.DataFrame:
        """Prioritize rows based on values in priority column"""
        def priority_func(group):
            for preferred in preferred_values:
                matching = group[group[priority_col] == preferred]
                if not matching.empty:
                    return matching.iloc[0]
            return group.iloc[0]
        
        return df.groupby(group_cols).apply(priority_func).reset_index(drop=True)
    
    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns for better presentation with improved logic"""
        from utils.constants import META_ORDER, ME_ORDER, ME_RATIO_ORDER, TE_ORDER, ISO_ORDER
        
        base_order = META_ORDER + ME_ORDER + ME_RATIO_ORDER + TE_ORDER + ISO_ORDER
        suffixes = ['_err', '_meta', '_conflict', '_calc', '_N', '_PM']
        
        df_cols_set = set(df.columns)
        ordered_cols = []
        
        # Add base columns and their variants
        for col in base_order:
            if col in df_cols_set:
                ordered_cols.append(col)
                # Add known suffixes immediately after base col
                for suf in suffixes:
                    suff_col = col + suf
                    if suff_col in df_cols_set:
                        ordered_cols.append(suff_col)
                # Add any other suffix columns starting with '_'
                other_suf_cols = [c for c in df_cols_set 
                                if c.startswith(col + '_') and 
                                all(not c.endswith(suf) for suf in suffixes)]
                ordered_cols.extend(sorted(other_suf_cols))
        
        # Add remaining columns not already ordered
        remaining_cols = [c for c in df.columns if c not in ordered_cols]
        ordered_cols.extend(remaining_cols)
        
        return df[ordered_cols]
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """Get summary statistics for the dataset"""
        summary = {
            'total_samples': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isna().sum().sum(),
            'duplicate_samples': df.duplicated().sum() if len(df) > 0 else 0,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }
        
        if 'Sample' in df.columns:
            summary['unique_samples'] = df['Sample'].nunique()
        
        if 'Lithology' in df.columns:
            summary['lithologies'] = df['Lithology'].nunique()
            summary['lithology_counts'] = df['Lithology'].value_counts().to_dict()
        
        if 'Zone' in df.columns:
            summary['zones'] = df['Zone'].nunique()
            summary['zone_counts'] = df['Zone'].value_counts().to_dict()
        
        return summary
    
    def export_data(self, df: pd.DataFrame, filename: str, format: str = 'csv') -> bytes:
        """Export data to specified format"""
        if format == 'csv':
            return df.to_csv(index=False).encode('utf-8')
        elif format == 'excel':
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")
