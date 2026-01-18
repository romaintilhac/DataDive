"""
Multi-file loading and combination utilities for geochemical data.
Implements catalogue-based logic for combining multiple datasets.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import streamlit as st

class MultiFileLoader:
    """Handle loading and combining multiple geochemical data files"""

    def __init__(self):
        self.loaded_files = {}
        self.combined_data = None
        self.catalogue_data = None
        self.merge_log = []

    def load_file(self, file_path: Union[str, Path, st.runtime.uploaded_file_manager.UploadedFile], 
                  file_type: str = 'auto', 
                  sheet_name: Union[str, int] = 0, 
                  file_label: Optional[str] = None) -> pd.DataFrame:
        """
        Load a single CSV or Excel file (XLS or XLSX).

        - Supports string paths and Streamlit uploaded files.
        - Automatically detects file type if 'auto'.
        - Note: .xls requires `xlrd<=1.2.0`.

        Returns:
            Loaded DataFrame.
        """
        try:
            # Handle Streamlit UploadedFile or path
            if hasattr(file_path, 'name'):
                file_name = file_path.name
                file_content = file_path
            else:
                file_name = str(file_path)
                file_content = file_path

            # Detect file extension
            if file_type == 'auto':
                suffix = Path(file_name).suffix.lower()
                if suffix == '.csv':
                    file_type = 'csv'
                elif suffix in ['.xls', '.xlsx']:
                    file_type = 'excel'
                else:
                    raise ValueError(f"Unsupported file type: {file_name}")

            # Load file
            if file_type == 'csv':
                df = pd.read_csv(file_content)
            elif file_type == 'excel':
                suffix = Path(file_name).suffix.lower()
                if suffix == '.xls':
                    df = pd.read_excel(file_content, sheet_name=sheet_name, engine='xlrd')
                else:
                    df = pd.read_excel(file_content, sheet_name=sheet_name, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            label = file_label or file_name
            self.loaded_files[label] = {
                'data': df,
                'file_name': file_name,
                'file_type': file_type,
                'sheet_name': sheet_name if file_type == 'excel' else None,
                'rows': len(df),
                'columns': len(df.columns)
            }

            return df

        except Exception as e:
            raise Exception(f"Error loading file {file_name}: {e}")

    def identify_catalogue_file(self, files_dict: Dict[str, pd.DataFrame]) -> Optional[str]:
        catalogue_indicators = ['catalogue', 'catalog', 'sample_info', 'metadata', 'samples']
        for label, df in files_dict.items():
            if any(indicator in label.lower() for indicator in catalogue_indicators):
                return label
        return None

    def combine_files_by_sample(self, files_dict: Dict[str, pd.DataFrame], 
                                sample_column: str = 'Sample',
                                catalogue_file: Optional[str] = None,
                                priority_order: Optional[List[str]] = None) -> pd.DataFrame:
        if not files_dict:
            raise ValueError("No files to combine.")

        if catalogue_file is None:
            catalogue_file = self.identify_catalogue_file(files_dict)

        if catalogue_file and catalogue_file in files_dict:
            base_df = files_dict[catalogue_file].copy()
            self.catalogue_data = base_df.copy()
            remaining = {k: v for k, v in files_dict.items() if k != catalogue_file}
        else:
            keys = list(files_dict.keys())
            if priority_order:
                ordered = [k for k in priority_order if k in keys]
                ordered += [k for k in keys if k not in ordered]
            else:
                ordered = keys
            base_df = files_dict[ordered[0]].copy()
            remaining = {k: files_dict[k] for k in ordered[1:]}

        for label, df in remaining.items():
            if sample_column not in df.columns:
                continue
            rename_cols = {col: f"{col}_{label}" for col in df.columns 
                           if col in base_df.columns and col != sample_column}
            df = df.rename(columns=rename_cols)
            base_df = base_df.merge(df, on=sample_column, how='outer')

        self.combined_data = base_df
        return base_df

    def get_merge_summary(self) -> Dict:
        return {
            'files_loaded': len(self.loaded_files),
            'combined_shape': self.combined_data.shape if self.combined_data is not None else None,
            'catalogue_loaded': self.catalogue_data is not None,
            'file_labels': list(self.loaded_files.keys())
        }

    def validate_sample_consistency(self, sample_column: str = 'Sample') -> Dict:
        if not self.loaded_files:
            return {'error': 'No files loaded.'}

        sample_sets = {
            label: set(info['data'][sample_column].dropna().astype(str))
            for label, info in self.loaded_files.items() if sample_column in info['data'].columns
        }

        if not sample_sets:
            return {'error': f"No '{sample_column}' column in any file."}

        all_files = list(sample_sets.keys())
        overlap_matrix = {
            f1: {f2: len(sample_sets[f1] & sample_sets[f2]) for f2 in all_files}
            for f1 in all_files
        }

        return {
            'sample_counts': {k: len(v) for k, v in sample_sets.items()},
            'common_samples': len(set.intersection(*sample_sets.values())) if len(sample_sets) > 1 else 0,
            'overlap_matrix': overlap_matrix
        }