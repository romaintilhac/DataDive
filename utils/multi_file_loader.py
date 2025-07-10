"""
Multi-file loading and combination utilities for geochemical data.
Implements catalogue-based logic for combining multiple datasets.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import streamlit as st

class MultiFileLoader:
    """Handle loading and combining multiple geochemical data files"""
    
    def __init__(self):
        self.loaded_files = {}
        self.combined_data = None
        self.catalogue_data = None
        self.merge_log = []
        
    def load_file(self, file_path: str, file_type: str = 'auto', 
                  sheet_name: Union[str, int] = 0, 
                  file_label: str = None) -> pd.DataFrame:
        """
        Load a single file with automatic type detection
        
        Parameters:
        - file_path: Path to the file or uploaded file object
        - file_type: 'csv', 'excel', or 'auto' for automatic detection
        - sheet_name: Sheet name or index for Excel files
        - file_label: Custom label for the file
        
        Returns:
        - DataFrame with loaded data
        """
        try:
            # Handle Streamlit uploaded files
            if hasattr(file_path, 'name'):
                file_name = file_path.name
                file_content = file_path
            else:
                file_name = str(file_path)
                file_content = file_path
            
            # Auto-detect file type
            if file_type == 'auto':
                if file_name.endswith('.csv'):
                    file_type = 'csv'
                elif file_name.endswith(('.xlsx', '.xls')):
                    file_type = 'excel'
                else:
                    raise ValueError(f"Unsupported file type: {file_name}")
            
            # Load the file
            if file_type == 'csv':
                df = pd.read_csv(file_content)
            elif file_type == 'excel':
                df = pd.read_excel(file_content, sheet_name=sheet_name)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Store the loaded file
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
            raise Exception(f"Error loading file {file_name}: {str(e)}")
    
    def identify_catalogue_file(self, files_dict: Dict[str, pd.DataFrame]) -> Optional[str]:
        """
        Identify which file is the catalogue based on common patterns
        
        Parameters:
        - files_dict: Dictionary of file_label -> DataFrame
        
        Returns:
        - Key of the catalogue file or None
        """
        catalogue_indicators = ['catalogue', 'catalog', 'sample_info', 'metadata', 'samples']
        
        # Check file names first
        for label, df in files_dict.items():
            label_lower = label.lower()
            if any(indicator in label_lower for indicator in catalogue_indicators):
                return label
        
        # Check for files with many metadata columns but fewer rows
        potential_catalogues = []
        for label, df in files_dict.items():
            # Look for common catalogue columns
            catalogue_cols = ['sample', 'location', 'latitude', 'longitude', 'lithology', 
                            'age', 'formation', 'unit', 'description', 'collector']
            
            matching_cols = sum(1 for col in df.columns if 
                              any(cat_col.lower() in col.lower() for cat_col in catalogue_cols))
            
            if matching_cols >= 3:  # Has at least 3 catalogue-type columns
                potential_catalogues.append((label, matching_cols, len(df)))
        
        if potential_catalogues:
            # Choose the one with most catalogue columns
            potential_catalogues.sort(key=lambda x: x[1], reverse=True)
            return potential_catalogues[0][0]
        
        return None
    
    def combine_files_by_sample(self, files_dict: Dict[str, pd.DataFrame], 
                               sample_column: str = 'Sample',
                               catalogue_file: Optional[str] = None,
                               priority_order: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Combine multiple files based on sample matching logic
        
        Parameters:
        - files_dict: Dictionary of file_label -> DataFrame
        - sample_column: Column name used for matching samples
        - catalogue_file: Key of the catalogue file (auto-detected if None)
        - priority_order: Order of file priority for conflict resolution
        
        Returns:
        - Combined DataFrame
        """
        if not files_dict:
            raise ValueError("No files to combine")
        
        # Auto-detect catalogue file if not specified
        if catalogue_file is None:
            catalogue_file = self.identify_catalogue_file(files_dict)
        
        # Start with catalogue file if available, otherwise use first file
        if catalogue_file and catalogue_file in files_dict:
            base_df = files_dict[catalogue_file].copy()
            self.catalogue_data = base_df.copy()
            remaining_files = {k: v for k, v in files_dict.items() if k != catalogue_file}
            merge_order = [catalogue_file] + list(remaining_files.keys())
        else:
            # Use priority order or default to first file
            file_keys = list(files_dict.keys())
            if priority_order:
                # Reorder based on priority
                ordered_keys = [key for key in priority_order if key in file_keys]
                ordered_keys.extend([key for key in file_keys if key not in ordered_keys])
                file_keys = ordered_keys
            
            base_df = files_dict[file_keys[0]].copy()
            remaining_files = {k: files_dict[k] for k in file_keys[1:]}
            merge_order = file_keys
        
        # Ensure sample column exists
        if sample_column not in base_df.columns:
            raise ValueError(f"Sample column '{sample_column}' not found in base file")
        
        # Initialize merge log
        self.merge_log = []
        self.merge_log.append(f"Starting with {merge_order[0]}: {len(base_df)} samples")
        
        # Merge each remaining file
        for file_label, df in remaining_files.items():
            if sample_column not in df.columns:
                self.merge_log.append(f"Skipping {file_label}: no '{sample_column}' column")
                continue
            
            # Prepare data for merge
            df_merge = df.copy()
            
            # Identify overlapping columns (excluding sample column)
            overlapping_cols = [col for col in df_merge.columns 
                              if col in base_df.columns and col != sample_column]
            
            # Add suffix to overlapping columns to track sources
            if overlapping_cols:
                suffix_map = {}
                for col in overlapping_cols:
                    new_col = f"{col}_{file_label}"
                    suffix_map[col] = new_col
                    df_merge = df_merge.rename(columns={col: new_col})
                
                self.merge_log.append(f"Renamed {len(overlapping_cols)} overlapping columns in {file_label}")
            
            # Perform merge
            before_merge = len(base_df)
            base_df = base_df.merge(df_merge, on=sample_column, how='left', suffixes=('', f'_{file_label}'))
            after_merge = len(base_df)
            
            # Check for new samples in the merging file
            new_samples = set(df_merge[sample_column]) - set(base_df[sample_column])
            if new_samples:
                # Add new samples
                new_sample_df = df_merge[df_merge[sample_column].isin(new_samples)]
                base_df = pd.concat([base_df, new_sample_df], ignore_index=True, sort=False)
                self.merge_log.append(f"Added {len(new_samples)} new samples from {file_label}")
            
            matched_samples = len(df_merge[df_merge[sample_column].isin(base_df[sample_column])])
            self.merge_log.append(f"Merged {file_label}: {matched_samples} samples matched")
        
        # Clean up the combined data
        combined_df = self._clean_combined_data(base_df, sample_column)
        
        self.combined_data = combined_df
        self.merge_log.append(f"Final combined dataset: {len(combined_df)} samples, {len(combined_df.columns)} columns")
        
        return combined_df
    
    def _clean_combined_data(self, df: pd.DataFrame, sample_column: str) -> pd.DataFrame:
        """
        Clean and organize the combined dataset
        
        Parameters:
        - df: Combined DataFrame
        - sample_column: Sample column name
        
        Returns:
        - Cleaned DataFrame
        """
        # Remove completely empty rows
        df = df.dropna(how='all', subset=[col for col in df.columns if col != sample_column])
        
        # Reorder columns: Sample first, then metadata, then analytical data
        cols = list(df.columns)
        
        # Priority column order
        priority_cols = [sample_column]
        metadata_cols = []
        analytical_cols = []
        
        for col in cols:
            if col == sample_column:
                continue
            elif any(meta in col.lower() for meta in ['location', 'latitude', 'longitude', 'lithology', 
                                                     'age', 'formation', 'unit', 'description', 'collector',
                                                     'date', 'method', 'reference']):
                metadata_cols.append(col)
            else:
                analytical_cols.append(col)
        
        # Reorder columns
        new_order = priority_cols + sorted(metadata_cols) + sorted(analytical_cols)
        df = df[new_order]
        
        return df
    
    def resolve_conflicts(self, df: pd.DataFrame, 
                         conflict_resolution: str = 'keep_first',
                         priority_files: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Resolve conflicts in overlapping columns
        
        Parameters:
        - df: DataFrame with potential conflicts
        - conflict_resolution: 'keep_first', 'keep_last', 'average', 'priority'
        - priority_files: File priority order for conflict resolution
        
        Returns:
        - DataFrame with resolved conflicts
        """
        # Find columns with suffixes indicating conflicts
        conflict_groups = {}
        
        for col in df.columns:
            if '_' in col:
                base_name = col.split('_')[0]
                if base_name in df.columns:
                    # This is a conflict column
                    if base_name not in conflict_groups:
                        conflict_groups[base_name] = [base_name]
                    conflict_groups[base_name].append(col)
        
        if not conflict_groups:
            return df
        
        # Resolve conflicts
        for base_col, conflict_cols in conflict_groups.items():
            if len(conflict_cols) <= 1:
                continue
            
            if conflict_resolution == 'keep_first':
                # Keep original column, drop others
                df = df.drop(columns=conflict_cols[1:])
            elif conflict_resolution == 'keep_last':
                # Keep last column, rename it, drop others
                df = df.rename(columns={conflict_cols[-1]: base_col})
                df = df.drop(columns=conflict_cols[:-1])
            elif conflict_resolution == 'average':
                # Average numeric columns
                numeric_cols = [col for col in conflict_cols if pd.api.types.is_numeric_dtype(df[col])]
                if numeric_cols:
                    df[base_col] = df[numeric_cols].mean(axis=1)
                    df = df.drop(columns=conflict_cols[1:])
            
        return df
    
    def get_merge_summary(self) -> Dict:
        """
        Get summary of the merge operation
        
        Returns:
        - Dictionary with merge statistics
        """
        summary = {
            'files_loaded': len(self.loaded_files),
            'file_details': self.loaded_files,
            'merge_log': self.merge_log,
            'combined_shape': self.combined_data.shape if self.combined_data is not None else None,
            'catalogue_file': self.catalogue_data is not None
        }
        
        return summary
    
    def validate_sample_consistency(self, sample_column: str = 'Sample') -> Dict:
        """
        Validate sample naming consistency across files
        
        Parameters:
        - sample_column: Column name for samples
        
        Returns:
        - Dictionary with validation results
        """
        if not self.loaded_files:
            return {'error': 'No files loaded'}
        
        all_samples = {}
        sample_stats = {}
        
        for file_label, file_info in self.loaded_files.items():
            df = file_info['data']
            if sample_column in df.columns:
                samples = set(df[sample_column].dropna().astype(str))
                all_samples[file_label] = samples
                sample_stats[file_label] = {
                    'count': len(samples),
                    'unique': len(samples),
                    'duplicates': len(df[sample_column]) - len(samples)
                }
        
        # Find common samples
        if all_samples:
            common_samples = set.intersection(*all_samples.values()) if len(all_samples) > 1 else set()
            all_unique_samples = set.union(*all_samples.values())
            
            validation_results = {
                'total_files': len(all_samples),
                'common_samples': len(common_samples),
                'unique_samples': len(all_unique_samples),
                'file_stats': sample_stats,
                'overlap_matrix': self._calculate_overlap_matrix(all_samples)
            }
        else:
            validation_results = {'error': f'No files contain column {sample_column}'}
        
        return validation_results
    
    def _calculate_overlap_matrix(self, all_samples: Dict[str, set]) -> Dict:
        """Calculate overlap matrix between files"""
        files = list(all_samples.keys())
        overlap_matrix = {}
        
        for i, file1 in enumerate(files):
            overlap_matrix[file1] = {}
            for j, file2 in enumerate(files):
                if i <= j:
                    overlap = len(all_samples[file1].intersection(all_samples[file2]))
                    overlap_matrix[file1][file2] = overlap
                    if file2 not in overlap_matrix:
                        overlap_matrix[file2] = {}
                    overlap_matrix[file2][file1] = overlap
        
        return overlap_matrix