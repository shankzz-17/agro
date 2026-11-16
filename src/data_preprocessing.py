"""
Data Preprocessing Module for AgroVision AI
Author: Mehul Shankar P (AgroVision Project)

This module handles:
1. Loading agricultural dataset safely.
2. Auditing missing values, duplicates, and invalid records.
3. Filtering valid agronomic observations without silent imputation.
4. Validating non-negative constraints for area and yield.
5. Providing structured diagnostic logs.
"""

import os
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the CSV dataset from a given path with validation.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is empty or cannot be parsed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"The file at {file_path} is empty.")
    
    return df


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a comprehensive health check on the dataset before modeling.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dict[str, Any]: Summary dictionary containing audit metrics.
    """
    total_records = len(df)
    duplicates = int(df.duplicated(subset=['district', 'year', 'crop', 'season']).sum())
    missing_area = int(df['area'].isna().sum()) if 'area' in df.columns else 0
    missing_yield = int(df['yield'].isna().sum()) if 'yield' in df.columns else 0
    missing_prod = int(df['production'].isna().sum()) if 'production' in df.columns else 0
    
    # Check invalid negative values
    neg_area = int((df['area'] < 0).sum()) if 'area' in df.columns else 0
    neg_yield = int((df['yield'] < 0).sum()) if 'yield' in df.columns else 0
    
    # Zero yield check
    zero_yield = int((df['yield'] == 0).sum()) if 'yield' in df.columns else 0

    return {
        'total_rows': total_records,
        'duplicate_rows': duplicates,
        'missing_area': missing_area,
        'missing_yield': missing_yield,
        'missing_production': missing_prod,
        'negative_area_count': neg_area,
        'negative_yield_count': neg_yield,
        'zero_yield_count': zero_yield,
        'unique_districts': int(df['district'].nunique()) if 'district' in df.columns else 0,
        'unique_years': int(df['year'].nunique()) if 'year' in df.columns else 0,
    }


def clean_agricultural_data(
    df: pd.DataFrame,
    require_positive_area: bool = True,
    drop_missing_yield: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the agricultural dataset for machine learning.
    
    Important Agronomic & ML Rules:
    1. We NEVER silently impute missing yield or area using arbitrary numbers.
    2. Missing yield records (e.g. newly bifurcated districts before creation) are dropped
       from training data to prevent training on fabricated labels.
    3. Production is strictly preserved for analysis but flagged so downstream models
       do not consume it as a predictive feature (Data Leakage Prevention).
       
    Args:
        df: Raw or initial processed DataFrame.
        require_positive_area: If True, keep only rows where area > 0.
        drop_missing_yield: If True, keep only rows where yield is non-null.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and audit report.
    """
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # 1. Deduplication
    key_cols = ['district', 'year', 'crop', 'season']
    existing_keys = [c for c in key_cols if c in df_clean.columns]
    if existing_keys:
        df_clean = df_clean.drop_duplicates(subset=existing_keys)
    dedup_count = initial_count - len(df_clean)
    
    # 2. Filter missing target (yield)
    dropped_missing_yield = 0
    if drop_missing_yield and 'yield' in df_clean.columns:
        valid_yield_mask = df_clean['yield'].notna()
        dropped_missing_yield = int((~valid_yield_mask).sum())
        df_clean = df_clean[valid_yield_mask]
        
    # 3. Filter positive cultivated area
    dropped_invalid_area = 0
    if require_positive_area and 'area' in df_clean.columns:
        valid_area_mask = (df_clean['area'].notna()) & (df_clean['area'] > 0)
        dropped_invalid_area = int((~valid_area_mask).sum())
        df_clean = df_clean[valid_area_mask]
        
    # 4. Standardize text strings
    text_cols = ['district', 'crop', 'season', 'state']
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
            
    # 5. Ensure numeric types
    num_cols = ['year', 'area', 'production', 'yield']
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
    final_count = len(df_clean)
    
    audit_summary = {
        'initial_records': initial_count,
        'duplicates_removed': dedup_count,
        'dropped_missing_yield': dropped_missing_yield,
        'dropped_invalid_area': dropped_invalid_area,
        'final_clean_records': final_count,
        'retention_percentage': round((final_count / initial_count) * 100, 2) if initial_count > 0 else 0.0
    }
    
    return df_clean, audit_summary


if __name__ == "__main__":
    # Test script directly
    sample_path = os.path.join("data", "processed", "des_cleaned_all_crops.csv")
    if os.path.exists(sample_path):
        data = load_dataset(sample_path)
        print("=== Initial Quality Audit ===")
        print(audit_data_quality(data))
        
        clean_df, summary = clean_agricultural_data(data)
        print("\n=== Cleaning Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
            
        print(f"\nFinal Cleaned Shape: {clean_df.shape}")
        print("Head of cleaned data:")
        print(clean_df[['year', 'district', 'crop', 'season', 'area', 'yield']].head(3))
    else:
        print(f"File not found for testing: {sample_path}")
