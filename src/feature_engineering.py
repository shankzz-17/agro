"""
Feature Engineering & Transformation Pipeline for AgroVision AI
Author: Mehul Shankar P (AgroVision Project)

This module handles:
1. Feature selection and strict exclusion of leakage variables (e.g., production).
2. Chronological train/validation/test splitting based on agricultural year.
3. Building and fitting reproducible scikit-learn ColumnTransformers.
4. Serializing feature pipelines and feature names for seamless inference.
"""

import os
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


# Strict list of potential features
BASE_CATEGORICAL_FEATURES = ['district', 'crop', 'season']
BASE_NUMERIC_FEATURES = ['year', 'area']
POTENTIAL_RAINFALL_FEATURES = ['rainfall_mm', 'rainy_days', 'rainfall_anomaly']
TARGET_COLUMN = 'yield'

# Explicit leakage columns to NEVER use as features
LEAKAGE_COLUMNS = [
    'production',
    'production_unit',
    'reported_yield_tonne_ha',
    'yield_status',
    'yield_unit',
    'source_file',
    'original_year',
    'original_district',
    'original_crop',
    'original_season',
    'crop_filtering_note'
]


def identify_model_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Dynamically discover available valid features in the dataframe while 
    strictly rejecting leakage columns.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple[List[str], List[str]]: (categorical_features, numeric_features)
    """
    cat_features = [col for col in BASE_CATEGORICAL_FEATURES if col in df.columns]
    
    num_features = [col for col in BASE_NUMERIC_FEATURES if col in df.columns]
    for rain_col in POTENTIAL_RAINFALL_FEATURES:
        if rain_col in df.columns and rain_col not in num_features:
            num_features.append(rain_col)
            
    return cat_features, num_features


def chronological_split(
    df: pd.DataFrame,
    train_end_year: int = 2016,
    val_end_year: int = 2019
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a chronological temporal split to strictly avoid data leakage.
    
    Why chronological splitting instead of random train_test_split?
    In agronomy, climate variability, technological advancements (irrigation, fertilizers),
    and yield dynamics happen along a temporal timeline. Random splitting leaks future
    unseen years into past training models, creating artificial, overly-optimistic metrics.
    
    Default split (23 years: 2000 to 2022):
    - Train: 2000 - 2016 (17 years, ~74%)
    - Validation: 2017 - 2019 (3 years, ~13%)
    - Test: 2020 - 2022 (3 years, ~13%)
    
    Args:
        df: Input cleaned DataFrame.
        train_end_year: Inclusive final year for training data.
        val_end_year: Inclusive final year for validation data.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    if 'year' not in df.columns:
        raise KeyError("Column 'year' is required for chronological splitting.")
        
    train_df = df[df['year'] <= train_end_year].copy()
    val_df = df[(df['year'] > train_end_year) & (df['year'] <= val_end_year)].copy()
    test_df = df[df['year'] > val_end_year].copy()
    
    return train_df, val_df, test_df


def build_preprocessor(
    categorical_features: List[str],
    numeric_features: List[str]
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer that standardizes numeric features
    and One-Hot encodes categorical features.
    
    Args:
        categorical_features: List of categorical column names.
        numeric_features: List of numerical column names.
        
    Returns:
        ColumnTransformer: Configured preprocessor pipeline.
    """
    transformers = []
    
    if numeric_features:
        transformers.append((
            'num',
            StandardScaler(),
            numeric_features
        ))
        
    if categorical_features:
        transformers.append((
            'cat',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            categorical_features
        ))
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop everything else, guaranteed leakage prevention
    )
    
    return preprocessor


def extract_feature_names(
    preprocessor: ColumnTransformer,
    categorical_features: List[str],
    numeric_features: List[str]
) -> List[str]:
    """
    Extract human-readable post-transformation feature names from the fitted preprocessor.
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        categorical_features: List of categorical column names.
        numeric_features: List of numeric column names.
        
    Returns:
        List[str]: Formatted feature names.
    """
    feature_names = list(numeric_features)
    
    if categorical_features and 'cat' in preprocessor.named_transformers_:
        cat_encoder = preprocessor.named_transformers_['cat']
        encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(list(encoded_cat_names))
        
    return feature_names


def save_pipeline(
    preprocessor: ColumnTransformer,
    feature_meta: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save the fitted preprocessor and feature metadata to disk.
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        feature_meta: Dictionary containing feature names, categorical maps, and split years.
        output_path: File path to save the artifact.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        'preprocessor': preprocessor,
        'metadata': feature_meta
    }
    joblib.dump(payload, output_path)


def load_pipeline(pipeline_path: str) -> Tuple[ColumnTransformer, Dict[str, Any]]:
    """
    Load a serialized preprocessor pipeline and feature metadata.
    
    Args:
        pipeline_path: Path to the .joblib file.
        
    Returns:
        Tuple[ColumnTransformer, Dict[str, Any]]: (preprocessor, metadata)
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline artifact not found at: {pipeline_path}")
    payload = joblib.load(pipeline_path)
    return payload['preprocessor'], payload['metadata']
