"""
Unit and Integration Test Suite for AgroVision AI
Author: Mehul Shankar P (AgroVision Project)

Covers:
1. Module imports and integrity.
2. Data preprocessing and filtering.
3. Zero-leakage verification (production exclusion).
4. Chronological splitting integrity.
5. End-to-end yield prediction and input validation.
6. Error handling for invalid parameters.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import load_dataset, clean_agricultural_data, audit_data_quality
from src.feature_engineering import (
    identify_model_features,
    chronological_split,
    build_preprocessor,
    LEAKAGE_COLUMNS
)
from src.evaluate import calculate_metrics, explain_metric
from src.predict import AgroYieldPredictor, predict_yield


def test_imports():
    """Verify core scientific and ML packages import without error."""
    import sklearn
    import xgboost
    import streamlit
    import plotly
    assert sklearn is not None
    assert xgboost is not None
    assert streamlit is not None
    assert plotly is not None


def test_load_dataset():
    """Test data loader handles existing and non-existing paths correctly."""
    valid_path = os.path.join("data", "processed", "des_cleaned_all_crops.csv")
    df = load_dataset(valid_path)
    assert not df.empty
    assert 'year' in df.columns
    assert 'district' in df.columns
    assert 'yield' in df.columns

    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_directory/file.csv")


def test_data_leakage_safeguard():
    """Ensure production and target derivatives are strictly omitted from features."""
    sample_df = pd.DataFrame({
        'district': ['thanjavur', 'madurai'],
        'year': [2015, 2016],
        'crop': ['total_group', 'total_group'],
        'season': ['whole_year', 'whole_year'],
        'area': [1000.0, 2000.0],
        'production': [3000.0, 6000.0],
        'yield': [3000.0, 3000.0],
        'reported_yield_tonne_ha': [3.0, 3.0]
    })
    
    cat_feats, num_feats = identify_model_features(sample_df)
    all_selected = cat_feats + num_feats
    
    # Assert none of the leakage columns are included
    for leakage_col in LEAKAGE_COLUMNS:
        assert leakage_col not in all_selected, f"LEAKAGE DETECTED: {leakage_col} was selected as a feature!"


def test_chronological_split():
    """Verify temporal split integrity: no future years in training set."""
    df_path = os.path.join("data", "processed", "des_cleaned_all_crops.csv")
    raw_df = load_dataset(df_path)
    clean_df, _ = clean_agricultural_data(raw_df)
    
    train_df, val_df, test_df = chronological_split(clean_df, train_end_year=2016, val_end_year=2019)
    
    assert train_df['year'].max() <= 2016
    assert val_df['year'].min() >= 2017
    assert val_df['year'].max() <= 2019
    assert test_df['year'].min() >= 2020
    
    # Assert no overlap
    train_years = set(train_df['year'].unique())
    val_years = set(val_df['year'].unique())
    test_years = set(test_df['year'].unique())
    assert len(train_years.intersection(val_years)) == 0
    assert len(val_years.intersection(test_years)) == 0


def test_metrics_calculation():
    """Verify regression metrics calculate correctly."""
    y_true = np.array([1000.0, 2000.0, 3000.0])
    y_pred = np.array([1100.0, 1900.0, 3100.0])
    
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics['mae_kg_ha'] == 100.0
    assert metrics['rmse_kg_ha'] == 100.0
    assert metrics['r2_score'] > 0.95


def test_predict_yield_valid():
    """Verify prediction function executes and returns positive realistic yield."""
    res = predict_yield(district="thanjavur", year=2024, area=50000.0)
    assert res['predicted_yield_kg_ha'] > 0
    assert res['estimated_production_tonnes'] > 0
    assert res['district'] == 'thanjavur'
    assert 'explanation' in res


def test_predict_yield_invalid_district():
    """Verify invalid district raises clear ValueError."""
    with pytest.raises(ValueError, match="not recognized"):
        predict_yield(district="invalid_district_name", year=2024, area=50000.0)


def test_predict_yield_invalid_area():
    """Verify negative or zero area raises clear ValueError."""
    with pytest.raises(ValueError, match="greater than 0"):
        predict_yield(district="thanjavur", year=2024, area=-500.0)
        
    with pytest.raises(ValueError, match="greater than 0"):
        predict_yield(district="thanjavur", year=2024, area=0.0)
