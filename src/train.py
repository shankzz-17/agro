"""
Model Training and Comparison Pipeline for AgroVision AI
Author: Mehul Shankar P (AgroVision Project)

This module handles:
1. End-to-end training of agricultural yield estimation models:
   - Linear Regression (Baseline)
   - Random Forest Regressor
   - XGBoost Regressor
2. Strict chronological validation (2000-2016 Train, 2017-2019 Val, 2020-2022 Test).
3. Leakage-free preprocessing (transformer fitted ONLY on train set).
4. Model comparison table exported to reports/model_comparison.csv.
5. Serializing best model and preprocessor to models/.
"""

import os
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Optional XGBoost import with graceful fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.data_preprocessing import load_dataset, clean_agricultural_data
from src.feature_engineering import (
    identify_model_features,
    chronological_split,
    build_preprocessor,
    extract_feature_names,
    TARGET_COLUMN
)
from src.evaluate import calculate_metrics, save_model_comparison, explain_metric


def train_and_evaluate_models(
    dataset_path: str = "data/processed/des_cleaned_all_crops.csv",
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Execute the full training, hyperparameter fit, validation, and benchmarking pipeline.
    
    Args:
        dataset_path: Path to cleaned dataset.
        models_dir: Directory where models and preprocessors will be stored.
        reports_dir: Directory where evaluation reports and charts will be stored.
        
    Returns:
        Dict[str, Any]: Summary containing best model name, test metrics, and comparison table.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("=" * 60)
    print("🌾 AGROVISION AI: MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 60)
    
    # 1. Load and clean data
    print("\n[Step 1/5] Loading and preparing agricultural dataset...")
    raw_df = load_dataset(dataset_path)
    clean_df, clean_audit = clean_agricultural_data(raw_df)
    print(f"-> Verified {len(clean_df)} valid records across 38 districts (2000-2022).")
    
    # 2. Identify features & Chronological Split
    cat_features, num_features = identify_model_features(clean_df)
    print(f"\n[Step 2/5] Feature identification & Leakage Prevention:")
    print(f"-> Categorical Features: {cat_features}")
    print(f"-> Numerical Features:   {num_features}")
    print(f"-> Target Feature:       '{TARGET_COLUMN}' (kg/ha)")
    print(f"-> Strictly Excluded:    ['production', 'reported_yield_tonne_ha'] (Leakage prevention)")
    
    # Chronological Split (Train: 2000-2016, Val: 2017-2019, Test: 2020-2022)
    train_df, val_df, test_df = chronological_split(clean_df, train_end_year=2016, val_end_year=2019)
    print(f"\n-> Chronological Split:")
    print(f"   Train set:      {len(train_df):3d} records (Years {train_df['year'].min()} to {train_df['year'].max()})")
    print(f"   Validation set: {len(val_df):3d} records (Years {val_df['year'].min()} to {val_df['year'].max()})")
    print(f"   Test set:       {len(test_df):3d} records (Years {test_df['year'].min()} to {test_df['year'].max()})")
    
    feature_cols = cat_features + num_features
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN].values
    
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COLUMN].values
    
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN].values
    
    # 3. Fit Preprocessor ONLY on training set
    print("\n[Step 3/5] Fitting preprocessor pipeline on training set...")
    preprocessor = build_preprocessor(cat_features, num_features)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    
    transformed_feature_names = extract_feature_names(preprocessor, cat_features, num_features)
    print(f"-> Transformed Feature Dimension: {X_train_trans.shape[1]} inputs.")
    
    # 4. Define Candidate Models
    models = {
        'Linear Regression (Baseline)': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost Regressor'] = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
    else:
        print("Note: XGBoost package not detected; running Linear Regression & Random Forest.")
        
    # 5. Train and Benchmark Models
    print("\n[Step 4/5] Training models and evaluating on validation set...")
    comparison_results = []
    trained_model_dict = {}
    
    for model_name, model in models.items():
        print(f"\n--- Training: {model_name} ---")
        model.fit(X_train_trans, y_train)
        trained_model_dict[model_name] = model
        
        # Train metrics
        train_preds = model.predict(X_train_trans)
        train_metrics = calculate_metrics(y_train, train_preds)
        
        # Validation metrics
        val_preds = model.predict(X_val_trans)
        val_metrics = calculate_metrics(y_val, val_preds)
        
        print(f"   Validation MAE:   {val_metrics['mae_kg_ha']:,.2f} kg/ha")
        print(f"   Validation RMSE:  {val_metrics['rmse_kg_ha']:,.2f} kg/ha")
        print(f"   Validation R²:    {val_metrics['r2_score']:.4f}")
        
        comparison_results.append({
            'model_name': model_name,
            'train_mae_kg_ha': train_metrics['mae_kg_ha'],
            'train_rmse_kg_ha': train_metrics['rmse_kg_ha'],
            'train_r2_score': train_metrics['r2_score'],
            'val_mae_kg_ha': val_metrics['mae_kg_ha'],
            'val_rmse_kg_ha': val_metrics['rmse_kg_ha'],
            'val_r2_score': val_metrics['r2_score'],
            'val_relative_error_pct': val_metrics['relative_error_pct']
        })
        
    # Save comparison CSV
    comp_csv_path = os.path.join(reports_dir, "model_comparison.csv")
    df_comparison = save_model_comparison(comparison_results, comp_csv_path)
    print(f"\n-> Saved model comparison matrix: {comp_csv_path}")
    
    # 6. Select Best Model Based on Validation RMSE
    best_row = df_comparison.sort_values(by='val_rmse_kg_ha', ascending=True).iloc[0]
    best_model_name = best_row['model_name']
    best_model = trained_model_dict[best_model_name]
    print(f"\n🏆 Best Selected Model: {best_model_name} (Lowest Validation RMSE: {best_row['val_rmse_kg_ha']:,.2f} kg/ha)")
    
    # 7. Final Unbiased Evaluation on Held-Out Test Set (2020-2022)
    print("\n[Step 5/5] Evaluating Best Model on Unbiased Test Set (2020-2022)...")
    test_preds = best_model.predict(X_test_trans)
    test_metrics = calculate_metrics(y_test, test_preds)
    
    print(f"   Test MAE:   {test_metrics['mae_kg_ha']:,.2f} kg/ha")
    print(f"   Test RMSE:  {test_metrics['rmse_kg_ha']:,.2f} kg/ha")
    print(f"   Test R²:    {test_metrics['r2_score']:.4f}")
    print(f"   Mean Test Actual Yield: {test_metrics['mean_actual_kg_ha']:,.2f} kg/ha")
    print(f"   Test Relative Error:    {test_metrics['relative_error_pct']:.2f}%")
    
    # 8. Save Artifacts for Prediction Module & Streamlit App
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    metadata_path = os.path.join(models_dir, "feature_metadata.json")
    
    joblib.dump(best_model, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save metadata
    unique_districts = sorted(clean_df['district'].unique().tolist())
    unique_crops = sorted(clean_df['crop'].unique().tolist())
    unique_seasons = sorted(clean_df['season'].unique().tolist())
    
    metadata = {
        'best_model_name': best_model_name,
        'categorical_features': cat_features,
        'numeric_features': num_features,
        'transformed_features': transformed_feature_names,
        'available_districts': unique_districts,
        'available_crops': unique_crops,
        'available_seasons': unique_seasons,
        'training_years': [int(train_df['year'].min()), int(train_df['year'].max())],
        'validation_years': [int(val_df['year'].min()), int(val_df['year'].max())],
        'test_years': [int(test_df['year'].min()), int(test_df['year'].max())],
        'test_metrics': test_metrics,
        'area_stats': {
            'min': float(clean_df['area'].min()),
            'max': float(clean_df['area'].max()),
            'mean': float(clean_df['area'].mean()),
            'median': float(clean_df['area'].median())
        },
        'yield_stats': {
            'min': float(clean_df['yield'].min()),
            'max': float(clean_df['yield'].max()),
            'mean': float(clean_df['yield'].mean())
        }
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"-> Saved model artifact:       {best_model_path}")
    print(f"-> Saved preprocessor:         {preprocessor_path}")
    print(f"-> Saved feature metadata:     {metadata_path}")
    
    return {
        'best_model_name': best_model_name,
        'comparison': df_comparison,
        'test_metrics': test_metrics,
        'metadata': metadata
    }


if __name__ == "__main__":
    train_and_evaluate_models()
