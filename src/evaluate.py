"""
Model Evaluation and Metric Calculation Module for AgroVision AI
Author: Mehul Shankar P (AgroVision Project)

This module calculates:
- Mean Absolute Error (MAE) in kg/ha
- Root Mean Squared Error (RMSE) in kg/ha
- Coefficient of Determination (R² Score)
- Generates reproducible model comparison CSV reports.
- Provides beginner-friendly natural language interpretations.
"""

import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate core regression metrics for agricultural yield estimation.
    
    Args:
        y_true: Ground truth target values (kg/ha).
        y_pred: Predicted target values (kg/ha).
        
    Returns:
        Dict[str, float]: Calculated MAE, RMSE, R2, and Mean Actual Yield.
    """
    mae = mean_absolute_error(y_true, y_pred)
    # Using root_mean_squared_error or sqrt of MSE for backward/forward compatibility
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    mean_actual = float(np.mean(y_true))
    
    # Calculate percentage error relative to mean yield
    mape_approx = (mae / mean_actual * 100.0) if mean_actual > 0 else 0.0
    
    return {
        'mae_kg_ha': round(float(mae), 2),
        'rmse_kg_ha': round(float(rmse), 2),
        'r2_score': round(float(r2), 4),
        'relative_error_pct': round(float(mape_approx), 2),
        'mean_actual_kg_ha': round(float(mean_actual), 2)
    }


def explain_metric(metric_name: str, value: float) -> str:
    """
    Provide an intuitive, beginner-friendly explanation for a metric value.
    
    Args:
        metric_name: Name of metric ('mae', 'rmse', 'r2').
        value: Numeric value.
        
    Returns:
        str: Conversational explanation.
    """
    if metric_name.lower() == 'mae':
        return (
            f"Mean Absolute Error (MAE = {value:,.2f} kg/ha): On average, the model's yield "
            f"prediction misses the true harvest by ±{value:,.2f} kilograms per hectare."
        )
    elif metric_name.lower() == 'rmse':
        return (
            f"Root Mean Squared Error (RMSE = {value:,.2f} kg/ha): Like MAE, but squares errors "
            f"first, giving higher penalty to large estimation blunders."
        )
    elif metric_name.lower() == 'r2':
        pct = value * 100.0
        if value < 0:
            return (
                f"R² Score ({value:.4f}): The model performs worse than simply guessing the average "
                f"state yield. This occurs when patterns are non-linear or feature set is incomplete."
            )
        else:
            return (
                f"R² Score ({value:.4f}): The model's features explain {pct:.1f}% of the historical "
                f"variation in crop yield across Tamil Nadu districts."
            )
    return f"{metric_name}: {value}"


def save_model_comparison(results: List[Dict[str, Any]], output_path: str) -> pd.DataFrame:
    """
    Compile evaluation results into a standardized comparison DataFrame and CSV.
    
    Args:
        results: List of dictionaries with model evaluation stats.
        output_path: File path to save the comparison CSV.
        
    Returns:
        pd.DataFrame: Formatted comparison table.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_comp = pd.DataFrame(results)
    
    # Sort by Validation RMSE ascending (lower error is better)
    if 'val_rmse_kg_ha' in df_comp.columns:
        df_comp = df_comp.sort_values(by='val_rmse_kg_ha', ascending=True).reset_index(drop=True)
        
    df_comp.to_csv(output_path, index=False)
    return df_comp
