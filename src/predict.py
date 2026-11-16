"""
Prediction and Inference Module for AgroVision AI
Author: Mehul Shankar P (AgroVision Project)

This module provides:
1. Reusable prediction function for agricultural yield estimation.
2. Input validation for district, year, cultivated area, crop, and season.
3. Preprocessing transformation identical to training pipeline.
4. Agronomic context and conversational explanation of predictions.
"""

import os
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib


class AgroYieldPredictor:
    """
    Inference engine that loads trained artifacts and generates validated yield estimates.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "best_model.joblib")
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        self.metadata_path = os.path.join(models_dir, "feature_metadata.json")
        
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self._load_artifacts()
        
    def _load_artifacts(self) -> None:
        """Load model, preprocessor, and metadata from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Please train the model first by running src/train.py."
            )
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor artifact not found at {self.preprocessor_path}. Run src/train.py first."
            )
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at {self.metadata_path}. Run src/train.py first."
            )
            
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
    def validate_inputs(
        self,
        district: str,
        year: int,
        area: float,
        crop: str = "total_group",
        season: str = "whole_year"
    ) -> Dict[str, Any]:
        """
        Validate incoming user inputs against trained domain constraints.
        
        Args:
            district: Tamil Nadu district name.
            year: Agricultural year.
            area: Cultivated area in hectares.
            crop: Crop name.
            season: Agricultural season.
            
        Returns:
            Dict[str, Any]: Normalized inputs.
            
        Raises:
            ValueError: If inputs violate agronomic bounds or district is invalid.
        """
        # Validate area
        if area is None or not isinstance(area, (int, float)) or area <= 0:
            raise ValueError(
                f"Cultivated area must be a positive number greater than 0 hectares. Received: {area}"
            )
            
        # Validate year
        if year is None or not isinstance(year, int) or year < 1990 or year > 2050:
            raise ValueError(
                f"Year must be a realistic agricultural year between 1990 and 2050. Received: {year}"
            )
            
        # Normalize and validate district
        clean_dist = str(district).strip().lower().replace(" ", "_")
        available_districts = self.metadata.get('available_districts', [])
        if available_districts and clean_dist not in available_districts:
            raise ValueError(
                f"District '{district}' is not recognized in Tamil Nadu 38-district database. "
                f"Available districts include: {', '.join(available_districts[:5])}..."
            )
            
        # Normalize crop and season
        clean_crop = str(crop).strip().lower().replace(" ", "_")
        clean_season = str(season).strip().lower().replace(" ", "_")
        
        return {
            'district': clean_dist,
            'year': int(year),
            'area': float(area),
            'crop': clean_crop,
            'season': clean_season
        }

    def predict(
        self,
        district: str,
        year: int,
        area: float,
        crop: str = "total_group",
        season: str = "whole_year"
    ) -> Dict[str, Any]:
        """
        Generate validated yield estimation in kg/ha.
        
        Args:
            district: Tamil Nadu district name.
            year: Agricultural year.
            area: Cultivated area in hectares.
            crop: Crop category.
            season: Agricultural season.
            
        Returns:
            Dict[str, Any]: Structured prediction payload.
        """
        # 1. Validate inputs
        validated = self.validate_inputs(district, year, area, crop, season)
        
        # 2. Construct single-row DataFrame matching the model's feature signature
        input_row = pd.DataFrame([{
            'district': validated['district'],
            'year': validated['year'],
            'area': validated['area'],
            'crop': validated['crop'],
            'season': validated['season']
        }])
        
        # 3. Transform features through identical preprocessor
        feature_cols = self.metadata['categorical_features'] + self.metadata['numeric_features']
        input_features = input_row[[c for c in feature_cols if c in input_row.columns]]
        X_trans = self.preprocessor.transform(input_features)
        
        # 4. Predict
        raw_pred = float(self.model.predict(X_trans)[0])
        # Agronomic constraint: Yield cannot be negative
        predicted_yield = max(0.0, round(raw_pred, 2))
        
        # Approximate total production estimation based on predicted yield
        # Production (tonnes) = Yield (kg/ha) * Area (ha) / 1000
        estimated_production_tonnes = round((predicted_yield * validated['area']) / 1000.0, 2)
        
        # Formulate beginner-friendly explanation
        state_mean = self.metadata['yield_stats']['mean']
        diff_pct = round(((predicted_yield - state_mean) / state_mean) * 100.0, 1)
        comparison_str = "above" if diff_pct >= 0 else "below"
        
        explanation = (
            f"For {validated['district'].replace('_', ' ').title()} in {validated['year']} "
            f"with {validated['area']:,.1f} ha cultivated area, the predicted crop yield is "
            f"{predicted_yield:,.2f} kg/hectare. This is {abs(diff_pct):.1f}% {comparison_str} "
            f"the historical Tamil Nadu statewide average ({state_mean:,.2f} kg/ha). "
            f"Total estimated regional harvest: ~{estimated_production_tonnes:,.1f} metric tonnes."
        )
        
        return {
            'predicted_yield_kg_ha': predicted_yield,
            'estimated_production_tonnes': estimated_production_tonnes,
            'district': validated['district'],
            'crop': validated['crop'],
            'season': validated['season'],
            'year': validated['year'],
            'area_ha': validated['area'],
            'model_used': self.metadata.get('best_model_name', 'Trained Model'),
            'explanation': explanation,
            'statewide_avg_yield_kg_ha': round(state_mean, 2)
        }


def predict_yield(
    district: str,
    year: int,
    area: float,
    crop: str = "total_group",
    season: str = "whole_year",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Convenience wrapper function for single-line yield prediction.
    """
    predictor = AgroYieldPredictor(models_dir=models_dir)
    return predictor.predict(district, year, area, crop, season)


if __name__ == "__main__":
    # Test script if models are already trained
    try:
        res = predict_yield(district="thanjavur", year=2024, area=50000.0)
        print("=== Prediction Result ===")
        for k, v in res.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"Prediction test info: {e}")
