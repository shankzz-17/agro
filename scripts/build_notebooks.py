"""
Notebook Builder Script for AgroVision AI
Generates 3 standard Jupyter Notebooks:
- 01_data_understanding.ipynb
- 02_exploratory_data_analysis.ipynb
- 03_model_experimentation.ipynb
"""

import os
import json


def create_cell(cell_type, source):
    if isinstance(source, str):
        source = [line + '\n' for line in source.split('\n')]
        if source and source[-1] == '\n':
            source[-1] = ''
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {})
    }


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def build_all_notebooks():
    nb_dir = "notebooks"
    os.makedirs(nb_dir, exist_ok=True)
    
    # ========================================================
    # 1. 01_data_understanding.ipynb
    # ========================================================
    nb1_cells = [
        create_cell("markdown", """# AgroVision AI: 01 - Data Understanding & Schema Inspection
**Project:** AI-Based Multi-Crop Yield Estimation and Rainfall Impact Analysis Using Machine Learning for Tamil Nadu  
**Author:** Mehul Shankar P (First-Year CSE Student)  
**Scope:** 38 Administrative Districts of Tamil Nadu (2000–2022)

---
### Notebook Objectives:
1. Load and inspect the Directorate of Economics and Statistics (DES) agricultural dataset.
2. Verify actual column names, data types, units, and row counts.
3. Audit missing values and understand administrative district bifurcation timelines.
4. Establish the target variable (`yield` in kg/ha) and verify zero-leakage rules.
"""),
        create_cell("code", """import os
import sys
import pandas as pd
import numpy as np

# Load the cleaned master dataset
data_path = os.path.join("..", "data", "processed", "des_cleaned_all_crops.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("data", "processed", "des_cleaned_all_crops.csv")

df = pd.read_csv(data_path)
print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
df.head()
"""),
        create_cell("markdown", """### 1. Data Schema and Types
Let's inspect the column names, non-null counts, and data types.
"""),
        create_cell("code", """df.info()
"""),
        create_cell("markdown", """### 2. Missing Value Analysis
Identify why missing values exist in the dataset without silently filling them with fake data.
"""),
        create_cell("code", """missing_summary = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Percentage (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
print("Missing Values Summary:")
display(missing_summary[missing_summary['Missing Count'] > 0])
"""),
        create_cell("markdown", """### 3. Geographical & Temporal Coverage
Verify the 38 Tamil Nadu districts and 23 historical years.
"""),
        create_cell("code", """print(f"Number of unique districts: {df['district'].nunique()}")
print("Districts:", sorted(df['district'].unique()))
print(f"\\nNumber of unique years: {df['year'].nunique()}")
print("Years range:", df['year'].min(), "to", df['year'].max())
"""),
        create_cell("markdown", """### 4. Target Variable Analysis: Yield (kg/ha)
Summary statistics of agricultural yield across valid observations.
"""),
        create_cell("code", """valid_yield = df['yield'].dropna()
print("Yield Summary (kg/ha):")
print(valid_yield.describe().round(2))
""")
    ]
    
    with open(os.path.join(nb_dir, "01_data_understanding.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb1_cells), f, indent=2)
        
    # ========================================================
    # 2. 02_exploratory_data_analysis.ipynb
    # ========================================================
    nb2_cells = [
        create_cell("markdown", """# AgroVision AI: 02 - Exploratory Data Analysis (EDA)
**Project:** AI-Based Multi-Crop Yield Estimation and Rainfall Impact Analysis  
**Focus:** District Yield Rankings, Historical Trends, and Correlation Analysis

---
### Visualizations in this Notebook:
1. District-wise average yield ranking across 38 districts.
2. 23-Year statewide yield trendline (2000–2022).
3. Cultivated area vs. yield productivity relationship.
4. Correlation heatmap of numerical features.
"""),
        create_cell("code", """import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

data_path = os.path.join("..", "data", "processed", "des_cleaned_all_crops.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("data", "processed", "des_cleaned_all_crops.csv")

df = pd.read_csv(data_path).dropna(subset=['yield'])
print(f"Loaded {len(df)} verified agricultural observations.")
"""),
        create_cell("markdown", """### 1. District-wise Yield Comparison
Which Tamil Nadu districts exhibit highest and lowest historical productivity?
"""),
        create_cell("code", """dist_avg = df.groupby('district')['yield'].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(14, 8))
sns.barplot(data=dist_avg, x='yield', y='district', palette='viridis')
plt.axvline(df['yield'].mean(), color='red', linestyle='--', label=f'State Mean ({df["yield"].mean():,.0f} kg/ha)')
plt.title('Average Agricultural Yield by District in Tamil Nadu (2000–2022)', fontsize=14, fontweight='bold')
plt.xlabel('Average Yield (kg/ha)')
plt.ylabel('District')
plt.legend()
plt.tight_layout()
plt.show()
"""),
        create_cell("markdown", """### 2. 23-Year Statewide Historical Productivity Trend
Tracking yield trajectory from 2000 to 2022.
"""),
        create_cell("code", """yearly = df.groupby('year')['yield'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(12, 5))
plt.plot(yearly['year'], yearly['mean'], marker='o', color='#1565C0', linewidth=2.5)
plt.title('Tamil Nadu Average Crop Yield Over 23 Years (2000–2022)', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Yield (kg/ha)')
plt.xticks(yearly['year'], rotation=45)
plt.tight_layout()
plt.show()
"""),
        create_cell("markdown", """### 3. Cultivated Area vs. Yield Scatter Plot
Examining whether farm acreage size correlates with higher productivity.
"""),
        create_cell("code", """plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='area', y='yield', alpha=0.6, color='#00897B')
plt.title('Cultivated Area (ha) vs. Crop Yield (kg/ha)', fontsize=14, fontweight='bold')
plt.xlabel('Area (Hectares)')
plt.ylabel('Yield (kg/ha)')
plt.tight_layout()
plt.show()
"""),
        create_cell("markdown", """### 4. Correlation Analysis
Correlation between numerical variables: Year, Area, Production, and Yield.
"""),
        create_cell("code", """plt.figure(figsize=(8, 6))
sns.heatmap(df[['year', 'area', 'production', 'yield']].corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Correlation Matrix of Agricultural Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
""")
    ]
    
    with open(os.path.join(nb_dir, "02_exploratory_data_analysis.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb2_cells), f, indent=2)
        
    # ========================================================
    # 3. 03_model_experimentation.ipynb
    # ========================================================
    nb3_cells = [
        create_cell("markdown", """# AgroVision AI: 03 - Model Experimentation & Evaluation
**Project:** AI-Based Multi-Crop Yield Estimation for Tamil Nadu  
**Models:** Linear Regression (Baseline), Random Forest Regressor, XGBoost Regressor  
**Methodology:** Chronological Train/Validation/Test Split with Zero Leakage Preprocessing

---
### Key ML Design Rules:
1. **Chronological Splitting:** 2000–2016 (Train), 2017–2019 (Val), 2020–2022 (Test).
2. **Leakage Exclusion:** `production` is strictly excluded from features because $\\text{Yield} = \\frac{\\text{Production} \\times 1000}{\\text{Area}}$.
3. **Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).
"""),
        create_cell("code", """import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath("..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import load_dataset, clean_agricultural_data
from src.feature_engineering import identify_model_features, chronological_split, build_preprocessor
from src.evaluate import calculate_metrics, save_model_comparison
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
"""),
        create_cell("markdown", """### 1. Data Ingestion & Chronological Split
Partition data by year to prevent future climate/economic leakage into past models.
"""),
        create_cell("code", """data_path = os.path.join("..", "data", "processed", "des_cleaned_all_crops.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("data", "processed", "des_cleaned_all_crops.csv")

clean_df, _ = clean_agricultural_data(load_dataset(data_path))
cat_feats, num_feats = identify_model_features(clean_df)
feature_cols = cat_feats + num_feats

train_df, val_df, test_df = chronological_split(clean_df, train_end_year=2016, val_end_year=2019)
print(f"Train records (2000-2016):      {len(train_df)}")
print(f"Validation records (2017-2019): {len(val_df)}")
print(f"Test records (2020-2022):       {len(test_df)}")

X_train, y_train = train_df[feature_cols], train_df['yield'].values
X_val, y_val = val_df[feature_cols], val_df['yield'].values
X_test, y_test = test_df[feature_cols], test_df['yield'].values
"""),
        create_cell("markdown", """### 2. Preprocessing Pipeline
Fit ColumnTransformer strictly on the training set.
"""),
        create_cell("code", """preprocessor = build_preprocessor(cat_feats, num_feats)
X_train_trans = preprocessor.fit_transform(X_train)
X_val_trans = preprocessor.transform(X_val)
X_test_trans = preprocessor.transform(X_test)
print(f"Transformed Feature Vector Dimension: {X_train_trans.shape[1]}")
"""),
        create_cell("markdown", """### 3. Model Training & Comparison
Train and benchmark Linear Regression, Random Forest, and XGBoost.
"""),
        create_cell("code", """models = {
    'Linear Regression (Baseline)': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
    'XGBoost Regressor': xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_trans, y_train)
    val_preds = model.predict(X_val_trans)
    m = calculate_metrics(y_val, val_preds)
    results.append({
        'Model': name,
        'Val MAE (kg/ha)': m['mae_kg_ha'],
        'Val RMSE (kg/ha)': m['rmse_kg_ha'],
        'Val R2 Score': m['r2_score']
    })

comparison_df = pd.DataFrame(results).sort_values(by='Val RMSE (kg/ha)')
display(comparison_df)
"""),
        create_cell("markdown", """### 4. Final Evaluation on Held-Out Test Set (2020–2022)
Evaluate the best model (Linear Regression) on unseen future test records.
"""),
        create_cell("code", """best_model = models['Linear Regression (Baseline)']
test_preds = best_model.predict(X_test_trans)
test_metrics = calculate_metrics(y_test, test_preds)

print("=== FINAL TEST EVALUATION (2020-2022) ===")
print(f"Test MAE:   {test_metrics['mae_kg_ha']:,.2f} kg/ha")
print(f"Test RMSE:  {test_metrics['rmse_kg_ha']:,.2f} kg/ha")
print(f"Test R2:    {test_metrics['r2_score']:.4f}")
print(f"Mean Yield: {test_metrics['mean_actual_kg_ha']:,.2f} kg/ha")
""")
    ]
    
    with open(os.path.join(nb_dir, "03_model_experimentation.ipynb"), "w", encoding="utf-8") as f:
        json.dump(make_notebook(nb3_cells), f, indent=2)
        
    print("✓ All 3 Jupyter Notebooks successfully generated in notebooks/")


if __name__ == "__main__":
    build_all_notebooks()
