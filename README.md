# AgroVision AI: Multi-Crop Yield Estimation & Rainfall Impact Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)
[![ML: Scikit--Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Full Title:** AI-Based Multi-Crop Yield Estimation and Rainfall Impact Analysis Using Machine Learning for Tamil Nadu  
> **Repository:** AgroVision  
> **Geographical Scope:** 38 Administrative Districts of Tamil Nadu, India  
> **Author:** Mehul Shankar P (First-Year Computer Science & Engineering Student)

---

## 1. Project Overview & Motivation

Agriculture is the economic foundation of Tamil Nadu, sustaining millions of farmers across diverse agro-climatic zones ranging from the Cauvery Delta to the dry Southern plains. However, crop yield fluctuates significantly due to erratic monsoon patterns, ground-water depletion, and acreage adjustments.

**AgroVision AI** provides an end-to-end, scientifically grounded machine learning pipeline and interactive decision-support dashboard (**VayalYield AI**) to forecast agricultural yield (in **kg/hectare**) before harvest, assess historical productivity trends across 38 districts over 23 years (2000–2022), and lay the architectural foundation for dual-stream rainfall impact fusion.

---

## 2. Problem Statement & Core Objectives

1. **Pre-Harvest Yield Estimation:** Accurately estimate crop yield in kilograms per hectare ($kg/ha$) using verified agricultural variables (`district`, `year`, `cultivated_area`).
2. **Zero Data Leakage:** Ensure models do not use `production` as an input. Since $\text{Yield} = \frac{\text{Production} \times 1000}{\text{Area}}$, including production is a classic ML error where the algorithm merely learns division arithmetic rather than agronomic patterns.
3. **Temporal Integrity:** Employ strict chronological train/validation/test splits (2000–2016 Train, 2017–2019 Validation, 2020–2022 Test) rather than random splits, reflecting real-world agricultural forecasting.
4. **Rainfall Integration Architecture:** Model rainfall dynamics conceptually and build modular hooks to integrate IMD seasonal rainfall metrics without rewriting code.

---

## 3. Dataset Description & Provenance

The primary agricultural dataset is sourced from the **Directorate of Economics and Statistics (DES)** Agricultural Statistics portal (APY Report for Tamil Nadu):

| Metric | Detail |
| :--- | :--- |
| **Geographic Scope** | 38 Districts of Tamil Nadu, India |
| **Temporal Range** | 2000 to 2022 (23 consecutive agricultural years) |
| **Total Structured Records** | 874 observations ($38 \text{ districts} \times 23 \text{ years}$) |
| **Valid Observations** | 716 verified records with non-null harvest yield |
| **Missing Records** | 158 rows (154 due to historical district bifurcations before formation date, 4 due to missing production) |
| **Target Variable ($y$)** | `yield` in kilograms per hectare ($kg/ha$) |
| **Feature Predictors ($X$)** | `district` (Categorical), `year` (Temporal), `area` (Continuous Hectares) |

---

## 4. System Architecture

```text
AgroVision/
│
├── data/
│   ├── raw/
│   │   └── apy_total_group_report.xls     # Original DES APY table
│   └── processed/
│       ├── des_cleaned_all_crops.csv      # Standardized master dataset (874 rows)
│       └── des_selected_8_crops.csv       # Filtered dataset with notes
│
├── notebooks/
│   ├── 01_data_understanding.ipynb        # Schema inspection & quality audit
│   ├── 02_exploratory_data_analysis.ipynb # Interactive visual analysis
│   └── 03_model_experimentation.ipynb     # Model training & comparison
│
├── src/
│   ├── __init__.py                        # Package initialization
│   ├── data_preprocessing.py              # Data loading, validation & cleaning
│   ├── feature_engineering.py             # Preprocessor & chronological split
│   ├── train.py                           # Multi-model training pipeline
│   ├── evaluate.py                        # MAE, RMSE, R² & comparison exporter
│   └── predict.py                         # Reusable prediction engine
│
├── models/
│   ├── best_model.joblib                  # Serialized best model artifact
│   ├── preprocessor.joblib                # Serialized ColumnTransformer
│   └── feature_metadata.json              # Schema & normalization bounds
│
├── app/
│   └── app.py                             # Interactive Streamlit Web App (VayalYield AI)
│
├── reports/
│   ├── figures/                           # 7 publication-quality EDA figures
│   └── model_comparison.csv               # Actual benchmark results matrix
│
├── tests/
│   └── test_pipeline.py                   # Pytest unit & integration test suite
│
├── .gitignore                             # Git ignore rules for caches & artifacts
├── LICENSE                                # MIT Open-Source License
├── README.md                              # Project documentation
└── requirements.txt                       # Project dependencies
```

---

## 5. Technology Stack

- **Language:** Python 3.10+
- **Data Engineering:** `pandas`, `numpy`, `openpyxl`
- **Machine Learning:** `scikit-learn`, `xgboost`, `joblib`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Web Application:** `streamlit`
- **Unit Testing:** `pytest`

---

## 6. Installation & Quickstart

### Step 1: Clone the Repository
```bash
git clone https://github.com/shankzz-17/AgroVision.git
cd AgroVision
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. How to Run the Project

### 1. Train and Benchmark the ML Models
To run data preprocessing, chronological splitting, model fitting, and export model comparison artifacts:
```bash
python src/train.py
```

### 2. Generate EDA Visualizations
To regenerate all 7 figures in `reports/figures/`:
```bash
python scripts/generate_eda_charts.py
```

### 3. Run the Unit Test Suite
To verify pipeline integrity, zero-leakage constraints, and input validation:
```bash
pytest tests/test_pipeline.py -v
```

### 4. Launch the Streamlit Web Application (VayalYield AI)
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 8. Actual Model Evaluation & Results

Models were evaluated on a **chronological holdout validation split** (Trained on 2000–2016, Validated on 2017–2019):

| Model Name | Train MAE ($kg/ha$) | Train RMSE ($kg/ha$) | Train $R^2$ | Validation MAE ($kg/ha$) | Validation RMSE ($kg/ha$) | Validation $R^2$ | Validation Rel. Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | **670.63** | **913.27** | **0.4631** | **874.24** | **1,319.33** | **+0.2868** | **23.41%** |
| **XGBoost Regressor** | 341.03 | 458.65 | 0.8646 | 1,431.31 | 1,831.31 | -0.3742 | 38.33% |
| **Random Forest Regressor** | 408.10 | 539.86 | 0.8124 | 1,656.16 | 2,081.53 | -0.7754 | 44.35% |

### Key Machine Learning Takeaway:
- **Decision Trees Overfitted:** Random Forest and XGBoost memorized the training years (high training $R^2 > 0.81$), but struggled to extrapolate on future temporal hold-out records when climate variables (like rainfall) are not yet in the feature matrix.
- **Linear Regression Won:** Linear Regression learned continuous yearly growth coefficients and regional baseline intercepts, achieving the lowest Validation RMSE ($1,319.33\text{ kg/ha}$) and positive $R^2$ ($0.2868$).

### Final Unbiased Test Set Evaluation (2020–2022, 113 records):
- **Test MAE:** $953.47\text{ kg/ha}$
- **Test RMSE:** $1,226.33\text{ kg/ha}$
- **Test Mean Actual Yield:** $3,575.42\text{ kg/ha}$
- **Relative Error:** $26.67\%$

---

## 9. Rainfall Impact Analysis & Roadmap

While rainfall is not yet present in the base DES agricultural table, the pipeline is constructed with modular dual-stream fusion:
1. **Monsoon Dynamics:** Captures South-West Monsoon (Jun–Sep) and North-East Monsoon (Oct–Dec).
2. **Correlation vs. Causation:** Explains why higher precipitation is non-linear (excess rainfall during flowering/harvest leads to pest outbreaks and crop damage).
3. **Pre-Harvest Temporal Integrity:** When rainfall data is fused, only rainfall up to the prediction cut-off date is utilized.

---

## 10. Limitations & Future Scope

1. **Crop Granularity:** The initial DES file provides aggregate crop group totals. Future extensions will ingest crop-specific files (Rice, Maize, Sugarcane, Cotton) to allow crop-level predictions.
2. **Rainfall Feature Fusion:** Ingestion of gridded IMD (India Meteorological Department) rainfall anomalies will improve tree model performance.
3. **Soil & Satellite Vegetation Indices:** Adding NDVI and soil moisture from Sentinel-2 or MODIS for real-time field monitoring.

---

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
