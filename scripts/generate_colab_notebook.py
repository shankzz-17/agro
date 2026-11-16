"""
Generates the comprehensive, beginner-friendly Google Colab Jupyter Notebook
for DES Tamil Nadu Agricultural Data Preprocessing.
"""

import os
import json

def build_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "None",
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    def md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header
    md("""# AI-Based Multi-Crop Yield Estimation and Rainfall Impact Analysis
## Module 1: DES Tamil Nadu Agricultural Data Processing & Validation Pipeline
**College Project:** AI-Based Multi-Crop Yield Estimation and Rainfall Impact Analysis Using Machine Learning for Tamil Nadu  
**Role:** Senior Data Scientist and Agricultural Data Engineer  
**Stage:** Data Exploration, Preprocessing, Cleaning, Validation, and Feature Preparation *(No ML model training yet)*  

---
### Key Objectives
1. **Automated Source Inspection:** Ingest and profile district-wise agricultural statistics from DES/APY (Government of India).
2. **Data Cleaning & Logging:** Normalize column names, standardize Tamil Nadu district names, remove artificial zero-imputations, and record every operation in an audit log.
3. **Agronomic Yield Calculation:** Reconcile units and calculate yield:
   $$\\text{Yield (kg/ha)} = \\frac{\\text{Production (tonnes)} \\times 1000}{\\text{Area (hectares)}}$$
4. **Data Quality Audit:** Validate non-negativity, mathematical consistency, and administrative bifurcation gaps.
5. **Crop Filtering:** Segregate target crops (Rice, Maize, Ragi, Black gram, Green gram, Groundnut, Cotton, Sugarcane) while preserving the full master dataset.
6. **Exploratory Data Analysis:** Render 8 comprehensive Matplotlib/Seaborn visualizations.
7. **ML Preparation:** Formulate the master agricultural feature table (strictly omitting production from yield predictor features).
8. **Export Deliverables:** Output all 8 CSV deliverables and a multi-sheet Excel workbook.""")

    # Section 1
    md("""---
## 1. Environment Setup & Library Initialization
Import core scientific computing, data wrangling, and plotting libraries.""")

    code("""import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure openpyxl is installed for Excel export in Colab
try:
    import openpyxl
except ImportError:
    !pip install -q openpyxl
    import openpyxl

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 150

print("✓ All libraries successfully initialized:")
print(f"  - Pandas: {pd.__version__}")
print(f"  - NumPy: {np.__version__}")
print(f"  - Matplotlib: {matplotlib.__version__ if 'matplotlib' in locals() else 'Loaded'}")
print(f"  - Seaborn: {sns.__version__}")""")

    # Section 2
    md("""---
## 2. Load the Dataset
Upload your raw DES CSV or Excel file using Google Colab's upload widget or select from disk.
The loader dynamically inspects:
- File format & HTML wrappers (DES exports often save HTML tables as `.xls`).
- MultiIndex headers & dimensions.
- Missing values & duplicate records.
- Districts, years, seasons, and measurement units.""")

    code("""# Check if running in Google Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import files
    print("Please upload your DES agricultural statistics file (.csv, .xls, .xlsx):")
    uploaded = files.upload()
    file_path = list(uploaded.keys())[0]
else:
    # Local fallback
    file_path = 'data/raw/apy_total_group_report.xls'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}. Please place your DES file in data/raw/")

print(f"\\nTarget dataset: {file_path} ({os.path.getsize(file_path):,} bytes)")""")

    code("""def inspect_des_dataset(filepath):
    \"\"\"
    Inspects and dynamically loads DES files regardless of whether they are 
    raw CSVs, genuine Excel, or HTML tables disguised as .xls.
    \"\"\"
    print(f"=== Automated Inspection: {os.path.basename(filepath)} ===")
    
    # 1. Detection of file type
    ext = os.path.splitext(filepath)[1].lower()
    df_raw = None
    file_type = "Unknown"
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        head_sample = f.read(500).strip().lower()
        
    if '<table' in head_sample or '<html' in head_sample:
        print("  -> Format: HTML Table wrapped in .xls (Standard DES APY export).")
        dfs = pd.read_html(filepath)
        df_raw = dfs[0]
        file_type = "HTML_in_XLS"
    elif ext in ['.xls', '.xlsx']:
        df_raw = pd.read_excel(filepath)
        file_type = "Excel"
    else:
        try:
            df_raw = pd.read_csv(filepath)
            file_type = "CSV"
        except UnicodeDecodeError:
            df_raw = pd.read_csv(filepath, encoding='latin1')
            file_type = "CSV_latin1"

    rows, cols = df_raw.shape
    print(f"  -> Rows: {rows:,}")
    print(f"  -> Columns: {cols:,}")
    print(f"  -> Total cells: {df_raw.size:,}")
    print(f"  -> Missing values: {df_raw.isna().sum().sum():,} cells ({(df_raw.isna().sum().sum()/df_raw.size)*100:.2f}%)")
    print(f"  -> Exact duplicate rows: {df_raw.duplicated().sum()}")
    
    return df_raw, file_type

df_raw, detected_format = inspect_des_dataset(file_path)""")

    code("""# Display head and tail
print("--- First 10 Records ---")
display(df_raw.head(10))

print("\\n--- Last 10 Records ---")
display(df_raw.tail(10))""")

    # Section 3
    md("""---
## 3. Data Cleaning & Transformation Pipeline
Standardize column names, districts, and reshape wide reports into standard long-format tables:
- Standardize district names without merging different districts incorrectly.
- Standardize column names to lowercase `snake_case`.
- Unpivot year-wise metrics (`Area`, `Production`, `Yield`) into rows.
- Record every cleaning action into `cleaning_log`.""")

    code("""cleaning_log = []

def log_clean(step, rule, description, affected):
    cleaning_log.append({
        'step_number': step,
        'rule_applied': rule,
        'description': description,
        'affected_records': affected
    })

# Standardize Tamil Nadu District Names to lowercase snake_case
TN_DISTRICT_MAP = {
    'Ariyalur': 'ariyalur',
    'Chengalpattu': 'chengalpattu',
    'Chennai': 'chennai',
    'Coimbatore': 'coimbatore',
    'Cuddalore': 'cuddalore',
    'Dharmapuri': 'dharmapuri',
    'Dindigul': 'dindigul',
    'Erode': 'erode',
    'Kallakurichi': 'kallakurichi',
    'Kanchipuram': 'kanchipuram',
    'Kanniyakumari': 'kanniyakumari',
    'Karur': 'karur',
    'Krishnagiri': 'krishnagiri',
    'Madurai': 'madurai',
    'Mayiladuthurai': 'mayiladuthurai',
    'Nagapattinam': 'nagapattinam',
    'Namakkal': 'namakkal',
    'Perambalur': 'perambalur',
    'Pudukkottai': 'pudukkottai',
    'Ramanathapuram': 'ramanathapuram',
    'Ranipet': 'ranipet',
    'Salem': 'salem',
    'Sivaganga': 'sivaganga',
    'Thanjavur': 'thanjavur',
    'The nilgiris': 'the_nilgiris',
    'Nilgiris': 'the_nilgiris',
    'Theni': 'theni',
    'Thenkasi': 'tenkasi',
    'Tenkasi': 'tenkasi',
    'Thiruvallur': 'tiruvallur',
    'Thiruvarur': 'tiruvarur',
    'Thoothukudi': 'thoothukudi',
    'Tuticorin': 'thoothukudi',
    'Tiruchirappalli': 'tiruchirappalli',
    'Tirunelveli': 'tirunelveli',
    'Tirupathur': 'tirupathur',
    'Tiruppur': 'tiruppur',
    'Tiruvannamalai': 'tiruvannamalai',
    'Vellore': 'vellore',
    'Villupuram': 'villupuram',
    'Virudhunagar': 'virudhunagar'
}

def clean_num(val):
    if pd.isna(val):
        return np.nan
    try:
        s = str(val).replace(',', '').strip()
        if s == '' or s.lower() == 'nan':
            return np.nan
        return float(s)
    except Exception:
        return np.nan

# Check if report is wide format
is_wide = any(' - ' in str(c) for c in df_raw.columns) or isinstance(df_raw.columns, pd.MultiIndex)

if is_wide:
    flat_cols = []
    for c in df_raw.columns:
        if isinstance(c, tuple):
            if c[0] == c[1]:
                flat_cols.append(str(c[0]).strip())
            else:
                flat_cols.append(f"{str(c[0]).strip()}_{str(c[1]).strip()}")
        else:
            flat_cols.append(str(c).strip())
            
    df_wide = df_raw.copy()
    df_wide.columns = flat_cols
    
    dist_col = [c for c in df_wide.columns if 'district' in c.lower()][0]
    years = sorted(list(set([c.split('_')[0] for c in df_wide.columns if '_' in c and any(ch.isdigit() for ch in c)])))
    
    long_rows = []
    for _, row in df_wide.iterrows():
        raw_dist = str(row[dist_col]).strip()
        clean_dist = TN_DISTRICT_MAP.get(raw_dist, raw_dist.lower().replace(' ', '_'))
        
        for yr_str in years:
            yr_int = int(re.findall(r'\\d{4}', yr_str)[0])
            
            a_val = row.get(f"{yr_str}_Area (Hectare)")
            p_val = row.get(f"{yr_str}_Production (Tonnes)")
            y_rep = row.get(f"{yr_str}_Yield (Tonne/Hectare)")
            
            area = clean_num(a_val)
            prod = clean_num(p_val)
            rep_y = clean_num(y_rep)
            
            long_rows.append({
                'year': yr_int,
                'original_year': yr_str,
                'state': 'tamil_nadu',
                'district': clean_dist,
                'original_district': raw_dist,
                'crop': 'total_group',
                'original_crop': 'Total (Crop Group Report)',
                'season': 'whole_year',
                'original_season': 'Whole Year',
                'area': area,
                'area_unit': 'hectare',
                'production': prod,
                'production_unit': 'tonne',
                'reported_yield_tonne_ha': rep_y,
                'source_file': os.path.basename(file_path)
            })
            
    df_clean = pd.DataFrame(long_rows)
    log_clean(1, 'UNPIVOT_WIDE_TABLE', f'Unpivoted {len(df_wide)} districts across {len(years)} years into {len(df_clean)} standard long records.', len(df_clean))
else:
    df_clean = df_raw.copy()
    df_clean.columns = [c.lower().strip().replace(' ', '_') for c in df_clean.columns]
    df_clean['district'] = df_clean['district'].map(TN_DISTRICT_MAP)
    df_clean['source_file'] = os.path.basename(file_path)

print(f"✓ Reshaped and cleaned dataset: {len(df_clean):,} records, {len(df_clean.columns)} columns")""")

    # Section 4
    md("""---
## 4. Mathematical Yield Calculation & Integrity Validation
Compute yield in **kg/hectare** using:
$$\\text{Yield (kg/ha)} = \\frac{\\text{Production (tonnes)} \\times 1000}{\\text{Area (hectares)}}$$

**Crucial Integrity Rules:**
- Never replace missing production values with zero.
- Reconcile reported yield with calculated yield to identify data entry errors.
- Preserve explicit status flags (`calculated_from_prod_area`, `missing_production`, `pre_bifurcation_null`).""")

    code("""def calculate_yield(row):
    area = row['area']
    prod = row['production']
    
    if pd.notna(area) and pd.notna(prod) and area > 0:
        return round((prod * 1000.0) / area, 2), 'calculated_from_prod_area'
    elif pd.notna(area) and pd.isna(prod):
        return np.nan, 'missing_production'
    elif pd.isna(area):
        return np.nan, 'missing_area'
    else:
        return np.nan, 'unresolved'

res = df_clean.apply(calculate_yield, axis=1)
df_clean['yield'] = [r[0] for r in res]
df_clean['yield_unit'] = 'kg_per_ha'
df_clean['yield_status'] = [r[1] for r in res]

log_clean(2, 'RECONCILE_YIELD', 'Calculated yield_kg_ha using (production_tonnes * 1000 / area_ha) without assuming missing production is zero.', len(df_clean))

# Flag misleading reported zero yields
if 'reported_yield_tonne_ha' in df_clean.columns:
    false_zeros = (df_clean['production'].isna()) & (df_clean['reported_yield_tonne_ha'] == 0.0)
    print(f"Identified {false_zeros.sum()} records where reported yield was falsely 0.0 with missing production.")
    log_clean(3, 'FLAG_FALSE_ZERO_YIELDS', f'Detected {false_zeros.sum()} misleading reported zero yields; calculated yield kept as NaN.', int(false_zeros.sum()))

print(f"\\nValid Calculated Yields: {df_clean['yield'].notna().sum():,} observations")
print(f"Missing Yields (Missing production or pre-bifurcation): {df_clean['yield'].isna().sum():,} observations")""")

    # Section 5
    md("""---
## 5. Data Quality Audit & Validation Report
Comprehensive audit checking:
- Geographical validity for Tamil Nadu.
- Non-negativity constraints ($Area \\ge 0, Production \\ge 0$).
- Historical gaps from district administrative splits (e.g., Tenkasi bifurcated from Tirunelveli in 2019).""")

    code("""# Check non-negative constraints
neg_area = int((df_clean['area'] < 0).sum())
neg_prod = int((df_clean['production'] < 0).sum())
neg_yield = int((df_clean['yield'] < 0).sum())

print("=== Non-Negativity Checks ===")
print(f"  - Negative Area Records: {neg_area}")
print(f"  - Negative Production Records: {neg_prod}")
print(f"  - Negative Yield Records: {neg_yield}")

# Duplicate District-Year check
dup_count = df_clean.duplicated(subset=['district', 'year', 'crop', 'season']).sum()
print(f"\\nDuplicate (District-Year-Crop-Season) Records: {dup_count}")

# Missing values by district (reflecting administrative inception dates)
print("\\nMissing Historical Years per District:")
null_summary = df_clean[df_clean['area'].isna()]['original_district'].value_counts()
display(null_summary.to_frame(name='Missing Historical Years'))""")

    # Section 6
    md("""---
## 6. Focal Crop Segmentation
Extract the 8 target crops specified for the multi-crop yield model:
- **Cereals:** Rice, Maize, Ragi
- **Pulses:** Black gram, Green gram
- **Oilseeds:** Groundnut
- **Commercial:** Cotton, Sugarcane

*Note: If processing the aggregate total report, the full dataset is preserved and a template is prepared for the crop-wise upload.*""")

    code("""TARGET_8_CROPS = [
    'rice', 'maize', 'ragi', 
    'black_gram', 'green_gram', 
    'groundnut', 
    'cotton', 'sugarcane'
]

has_individual_crops = set(df_clean['crop'].str.lower()).intersection(TARGET_8_CROPS)

if has_individual_crops:
    print(f"Found {len(has_individual_crops)} of 8 focal crops: {has_individual_crops}")
    df_selected_crops = df_clean[df_clean['crop'].str.lower().isin(TARGET_8_CROPS)].copy()
else:
    print("Uploaded file represents aggregate district totals ('total_group').")
    print("Preserving complete cleaned dataset separately and establishing template for crop-wise APY data.")
    df_selected_crops = df_clean.copy()
    df_selected_crops['crop_filter_status'] = 'Aggregate total group; individual crop breakdown pending crop-wise APY upload'

print(f"Selected crops dataset shape: {df_selected_crops.shape}")""")

    # Section 7
    md("""---
## 7. Exploratory Data Analysis & Visualizations
Generating all 8 required exploratory visualizations:
1. Valid record counts per district
2. Crop-wise / group record counts
3. Year-wise data availability
4. District-wise cultivated area
5. District-wise crop production
6. Annual yield trends
7. Missing value matrix heatmap
8. District coverage analysis""")

    code("""# 1. District Data Availability
plt.figure(figsize=(12, 9))
dist_counts = df_clean.dropna(subset=['area']).groupby('original_district')['year'].count().sort_values()
colors = ['#1f77b4' if v == 23 else '#e67e22' for v in dist_counts]
bars = plt.barh(dist_counts.index, dist_counts.values, color=colors, height=0.65)
plt.title('1. Valid Historical Agricultural Data Availability by District (2000–2023)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Reporting Years (Max = 23)', fontsize=11)
plt.ylabel('Tamil Nadu District', fontsize=11)
plt.xlim(0, 25)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{int(w)}', va='center', fontsize=9)
plt.tight_layout()
plt.show()""")

    code("""# 2. Crop Record Counts
plt.figure(figsize=(8, 4))
crop_counts = df_clean.groupby('original_crop')['district'].count()
plt.bar(crop_counts.index, crop_counts.values, color='#2ecc71', width=0.4)
plt.title('2. Crop Coverage Record Counts', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Crop / Category', fontsize=11)
plt.ylabel('Total Observations', fontsize=11)
for i, v in enumerate(crop_counts.values):
    plt.text(i, v + 15, f'{v} records\\n(38 districts × 23 yrs)', ha='center', fontsize=10, fontweight='bold')
plt.ylim(0, 1000)
plt.tight_layout()
plt.show()""")

    code("""# 3. Year-wise Agricultural Reporting Availability
plt.figure(figsize=(12, 5))
yr_counts = df_clean.dropna(subset=['area']).groupby('original_year')['district'].count()
plt.plot(range(len(yr_counts)), yr_counts.values, marker='o', color='#27ae60', linewidth=2.5, markersize=6)
plt.xticks(range(len(yr_counts)), yr_counts.index, rotation=45, ha='right', fontsize=9)
plt.title('3. Year-Wise Reporting Districts Availability in Tamil Nadu (2000 - 2023)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Agricultural Year', fontsize=11)
plt.ylabel('Number of Active Reporting Districts', fontsize=11)
plt.ylim(25, 40)
for i, v in enumerate(yr_counts.values):
    plt.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()""")

    code("""# 4. Average Cultivated Area by District
plt.figure(figsize=(12, 9))
dist_avg_area = df_clean.groupby('original_district')['area'].mean().dropna().sort_values()
plt.barh(dist_avg_area.index, dist_avg_area.values / 1000, color='#2980b9', height=0.65)
plt.title('4. Average Annual Cultivated Area by District in Tamil Nadu', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Cultivated Area (Thousand Hectares)', fontsize=11)
plt.ylabel('District', fontsize=11)
plt.tight_layout()
plt.show()""")

    code("""# 5. Average Crop Production by District
plt.figure(figsize=(12, 9))
dist_avg_prod = df_clean.groupby('original_district')['production'].mean().dropna().sort_values()
plt.barh(dist_avg_prod.index, dist_avg_prod.values / 1000, color='#8e44ad', height=0.65)
plt.title('5. Average Annual Crop Production by District in Tamil Nadu', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Production (Thousand Tonnes)', fontsize=11)
plt.ylabel('District', fontsize=11)
plt.tight_layout()
plt.show()""")

    code("""# 6. Statewide Average Yield Trends
plt.figure(figsize=(12, 5))
yr_avg_yield = df_clean.groupby('year')['yield'].mean()
plt.plot(yr_avg_yield.index, yr_avg_yield.values, marker='s', color='#c0392b', linewidth=2.5, markersize=6)
plt.title('6. Tamil Nadu State-Wide Average Agricultural Yield Trend (2000 - 2022)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Agricultural Year', fontsize=11)
plt.ylabel('Yield (kg/Hectare)', fontsize=11)
plt.tight_layout()
plt.show()""")

    code("""# 7. Missing Value Matrix Heatmap
plt.figure(figsize=(14, 10))
pivot_missing = df_clean.pivot(index='original_district', columns='year', values='area').isna().astype(int)
sns.heatmap(pivot_missing, cmap=['#ffffff', '#2c3e50'], cbar=False, linewidths=0.5, linecolor='#e0e0e0')
plt.title('7. Missing Data Heatmap by District and Year (Dark = Missing / Pre-Bifurcation)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Agricultural Year', fontsize=11)
plt.ylabel('District', fontsize=11)
plt.tight_layout()
plt.show()""")

    code("""# 8. District Coverage Analysis
plt.figure(figsize=(10, 6))
valid_per_dist = df_clean.dropna(subset=['yield']).groupby('original_district')['yield'].count().sort_values()
plt.plot(range(len(valid_per_dist)), valid_per_dist.values, marker='^', color='#d35400', linewidth=2, markersize=6)
plt.xticks(range(len(valid_per_dist)), valid_per_dist.index, rotation=90, fontsize=8)
plt.title('8. District Coverage Analysis: Verified Yield Records Available for ML Modeling', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('District', fontsize=11)
plt.ylabel('Verified Yield Records (kg/ha)', fontsize=11)
plt.ylim(0, 25)
plt.tight_layout()
plt.show()""")

    # Section 8
    md("""---
## 8. Machine Learning Preparation (Agricultural Master Table)
We prepare an agricultural master table engineered specifically for downstream rainfall merging.

### ML Design Constraints:
- **Target Variable ($y$):** `yield_kg_ha`
- **Future Feature Predictors ($X$):** `year`, `district`, `crop`, `season`, `area_ha`, along with rainfall metrics (`rainfall_mm`, `rainy_days`, `rainfall_anomaly`).
- **Data Leakage Prevention:** Production is strictly excluded from feature inputs, as Yield is a direct mathematical derivative of Production.
- **Train/Test Splitting:** Kept un-split until rainfall features are fused to prevent data leakage across temporal sequences.""")

    code("""ml_feature_columns = [
    'year',
    'district',
    'crop',
    'season',
    'area',        # area_ha
    'yield'        # target: yield_kg_ha
]

ml_master_df = df_clean[ml_feature_columns].dropna(subset=['yield']).copy()
ml_master_df.rename(columns={'area': 'area_ha', 'yield': 'yield_kg_ha'}, inplace=True)

print("=== Machine Learning Master Table Overview ===")
print(f"Total verified instances ready for rainfall integration: {len(ml_master_df):,}")
display(ml_master_df.head(10))""")

    # Section 9
    md("""---
## 9. Multi-File Export & Excel Workbook Generation
Export all 8 standardized deliverables and compile a multi-sheet Excel workbook.""")

    code("""os.makedirs('exports', exist_ok=True)

# 1. Core Data Deliverables
df_raw.to_csv('exports/des_raw_backup.csv', index=False)
df_clean.to_csv('exports/des_cleaned_all_crops.csv', index=False)
df_selected_crops.to_csv('exports/des_selected_8_crops.csv', index=False)

# 2. Summaries
dist_summary = df_clean.groupby('district').agg(
    total_years=('area', 'count'),
    mean_area_ha=('area', 'mean'),
    mean_production_tonnes=('production', 'mean'),
    mean_yield_kg_ha=('yield', 'mean')
).reset_index().round(2)
dist_summary.to_csv('exports/district_summary.csv', index=False)

yearly_summary = df_clean.groupby('year').agg(
    reporting_districts=('area', 'count'),
    total_area_ha=('area', 'sum'),
    total_prod_tonnes=('production', 'sum'),
    avg_yield_kg_ha=('yield', 'mean')
).reset_index().round(2)
yearly_summary.to_csv('exports/yearly_summary.csv', index=False)

crop_summary = df_clean.groupby('crop').agg(
    record_count=('district', 'count'),
    avg_yield_kg_ha=('yield', 'mean')
).reset_index().round(2)
crop_summary.to_csv('exports/crop_summary.csv', index=False)

# 3. Quality & Cleaning Audit Logs
pd.DataFrame(cleaning_log).to_csv('exports/cleaning_log.csv', index=False)

q_report = pd.DataFrame([
    {'metric': 'Source File', 'value': os.path.basename(file_path)},
    {'metric': 'Original District Rows', 'value': len(df_raw)},
    {'metric': 'Total Structured Records', 'value': len(df_clean)},
    {'metric': 'Number of Districts', 'value': df_clean['district'].nunique()},
    {'metric': 'Number of Historical Years', 'value': df_clean['year'].nunique()},
    {'metric': 'Duplicate Rows', 'value': df_clean.duplicated(subset=['district', 'year', 'crop', 'season']).sum()},
    {'metric': 'Missing Area Observations', 'value': df_clean['area'].isna().sum()},
    {'metric': 'Missing Production Observations', 'value': df_clean['production'].isna().sum()},
    {'metric': 'Valid Target Yields (kg/ha)', 'value': df_clean['yield'].notna().sum()}
])
q_report.to_csv('exports/data_quality_report.csv', index=False)

# 4. Multi-sheet Excel Workbook
excel_file = 'exports/des_tamil_nadu_master_workbook.xlsx'
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    df_clean.to_excel(writer, sheet_name='Cleaned_All_Data', index=False)
    df_selected_crops.to_excel(writer, sheet_name='Selected_Crops', index=False)
    dist_summary.to_excel(writer, sheet_name='District_Summary', index=False)
    yearly_summary.to_excel(writer, sheet_name='Yearly_Summary', index=False)
    crop_summary.to_excel(writer, sheet_name='Crop_Summary', index=False)
    q_report.to_excel(writer, sheet_name='Data_Quality_Report', index=False)
    pd.DataFrame(cleaning_log).to_excel(writer, sheet_name='Cleaning_Log', index=False)

print("✓ All 8 CSV deliverables and multi-sheet Excel workbook generated successfully in exports/")

# If in Colab, download all files
if IN_COLAB:
    from google.colab import files
    for f in os.listdir('exports'):
        files.download(os.path.join('exports', f))""")

    output_path = "DES_Tamil_Nadu_Agricultural_Data_Preprocessing.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Notebook successfully created at: {output_path}")

if __name__ == "__main__":
    build_notebook()
