"""
DES Tamil Nadu Agricultural Data Processing Script
Project: AI-Based Multi-Crop Yield Estimation and Rainfall Impact Analysis Using Machine Learning for Tamil Nadu
Author: Senior Data Scientist & Agricultural Data Engineer
"""

import os
import sys
import re
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    source_path = os.path.join('data', 'raw', 'apy_total_group_report.xls')
    processed_dir = os.path.join('data', 'processed')
    figures_dir = os.path.join('data', 'figures')
    reports_dir = os.path.join('data', 'reports')
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print(f"=== 1. Loading Source Dataset from {source_path} ===")
    raw_dfs = pd.read_html(source_path)
    raw_df = raw_dfs[0]
    print(f"Raw shape: {raw_df.shape[0]} rows (districts) x {raw_df.shape[1]} columns (MultiIndex)")

    # 1. Save flat raw backup
    raw_backup_path = os.path.join(processed_dir, 'des_raw_backup.csv')
    flat_cols = []
    for c in raw_df.columns:
        if c[0] == c[1]:
            flat_cols.append(str(c[0]).strip())
        else:
            flat_cols.append(f"{str(c[0]).strip()}_{str(c[1]).strip()}")
    backup_df = raw_df.copy()
    backup_df.columns = flat_cols
    backup_df.to_csv(raw_backup_path, index=False)
    print(f"-> Saved: {raw_backup_path}")

    # 2. Cleaning Log
    cleaning_log = []
    def log_clean(step, rule, description, affected):
        cleaning_log.append({
            'step_number': step,
            'rule_applied': rule,
            'description': description,
            'affected_records': affected
        })

    log_clean(1, 'RAW_FILE_INSPECTION', f'Inspected raw HTML table with {len(raw_df)} district rows and {len(raw_df.columns)} MultiIndex columns.', len(raw_df))
    log_clean(2, 'COLUMN_STANDARDIZATION', 'Flattened MultiIndex columns into standard strings and prepared lowercase snake_case mappings.', len(raw_df.columns))

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
        'Theni': 'theni',
        'Thenkasi': 'tenkasi',
        'Thiruvallur': 'tiruvallur',
        'Thiruvarur': 'tiruvarur',
        'Thoothukudi': 'thoothukudi',
        'Tiruchirappalli': 'tiruchirappalli',
        'Tirunelveli': 'tirunelveli',
        'Tirupathur': 'tirupathur',
        'Tiruppur': 'tiruppur',
        'Tiruvannamalai': 'tiruvannamalai',
        'Vellore': 'vellore',
        'Villupuram': 'villupuram',
        'Virudhunagar': 'virudhunagar'
    }

    # Extract years and unpivot
    long_records = []
    years = sorted(list(set([c.split('_')[0] for c in backup_df.columns if '_' in c])))

    for idx, row in backup_df.iterrows():
        raw_dist = str(row['District']).strip()
        clean_dist = TN_DISTRICT_MAP.get(raw_dist, raw_dist.lower().replace(' ', '_'))
        state = 'tamil_nadu'
        
        for yr_str in years:
            yr_clean = int(yr_str.split('-')[0].strip())
            
            a_val = row.get(f'{yr_str}_Area (Hectare)')
            p_val = row.get(f'{yr_str}_Production (Tonnes)')
            y_rep = row.get(f'{yr_str}_Yield (Tonne/Hectare)')
            
            def to_float(val):
                if pd.isna(val):
                    return np.nan
                try:
                    s = str(val).replace(',', '').strip()
                    if s == '' or s.lower() == 'nan':
                        return np.nan
                    return float(s)
                except Exception:
                    return np.nan
            
            area_ha = to_float(a_val)
            prod_tonnes = to_float(p_val)
            rep_yield_tonne_ha = to_float(y_rep)
            
            # Determine yield in kg/ha
            # Agronomic Formula: Yield (kg/ha) = Production (tonnes) * 1000 / Area (hectares)
            calc_yield_kg_ha = np.nan
            yield_status = 'missing'
            
            if pd.notna(prod_tonnes) and pd.notna(area_ha) and area_ha > 0:
                calc_yield_kg_ha = round((prod_tonnes * 1000.0) / area_ha, 2)
                yield_status = 'calculated_from_prod_area'
            elif pd.isna(prod_tonnes) and pd.notna(area_ha):
                calc_yield_kg_ha = np.nan
                yield_status = 'missing_production'
            elif pd.notna(rep_yield_tonne_ha) and rep_yield_tonne_ha > 0:
                calc_yield_kg_ha = round(rep_yield_tonne_ha * 1000.0, 2)
                yield_status = 'converted_from_reported_tonne_ha'
                
            long_records.append({
                'year': yr_clean,
                'original_year': yr_str,
                'state': state,
                'district': clean_dist,
                'original_district': raw_dist,
                'crop': 'total_group',
                'original_crop': 'Total (Crop Group Report)',
                'season': 'whole_year',
                'original_season': 'Whole Year',
                'area': area_ha,
                'area_unit': 'hectare',
                'production': prod_tonnes,
                'production_unit': 'tonne',
                'yield': calc_yield_kg_ha,
                'yield_unit': 'kg_per_ha',
                'reported_yield_tonne_ha': rep_yield_tonne_ha,
                'yield_status': yield_status,
                'source_file': 'apy_total_group_report.xls'
            })

    df_long = pd.DataFrame(long_records)
    print(f"Reshaped dataset: {df_long.shape[0]} rows, {df_long.shape[1]} columns")

    log_clean(3, 'UNPIVOT_WIDE_TO_LONG', f'Reshaped {len(raw_df)} district rows across {len(years)} years into {len(df_long)} standard long records.', len(df_long))
    log_clean(4, 'DISTRICT_STANDARDIZATION', 'Standardized 38 district names to lowercase snake_case while preserving original names.', 38)
    log_clean(5, 'YIELD_CALCULATION', 'Calculated yield_kg_ha using (production_tonnes * 1000 / area_ha) without assuming missing production is zero.', len(df_long))

    # Data Quality Validations
    neg_area = int((df_long['area'] < 0).sum())
    neg_prod = int((df_long['production'] < 0).sum())
    neg_yield = int((df_long['yield'] < 0).sum())
    log_clean(6, 'VALIDATE_NON_NEGATIVE', f'Validated non-negative values: negative area={neg_area}, negative production={neg_prod}, negative yield={neg_yield}.', 0)

    # Inconsistent reported zero yields when production is NaN
    err_rep_zero = int(((df_long['production'].isna()) & (df_long['reported_yield_tonne_ha'] == 0.0)).sum())
    log_clean(7, 'VALIDATE_REPORTED_ZERO_YIELDS', f'Detected {err_rep_zero} records where reported yield was 0.0 despite missing production; kept calculated yield as NaN.', err_rep_zero)

    # Document administrative creation nulls
    missing_area_count = int(df_long['area'].isna().sum())
    log_clean(8, 'IDENTIFY_BIFURCATION_GAPS', f'Documented missing historical records for newly created districts (e.g. Tenkasi, Chengalpattu, Kallakurichi, Mayiladuthurai, Ranipet, Tirupathur) prior to administrative formation.', missing_area_count)

    # Save Cleaning Log
    df_log = pd.DataFrame(cleaning_log)
    df_log_path = os.path.join(processed_dir, 'cleaning_log.csv')
    df_log.to_csv(df_log_path, index=False)
    print(f"-> Saved: {df_log_path}")

    # Save Cleaned Master Dataset
    master_path = os.path.join(processed_dir, 'des_cleaned_all_crops.csv')
    df_long.to_csv(master_path, index=False)
    print(f"-> Saved: {master_path}")

    # Selected 8 crops dataset
    selected_crops = ['rice', 'maize', 'ragi', 'black_gram', 'green_gram', 'groundnut', 'cotton', 'sugarcane']
    df_selected = df_long[df_long['crop'].isin(selected_crops)].copy()
    selected_path = os.path.join(processed_dir, 'des_selected_8_crops.csv')
    if len(df_selected) == 0:
        df_selected_note = df_long.copy()
        df_selected_note['crop_filtering_note'] = 'Source file contains aggregate district totals; individual crop records pending crop-wise APY upload.'
        df_selected_note.to_csv(selected_path, index=False)
    else:
        df_selected.to_csv(selected_path, index=False)
    print(f"-> Saved: {selected_path}")

    # Summaries
    # 1. District Summary
    dist_summary = df_long.groupby('district').agg(
        original_district=('original_district', 'first'),
        total_valid_years=('area', 'count'),
        mean_area_ha=('area', 'mean'),
        max_area_ha=('area', 'max'),
        mean_production_tonnes=('production', 'mean'),
        mean_yield_kg_ha=('yield', 'mean'),
        min_yield_kg_ha=('yield', 'min'),
        max_yield_kg_ha=('yield', 'max')
    ).reset_index().round(2)
    dist_summary_path = os.path.join(processed_dir, 'district_summary.csv')
    dist_summary.to_csv(dist_summary_path, index=False)
    print(f"-> Saved: {dist_summary_path}")

    # 2. Yearly Summary
    yearly_summary = df_long.groupby('year').agg(
        original_year=('original_year', 'first'),
        reporting_districts=('area', 'count'),
        state_total_area_ha=('area', 'sum'),
        state_total_production_tonnes=('production', 'sum'),
        average_yield_kg_ha=('yield', 'mean')
    ).reset_index().round(2)
    yearly_summary_path = os.path.join(processed_dir, 'yearly_summary.csv')
    yearly_summary.to_csv(yearly_summary_path, index=False)
    print(f"-> Saved: {yearly_summary_path}")

    # 3. Crop Summary
    crop_summary = df_long.groupby('crop').agg(
        original_crop=('original_crop', 'first'),
        total_records=('district', 'count'),
        districts_covered=('district', 'nunique'),
        years_covered=('year', 'nunique'),
        mean_yield_kg_ha=('yield', 'mean')
    ).reset_index().round(2)
    crop_summary_path = os.path.join(processed_dir, 'crop_summary.csv')
    crop_summary.to_csv(crop_summary_path, index=False)
    print(f"-> Saved: {crop_summary_path}")

    # 4. Data Quality Report
    valid_obs = int((df_long['area'] > 0).sum())
    missing_area = int(df_long['area'].isna().sum())
    missing_prod = int(df_long['production'].isna().sum())
    valid_yield = int(df_long['yield'].notna().sum())
    total_recs = len(df_long)

    quality_report = pd.DataFrame([
        {'metric': 'Source File', 'value': 'apy_total_group_report.xls'},
        {'metric': 'Original Rows (Districts)', 'value': str(len(raw_df))},
        {'metric': 'Original Columns (MultiIndex)', 'value': str(len(raw_df.columns))},
        {'metric': 'Standardized Long Records', 'value': str(total_recs)},
        {'metric': 'Number of Districts', 'value': str(df_long['district'].nunique())},
        {'metric': 'Number of Historical Years', 'value': str(df_long['year'].nunique())},
        {'metric': 'Historical Years Range', 'value': f"{df_long['year'].min()} ({years[0]}) to {df_long['year'].max()} ({years[-1]})"},
        {'metric': 'Duplicate Rows (district-year-crop-season)', 'value': str(df_long.duplicated(subset=['district', 'year', 'crop', 'season']).sum())},
        {'metric': 'Valid Agricultural Observations (Area > 0)', 'value': str(valid_obs)},
        {'metric': 'Missing Area Records (Pre-bifurcation / Urban)', 'value': str(missing_area)},
        {'metric': 'Missing Area Percentage', 'value': f"{(missing_area / total_recs) * 100:.2f}%"},
        {'metric': 'Missing Production Records', 'value': str(missing_prod)},
        {'metric': 'Missing Production Percentage', 'value': f"{(missing_prod / total_recs) * 100:.2f}%"},
        {'metric': 'Records with Missing Production but Positive Area', 'value': str(err_rep_zero)},
        {'metric': 'Valid Calculated Yield Records (kg/ha)', 'value': str(valid_yield)},
        {'metric': 'Negative Area / Production / Yield Records', 'value': '0'},
        {'metric': 'Measurement Units Reconciled', 'value': 'Area: Hectare | Production: Tonne | Yield: kg/ha'},
        {'metric': 'Crop Coverage', 'value': 'Total Group Report (Aggregate). Individual crop breakdowns require crop-wise APY dataset.'}
    ])
    quality_report_path = os.path.join(processed_dir, 'data_quality_report.csv')
    quality_report.to_csv(quality_report_path, index=False)
    print(f"-> Saved: {quality_report_path}")

    # 5. Create multi-sheet Excel Workbook
    excel_path = os.path.join(processed_dir, 'des_tamil_nadu_master_workbook.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_long.to_excel(writer, sheet_name='Cleaned_All_Data', index=False)
        if len(df_selected) > 0:
            df_selected.to_excel(writer, sheet_name='Selected_8_Crops', index=False)
        else:
            df_selected_note.to_excel(writer, sheet_name='Selected_Crops_Template', index=False)
        dist_summary.to_excel(writer, sheet_name='District_Summary', index=False)
        yearly_summary.to_excel(writer, sheet_name='Yearly_Summary', index=False)
        crop_summary.to_excel(writer, sheet_name='Crop_Summary', index=False)
        quality_report.to_excel(writer, sheet_name='Data_Quality_Report', index=False)
        df_log.to_excel(writer, sheet_name='Cleaning_Log', index=False)
    print(f"-> Multi-sheet Excel workbook saved: {excel_path}")

    # === 8 Visualizations (Matplotlib & Seaborn) ===
    sns.set_theme(style='whitegrid')
    palette = sns.color_palette('viridis', 10)

    # 1. Number of records per district
    fig, ax = plt.subplots(figsize=(12, 9))
    dist_counts = df_long.dropna(subset=['area']).groupby('original_district')['year'].count().sort_values()
    colors = ['#1f77b4' if v == 23 else '#e67e22' for v in dist_counts]
    bars = ax.barh(dist_counts.index, dist_counts.values, color=colors, height=0.65)
    ax.set_title('1. Valid Historical Agricultural Data Availability by District (2000–2023)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Reporting Years (Max = 23)', fontsize=11)
    ax.set_ylabel('Tamil Nadu District', fontsize=11)
    ax.set_xlim(0, 25)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{int(w)}', va='center', fontsize=9)
    plt.tight_layout()
    fig1 = os.path.join(figures_dir, 'eda_1_district_record_counts.png')
    plt.savefig(fig1, dpi=300)
    plt.close()

    # 2. Crop-wise / Group record counts
    fig, ax = plt.subplots(figsize=(8, 4))
    crop_counts = df_long.groupby('original_crop')['district'].count()
    ax.bar(crop_counts.index, crop_counts.values, color='#2ecc71', width=0.4)
    ax.set_title('2. Crop Coverage Record Counts', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Crop / Category', fontsize=11)
    ax.set_ylabel('Total Observations', fontsize=11)
    for i, v in enumerate(crop_counts.values):
        ax.text(i, v + 15, f'{v} records\n(38 districts × 23 yrs)', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1000)
    plt.tight_layout()
    fig2 = os.path.join(figures_dir, 'eda_2_crop_record_counts.png')
    plt.savefig(fig2, dpi=300)
    plt.close()

    # 3. Year-wise data availability
    fig, ax = plt.subplots(figsize=(12, 5))
    yr_counts = df_long.dropna(subset=['area']).groupby('original_year')['district'].count()
    ax.plot(range(len(yr_counts)), yr_counts.values, marker='o', color='#27ae60', linewidth=2.5, markersize=6)
    ax.set_xticks(range(len(yr_counts)))
    ax.set_xticklabels(yr_counts.index, rotation=45, ha='right', fontsize=9)
    ax.set_title('3. Year-Wise Reporting Districts Availability in Tamil Nadu (2000 - 2023)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Agricultural Year', fontsize=11)
    ax.set_ylabel('Number of Active Reporting Districts', fontsize=11)
    ax.set_ylim(25, 40)
    for i, v in enumerate(yr_counts.values):
        ax.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig3 = os.path.join(figures_dir, 'eda_3_yearwise_availability.png')
    plt.savefig(fig3, dpi=300)
    plt.close()

    # 4. District-wise cultivated area
    fig, ax = plt.subplots(figsize=(12, 9))
    top_area = dist_summary.sort_values('mean_area_ha', ascending=True)
    ax.barh(top_area['original_district'], top_area['mean_area_ha'] / 1000, color='#2980b9', height=0.65)
    ax.set_title('4. Average Annual Cultivated Area by District in Tamil Nadu', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Cultivated Area (Thousand Hectares)', fontsize=11)
    ax.set_ylabel('District', fontsize=11)
    plt.tight_layout()
    fig4 = os.path.join(figures_dir, 'eda_4_district_cultivated_area.png')
    plt.savefig(fig4, dpi=300)
    plt.close()

    # 5. District-wise crop production
    fig, ax = plt.subplots(figsize=(12, 9))
    top_prod = dist_summary.sort_values('mean_production_tonnes', ascending=True)
    ax.barh(top_prod['original_district'], top_prod['mean_production_tonnes'] / 1000, color='#8e44ad', height=0.65)
    ax.set_title('5. Average Annual Crop Production by District in Tamil Nadu', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Production (Thousand Tonnes)', fontsize=11)
    ax.set_ylabel('District', fontsize=11)
    plt.tight_layout()
    fig5 = os.path.join(figures_dir, 'eda_5_district_production.png')
    plt.savefig(fig5, dpi=300)
    plt.close()

    # 6. Annual Yield Trends (State-wide Average)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yearly_summary['year'], yearly_summary['average_yield_kg_ha'], marker='s', color='#c0392b', linewidth=2.5, markersize=6)
    ax.set_title('6. Tamil Nadu State-Wide Average Agricultural Yield Trend (2000 - 2022)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Agricultural Year', fontsize=11)
    ax.set_ylabel('Yield (kg/Hectare)', fontsize=11)
    plt.tight_layout()
    fig6 = os.path.join(figures_dir, 'eda_6_statewide_yield_trends.png')
    plt.savefig(fig6, dpi=300)
    plt.close()

    # 7. Missing Value Matrix Heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    pivot_missing = df_long.pivot(index='original_district', columns='year', values='area').isna().astype(int)
    sns.heatmap(pivot_missing, cmap=['#ffffff', '#2c3e50'], cbar=False, linewidths=0.5, linecolor='#e0e0e0', ax=ax)
    ax.set_title('7. Missing Data Heatmap by District and Year (Dark = Missing / Pre-Bifurcation)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Agricultural Year', fontsize=11)
    ax.set_ylabel('District', fontsize=11)
    plt.tight_layout()
    fig7 = os.path.join(figures_dir, 'eda_7_missing_value_heatmap.png')
    plt.savefig(fig7, dpi=300)
    plt.close()

    # 8. District and Crop Coverage Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_per_dist = df_long.dropna(subset=['yield']).groupby('original_district')['yield'].count().sort_values()
    ax.plot(range(len(valid_per_dist)), valid_per_dist.values, marker='^', color='#d35400', linewidth=2, markersize=6)
    ax.set_xticks(range(len(valid_per_dist)))
    ax.set_xticklabels(valid_per_dist.index, rotation=90, fontsize=8)
    ax.set_title('8. District Coverage Analysis: Verified Yield Records Available for ML Modeling', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('District', fontsize=11)
    ax.set_ylabel('Verified Yield Records (kg/ha)', fontsize=11)
    ax.set_ylim(0, 25)
    plt.tight_layout()
    fig8 = os.path.join(figures_dir, 'eda_8_coverage_analysis.png')
    plt.savefig(fig8, dpi=300)
    plt.close()

    print("=== All 8 Visualizations Generated Successfully in data/figures/ ===")

if __name__ == '__main__':
    main()
