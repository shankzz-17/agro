"""
Publication-Quality EDA Chart Generator for AgroVision AI
Author: Mehul Shankar P (AgroVision Project)

Saves analytical figures into reports/figures/:
1. District-wise average yield comparison.
2. 23-Year statewide yield trend (2000-2022).
3. Cultivated Area vs. Yield scatter relationship.
4. Missing values audit & bifurcation timeline.
5. Correlation matrix of agricultural features.
6. Model benchmark comparison (Train vs Validation error).
7. Rainfall integration roadmap architecture.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['MPLCONFIGDIR'] = os.path.join(PROJECT_ROOT, "data", ".mpl")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def generate_all_figures():
    output_dir = os.path.join("reports", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load cleaned dataset
    data_path = os.path.join("data", "processed", "des_cleaned_all_crops.csv")
    df = pd.read_csv(data_path)
    clean_df = df.dropna(subset=['yield']).copy()
    
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})
    
    # ----------------------------------------------------
    # 1. District-wise Average Yield Comparison
    # ----------------------------------------------------
    print("[1/7] Generating District Yield Comparison...")
    dist_avg = clean_df.groupby('district')['yield'].mean().sort_values(ascending=False).reset_index()
    state_mean = clean_df['yield'].mean()
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(dist_avg['district'].str.replace('_', ' ').str.title(), dist_avg['yield'], color='#2E7D32', alpha=0.85)
    plt.axvline(state_mean, color='#D32F2F', linestyle='--', linewidth=2, label=f'Statewide Mean ({state_mean:,.0f} kg/ha)')
    plt.title('Tamil Nadu: Average Agricultural Yield by District (2000–2022)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Average Yield (kg/hectare)', fontsize=12)
    plt.ylabel('District', fontsize=12)
    plt.gca().invert_yaxis()
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_district_yield_comparison.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # 2. 23-Year Historical Yield Trend (2000-2022)
    # ----------------------------------------------------
    print("[2/7] Generating Statewide Historical Yield Trend...")
    yearly = clean_df.groupby('year')['yield'].agg(['mean', 'std', 'count']).reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(yearly['year'], yearly['mean'], marker='o', color='#1565C0', linewidth=2.5, label='Mean Yield (kg/ha)')
    plt.fill_between(
        yearly['year'],
        yearly['mean'] - (yearly['std'] / np.sqrt(yearly['count'])),
        yearly['mean'] + (yearly['std'] / np.sqrt(yearly['count'])),
        color='#90CAF9',
        alpha=0.4,
        label='Standard Error Band'
    )
    plt.title('Tamil Nadu Statewide Yield Trend Over 23 Years (2000–2022)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Agricultural Year', fontsize=12)
    plt.ylabel('Mean Yield (kg/hectare)', fontsize=12)
    plt.xticks(yearly['year'], rotation=45)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_yearly_yield_trend.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # 3. Cultivated Area vs. Yield Relationship
    # ----------------------------------------------------
    print("[3/7] Generating Area vs. Yield Scatter Analysis...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=clean_df,
        x='area',
        y='yield',
        alpha=0.6,
        color='#00897B',
        edgecolor='w',
        s=60
    )
    # Add trend line
    sns.regplot(
        data=clean_df,
        x='area',
        y='yield',
        scatter=False,
        color='#D81B60',
        line_kws={'linewidth': 2, 'label': 'Linear Trendline'}
    )
    plt.title('Relationship Between Cultivated Area (ha) and Crop Yield (kg/ha)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Cultivated Area (Hectares)', fontsize=12)
    plt.ylabel('Yield (kg/hectare)', fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_area_vs_yield_scatter.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # 4. Missing Values Audit & Bifurcation Timeline
    # ----------------------------------------------------
    print("[4/7] Generating Missing Values Audit Chart...")
    missing_by_year = df.groupby('year')['yield'].apply(lambda s: s.isna().sum()).reset_index()
    
    plt.figure(figsize=(12, 5))
    plt.bar(missing_by_year['year'], missing_by_year['yield'], color='#FB8C00', alpha=0.85, edgecolor='black', linewidth=0.5)
    plt.title('Count of Missing Yield Observations per Year Across 38 Districts', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Missing Records Count', fontsize=12)
    plt.xticks(missing_by_year['year'], rotation=45)
    plt.annotate(
        'Missing records primarily represent newly bifurcated districts\n(Tenkasi, Chengalpattu, Kallakurichi, Mayiladuthurai, etc.)\nbefore official government establishment',
        xy=(2008, 7),
        xytext=(2004, 11),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="orange", lw=1)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_missing_values_audit.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # 5. Correlation Heatmap
    # ----------------------------------------------------
    print("[5/7] Generating Correlation Heatmap...")
    numeric_cols = ['year', 'area', 'production', 'yield']
    corr_matrix = clean_df[numeric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Pearson Correlation (r)'}
    )
    plt.title('Correlation Matrix of Agricultural Numerical Variables', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # 6. Model Comparison Chart (Train vs. Validation Error)
    # ----------------------------------------------------
    print("[6/7] Generating Model Comparison Chart...")
    comp_path = os.path.join("reports", "model_comparison.csv")
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(df_comp))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, df_comp['train_rmse_kg_ha'], width, label='Train RMSE', color='#42A5F5')
        rects2 = ax.bar(x + width/2, df_comp['val_rmse_kg_ha'], width, label='Validation RMSE', color='#EF5350')
        
        ax.set_ylabel('RMSE (kg/hectare)', fontsize=12)
        ax.set_title('Model Benchmark: Train vs. Chronological Validation RMSE', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(df_comp['model_name'], fontsize=11)
        ax.legend(frameon=True)
        
        # Add labels on top of bars
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:,.0f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:,.0f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '06_model_performance_comparison.png'), dpi=300)
        plt.close()
        
    # ----------------------------------------------------
    # 7. Rainfall Integration Conceptual Diagram
    # ----------------------------------------------------
    print("[7/7] Generating Rainfall Integration Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis('off')
    
    box_blue = dict(boxstyle="square,pad=0.6", fc="#E3F2FD", ec="#1E88E5", lw=2)
    box_green = dict(boxstyle="square,pad=0.6", fc="#E8F5E9", ec="#43A047", lw=2)
    box_purple = dict(boxstyle="square,pad=0.6", fc="#F3E5F5", ec="#8E24AA", lw=2)
    box_gold = dict(boxstyle="square,pad=0.6", fc="#FFF8E1", ec="#FBC02D", lw=2)
    
    ax.text(0.12, 0.75, "DES Agricultural Base\n- District (38)\n- Year (2000-2022)\n- Cultivated Area (ha)", ha="center", va="center", bbox=box_blue, fontsize=10)
    ax.text(0.12, 0.25, "Weather / Rainfall Feed\n- Seasonal Rainfall (mm)\n- Rainy Days Count\n- Rainfall Anomaly (%)", ha="center", va="center", bbox=box_green, fontsize=10)
    ax.text(0.50, 0.50, "AgroVision Feature Fusion\n- Chronological Split\n- District One-Hot Encoding\n- Standard Scaler (No Leakage)", ha="center", va="center", bbox=box_purple, fontsize=10)
    ax.text(0.88, 0.50, "Yield Estimation\n- Predicted Yield (kg/ha)\n- Regional Harvest (tonnes)\n- Impact Interpretability", ha="center", va="center", bbox=box_gold, fontsize=10)
    
    # Arrows
    ax.annotate('', xy=(0.33, 0.65), xytext=(0.25, 0.75), arrowprops=dict(arrowstyle="->", lw=2, color="#1E88E5"))
    ax.annotate('', xy=(0.33, 0.35), xytext=(0.25, 0.25), arrowprops=dict(arrowstyle="->", lw=2, color="#43A047"))
    ax.annotate('', xy=(0.73, 0.50), xytext=(0.67, 0.50), arrowprops=dict(arrowstyle="->", lw=2, color="#8E24AA"))
    
    plt.title("AgroVision AI: Modular Dual-Stream System Architecture", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_rainfall_integration_architecture.png'), dpi=300)
    plt.close()
    
    print("✓ All 7 EDA figures successfully generated in reports/figures/")


if __name__ == "__main__":
    generate_all_figures()
