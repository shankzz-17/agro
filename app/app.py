"""
VayalYield AI - Streamlit Dashboard
Project: AI-Based Multi-Crop Yield Estimation and Rainfall Impact Analysis Using Machine Learning for Tamil Nadu
Author: Mehul Shankar P (AgroVision Project)
"""

import os
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.predict import AgroYieldPredictor, predict_yield

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="VayalYield AI - Tamil Nadu Crop Yield Estimator",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean college-ready presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_app_data():
    """Load pre-processed datasets for the dashboard."""
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "des_cleaned_all_crops.csv")
    df = pd.read_csv(data_path)
    clean_df = df.dropna(subset=['yield']).copy()
    
    comp_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison.csv")
    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else None
    
    meta_path = os.path.join(PROJECT_ROOT, "models", "feature_metadata.json")
    metadata = None
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
    return df, clean_df, comp_df, metadata


df_raw, df_clean, df_comparison, metadata = load_app_data()


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/wheat.png", width=70)
st.sidebar.title("VayalYield AI")
st.sidebar.caption("Tamil Nadu Agricultural ML Platform")

menu_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "📈 Exploratory Analysis",
        "🔮 Yield Prediction Simulator",
        "🌧️ Rainfall Impact Analysis",
        "🏆 Model Performance",
        "ℹ️ About & Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** AgroVision")
st.sidebar.markdown("**Scope:** 38 Districts, Tamil Nadu")
st.sidebar.markdown("**Author:** Mehul Shankar P (CSE)")


# ==============================================================================
# 1. HOME PAGE
# ==============================================================================
if menu_choice == "🏠 Home":
    st.markdown('<div class="main-title">🌾 VayalYield AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Based Multi-Crop Yield Estimation and Rainfall Impact Analysis Using Machine Learning for Tamil Nadu</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Geographic Scope", "Tamil Nadu, India")
    with col2:
        st.metric("Districts Covered", "38 Administrative Districts")
    with col3:
        st.metric("Temporal Timeline", "2000 – 2022 (23 Years)")
    with col4:
        st.metric("Verified Records", f"{len(df_clean):,} Observations")
        
    st.markdown("---")
    
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("📌 Project Motivation & Objectives")
        st.write(
            """
            Agriculture is the backbone of Tamil Nadu's rural economy, yet crop yield is subject to severe 
            fluctuations caused by changing monsoon patterns, regional soil variations, and acreage shifts. 
            Accurate, pre-harvest yield estimation is crucial for:
            
            - **District Agricultural Planners:** Allocating storage, procurement centers, and minimum support prices.
            - **Farmers & Producer Organizations:** Optimizing input investments and crop insurance coverage.
            - **Disaster Response Teams:** Formulating proactive drought and flood relief measures.
            """
        )
        st.markdown("""
        <div class="info-box">
            <b>Core ML Formulation:</b> Predict crop yield in <b>kg/hectare</b> using strictly valid features 
            (District, Historical Year, Cultivated Area in hectares) while enforcing 
            <b>Zero Data Leakage</b> (production is never consumed as an input because yield is mathematically derived from production).
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        st.subheader("💡 Key Architectural Strengths")
        st.success("✅ **Chronological Splitting:** Trained on 2000–2016, validated on 2017–2019, tested on 2020–2022.")
        st.success("✅ **Leakage-Free Preprocessing:** Standardized with scikit-learn ColumnTransformer fitted only on train set.")
        st.success("✅ **Multi-Model Benchmark:** Evaluates Linear Regression, Random Forest, and XGBoost.")
        st.success("✅ **Modular Dual-Stream Design:** Ready to fuse IMD/satellite rainfall metrics seamlessly.")


# ==============================================================================
# 2. DATASET OVERVIEW
# ==============================================================================
elif menu_choice == "📊 Dataset Overview":
    st.title("📊 Dataset Inspection & Quality Audit")
    st.write("Source: Directorate of Economics and Statistics (DES) APY Portal, Tamil Nadu.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df_raw))
    c2.metric("Valid Yield Records", len(df_clean))
    c3.metric("Statewide Mean Yield", f"{df_clean['yield'].mean():,.1f} kg/ha")
    c4.metric("Mean Cultivated Area", f"{df_clean['area'].mean():,.0f} ha")
    
    tab1, tab2, tab3 = st.tabs(["📋 Clean Agricultural Data", "🏛️ District Summary", "📅 Yearly Summary"])
    
    with tab1:
        st.write("Displaying verified records without missing values or data leakage:")
        st.dataframe(
            df_clean[['year', 'district', 'crop', 'season', 'area', 'yield', 'yield_status']].head(100),
            use_container_width=True
        )
        
    with tab2:
        dist_summary = df_clean.groupby('district').agg(
            valid_years=('year', 'count'),
            avg_area_ha=('area', 'mean'),
            avg_yield_kg_ha=('yield', 'mean'),
            max_yield_kg_ha=('yield', 'max')
        ).reset_index().round(1)
        st.dataframe(dist_summary, use_container_width=True)
        
    with tab3:
        yearly_summary = df_clean.groupby('year').agg(
            districts_reporting=('district', 'count'),
            total_state_area_ha=('area', 'sum'),
            avg_state_yield_kg_ha=('yield', 'mean')
        ).reset_index().round(1)
        st.dataframe(yearly_summary, use_container_width=True)


# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ==============================================================================
elif menu_choice == "📈 Exploratory Analysis":
    st.title("📈 Interactive Exploratory Data Analysis")
    st.write("Gain interactive agronomic insights across districts, years, and cultivated area.")
    
    chart_view = st.selectbox(
        "Select Visualization Angle:",
        [
            "1. District-wise Yield Comparison (All 38 Districts)",
            "2. 23-Year Statewide Historical Yield Trend (2000-2022)",
            "3. Cultivated Area vs. Crop Yield Relationship",
            "4. Missing Observations & District Bifurcation Audit"
        ]
    )
    
    if "1. District-wise Yield" in chart_view:
        st.subheader("Tamil Nadu District Agricultural Productivity Ranking")
        dist_agg = df_clean.groupby('district')['yield'].mean().sort_values(ascending=False).reset_index()
        dist_agg['district_title'] = dist_agg['district'].str.replace('_', ' ').str.title()
        
        fig = px.bar(
            dist_agg,
            x='yield',
            y='district_title',
            orientation='h',
            color='yield',
            color_continuous_scale='Greens',
            labels={'yield': 'Average Yield (kg/ha)', 'district_title': 'District'},
            title="Average Crop Yield Across 38 Tamil Nadu Districts (2000–2022)"
        )
        fig.add_vline(x=df_clean['yield'].mean(), line_dash="dash", line_color="red", annotation_text="State Mean")
        fig.update_layout(height=800, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Insight:** Delta districts like Thanjavur and Tiruvarur along with Western districts consistently demonstrate higher average yields due to established river canal networks and intensive cultivation.")
        
    elif "2. 23-Year Statewide" in chart_view:
        st.subheader("23-Year Historical Productivity Trajectory")
        yearly_agg = df_clean.groupby('year')['yield'].agg(['mean', 'median', 'std']).reset_index()
        
        fig = px.line(
            yearly_agg,
            x='year',
            y='mean',
            markers=True,
            title="Statewide Average Crop Yield Over 23 Years (2000–2022)",
            labels={'mean': 'Mean Yield (kg/ha)', 'year': 'Agricultural Year'}
        )
        fig.add_trace(go.Scatter(
            x=yearly_agg['year'],
            y=yearly_agg['median'],
            mode='lines',
            line=dict(dash='dot', color='orange'),
            name='Median Yield'
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Insight:** The upward trajectory over 23 years reflects technological progress in hybrid seeds, drip irrigation adoption, and mechanization, interspersed with sharp dips during severe drought years (e.g., 2002–03 and 2016–17).")
        
    elif "3. Cultivated Area vs. Crop Yield" in chart_view:
        st.subheader("Acreage Scale vs. Land Productivity")
        fig = px.scatter(
            df_clean,
            x='area',
            y='yield',
            color='district',
            hover_data=['year', 'district'],
            title="Cultivated Area (Hectares) vs. Crop Yield (kg/ha)",
            labels={'area': 'Cultivated Area (ha)', 'yield': 'Yield (kg/ha)'}
        )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Insight:** Yield is non-linearly distributed across acreage sizes. High cultivated area does not automatically guarantee high yield per hectare, demonstrating that farm management practices and water availability dominate over sheer land scale.")
        
    elif "4. Missing Observations" in chart_view:
        st.subheader("Data Quality & District Formation History")
        missing_by_year = df_raw.groupby('year')['yield'].apply(lambda s: s.isna().sum()).reset_index()
        
        fig = px.bar(
            missing_by_year,
            x='year',
            y='yield',
            title="Count of Missing Yield Records Per Year (Total = 158)",
            labels={'yield': 'Missing Records Count', 'year': 'Year'},
            color='yield',
            color_continuous_scale='Oranges'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.warning("⚠️ **Data Governance Note:** Missing records prior to 2019 are due to administrative bifurcation. For instance, Tenkasi was bifurcated from Tirunelveli, and Chengalpattu from Kanchipuram in 2019; prior to this date, records were recorded under parent districts.")


# ==============================================================================
# 4. YIELD PREDICTION SIMULATOR
# ==============================================================================
elif menu_choice == "🔮 Yield Prediction Simulator":
    st.title("🔮 Agricultural Yield Prediction Simulator")
    st.write("Estimate crop yield in kilograms per hectare using trained machine learning models.")
    
    predictor = AgroYieldPredictor(models_dir=os.path.join(PROJECT_ROOT, "models"))
    available_districts = metadata.get('available_districts', sorted(df_clean['district'].unique().tolist()))
    
    with st.form("yield_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            selected_district = st.selectbox(
                "Select Tamil Nadu District:",
                options=available_districts,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            selected_year = st.slider(
                "Select Target Agricultural Year:",
                min_value=2000,
                max_value=2030,
                value=2024,
                step=1
            )
            
        with col2:
            # Provide sensible default area based on district average
            dist_mean_area = df_clean[df_clean['district'] == selected_district]['area'].mean()
            dist_mean_area = float(dist_mean_area) if pd.notna(dist_mean_area) else 50000.0
            
            cultivated_area = st.number_input(
                f"Cultivated Area in Hectares (District historical avg: ~{dist_mean_area:,.0f} ha):",
                min_value=1.0,
                max_value=300000.0,
                value=round(dist_mean_area, 0),
                step=100.0
            )
            selected_crop = st.text_input("Crop Group / Crop Type:", value="total_group (Aggregate Baseline)", disabled=True)
            selected_season = st.text_input("Season:", value="whole_year", disabled=True)
            
        submit_btn = st.form_submit_button("🌱 Estimate Yield", use_container_width=True)
        
    if submit_btn:
        with st.spinner("Calculating validated yield prediction..."):
            pred_res = predictor.predict(
                district=selected_district,
                year=selected_year,
                area=cultivated_area,
                crop="total_group",
                season="whole_year"
            )
            
            st.success("✅ Prediction Successfully Generated!")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(
                    "Estimated Yield",
                    f"{pred_res['predicted_yield_kg_ha']:,.2f} kg/ha"
                )
            with res_col2:
                st.metric(
                    "Estimated Regional Production",
                    f"{pred_res['estimated_production_tonnes']:,.1f} Tonnes"
                )
            with res_col3:
                st.metric(
                    "Model Deployed",
                    pred_res['model_used']
                )
                
            st.markdown(f"""
            <div class="info-box">
                <b>🌾 Agronomic Interpretation:</b><br>
                {pred_res['explanation']}
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# 5. RAINFALL IMPACT ANALYSIS
# ==============================================================================
elif menu_choice == "🌧️ Rainfall Impact Analysis":
    st.title("🌧️ Rainfall Impact Analysis & Integration Roadmap")
    st.write("Examining monsoon dynamics, precipitation anomalies, and crop yield relationships.")
    
    st.markdown("""
    <div class="info-box">
        <b>Current Status:</b> Rainfall observations are not part of the base APY statistical report. 
        In strict adherence to engineering integrity, <b>we do not fabricate synthetic rainfall numbers</b>. 
        Below is the scientific methodology and architectural pipeline prepared for rainfall feature fusion.
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌦️ Tamil Nadu Monsoon Seasons")
        st.markdown("""
        Tamil Nadu has a unique rainfall regime in India:
        1. **South-West Monsoon (June – September):** Benefits Western Ghats and North-Western districts (Nilgiris, Coimbatore, Dharmapuri).
        2. **North-East Monsoon (October – December):** The primary lifeblood for Coastal and Delta Tamil Nadu (Thanjavur, Nagapattinam, Cuddalore, Tiruvarur), providing over 50% of annual precipitation.
        3. **Summer & Winter Showers (January – May):** Crucial for soil moisture preceding main Kharif sowing.
        """)
        
    with c2:
        st.subheader("⚠️ Core Scientific Principle: Correlation vs. Causation")
        st.markdown("""
        - Higher rainfall does **not** linearly equal higher crop yield.
        - **Excess Rainfall / Flooding:** Damages standing paddy crops during harvest stage (lodging, fungal blast).
        - **Deficit Rainfall / Drought:** Causes moisture stress during flowering and grain-filling stages.
        - **Pre-Harvest Availability Constraint:** To forecast yield *before* harvest, only rainfall recorded up to the prediction cut-off date can be used (never post-harvest rainfall).
        """)
        
    st.subheader("📐 Dual-Stream Architectural Blueprint")
    diag_path = os.path.join(PROJECT_ROOT, "reports", "figures", "07_rainfall_integration_architecture.png")
    if os.path.exists(diag_path):
        st.image(diag_path, use_container_width=True)


# ==============================================================================
# 6. MODEL PERFORMANCE
# ==============================================================================
elif menu_choice == "🏆 Model Performance":
    st.title("🏆 Model Benchmarking & Performance Evaluation")
    st.write("Rigorous evaluation comparing Linear Regression Baseline, Random Forest, and XGBoost.")
    
    if df_comparison is not None:
        st.subheader("📊 Model Comparison Matrix (Chronological Hold-out Validation)")
        st.dataframe(df_comparison, use_container_width=True)
        
        # Plotly comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_comparison['model_name'],
            y=df_comparison['train_rmse_kg_ha'],
            name='Train RMSE (kg/ha)',
            marker_color='#3B82F6'
        ))
        fig.add_trace(go.Bar(
            x=df_comparison['model_name'],
            y=df_comparison['val_rmse_kg_ha'],
            name='Validation RMSE (kg/ha)',
            marker_color='#EF4444'
        ))
        fig.update_layout(
            barmode='group',
            title="Train RMSE vs. Chronological Validation RMSE",
            yaxis_title="RMSE (kg/hectare)",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <b>Senior ML Engineering Analysis:</b><br>
            Notice that <b>Random Forest</b> and <b>XGBoost</b> achieved very low Train RMSE (high training R² of ~0.86), 
            but performed poorly on future validation years (negative R²). 
            <b>Why did this happen?</b><br>
            Decision trees partition feature space using orthogonal splits; they cannot extrapolate global temporal trends 
            into future years when key physical climate variables (like rainfall) are not yet in the feature matrix. 
            <b>Linear Regression</b> learns continuous temporal coefficients and district baselines, delivering superior generalization 
            on the chronological holdout set.
        </div>
        """, unsafe_allow_html=True)
        
        if metadata and 'test_metrics' in metadata:
            st.subheader("🎯 Unbiased Test Set Evaluation (2020–2022)")
            tm = metadata['test_metrics']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Test MAE", f"{tm['mae_kg_ha']:,.1f} kg/ha")
            col2.metric("Test RMSE", f"{tm['rmse_kg_ha']:,.1f} kg/ha")
            col3.metric("Relative Error", f"{tm['relative_error_pct']:.1f}%")
            col4.metric("Test Mean Actual", f"{tm['mean_actual_kg_ha']:,.1f} kg/ha")


# ==============================================================================
# 7. ABOUT & METHODOLOGY
# ==============================================================================
elif menu_choice == "ℹ️ About & Methodology":
    st.title("ℹ️ About AgroVision AI")
    st.write("Comprehensive methodology, project architecture, and engineering documentation.")
    
    st.markdown("""
    ### 🏛️ Project Details
    - **Project Name:** AgroVision AI
    - **Dashboard Name:** VayalYield AI (*Vayal* is the Tamil word for agricultural field/farmland)
    - **Author:** Mehul Shankar P (First-Year Computer Science and Engineering Student)
    - **Scope:** Tamil Nadu, India (38 Administrative Districts)
    
    ### 🔬 Machine Learning Pipeline & Zero-Leakage Policy
    1. **Data Ingestion:** Cleaned 23-year historical records (2000–2022) from Directorate of Economics and Statistics.
    2. **Anti-Leakage Safeguard:** Yield is defined as $(\\text{Production} \\times 1000) / \\text{Area}$. Including production as an input feature is a fatal data leakage error. Production is strictly excluded from all training matrices.
    3. **Chronological Splitting:**
       - Train: 2000–2016 (506 records)
       - Validation: 2017–2019 (97 records)
       - Test: 2020–2022 (113 records)
    4. **Transformation:** OneHotEncoder for categorical districts; StandardScaler for numeric acreage.
    5. **Artifact Persistence:** Serialized pipelines stored with joblib for instant deterministic inference.
    """)
