"""
Generate the Master Jupyter Notebook: notebooks/01_crop_yield_prediction_master.ipynb
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cells definition
cells = []

# Title & Context
cells.append(nbf.v4.new_markdown_cell("""# District-Level Crop Yield Prediction & Procurement Driver Intelligence
### Nagpur Agritech Machine Learning & Business Procurement Decision Engine

---

## 1. Executive Summary & Business Context
Agritech startups and agricultural commodity procurement networks in **Vidarbha (Nagpur region, Maharashtra)** traditionally operate reactively. When crop failure strikes due to monsoon dry spells or unseasonal weather, spot mandi prices surge, supply chains break, and fulfillment contracts default.

**Goal:** Build a robust, leakage-free early yield forecasting system across **6 core districts** (Nagpur, Wardha, Amravati, Chandrapur, Bhandara, Yavatmal) covering ~350,000 multi-modal records across crop surveys, daily rainfall stations, APMC mandi transactions, and soil health cards.
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', 35)

print("Environment initialized successfully!")
"""))

# Data Quality & Anomaly Discovery
cells.append(nbf.v4.new_markdown_cell("""## 2. Multi-Modal Data Ingestion & Dirty Data Anomalies

We ingest four transactional tables spanning 2017 to 2023:
1. `crop_yield.csv` (150,000 rows): Surveyed farm plots with **erratic recorded yield units**.
2. `rainfall.csv` (61,344 rows): Daily weather stations with **monsoon sensor dropout missingness**.
3. `mandi_prices.csv` (120,000 rows): APMC daily transactions with **bimodal Rs/kg vs Rs/quintal price errors**.
4. `soil_tests.csv` (20,000 rows): Soil health card fertility records.
"""))

cells.append(nbf.v4.new_code_cell("""# Inspect Raw Tables
df_yield_raw = pd.read_csv('data/raw/crop_yield.csv')
df_rain_raw = pd.read_csv('data/raw/rainfall.csv')
df_mandi_raw = pd.read_csv('data/raw/mandi_prices.csv')
df_soil_raw = pd.read_csv('data/raw/soil_tests.csv')

print(f"Crop Yield Records:  {len(df_yield_raw):,}")
print(f"Rainfall Records:    {len(df_rain_raw):,}")
print(f"Mandi Price Records: {len(df_mandi_raw):,}")
print(f"Soil Test Records:   {len(df_soil_raw):,}")
print(f"Total Raw Rows:      {len(df_yield_raw)+len(df_rain_raw)+len(df_mandi_raw)+len(df_soil_raw):,}")
"""))

# Anomaly inspection code
cells.append(nbf.v4.new_markdown_cell("""### 2.1 Anomaly Diagnostics: Mandi Price Bimodal Units & Yield Unit Scaling"""))

cells.append(nbf.v4.new_code_cell("""# Anomaly 1: Mixed price units in Mandi records
sub_250_prices = (df_mandi_raw['modal_price'] < 250).sum()
print(f"Mandi rows entered per kg (< Rs 250): {sub_250_prices:,} ({sub_250_prices/len(df_mandi_raw)*100:.1f}%)")

# Anomaly 2: Monsoon rainfall dropouts
monsoon_mask = pd.to_datetime(df_rain_raw['date']).dt.month.isin([6, 7, 8, 9])
missing_monsoon_rain = df_rain_raw.loc[monsoon_mask, 'rainfall_mm'].isna().sum()
print(f"Monsoon rainfall sensor dropouts: {missing_monsoon_rain:,} ({missing_monsoon_rain/monsoon_mask.sum()*100:.1f}%)")

# Anomaly 3: Yield unit discrepancies (Tonnes vs kg vs Quintal)
high_yield = (df_yield_raw['recorded_yield'] > 200).sum()
low_yield = (df_yield_raw['recorded_yield'] < 3.0).sum()
print(f"Yield entered in kg/ha (> 200):     {high_yield:,} ({high_yield/len(df_yield_raw)*100:.1f}%)")
print(f"Yield entered in Tonnes/ha (< 3.0): {low_yield:,} ({low_yield/len(df_yield_raw)*100:.1f}%)")
"""))

# Data Cleaning & Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Harmonization, Imputation & Leakage-Free Splitting

1. **Price Harmonization**: All prices with `modal_price < 250` scaled by $\\times 100$ to standardize to **Rs / Quintal**.
2. **Spatial Neighbor Imputation**: Daily rainfall interpolated across weather stations within the same district to avoid zero-fill drought distortion.
3. **Biological Envelope Normalization**: Enforcing physiological yield boundaries (e.g. Soybean 3-45 q/ha) to standardize to **Quintals / Hectare**.
4. **Time-Aware Splitting**:
   - **Training Set (2018 - 2021)**: 86,996 records.
   - **Validation Set (2022)**: 20,893 records (Used for model selection).
   - **Out-of-Time Blind Test (2023)**: 22,634 records (Final business evaluation).
"""))

cells.append(nbf.v4.new_code_cell("""# Load Clean Master Splits
train_df = pd.read_parquet('data/processed/train.parquet')
val_df = pd.read_parquet('data/processed/val.parquet')
test_df = pd.read_parquet('data/processed/test_with_predictions.parquet')

print(f"Train Partition (2018-2021): {len(train_df):,} rows")
print(f"Validation Set (2022):       {len(val_df):,} rows")
print(f"Blind Test Set (2023):       {len(test_df):,} rows")
"""))

# Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 4. Feature Engineering: Agronomic & Market Indicators
Over 15 domain-specific features engineered:
- **Historical Yield Lags**: `yield_lag_1y`, `yield_lag_2y`, `yield_rolling_mean_3y`, `yield_rolling_std_3y`
- **Weather Shock Indices**: `monsoon_cumulative_rain_mm`, `max_consecutive_dry_days`, `dry_spell_severity_index`, `monsoon_rainfall_anomaly_ratio`
- **Soil Balance Metrics**: `soil_npk_ratio`, `soil_ph_stress`, `soil_productivity_index`, `soil_organic_carbon_pct`
- **Economic Incentives**: `pre_sowing_mandi_price`, `mandi_price_momentum_ratio`, `district_mandi_price_spread`
- **High-Cardinality Target Encoding**: Out-of-fold target encoding for 51 Talukas strictly fitted on the training split.
"""))

cells.append(nbf.v4.new_code_cell("""feature_sample_cols = [
    'crop', 'district', 'clean_yield_q_ha', 'yield_rolling_mean_3y',
    'monsoon_cumulative_rain_mm', 'max_consecutive_dry_days',
    'soil_npk_ratio', 'mandi_price_momentum_ratio', 'taluka_target_encoded'
]
train_df[feature_sample_cols].head(8)
"""))

# Model Benchmark
cells.append(nbf.v4.new_markdown_cell("""## 5. Model Benchmarking & Business Cost Evaluation

We evaluate four model families:
1. **Baseline**: 3-Year District-Crop Rolling Median Predictor
2. **Linear Regularized**: Ridge Regression
3. **Random Forest**: RandomForestRegressor
4. **Gradient Boosting**: Tuned LightGBM Regressor
"""))

cells.append(nbf.v4.new_code_cell("""with open('models/benchmark_results.json', 'r') as f:
    benchmarks = json.load(f)

bench_df = []
for model_name, metrics in benchmarks['benchmark_comparison'].items():
    bench_df.append({
        'Model': model_name,
        'Val MAE (q/ha)': metrics['val']['mae_q_ha'],
        'Val RMSE (q/ha)': metrics['val']['rmse_q_ha'],
        'Val WAPE (%)': metrics['val']['wape_pct'],
        'Test MAE (q/ha)': metrics['test']['mae_q_ha'],
        'Test WAPE (%)': metrics['test']['wape_pct'],
        'Asymmetric Risk Cost': metrics['test']['asymmetric_cost']
    })

pd.DataFrame(bench_df).set_index('Model')
"""))

# Success Verification
cells.append(nbf.v4.new_markdown_cell("""### 5.1 Success Criteria Verification Check
- **Criterion 1**: Beat Baseline by $\\ge 15\\%$ on Validation.
- **Criterion 2**: Generalization gap between Test and Validation $\\le 5\\%$.
"""))

cells.append(nbf.v4.new_code_cell("""verif = benchmarks['verification']
print(f"Improvement over Baseline on Validation: {verif['pct_improvement_val']:.2f}% (Target: >= 15.0%) -> {'PASS' if verif['baseline_beat_passed'] else 'FAIL'}")
print(f"Generalization Gap (Test vs Val):        {verif['generalization_gap_pct']:.2f}% (Target: <= 5.0%)  -> {'PASS' if verif['generalization_passed'] else 'FAIL'}")
"""))

# Explainability & SHAP
cells.append(nbf.v4.new_markdown_cell("""## 6. Model Explainability: SHAP Values & Partial Dependence Plots

Tree SHAP identifies the top global drivers of crop yield, while Partial Dependence Plots reveal non-linear agronomic response curves.
"""))

cells.append(nbf.v4.new_code_cell("""# Display SHAP Global Feature Importance & PDP Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

img_shap = plt.imread('docs/shap_feature_importance.png')
ax1.imshow(img_shap)
ax1.axis('off')
ax1.set_title("Global Feature Importance (Tree SHAP)", fontsize=14, weight='bold')

img_pdp = plt.imread('docs/pdp_curves.png')
ax2.imshow(img_pdp)
ax2.axis('off')
ax2.set_title("Partial Dependence Curves (PDP)", fontsize=14, weight='bold')

plt.tight_layout()
plt.show()
"""))

# Forensic Case Study
cells.append(nbf.v4.new_markdown_cell("""## 7. Outlier Prediction Forensic Autopsy

Every production machine learning model will encounter outlier failure cases. Here we analyze the worst single prediction on the blind 2023 test set to determine root cause and design operational safeguards.
"""))

cells.append(nbf.v4.new_code_cell("""with open('docs/outlier_prediction_autopsy.json', 'r') as f:
    autopsy = json.load(f)

meta = autopsy['case_metadata']
print(f"Record ID:        {meta['record_id']}")
print(f"Location:         {meta['taluka']} Taluka, {meta['district']} District")
print(f"Crop & System:    {meta['crop']} ({meta['irrigation_type']} Irrigation, {meta['farm_area_ha']} ha)")
print(f"Actual Realized:  {meta['actual_yield_q_ha']:.2f} quintals/ha")
print(f"Model Prediction: {meta['predicted_yield_q_ha']:.2f} quintals/ha")
print(f"Prediction Error: +{meta['prediction_error_q_ha']:.2f} quintals/ha ({meta['pct_error']}%)")
print(\"\\nRoot Cause Analysis:\")
print(autopsy['root_cause_analysis'])
print(\"\\nOperational Mitigations:\")
for m in autopsy['operational_guardrails_and_mitigation']:
    print(f"  • {m}")
"""))

# Business Procurement Prioritization
cells.append(nbf.v4.new_markdown_cell("""## 8. Business Translation: Procurement Prioritization Matrix

Translating model predictions into actionable procurement decisions across the 6 Vidarbha districts.
"""))

cells.append(nbf.v4.new_code_cell("""procurement_df = pd.read_csv('docs/procurement_prioritization_matrix.csv')
display_cols = [
    'district', 'crop', 'expected_production_k_tonnes', 'surplus_pct',
    'mean_mandi_price_rs_q', 'procurement_priority_score', 'procurement_tier'
]
procurement_df[display_cols].head(10)
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(14, 8))
img_matrix = plt.imread('docs/procurement_matrix.png')
plt.imshow(img_matrix)
plt.axis('off')
plt.title("Executive Procurement Prioritization Grid", fontsize=15, weight='bold')
plt.show()
"""))

nb['cells'] = cells

os.makedirs('notebooks', exist_ok=True)
notebook_path = 'notebooks/01_crop_yield_prediction_master.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated master notebook: {notebook_path}")

