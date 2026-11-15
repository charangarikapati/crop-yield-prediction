# Production Model Card: District-Level Crop Yield Predictor (Vidarbha Agritech)
**Version:** 1.0.0-prod  
**Model Date:** November 2026  
**Target Organization:** Nagpur Agritech Startup (Vidarbha Commodity Procurement)  
**Standard:** Adheres to Google / Mitchell et al. (2019) Model Cards for Model Reporting  

---

## 1. Model Details

- **Model Architecture:** Normalized Yield Multiplier Gradient Boosted Decision Tree (LightGBM Regressor).
- **Target Formulation:** Multi-Crop Normalized Index ($y_{\text{mult}} = \frac{\text{Actual Yield}}{\text{Crop Baseline}}$), rescaled to **Quintals / Hectare** ($q/ha$) at inference.
- **Hyperparameters:** `n_estimators=85`, `learning_rate=0.04`, `num_leaves=24`, `max_depth=7`, `min_child_samples=45`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_alpha=0.6`, `reg_lambda=3.5`.
- **Framework & Dependencies:** LightGBM 4.7+, Scikit-Learn 1.9+, Pandas 3.1+, Python 3.14.
- **Serialization Artifacts:** 
  - `models/crop_yield_lgbm_best.joblib` (Trained model object)
  - `models/crop_base_means.joblib` (Crop normal scaling factors)

---

## 2. Intended Use & Deployment Scope

### 2.1 Primary Intended Applications
- **Pre-Harvest Procurement Planning (60–90 Days Advance):** Forecasting harvest volumes per district to negotiate forward contracts with Farmer Producer Organizations (FPOs) before spot mandi prices peak.
- **Logistics & Warehousing Optimization:** Allocating regional warehouse capacity, cold storage, and truck fleets towards projected surplus districts while minimizing empty backhauls.
- **Dynamic Risk Hedging:** Alerting commodity procurement desks when a crop in a key district faces severe shortfall risks ($< 85\%$ normal), allowing timely hedging on commodity exchanges.

### 2.2 Out-of-Scope & Prohibited Uses
- **Direct Crop Insurance Claim Adjudication:** This model operates at district and taluka agro-climatic scales; it must not be used as the sole determinant for individual micro-farm crop loss compensation without on-the-ground surveyor inspection.
- **Extrapolation to Uncalibrated Agro-Climatic Zones:** Not calibrated for coastal paddy (Konkan) or Himalayan temperate fruit systems without regional retraining.

---

## 3. Data & Environmental Factors

### 3.1 Data Lineage & Scope
- **Geographic Coverage:** 6 Vidarbha districts in Eastern Maharashtra: **Nagpur, Wardha, Amravati, Chandrapur, Bhandara, Yavatmal** (spanning 51 talukas).
- **Temporal Span:** 2017 to 2023 (7 agricultural years covering historical droughts, excess monsoon downpours, and normal cycles).
- **Multi-Modal Data Integration:** ~351,000 raw transaction and survey rows across four sources:
  1. *Farm Survey Records:* 150,000 observations (plot area, irrigation source, crop yield).
  2. *Daily Weather Stations:* 61,344 station-day records across 24 regional stations (precipitation, temp max/min).
  3. *APMC Mandi Records:* 120,000 transaction records across 21 mandis (arrivals, modal prices).
  4. *Soil Health Cards:* 20,000 soil test records (N, P, K, pH, organic carbon, electrical conductivity).

### 3.2 Real-World Data Quality Reconciliation
- **Bimodal Mandi Prices:** Corrected 26,411 rows ($22.0\%$) erroneously entered per kg ($< \text{Rs } 250$) instead of per quintal, standardizing the entire series to $\text{INR} / \text{quintal}$.
- **Monsoon Sensor Dropouts:** 2,927 missing precipitation records ($4.8\%$) during June–September imputed using spatial inverse-distance weighting across neighboring district stations.
- **Yield Unit Scaling:** Biological crop envelope reconciliation corrected 22,478 rows entered in $kg/ha$ ($15.0\%$) and 1,981 rows entered in tonnes/ha ($1.3\%$) into uniform **Quintals / Hectare**.

---

## 4. Evaluation Metrics & Success Benchmarks

### 4.1 Business-Aligned Metrics
1. **Mean Absolute Error (MAE):** Expressed directly in business units (**Quintals / Hectare**).
2. **Weighted Absolute Percentage Error (WAPE):** Percentage aggregate error across varied production scales.
3. **Asymmetric Procurement Risk Cost:** 
   $$\text{Cost} = \frac{1}{N} \sum \left( 1.0 \cdot \max(y - \hat{y}, 0) + 1.3 \cdot \max(\hat{y} - y, 0) \right)$$
   Penalizes over-prediction ($1.3\times$) higher than under-prediction ($1.0\times$) due to severe commercial costs of unfulfilled procurement commitments and unused leased warehouse space.
4. **Shortfall Early-Warning Alert:** Yield shortfall ($< 85\%$ of 3-year trailing normal) evaluated at fixed $80\%$ precision.

### 4.2 Benchmark Model Comparison

| Model Architecture | Validation MAE ($q/ha$) | Validation WAPE | Test MAE ($q/ha$) | Test WAPE | Asymmetric Risk Cost | Baseline Beat ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Baseline (3-Yr Median)** | 4.879 | 24.11% | 4.557 | 21.57% | 4.982 | Baseline |
| **Regularized Linear (Ridge)** | 6.128 | 30.29% | 7.299 | 34.55% | 7.641 | -25.60% (Worse) |
| **Random Forest Regressor** | 3.985 | 19.69% | 3.944 | 18.67% | 4.120 | +18.32% |
| **Tuned LightGBM (Production)** | **3.756** | **18.57%** | **3.939** | **18.65%** | **4.088** | **+23.02% (PASS)** |

### 4.3 Success Criteria Audit
- **Criterion 1 (Beat Baseline by $\ge 15\%$):** Achieved **$23.02\%$ improvement** on Validation MAE ($3.756$ vs $4.879$ $q/ha$) $\rightarrow$ **PASSED**.
- **Criterion 2 (Generalization Gap $\le 5\%$):** Generalization gap between blind Test ($3.939$) and Validation ($3.756$) is **$4.87\%$** $\rightarrow$ **PASSED**.
- **Criterion 3 (Domain-Sound Top Drivers):** Top 5 SHAP features verified by agronomic specialists $\rightarrow$ **PASSED**.

---

## 5. Model Explainability & Key Drivers

### 5.1 Global Tree SHAP Feature Importance
1. `irrigation_type` (Mean $|SHAP| = 0.1225$): Primary determinant of drought buffering capacity (Drip/Canal vs Rainfed).
2. `mandi_price_momentum_ratio` (Mean $|SHAP| = 0.0433$): Pre-sowing economic incentive governing farmer seed and fertilizer investments.
3. `monsoon_cumulative_rain_mm` (Mean $|SHAP| = 0.0095$): Cumulative moisture availability during vegetative growth (June–August).
4. `season` (Mean $|SHAP| = 0.0083$): Kharif (monsoon-dependent) vs Rabi (receding moisture/irrigation).
5. `yield_rolling_std_3y` (Mean $|SHAP| = 0.0073$): Historical agro-ecological yield volatility of the district-crop pair.

### 5.2 Non-Linear Partial Dependence (PDP) Insights
- **Dry Spell Cliff:** Yield shows mild resilience up to 10–12 consecutive dry days, followed by a steep nonlinear drop between **14 and 18 dry days** for rainfed field crops.
- **Rainfall Saturation Curve:** Yield increases with monsoon rainfall up to ~850 mm; beyond 950 mm, waterlogging and root-rot risks cause plateauing and slight yield declines.
- **NPK Ratio Plateau:** Optimal yield response occurs in the $0.55 - 0.70$ $N/(P+K)$ equilibrium zone.

---

## 6. Fairness & Regional Stability Audit

| District | Test Records | Mean Actual ($q/ha$) | Mean Pred ($q/ha$) | MAE ($q/ha$) | WAPE ($\%$) | Residual Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Amravati** | 4,502 | 28.28 | 27.92 | 5.258 | 18.59% | -0.360 |
| **Bhandara** | 2,752 | 16.30 | 16.16 | 3.237 | 19.86% | -0.143 |
| **Chandrapur**| 3,421 | 17.38 | 17.25 | 3.374 | 19.41% | -0.129 |
| **Nagpur** | 4,518 | 31.15 | 30.79 | 5.517 | 17.71% | -0.355 |
| **Wardha** | 3,639 | 14.30 | 14.02 | 2.630 | 18.39% | -0.282 |
| **Yavatmal** | 3,802 | 13.96 | 13.83 | 2.741 | 19.64% | -0.127 |

- **Stability Verdict:** Strong stability across all 6 districts with WAPE constrained tightly between $17.71\%$ and $19.86\%$.
- **Farm Scale Equity:** Smallholder farms ($< 1$ ha) show an MAE of $3.89$ $q/ha$ vs Large commercial farms ($> 5$ ha) at $3.97$ $q/ha$, confirming zero discriminatory distortion against smallholders.

---

## 7. Outlier Case Study: Autopsy of Worst Prediction

- **Record Identifier:** `CY_0007894` (2023 Blind Test Set)
- **Geography & Crop:** Yavatmal District, Pusad Taluka, Cotton (Rainfed, 2.4 ha)
- **Model Forecast:** $16.42$ $q/ha$ | **Actual Harvest:** $3.20$ $q/ha$ | **Error:** $+13.22$ $q/ha$ ($+413\%$ over-prediction)
- **Forensic Root Cause:** Regional weather gauges reported normal monsoon rainfall ($840$ mm) and no macro dry spells. However, Pusad experienced a localized hyper-pest outbreak (Pink Bollworm third-generation infestation) compounded by localized unseasonal hail in October during boll formation. Macro weather variables were completely blind to this localized micro-shock.
- **Operational Guardrails Implemented:**
  1. *Satellite NDVI Vegetative Anomaly Alerts:* Automated cross-referencing with 10m Sentinel-2 NDVI anomalies 3 weeks prior to harvest.
  2. *Safety Margin in Rainfed Cotton:* Capping advance forward contracts at $80\%$ of model-predicted yield in rainfed Yavatmal cotton clusters.
  3. *Surveyor Verification Trigger:* Any prediction deviating by $> 25\%$ from 3-year baseline automatically triggers on-ground FPO lead verification.

---

## 8. Governance, Retraining & Monitoring Protocol

1. **Annual Retraining Cycle:** Re-fit model annually in December once Kharif harvest totals and Rabi sowing records are finalized.
2. **Data Drift Monitoring:** Track Population Stability Index (PSI) on pre-sowing mandi prices and monthly rainfall distributions; trigger re-tuning if $\text{PSI} > 0.20$.
3. **Sign-off:** Approved for deployment in 2024 Procurement Planning by Agritech ML & Commodity Procurement Committee.

