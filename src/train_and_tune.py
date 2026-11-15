"""
Production Model Benchmarking, Tuning and Business Evaluation Pipeline
Implements:
1. Multi-Crop Normalized Yield Index Modeling (resolving scale disparities across crops).
2. Four-Model Comparative Benchmark:
   - Baseline: Historical 3-Year District-Crop Rolling Median.
   - Linear Regularized: Ridge Regression.
   - Random Forest: RandomForestRegressor.
   - Tuned Gradient Boosting: LightGBM Regressor.
3. Business Metric Alignment:
   - MAE in Quintals / Hectare (Primary Business Unit).
   - RMSE and WAPE (%).
   - Asymmetric Procurement Risk Cost (Over-prediction penalty 1.3 vs Under-prediction penalty 1.0).
   - Shortfall Alert Classification (Recall at fixed 80% Precision).
4. Success Verification:
   - Beats baseline by >= 15%.
   - Test within 5% of Validation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_curve
import lightgbm as lgb

FEATURE_COLS_NUM = [
    'area_hectares',
    'pre_sowing_mandi_price',
    'district_mandi_price_spread',
    'mandi_price_momentum_ratio',
    'monsoon_cumulative_rain_mm',
    'heavy_rain_days_count',
    'max_consecutive_dry_days',
    'monsoon_rainfall_anomaly_ratio',
    'dry_spell_severity_index',
    'is_severe_dry_spell',
    'heavy_rain_intensity_ratio',
    'rainfall_deficit_pct',
    'soil_nitrogen_kg_ha',
    'soil_phosphorus_kg_ha',
    'soil_potassium_kg_ha',
    'soil_ph',
    'soil_organic_carbon_pct',
    'soil_npk_ratio',
    'soil_ph_stress',
    'soil_productivity_index',
    'yield_lag_1y',
    'yield_lag_2y',
    'yield_rolling_mean_3y',
    'yield_rolling_std_3y',
    'taluka_target_encoded'
]

FEATURE_COLS_CAT = [
    'district',
    'crop',
    'season',
    'irrigation_type',
    'farm_scale_tier'
]


def calculate_business_metrics(y_true, y_pred, rolling_mean):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100.0
    
    # Asymmetric Procurement Cost:
    # Over-predicting (penalty 1.3) -> warehouse over-allocation, defaulted contracts
    # Under-predicting (penalty 1.0) -> missed pre-harvest discount
    errors = y_pred - y_true
    asymmetric_cost = np.mean(np.where(errors > 0, 1.3 * errors, -1.0 * errors))
    
    # Yield shortfall alert (< 85% of normal)
    is_shortfall_true = (y_true < 0.85 * rolling_mean).astype(int)
    predicted_shortfall_ratio = (rolling_mean - y_pred) / (rolling_mean + 1e-5)
    
    precisions, recalls, _ = precision_recall_curve(is_shortfall_true, predicted_shortfall_ratio)
    valid_idx = np.where(precisions >= 0.80)[0]
    if len(valid_idx) > 0:
        recall_at_80_prec = float(recalls[valid_idx[0]])
    else:
        recall_at_80_prec = float(recalls[np.argmax(precisions)])
        
    return {
        'mae_q_ha': round(float(mae), 3),
        'rmse_q_ha': round(float(rmse), 3),
        'wape_pct': round(float(wape), 2),
        'asymmetric_cost': round(float(asymmetric_cost), 3),
        'recall_at_80_precision': round(recall_at_80_prec, 3)
    }


def train_benchmark_pipeline():
    print("Loading engineered train, val, and test partitions...")
    train_df = pd.read_parquet('data/processed/train.parquet')
    val_df = pd.read_parquet('data/processed/val.parquet')
    test_df = pd.read_parquet('data/processed/test.parquet')
    
    # Calculate crop base normal yields strictly on train
    crop_means = train_df.groupby('crop', observed=False)['clean_yield_q_ha'].mean().to_dict()
    
    for df in [train_df, val_df, test_df]:
        df['crop_base_mean'] = df['crop'].astype(str).map(crop_means).astype(float)
        df['yield_multiplier_target'] = df['clean_yield_q_ha'] / df['crop_base_mean']
        for col in FEATURE_COLS_CAT:
            df[col] = df[col].astype('category')
            
    all_features = FEATURE_COLS_NUM + FEATURE_COLS_CAT
    
    X_train = train_df[all_features]
    y_train_target = train_df['yield_multiplier_target'].values
    y_train_actual = train_df['clean_yield_q_ha'].values
    
    X_val = val_df[all_features]
    y_val_actual = val_df['clean_yield_q_ha'].values
    val_rolling_mean = val_df['yield_rolling_mean_3y'].values
    val_crop_base = val_df['crop_base_mean'].values
    
    X_test = test_df[all_features]
    y_test_actual = test_df['clean_yield_q_ha'].values
    test_rolling_mean = test_df['yield_rolling_mean_3y'].values
    test_crop_base = test_df['crop_base_mean'].values
    
    # For Linear and RF models, build one-hot encoded design matrices
    X_train_encoded = pd.get_dummies(X_train, columns=FEATURE_COLS_CAT, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, columns=FEATURE_COLS_CAT, drop_first=True).reindex(columns=X_train_encoded.columns, fill_value=0)
    X_test_encoded = pd.get_dummies(X_test, columns=FEATURE_COLS_CAT, drop_first=True).reindex(columns=X_train_encoded.columns, fill_value=0)
    
    results = {}
    
    # ---------------------------------------------------------
    # 1. Baseline: 3-Year Rolling Median
    # ---------------------------------------------------------
    print("\n--- 1. Baseline: Historical Rolling Median ---")
    val_base_pred = val_rolling_mean
    test_base_pred = test_rolling_mean
    
    val_base_metrics = calculate_business_metrics(y_val_actual, val_base_pred, val_rolling_mean)
    test_base_metrics = calculate_business_metrics(y_test_actual, test_base_pred, test_rolling_mean)
    results['Baseline'] = {'val': val_base_metrics, 'test': test_base_metrics}
    print(f"Val MAE: {val_base_metrics['mae_q_ha']} q/ha | Test MAE: {test_base_metrics['mae_q_ha']} q/ha")
    
    # ---------------------------------------------------------
    # 2. Linear Regularized (Ridge Regression)
    # ---------------------------------------------------------
    print("\n--- 2. Regularized Linear (Ridge) ---")
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train_encoded, y_train_target)
    
    val_ridge_pred = ridge.predict(X_val_encoded) * val_crop_base
    test_ridge_pred = ridge.predict(X_test_encoded) * test_crop_base
    
    val_ridge_metrics = calculate_business_metrics(y_val_actual, val_ridge_pred, val_rolling_mean)
    test_ridge_metrics = calculate_business_metrics(y_test_actual, test_ridge_pred, test_rolling_mean)
    results['Ridge'] = {'val': val_ridge_metrics, 'test': test_ridge_metrics}
    print(f"Val MAE: {val_ridge_metrics['mae_q_ha']} q/ha | Test MAE: {test_ridge_metrics['mae_q_ha']} q/ha")
    
    # ---------------------------------------------------------
    # 3. Random Forest Regressor
    # ---------------------------------------------------------
    print("\n--- 3. Random Forest Regressor ---")
    rf = RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=42)
    rf.fit(X_train_encoded, y_train_target)
    
    val_rf_pred = rf.predict(X_val_encoded) * val_crop_base
    test_rf_pred = rf.predict(X_test_encoded) * test_crop_base
    
    val_rf_metrics = calculate_business_metrics(y_val_actual, val_rf_pred, val_rolling_mean)
    test_rf_metrics = calculate_business_metrics(y_test_actual, test_rf_pred, test_rolling_mean)
    results['RandomForest'] = {'val': val_rf_metrics, 'test': test_rf_metrics}
    print(f"Val MAE: {val_rf_metrics['mae_q_ha']} q/ha | Test MAE: {test_rf_metrics['mae_q_ha']} q/ha")
    
    # ---------------------------------------------------------
    # 4. Tuned Gradient Boosting (LightGBM)
    # ---------------------------------------------------------
    print("\n--- 4. Tuned Gradient Boosting (LightGBM) ---")
    lgbm = lgb.LGBMRegressor(
        n_estimators=85,
        learning_rate=0.04,
        num_leaves=24,
        max_depth=7,
        min_child_samples=45,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.6,
        reg_lambda=3.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm.fit(X_train, y_train_target)
    
    val_lgbm_pred = lgbm.predict(X_val) * val_crop_base
    test_lgbm_pred = lgbm.predict(X_test) * test_crop_base
    
    val_lgbm_metrics = calculate_business_metrics(y_val_actual, val_lgbm_pred, val_rolling_mean)
    test_lgbm_metrics = calculate_business_metrics(y_test_actual, test_lgbm_pred, test_rolling_mean)
    results['LightGBM'] = {'val': val_lgbm_metrics, 'test': test_lgbm_metrics}
    print(f"Val MAE: {val_lgbm_metrics['mae_q_ha']} q/ha | Test MAE: {test_lgbm_metrics['mae_q_ha']} q/ha")
    
    # ---------------------------------------------------------
    # Verification of Success Criteria
    # ---------------------------------------------------------
    base_val_mae = results['Baseline']['val']['mae_q_ha']
    lgb_val_mae = results['LightGBM']['val']['mae_q_ha']
    lgb_test_mae = results['LightGBM']['test']['mae_q_ha']
    
    pct_impr_val = ((base_val_mae - lgb_val_mae) / base_val_mae) * 100.0
    base_test_mae = results['Baseline']['test']['mae_q_ha']
    pct_impr_test = ((base_test_mae - lgb_test_mae) / base_test_mae) * 100.0
    generalization_gap = (abs(lgb_test_mae - lgb_val_mae) / lgb_val_mae) * 100.0
    
    print("\n=======================================================")
    print("SUCCESS CRITERIA VERIFICATION:")
    print(f"1. Beat Baseline by >= 15% on Val:  Achieved {pct_impr_val:.2f}% improvement! ({'PASS' if pct_impr_val >= 15 else 'FAIL'})")
    print(f"2. Test within 5% of Validation:    Generalization gap is {generalization_gap:.2f}%! ({'PASS' if generalization_gap <= 5 else 'FAIL'})")
    print(f"3. Shortfall Recall @ 80% Prec:     {val_lgbm_metrics['recall_at_80_precision']*100:.1f}%")
    print("=======================================================")
    
    # Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(lgbm, 'models/crop_yield_lgbm_best.joblib')
    joblib.dump(crop_means, 'models/crop_base_means.joblib')
    
    summary = {
        'benchmark_comparison': results,
        'features': all_features,
        'verification': {
            'pct_improvement_val': round(pct_impr_val, 2),
            'baseline_beat_passed': bool(pct_impr_val >= 15.0),
            'generalization_gap_pct': round(generalization_gap, 2),
            'generalization_passed': bool(generalization_gap <= 5.0)
        }
    }
    with open('models/benchmark_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("Trained model saved to: models/crop_yield_lgbm_best.joblib")
    print("Benchmark summary saved to: models/benchmark_results.json")
    
    # Save predictions into test dataframe for explainability
    test_df['predicted_yield_q_ha'] = test_lgbm_pred
    test_df['prediction_error'] = test_df['predicted_yield_q_ha'] - test_df['clean_yield_q_ha']
    test_df['abs_prediction_error'] = test_df['prediction_error'].abs()
    test_df.to_parquet('data/processed/test_with_predictions.parquet', index=False)
    
    return summary


if __name__ == '__main__':
    train_benchmark_pipeline()
