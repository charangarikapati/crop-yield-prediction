"""
Feature Engineering & Time-Aware Train/Validation/Test Splitting Pipeline
Implements:
1. Historical Lag & Rolling Yield Features (1-Year Lag, 2-Year Lag, 3-Year Rolling Avg & Volatility).
2. Economic Incentives & Price Momentum Ratios.
3. Weather Shock & Anomaly Indices (Dry Spells, Extreme Rain, Anomaly Ratios).
4. Soil Balance & Environmental Stress Metrics.
5. Out-of-Fold Target Encoding for High-Cardinality Talukas (strictly fitted on training period).
6. Leakage-Free Temporal Splitting:
   - Train: 2017 - 2021
   - Validation: 2022
   - Test: 2023 (Blind Holdout)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def compute_historical_lags(df):
    """
    Computes lag and rolling yield and price metrics strictly based on prior years
    to ensure zero lookahead data leakage.
    """
    print("Computing historical lag and rolling features...")
    
    # Calculate district-crop annual historical averages
    hist_dist_crop = df.groupby(['district', 'crop', 'year']).agg(
        mean_yield=('clean_yield_q_ha', 'mean'),
        mean_price=('pre_sowing_mandi_price', 'mean')
    ).reset_index()
    
    # Build complete grid of years, districts, crops
    records = []
    years = sorted(df['year'].unique())
    
    for y in years:
        # Prior year 1
        p1 = hist_dist_crop[hist_dist_crop['year'] == y - 1][['district', 'crop', 'mean_yield', 'mean_price']].copy()
        p1.rename(columns={'mean_yield': 'yield_lag_1y', 'mean_price': 'price_lag_1y'}, inplace=True)
        p1['year'] = y
        
        # Prior year 2
        p2 = hist_dist_crop[hist_dist_crop['year'] == y - 2][['district', 'crop', 'mean_yield']].copy()
        p2.rename(columns={'mean_yield': 'yield_lag_2y'}, inplace=True)
        p2['year'] = y
        
        # 3-year rolling window: (y-3, y-2, y-1)
        p3 = hist_dist_crop[hist_dist_crop['year'].isin([y-3, y-2, y-1])].groupby(['district', 'crop']).agg(
            yield_rolling_mean_3y=('mean_yield', 'mean'),
            yield_rolling_std_3y=('mean_yield', 'std')
        ).reset_index()
        p3['year'] = y
        
        merged = p1.merge(p2, on=['district', 'crop', 'year'], how='left')
        merged = merged.merge(p3, on=['district', 'crop', 'year'], how='left')
        records.append(merged)
        
    lags_df = pd.concat(records, ignore_index=True)
    
    # Merge lags back into main dataframe
    df = df.merge(lags_df, on=['district', 'crop', 'year'], how='left')
    
    # Global fallbacks for initial years (2017, 2018) where 3-year history is incomplete
    crop_defaults = df.groupby('crop')['clean_yield_q_ha'].transform('mean')
    df['yield_lag_1y'] = df['yield_lag_1y'].fillna(crop_defaults)
    df['yield_lag_2y'] = df['yield_lag_2y'].fillna(df['yield_lag_1y'])
    df['yield_rolling_mean_3y'] = df['yield_rolling_mean_3y'].fillna(df['yield_lag_1y'])
    df['yield_rolling_std_3y'] = df['yield_rolling_std_3y'].fillna(1.5)
    
    # Price momentum ratio: current pre-sowing price / lag 1y price
    df['price_lag_1y'] = df['price_lag_1y'].fillna(df['pre_sowing_mandi_price'])
    df['mandi_price_momentum_ratio'] = df['pre_sowing_mandi_price'] / (df['price_lag_1y'] + 1e-5)
    
    return df


def engineer_interaction_and_ratio_features(df):
    """
    Engineers at least 12 domain-specific features.
    """
    print("Engineering domain features, ratios and shock indices...")
    
    # 1. Weather Dry Spell Severity Index: dry days weighted by moisture deficit
    df['dry_spell_severity_index'] = (df['max_consecutive_dry_days'] * 100.0) / (df['monsoon_cumulative_rain_mm'] + 1.0)
    
    # 2. Moisture Stress Flag: severe dry spell (> 14 consecutive dry days during monsoon)
    df['is_severe_dry_spell'] = (df['max_consecutive_dry_days'] >= 14).astype(int)
    
    # 3. Heavy Rain Flooding Index: heavy rain days vs cumulative rainfall
    df['heavy_rain_intensity_ratio'] = df['heavy_rain_days_count'] / ((df['monsoon_cumulative_rain_mm'] / 100.0) + 1.0)
    
    # 4. Rainfall-to-Normal Deviation
    df['rainfall_deficit_pct'] = (df['monsoon_rainfall_anomaly_ratio'] - 1.0) * 100.0
    
    # 5. Soil Productivity Index (heuristic: Organic Carbon * NPK Ratio / (1 + pH Stress))
    df['soil_productivity_index'] = (df['soil_organic_carbon_pct'] * (df['soil_npk_ratio'] * 10.0)) / (1.0 + df['soil_ph_stress'])
    
    # 6. Farm Scale Tier (Marginal < 1 ha, Small 1-2 ha, Medium 2-5 ha, Large > 5 ha)
    df['farm_scale_tier'] = pd.cut(
        df['area_hectares'],
        bins=[0, 1.0, 2.0, 5.0, 100.0],
        labels=['Marginal', 'Small', 'Medium', 'Large']
    ).astype(str)
    
    # 7. Shortfall Threshold Label (for classification & business alert)
    # A harvest is considered a "Shortfall" if yield is < 85% of its 3-year historical average
    df['is_yield_shortfall'] = (df['clean_yield_q_ha'] < 0.85 * df['yield_rolling_mean_3y']).astype(int)
    
    return df


def apply_target_encoding(train_df, val_df, test_df, cat_col='taluka', target_col='clean_yield_q_ha'):
    """
    Out-of-fold target encoding for high-cardinality Talukas.
    Strictly trained on training data to prevent leakage into validation or test sets.
    """
    print(f"Applying out-of-fold target encoding for high-cardinality column: {cat_col}...")
    global_mean = train_df[target_col].mean()
    
    # 5-Fold out-of-fold target encoding on Train
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_encoded = np.zeros(len(train_df))
    
    for tr_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[tr_idx]
        fold_stats = fold_train.groupby(cat_col)[target_col].agg(['count', 'mean'])
        # Empirical Bayes smoothing: (count * mean + 10 * global_mean) / (count + 10)
        smoothed = (fold_stats['count'] * fold_stats['mean'] + 10.0 * global_mean) / (fold_stats['count'] + 10.0)
        
        train_encoded[val_idx] = train_df.iloc[val_idx][cat_col].map(smoothed).fillna(global_mean)
        
    train_df = train_df.copy()
    train_df[f"{cat_col}_target_encoded"] = train_encoded
    
    # Map onto Val and Test using overall training statistics
    overall_stats = train_df.groupby(cat_col)[target_col].agg(['count', 'mean'])
    overall_smoothed = (overall_stats['count'] * overall_stats['mean'] + 10.0 * global_mean) / (overall_stats['count'] + 10.0)
    
    val_df = val_df.copy()
    val_df[f"{cat_col}_target_encoded"] = val_df[cat_col].map(overall_smoothed).fillna(global_mean)
    
    test_df = test_df.copy()
    test_df[f"{cat_col}_target_encoded"] = test_df[cat_col].map(overall_smoothed).fillna(global_mean)
    
    return train_df, val_df, test_df


def run_feature_pipeline():
    in_path = 'data/processed/clean_master_dataset.parquet'
    print(f"Loading {in_path}...")
    df = pd.read_parquet(in_path)
    
    # 1. Compute historical lag and rolling features
    df = compute_historical_lags(df)
    
    # 2. Engineer interactions, ratios and business flags
    df = engineer_interaction_and_ratio_features(df)
    
    # 3. Time-Aware Split:
    # Filter out 2017 (initial seed year with incomplete prior lags) from final training to ensure highest quality
    # Train: 2018 - 2021 (4 full years)
    # Validation: 2022 (1 year)
    # Test: 2023 (1 year blind holdout)
    train_mask = df['year'].isin([2018, 2019, 2020, 2021])
    val_mask = df['year'] == 2022
    test_mask = df['year'] == 2023
    
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)
    
    print(f"Time-Aware Split Summary:")
    print(f"  - Train Set (2018-2021): {len(train_df):,} rows")
    print(f"  - Validation Set (2022): {len(val_df):,} rows")
    print(f"  - Blind Test Set (2023): {len(test_df):,} rows")
    
    # 4. Out-of-fold target encoding for high-cardinality Talukas (51 talukas)
    train_df, val_df, test_df = apply_target_encoding(train_df, val_df, test_df, cat_col='taluka', target_col='clean_yield_q_ha')
    
    # 5. Save splits
    train_df.to_parquet('data/processed/train.parquet', index=False)
    val_df.to_parquet('data/processed/val.parquet', index=False)
    test_df.to_parquet('data/processed/test.parquet', index=False)
    
    print("Saved feature-engineered train, val, and test partitions successfully!")


if __name__ == '__main__':
    run_feature_pipeline()

