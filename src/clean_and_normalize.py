"""
Data Cleaning, Normalization and Spatial-Temporal Imputation Pipeline
Handles:
1. Mandi Price Harmonization: Bimodal Rs/kg vs Rs/quintal in the same column.
2. Monsoon Rainfall Imputation: Spatial neighbor interpolation & seasonal moving averages.
3. Crop Yield Unit Reconciliation: Biological envelope-based correction (Tonnes, Quintals, kg -> Quintals/ha).
4. Soil Health Metrics: NPK balance, organic carbon, and crop pH stress.
5. Unified Master Dataset Construction.
"""

import os
import numpy as np
import pandas as pd

CROP_OPTIMAL_PH = {
    'Soybean': 6.8,
    'Cotton': 7.2,
    'Tur': 6.9,
    'Gram': 7.0,
    'Wheat': 7.1,
    'Mandarin Orange': 6.5
}

CROP_NORMAL_YIELD_BOUNDS = {
    # (min_plausible_q_ha, max_plausible_q_ha)
    'Soybean': (3.0, 45.0),
    'Cotton': (4.0, 50.0),
    'Tur': (2.0, 35.0),
    'Gram': (2.5, 40.0),
    'Wheat': (6.0, 70.0),
    'Mandarin Orange': (15.0, 220.0)
}


def clean_mandi_prices(mandi_csv_path='data/raw/mandi_prices.csv'):
    print("--- 1. Cleaning Mandi Prices ---")
    df = pd.read_csv(mandi_csv_path)
    print(f"Loaded {len(df):,} mandi price records.")
    
    # Analyze bimodal price trap
    sub_250_mask = df['modal_price'] < 250
    print(f"Detected {sub_250_mask.sum():,} rows ({sub_250_mask.mean()*100:.1f}%) entered per kg (< Rs 250).")
    
    # Harmonize to Rs. / Quintal (1 Quintal = 100 kg)
    df['clean_modal_price'] = df['modal_price'].copy()
    df.loc[sub_250_mask, 'clean_modal_price'] = df.loc[sub_250_mask, 'modal_price'] * 100.0
    
    df['clean_min_price'] = df['min_price'].copy()
    df.loc[df['min_price'] < 250, 'clean_min_price'] = df.loc[df['min_price'] < 250, 'min_price'] * 100.0
    
    df['clean_max_price'] = df['max_price'].copy()
    df.loc[df['max_price'] < 250, 'clean_max_price'] = df.loc[df['max_price'] < 250, 'max_price'] * 100.0
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Pre-sowing price indicators:
    # Kharif sowing occurs in June. Pre-sowing price window is March to May (months 3-5).
    # Rabi sowing occurs in October-November. Pre-sowing price window is August to October (months 8-10).
    df['season_window'] = 'Other'
    df.loc[df['month'].isin([3, 4, 5]), 'season_window'] = 'Pre_Kharif'
    df.loc[df['month'].isin([8, 9, 10]), 'season_window'] = 'Pre_Rabi'
    
    # Compute district-crop annual pre-sowing price averages
    pre_sowing_kharif = df[df['season_window'] == 'Pre_Kharif'].groupby(
        ['district', 'crop', 'year']
    )['clean_modal_price'].agg(['mean', 'std']).reset_index().rename(
        columns={'mean': 'pre_sowing_mandi_price', 'std': 'pre_sowing_mandi_price_std'}
    )
    pre_sowing_kharif['season'] = 'Kharif'
    
    pre_sowing_rabi = df[df['season_window'] == 'Pre_Rabi'].groupby(
        ['district', 'crop', 'year']
    )['clean_modal_price'].agg(['mean', 'std']).reset_index().rename(
        columns={'mean': 'pre_sowing_mandi_price', 'std': 'pre_sowing_mandi_price_std'}
    )
    pre_sowing_rabi['season'] = 'Rabi'
    
    annual_orange = df[df['crop'] == 'Mandarin Orange'].groupby(
        ['district', 'crop', 'year']
    )['clean_modal_price'].agg(['mean', 'std']).reset_index().rename(
        columns={'mean': 'pre_sowing_mandi_price', 'std': 'pre_sowing_mandi_price_std'}
    )
    annual_orange['season'] = 'Annual'
    
    pre_sowing_df = pd.concat([pre_sowing_kharif, pre_sowing_rabi, annual_orange], ignore_index=True)
    
    # Also compute 6-district regional average per crop per year to calculate price spread
    reg_avg = pre_sowing_df.groupby(['crop', 'year', 'season'])['pre_sowing_mandi_price'].transform('mean')
    pre_sowing_df['district_mandi_price_spread'] = pre_sowing_df['pre_sowing_mandi_price'] / reg_avg
    
    print("Mandi prices successfully normalized to Rs/quintal.")
    return df, pre_sowing_df


def clean_and_impute_rainfall(rainfall_csv_path='data/raw/rainfall.csv'):
    print("--- 2. Cleaning & Imputing Rainfall ---")
    df = pd.read_csv(rainfall_csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    n_missing = df['rainfall_mm'].isna().sum()
    pct_missing = (n_missing / len(df)) * 100
    print(f"Missing rainfall readings: {n_missing:,} ({pct_missing:.2f}% of total).")
    
    # Step 1: Spatial imputation within district on same date
    district_date_median = df.groupby(['district', 'date'])['rainfall_mm'].transform('median')
    df['clean_rainfall_mm'] = df['rainfall_mm'].fillna(district_date_median)
    
    # Step 2: Regional date median if all stations in a district dropped out
    regional_date_median = df.groupby('date')['clean_rainfall_mm'].transform('median')
    df['clean_rainfall_mm'] = df['clean_rainfall_mm'].fillna(regional_date_median)
    
    # Step 3: Historical day-of-year median fallback if any remain
    df['day_of_year'] = df['date'].dt.dayofyear
    doy_median = df.groupby(['district', 'day_of_year'])['clean_rainfall_mm'].transform('median')
    df['clean_rainfall_mm'] = df['clean_rainfall_mm'].fillna(doy_median).fillna(0.0)
    
    print(f"Remaining missing rainfall readings after spatial imputation: {df['clean_rainfall_mm'].isna().sum()}")
    
    # Feature derivation:
    # Critical vegetative monsoon window (June 1 - August 31)
    monsoon_df = df[df['month'].isin([6, 7, 8])].copy()
    
    # Aggregate by district, station, year
    station_monsoon = monsoon_df.groupby(['district', 'station_id', 'year']).agg(
        cumulative_monsoon_rain=('clean_rainfall_mm', 'sum'),
        heavy_rain_days=('clean_rainfall_mm', lambda x: (x >= 65.0).sum()),
        rainy_days=('clean_rainfall_mm', lambda x: (x >= 2.5).sum())
    ).reset_index()
    
    # Max consecutive dry days during June-August (< 2.5 mm rain per day)
    def calc_max_dry_spell(series):
        is_dry = (series < 2.5).astype(int)
        max_streak = 0
        curr_streak = 0
        for val in is_dry:
            if val == 1:
                curr_streak += 1
                if curr_streak > max_streak:
                    max_streak = curr_streak
            else:
                curr_streak = 0
        return max_streak
    
    dry_spell_df = monsoon_df.groupby(['district', 'station_id', 'year'])['clean_rainfall_mm'].apply(
        calc_max_dry_spell
    ).reset_index().rename(columns={'clean_rainfall_mm': 'max_consecutive_dry_days'})
    
    station_monsoon = station_monsoon.merge(dry_spell_df, on=['district', 'station_id', 'year'])
    
    # District-level seasonal weather aggregates
    district_weather = station_monsoon.groupby(['district', 'year']).agg(
        monsoon_cumulative_rain_mm=('cumulative_monsoon_rain', 'mean'),
        heavy_rain_days_count=('heavy_rain_days', 'mean'),
        max_consecutive_dry_days=('max_consecutive_dry_days', 'mean')
    ).reset_index()
    
    # 7-year normal district monsoon rainfall to calculate rainfall anomaly
    normal_rain = district_weather.groupby('district')['monsoon_cumulative_rain_mm'].transform('mean')
    district_weather['monsoon_rainfall_anomaly_ratio'] = district_weather['monsoon_cumulative_rain_mm'] / normal_rain
    
    print("Rainfall successfully imputed and seasonal features aggregated.")
    return df, district_weather


def clean_crop_yield_units(yield_csv_path='data/raw/crop_yield.csv'):
    print("--- 3. Reconciling Crop Yield Units ---")
    df = pd.read_csv(yield_csv_path)
    print(f"Loaded {len(df):,} crop yield records.")
    
    clean_yield = []
    unit_corrected = []
    
    for _, row in df.iterrows():
        crop = row['crop']
        val = row['recorded_yield']
        min_p, max_p = CROP_NORMAL_YIELD_BOUNDS[crop]
        
        # Check if entered in kg/ha (typical values 800 - 15000)
        if val > max_p * 4.0:
            norm_val = val / 100.0
            corrected_flag = 'From_kg_ha'
        # Check if entered in Tonnes/ha (typical values 0.3 - 15.0)
        elif val < min_p * 0.40:
            norm_val = val * 10.0
            corrected_flag = 'From_tonne_ha'
        else:
            norm_val = val
            corrected_flag = 'Standard_q_ha'
            
        clean_yield.append(round(norm_val, 2))
        unit_corrected.append(corrected_flag)
        
    df['clean_yield_q_ha'] = clean_yield
    df['unit_correction_status'] = unit_corrected
    
    counts = df['unit_correction_status'].value_counts()
    print("Unit correction breakdown:")
    for k, v in counts.items():
        print(f"  - {k}: {v:,} ({v/len(df)*100:.1f}%)")
        
    return df


def aggregate_soil_tests(soil_csv_path='data/raw/soil_tests.csv'):
    print("--- 4. Aggregating Soil Tests ---")
    df = pd.read_csv(soil_csv_path)
    print(f"Loaded {len(df):,} soil test records.")
    
    # Group by district and taluka
    soil_taluka = df.groupby(['district', 'taluka']).agg(
        soil_nitrogen_kg_ha=('nitrogen_kg_ha', 'mean'),
        soil_phosphorus_kg_ha=('phosphorus_kg_ha', 'mean'),
        soil_potassium_kg_ha=('potassium_kg_ha', 'mean'),
        soil_ph=('ph', 'mean'),
        soil_organic_carbon_pct=('organic_carbon_pct', 'mean'),
        soil_ec_ds_m=('electrical_conductivity', 'mean')
    ).reset_index()
    
    # NPK Balance Ratio: N / (P + K)
    soil_taluka['soil_npk_ratio'] = soil_taluka['soil_nitrogen_kg_ha'] / (
        soil_taluka['soil_phosphorus_kg_ha'] + soil_taluka['soil_potassium_kg_ha']
    )
    
    # Also calculate district-level backup
    soil_district = df.groupby('district').agg(
        dist_soil_nitrogen_kg_ha=('nitrogen_kg_ha', 'mean'),
        dist_soil_phosphorus_kg_ha=('phosphorus_kg_ha', 'mean'),
        dist_soil_potassium_kg_ha=('potassium_kg_ha', 'mean'),
        dist_soil_ph=('ph', 'mean'),
        dist_soil_organic_carbon_pct=('organic_carbon_pct', 'mean'),
        dist_soil_ec_ds_m=('electrical_conductivity', 'mean')
    ).reset_index()
    soil_district['dist_soil_npk_ratio'] = soil_district['dist_soil_nitrogen_kg_ha'] / (
        soil_district['dist_soil_phosphorus_kg_ha'] + soil_district['dist_soil_potassium_kg_ha']
    )
    
    return soil_taluka, soil_district


def build_master_dataset():
    print("\n--- 5. Joining Tables into Master Clean Dataset ---")
    os.makedirs('data/processed', exist_ok=True)
    
    _, pre_sowing_df = clean_mandi_prices()
    _, district_weather = clean_and_impute_rainfall()
    df_yield = clean_crop_yield_units()
    soil_taluka, soil_district = aggregate_soil_tests()
    
    # Merge Yield with Pre-sowing Mandi prices
    m1 = df_yield.merge(
        pre_sowing_df[['district', 'crop', 'year', 'season', 'pre_sowing_mandi_price', 'district_mandi_price_spread']],
        on=['district', 'crop', 'year', 'season'],
        how='left'
    )
    
    # Merge with Seasonal Weather
    m2 = m1.merge(
        district_weather,
        on=['district', 'year'],
        how='left'
    )
    
    # Merge with Soil Tests by Taluka
    m3 = m2.merge(
        soil_taluka,
        on=['district', 'taluka'],
        how='left'
    )
    
    # Fallback to district soil if any taluka missing
    m3 = m3.merge(soil_district, on='district', how='left')
    for col in ['soil_nitrogen_kg_ha', 'soil_phosphorus_kg_ha', 'soil_potassium_kg_ha', 
                'soil_ph', 'soil_organic_carbon_pct', 'soil_npk_ratio']:
        dist_col = f"dist_{col}"
        m3[col] = m3[col].fillna(m3[dist_col])
        m3.drop(columns=[dist_col], inplace=True)
        
    m3.drop(columns=['dist_soil_ec_ds_m'], inplace=True, errors='ignore')
    
    # Soil pH stress: deviation from crop's optimal pH
    m3['optimal_ph'] = m3['crop'].map(CROP_OPTIMAL_PH)
    m3['soil_ph_stress'] = (m3['soil_ph'] - m3['optimal_ph']).abs()
    
    # Save master cleaned table
    out_path = 'data/processed/clean_master_dataset.parquet'
    m3.to_parquet(out_path, index=False)
    print(f"\nSuccessfully generated master clean dataset: {len(m3):,} rows -> {out_path}")
    
    # Also save a sample CSV for inspection
    sample_path = 'data/processed/clean_master_sample.csv'
    m3.head(2000).to_csv(sample_path, index=False)
    print(f"Saved sample (2,000 rows) for quick review -> {sample_path}")
    return m3


if __name__ == '__main__':
    build_master_dataset()

