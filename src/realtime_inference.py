"""
Real-Time Inference Engine for District-Level Crop Yield & Procurement Decisioning
Allows business users, procurement managers, and automated data pipelines to:
1. Score individual farm/district scenarios interactively.
2. Run batch predictions on newly incoming mid-season weather/mandi data.
3. Classify shortfall risk and assign strategic procurement action tiers.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = 'models/crop_yield_lgbm_best.joblib'
CROP_MEANS_PATH = 'models/crop_base_means.joblib'
HISTORY_PATH = 'models/dist_crop_history.joblib'
TRAIN_DATA_PATH = 'data/processed/train.parquet'

DISTRICT_DEFAULTS = {
    'Nagpur': {'soil_n': 240.0, 'soil_p': 22.0, 'soil_k': 380.0, 'ph': 7.4, 'oc': 0.52, 'normal_rain': 1050.0},
    'Wardha': {'soil_n': 235.0, 'soil_p': 20.0, 'soil_k': 390.0, 'ph': 7.5, 'oc': 0.50, 'normal_rain': 980.0},
    'Amravati': {'soil_n': 245.0, 'soil_p': 23.0, 'soil_k': 375.0, 'ph': 7.3, 'oc': 0.54, 'normal_rain': 920.0},
    'Chandrapur': {'soil_n': 230.0, 'soil_p': 21.0, 'soil_k': 360.0, 'ph': 7.1, 'oc': 0.56, 'normal_rain': 1200.0},
    'Bhandara': {'soil_n': 255.0, 'soil_p': 24.0, 'soil_k': 385.0, 'ph': 7.0, 'oc': 0.58, 'normal_rain': 1300.0},
    'Yavatmal': {'soil_n': 225.0, 'soil_p': 19.0, 'soil_k': 370.0, 'ph': 7.6, 'oc': 0.48, 'normal_rain': 890.0}
}

CROP_OPTIMAL_PH = {
    'Soybean': 6.8, 'Cotton': 7.2, 'Tur': 6.9,
    'Gram': 7.0, 'Wheat': 7.1, 'Mandarin Orange': 6.5
}


class RealtimeYieldPredictor:
    def __init__(self, model_path=MODEL_PATH, crop_means_path=CROP_MEANS_PATH, history_path=HISTORY_PATH):
        if not os.path.exists(model_path) or not os.path.exists(crop_means_path):
            raise FileNotFoundError("Model artifacts not found. Please run src/train_and_tune.py first.")
        self.model = joblib.load(model_path)
        self.crop_means = joblib.load(crop_means_path)
        
        # Load historical baselines from fast cache
        if os.path.exists(history_path):
            self.dist_crop_history = joblib.load(history_path)
        else:
            train_df = pd.read_parquet(TRAIN_DATA_PATH)
            self.dist_crop_history = train_df.groupby(['district', 'crop'], observed=False).agg(
                hist_mean_yield=('clean_yield_q_ha', 'mean'),
                hist_std_yield=('clean_yield_q_ha', 'std'),
                hist_price=('pre_sowing_mandi_price', 'mean'),
                taluka_encoded=('taluka_target_encoded', 'mean')
            ).to_dict('index')

    def predict_scenario(self, scenario: dict) -> dict:
        """
        Takes raw real-time operational inputs:
        - district: str ('Nagpur', 'Wardha', 'Amravati', 'Chandrapur', 'Bhandara', 'Yavatmal')
        - crop: str ('Soybean', 'Cotton', 'Tur', 'Gram', 'Wheat', 'Mandarin Orange')
        - season: str ('Kharif', 'Rabi', 'Annual')
        - irrigation_type: str ('Rainfed', 'Well/Borewell', 'Canal', 'Drip/Sprinkler')
        - area_hectares: float (e.g. 2.5)
        - monsoon_cumulative_rain_mm: float (e.g. 750.0)
        - max_consecutive_dry_days: int (e.g. 16)
        - current_mandi_price_rs_q: float (e.g. 5200.0)
        """
        district = scenario.get('district', 'Nagpur')
        crop = scenario.get('crop', 'Soybean')
        season = scenario.get('season', 'Kharif')
        irrigation = scenario.get('irrigation_type', 'Rainfed')
        area = float(scenario.get('area_hectares', 2.0))
        rain_mm = float(scenario.get('monsoon_cumulative_rain_mm', 850.0))
        dry_days = int(scenario.get('max_consecutive_dry_days', 8))
        price = float(scenario.get('current_mandi_price_rs_q', 5000.0))
        
        # Look up baseline & soil defaults
        dist_meta = DISTRICT_DEFAULTS.get(district, DISTRICT_DEFAULTS['Nagpur'])
        hist_key = (district, crop)
        hist_meta = self.dist_crop_history.get(hist_key, {
            'hist_mean_yield': self.crop_means.get(crop, 15.0),
            'hist_std_yield': 2.5,
            'hist_price': price,
            'taluka_encoded': 15.0
        })
        
        hist_yield = hist_meta['hist_mean_yield']
        hist_price = hist_meta['hist_price']
        normal_rain = dist_meta['normal_rain']
        
        # Real-time feature derivation
        rain_anomaly = rain_mm / normal_rain
        dry_severity = (dry_days * 100.0) / (rain_mm + 1.0)
        is_severe_dry = 1 if dry_days >= 14 else 0
        heavy_rain_count = 3.0 if rain_mm > 950 else 1.0
        heavy_intensity = heavy_rain_count / ((rain_mm / 100.0) + 1.0)
        rain_deficit_pct = (rain_anomaly - 1.0) * 100.0
        
        # Price dynamics
        price_momentum = price / (hist_price + 1e-5)
        price_spread = 1.02  # regional benchmark parity
        
        # Soil metrics
        n = dist_meta['soil_n']
        p = dist_meta['soil_p']
        k = dist_meta['soil_k']
        ph = dist_meta['ph']
        oc = dist_meta['oc']
        npk_ratio = n / (p + k)
        opt_ph = CROP_OPTIMAL_PH.get(crop, 7.0)
        ph_stress = abs(ph - opt_ph)
        soil_prod_index = (oc * (npk_ratio * 10.0)) / (1.0 + ph_stress)
        
        # Scale tier
        if area <= 1.0:
            scale_tier = 'Marginal'
        elif area <= 2.0:
            scale_tier = 'Small'
        elif area <= 5.0:
            scale_tier = 'Medium'
        else:
            scale_tier = 'Large'
            
        # Assemble feature row matching model schema
        features_dict = {
            'area_hectares': area,
            'pre_sowing_mandi_price': price,
            'district_mandi_price_spread': price_spread,
            'mandi_price_momentum_ratio': price_momentum,
            'monsoon_cumulative_rain_mm': rain_mm,
            'heavy_rain_days_count': heavy_rain_count,
            'max_consecutive_dry_days': dry_days,
            'monsoon_rainfall_anomaly_ratio': rain_anomaly,
            'dry_spell_severity_index': dry_severity,
            'is_severe_dry_spell': is_severe_dry,
            'heavy_rain_intensity_ratio': heavy_intensity,
            'rainfall_deficit_pct': rain_deficit_pct,
            'soil_nitrogen_kg_ha': n,
            'soil_phosphorus_kg_ha': p,
            'soil_potassium_kg_ha': k,
            'soil_ph': ph,
            'soil_organic_carbon_pct': oc,
            'soil_npk_ratio': npk_ratio,
            'soil_ph_stress': ph_stress,
            'soil_productivity_index': soil_prod_index,
            'yield_lag_1y': hist_yield,
            'yield_lag_2y': hist_yield,
            'yield_rolling_mean_3y': hist_yield,
            'yield_rolling_std_3y': hist_meta['hist_std_yield'],
            'taluka_target_encoded': hist_meta['taluka_encoded'],
            'district': district,
            'crop': crop,
            'season': season,
            'irrigation_type': irrigation,
            'farm_scale_tier': scale_tier
        }
        
        input_df = pd.DataFrame([features_dict])
        for c in ['district', 'crop', 'season', 'irrigation_type', 'farm_scale_tier']:
            input_df[c] = input_df[c].astype('category')
            
        # Model inference
        predicted_multiplier = float(self.model.predict(input_df)[0])
        crop_base = float(self.crop_means.get(crop, 15.0))
        predicted_yield_q_ha = round(predicted_multiplier * crop_base, 2)
        total_production_quintals = round(predicted_yield_q_ha * area, 1)
        
        # Business calculations
        surplus_pct = round(((predicted_yield_q_ha - hist_yield) / hist_yield) * 100.0, 1)
        
        # Risk & Action classification
        if predicted_yield_q_ha < 0.85 * hist_yield:
            risk_alert = "CRITICAL SHORTFALL ALERT"
            action_tier = "Tier 3: Defensive / Deficit Hedging"
            action_recommendation = (
                f"Anticipated {abs(surplus_pct)}% yield drop. Divert warehouse logistics to neighboring districts; "
                f"avoid forward purchase volume commitments and initiate commodity exchange hedging."
            )
        elif surplus_pct >= 8.0:
            risk_alert = "HIGH YIELD SURPLUS"
            action_tier = "Tier 1: Aggressive Forward Procurement"
            action_recommendation = (
                f"Surplus of +{surplus_pct}% projected. Lock in 60-day pre-harvest forward contracts with local FPOs "
                f"at 6–8% pre-harvest discount; reserve district warehouse capacity immediately."
            )
        else:
            risk_alert = "NORMAL / BALANCED PRODUCTION"
            action_tier = "Tier 2: Balanced Dynamic Sourcing"
            action_recommendation = (
                f"Production aligned with historical baseline ({surplus_pct:+.1f}%). "
                f"Execute flexible spot purchasing during post-harvest peak mandi arrival dips."
            )
            
        return {
            'input_scenario': scenario,
            'predicted_yield_quintals_per_hectare': predicted_yield_q_ha,
            'projected_production_quintals': total_production_quintals,
            'historical_baseline_yield_q_ha': round(hist_yield, 2),
            'projected_surplus_deficit_pct': surplus_pct,
            'risk_status': risk_alert,
            'strategic_procurement_tier': action_tier,
            'procurement_action': action_recommendation,
            'key_driving_signals': {
                'monsoon_rainfall_anomaly': f"{rain_anomaly*100:.1f}% of normal",
                'max_dry_spell': f"{dry_days} consecutive dry days",
                'irrigation_buffer': irrigation,
                'price_momentum_incentive': f"{price_momentum:.2f}x vs 3-yr price"
            }
        }


def run_interactive_cli():
    print("=" * 70)
    print("   NAGPUR AGRITECH: REAL-TIME CROP YIELD & PROCUREMENT PREDICTOR")
    print("=" * 70)
    
    predictor = RealtimeYieldPredictor()
    
    print("\nSelect a sample scenario or enter custom inputs:")
    print("1. Severe Drought Stress (Yavatmal Rainfed Cotton - Dry Spell 21 Days)")
    print("2. Favorable Bumper Monsoon (Bhandara Irrigated Soybean - Rain 1100mm)")
    print("3. High Market Price Incentive (Nagpur Mandarin Orange - High Price)")
    print("4. Custom Real-Time Scenario")
    
    choice = input("\nEnter choice (1-4) [Default 1]: ").strip() or "1"
    
    if choice == "1":
        scenario = {
            'district': 'Yavatmal',
            'crop': 'Cotton',
            'season': 'Kharif',
            'irrigation_type': 'Rainfed',
            'area_hectares': 3.5,
            'monsoon_cumulative_rain_mm': 620.0,
            'max_consecutive_dry_days': 21,
            'current_mandi_price_rs_q': 7100.0
        }
    elif choice == "2":
        scenario = {
            'district': 'Bhandara',
            'crop': 'Soybean',
            'season': 'Kharif',
            'irrigation_type': 'Canal',
            'area_hectares': 4.0,
            'monsoon_cumulative_rain_mm': 1250.0,
            'max_consecutive_dry_days': 5,
            'current_mandi_price_rs_q': 5100.0
        }
    elif choice == "3":
        scenario = {
            'district': 'Nagpur',
            'crop': 'Mandarin Orange',
            'season': 'Annual',
            'irrigation_type': 'Drip/Sprinkler',
            'area_hectares': 5.0,
            'monsoon_cumulative_rain_mm': 1050.0,
            'max_consecutive_dry_days': 8,
            'current_mandi_price_rs_q': 4200.0
        }
    else:
        print("\nEnter custom scenario parameters:")
        dist = input("District [Nagpur/Wardha/Amravati/Chandrapur/Bhandara/Yavatmal]: ").strip() or "Nagpur"
        crop = input("Crop [Soybean/Cotton/Tur/Gram/Wheat/Mandarin Orange]: ").strip() or "Soybean"
        irr = input("Irrigation [Rainfed/Well/Borewell/Canal/Drip/Sprinkler]: ").strip() or "Rainfed"
        rain = float(input("Monsoon Rainfall (mm) [e.g. 850]: ").strip() or "850")
        dry = int(input("Max Consecutive Dry Days during Monsoon [e.g. 10]: ").strip() or "10")
        price = float(input("Current Mandi Price (Rs/quintal) [e.g. 5200]: ").strip() or "5200")
        scenario = {
            'district': dist,
            'crop': crop,
            'season': 'Kharif' if crop in ['Soybean', 'Cotton', 'Tur'] else ('Rabi' if crop in ['Gram', 'Wheat'] else 'Annual'),
            'irrigation_type': irr,
            'area_hectares': 2.5,
            'monsoon_cumulative_rain_mm': rain,
            'max_consecutive_dry_days': dry,
            'current_mandi_price_rs_q': price
        }
        
    print("\nRunning real-time inference...")
    result = predictor.predict_scenario(scenario)
    
    print("\n" + "=" * 70)
    print("                     REAL-TIME INFERENCE RESULT")
    print("=" * 70)
    print(f"Location & Crop:       {scenario['district']} | {scenario['crop']} ({scenario['irrigation_type']} Irrigation)")
    print(f"Farm Scale:            {scenario['area_hectares']} Hectares")
    print(f"----------------------------------------------------------------------")
    print(f"PREDICTED YIELD:       {result['predicted_yield_quintals_per_hectare']} Quintals / Hectare")
    print(f"Historical Baseline:   {result['historical_baseline_yield_q_ha']} Quintals / Hectare")
    print(f"Projected Variance:    {result['projected_surplus_deficit_pct']:+.1f}% vs Normal")
    print(f"Total Production Est:  {result['projected_production_quintals']} Quintals")
    print(f"----------------------------------------------------------------------")
    print(f"RISK ALERT LEVEL:      {result['risk_status']}")
    print(f"STRATEGIC TIER:        {result['strategic_procurement_tier']}")
    print(f"RECOMMENDED ACTION:    {result['procurement_action']}")
    print(f"----------------------------------------------------------------------")
    print("KEY DRIVER SIGNALS:")
    for k, v in result['key_driving_signals'].items():
        print(f"  • {k:<25}: {v}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_interactive_cli()
