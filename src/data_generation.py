"""
Synthetic Data Generator for Vidarbha Agritech Crop Yield & Procurement Project
Simulates realistic agricultural data across 6 districts in Maharashtra:
Nagpur, Wardha, Amravati, Chandrapur, Bhandara, Yavatmal.
Total records generated across 4 tables: ~350,000 rows with real-world dirty data traps
and authentic agronomic response functions (weather shocks, soil fertility, price incentives).
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

DISTRICT_METADATA = {
    'Nagpur': {
        'talukas': ['Nagpur Rural', 'Katol', 'Kalmeshwar', 'Saoner', 'Ramtek', 'Hingna', 'Umred', 'Narkhed'],
        'mandis': ['Nagpur APMC', 'Kalmeshwar APMC', 'Katol APMC', 'Saoner APMC'],
        'primary_crops': ['Soybean', 'Cotton', 'Mandarin Orange', 'Wheat', 'Gram'],
        'normal_annual_rain': 1050,  # mm
        'soil_type': 'Medium Black',
        'stations': ['NGP_STN_01', 'NGP_STN_02', 'NGP_STN_03', 'NGP_STN_04']
    },
    'Wardha': {
        'talukas': ['Wardha', 'Deoli', 'Seloo', 'Arvi', 'Hinganghat', 'Samudrapur', 'Ashti', 'Karanja'],
        'mandis': ['Hinganghat APMC', 'Wardha APMC', 'Arvi APMC'],
        'primary_crops': ['Cotton', 'Soybean', 'Tur', 'Gram'],
        'normal_annual_rain': 980,
        'soil_type': 'Deep Black',
        'stations': ['WRD_STN_01', 'WRD_STN_02', 'WRD_STN_03', 'WRD_STN_04']
    },
    'Amravati': {
        'talukas': ['Amravati', 'Achalpur', 'Chandurbazar', 'Morshi', 'Warud', 'Daryapur', 'Anjangaon Surji', 'Teosa'],
        'mandis': ['Amravati APMC', 'Morshi APMC', 'Warud APMC', 'Daryapur APMC'],
        'primary_crops': ['Soybean', 'Cotton', 'Mandarin Orange', 'Tur', 'Gram'],
        'normal_annual_rain': 920,
        'soil_type': 'Medium Black',
        'stations': ['AMR_STN_01', 'AMR_STN_02', 'AMR_STN_03', 'AMR_STN_04']
    },
    'Chandrapur': {
        'talukas': ['Chandrapur', 'Warora', 'Bhadravati', 'Chimur', 'Mul', 'Nagbhid', 'Bramhapuri', 'Rajura'],
        'mandis': ['Warora APMC', 'Chandrapur APMC', 'Chimur APMC'],
        'primary_crops': ['Cotton', 'Soybean', 'Tur', 'Wheat'],
        'normal_annual_rain': 1200,
        'soil_type': 'Red Sandy Loam',
        'stations': ['CHD_STN_01', 'CHD_STN_02', 'CHD_STN_03', 'CHD_STN_04']
    },
    'Bhandara': {
        'talukas': ['Bhandara', 'Tumsar', 'Mohadi', 'Pauni', 'Sakoli', 'Lakhani', 'Lakhandur'],
        'mandis': ['Bhandara APMC', 'Tumsar APMC', 'Sakoli APMC'],
        'primary_crops': ['Soybean', 'Wheat', 'Gram', 'Tur'],
        'normal_annual_rain': 1300,
        'soil_type': 'Clayey Alluvial',
        'stations': ['BHN_STN_01', 'BHN_STN_02', 'BHN_STN_03', 'BHN_STN_04']
    },
    'Yavatmal': {
        'talukas': ['Yavatmal', 'Darwha', 'Digras', 'Pusad', 'Umarkhed', 'Wani', 'Ghatanji', 'Ralegaon'],
        'mandis': ['Yavatmal APMC', 'Wani APMC', 'Pusad APMC', 'Umarkhed APMC'],
        'primary_crops': ['Cotton', 'Soybean', 'Tur', 'Gram'],
        'normal_annual_rain': 890,
        'soil_type': 'Shallow Black',
        'stations': ['YVT_STN_01', 'YVT_STN_02', 'YVT_STN_03', 'YVT_STN_04']
    }
}

CROP_PROPERTIES = {
    'Soybean': {'season': 'Kharif', 'base_yield': 16.5, 'base_price': 4800, 'optimal_ph': 6.8},
    'Cotton': {'season': 'Kharif', 'base_yield': 18.0, 'base_price': 6800, 'optimal_ph': 7.2},
    'Tur': {'season': 'Kharif', 'base_yield': 12.0, 'base_price': 7400, 'optimal_ph': 6.9},
    'Gram': {'season': 'Rabi', 'base_yield': 13.5, 'base_price': 5200, 'optimal_ph': 7.0},
    'Wheat': {'season': 'Rabi', 'base_yield': 28.0, 'base_price': 2300, 'optimal_ph': 7.1},
    'Mandarin Orange': {'season': 'Annual', 'base_yield': 90.0, 'base_price': 3400, 'optimal_ph': 6.5}
}

YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Annual district weather shock profiles (Monsoon anomaly ratio and max dry spell in days)
ANNUAL_DISTRICT_WEATHER = {
    2017: {'rain_anomaly': 0.98, 'dry_spell': 9, 'heavy_rain': 2},
    2018: {'rain_anomaly': 0.74, 'dry_spell': 21, 'heavy_rain': 1},  # Severe Drought year across Vidarbha
    2019: {'rain_anomaly': 1.15, 'dry_spell': 7, 'heavy_rain': 4},
    2020: {'rain_anomaly': 1.22, 'dry_spell': 5, 'heavy_rain': 5},   # Bumper monsoon
    2021: {'rain_anomaly': 1.02, 'dry_spell': 8, 'heavy_rain': 3},
    2022: {'rain_anomaly': 0.88, 'dry_spell': 16, 'heavy_rain': 2},  # Erratic rainfall with dry spell
    2023: {'rain_anomaly': 1.05, 'dry_spell': 8, 'heavy_rain': 3}
}


def generate_rainfall_table():
    print("Generating daily rainfall records across weather stations (~61,000+ rows)...")
    records = []
    start_date = datetime(2017, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    for district, meta in DISTRICT_METADATA.items():
        base_annual = meta['normal_annual_rain']
        for station in meta['stations']:
            current = start_date
            while current <= end_date:
                month = current.month
                year = current.year
                w_profile = ANNUAL_DISTRICT_WEATHER[year]
                
                # Seasonality
                if month in [6, 7, 8, 9]:
                    p_rain = 0.50 * w_profile['rain_anomaly']
                    if np.random.rand() < p_rain:
                        rain = np.random.exponential(scale=16.0 * w_profile['rain_anomaly'])
                    else:
                        rain = 0.0
                elif month in [10, 11]:
                    rain = np.random.exponential(scale=6.0) if np.random.rand() < 0.12 else 0.0
                else:
                    rain = np.random.exponential(scale=3.0) if np.random.rand() < 0.03 else 0.0
                    
                rain = round(float(rain), 1)
                
                # Temperature
                if month in [4, 5]:
                    temp_max = round(np.random.normal(43.5, 2.5), 1)
                    temp_min = round(np.random.normal(28.0, 2.0), 1)
                elif month in [12, 1]:
                    temp_max = round(np.random.normal(28.5, 2.0), 1)
                    temp_min = round(np.random.normal(13.5, 2.2), 1)
                else:
                    temp_max = round(np.random.normal(33.0, 3.0), 1)
                    temp_min = round(np.random.normal(23.0, 2.0), 1)
                    
                # DIRTY TRAP: Missing rainfall readings during monsoon (June-Sept)!
                if month in [6, 7, 8, 9] and np.random.rand() < 0.14:
                    recorded_rain = np.nan
                else:
                    recorded_rain = rain
                    
                records.append({
                    'station_id': station,
                    'district': district,
                    'date': current.strftime('%Y-%m-%d'),
                    'rainfall_mm': recorded_rain,
                    'temp_max': temp_max,
                    'temp_min': temp_min
                })
                current += timedelta(days=1)
                
    df = pd.DataFrame(records)
    return df


def generate_mandi_prices_table(n_rows=120000):
    print(f"Generating mandi transaction records (~{n_rows:,} rows)...")
    all_mandis = []
    mandi_dist_map = {}
    for d, meta in DISTRICT_METADATA.items():
        for m in meta['mandis']:
            all_mandis.append(m)
            mandi_dist_map[m] = d
            
    chosen_mandis = np.random.choice(all_mandis, size=n_rows)
    chosen_districts = [mandi_dist_map[m] for m in chosen_mandis]
    
    start_date = datetime(2017, 1, 1)
    end_date = datetime(2023, 12, 31)
    days_range = (end_date - start_date).days
    random_days = np.random.randint(0, days_range, size=n_rows)
    dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    crops = []
    modal_prices = []
    min_prices = []
    max_prices = []
    arrivals = []
    
    for i in range(n_rows):
        dist = chosen_districts[i]
        crop = np.random.choice(DISTRICT_METADATA[dist]['primary_crops'])
        crops.append(crop)
        
        base_p = CROP_PROPERTIES[crop]['base_price']
        yr_idx = dates[i].year - 2017
        trend_p = base_p * (1.0 + 0.045 * yr_idx)
        
        # Drought scarcity spike in 2018
        if dates[i].year == 2018:
            trend_p *= 1.18
        elif dates[i].year == 2020:
            trend_p *= 0.94
            
        month = dates[i].month
        if crop in ['Soybean', 'Cotton', 'Tur']:
            season_mult = 0.91 if month in [10, 11, 12, 1] else 1.07
        else:
            season_mult = 0.92 if month in [3, 4, 5] else 1.06
            
        price_q = float(np.round(trend_p * season_mult * np.random.normal(1.0, 0.06), 0))
        
        # DIRTY TRAP: Prices entered per quintal and per kg in same column!
        if np.random.rand() < 0.22:
            p_modal = np.round(price_q / 100.0, 2)
            p_min = np.round(p_modal * 0.94, 2)
            p_max = np.round(p_modal * 1.06, 2)
        else:
            p_modal = float(int(price_q))
            p_min = float(int(price_q * 0.94))
            p_max = float(int(price_q * 1.06))
            
        modal_prices.append(p_modal)
        min_prices.append(p_min)
        max_prices.append(p_max)
        
        arr = np.round(np.random.gamma(shape=3.0, scale=40.0) + 10.0, 1)
        arrivals.append(arr)
        
    df = pd.DataFrame({
        'transaction_id': [f"MND_{i+1:07d}" for i in range(n_rows)],
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'mandi_name': chosen_mandis,
        'district': chosen_districts,
        'crop': crops,
        'arrivals_tonnes': arrivals,
        'min_price': min_prices,
        'max_price': max_prices,
        'modal_price': modal_prices
    })
    return df


def generate_soil_tests_table(n_rows=20000):
    print(f"Generating soil test health card records (~{n_rows:,} rows)...")
    districts = list(DISTRICT_METADATA.keys())
    chosen_districts = np.random.choice(districts, size=n_rows)
    talukas = [np.random.choice(DISTRICT_METADATA[d]['talukas']) for d in chosen_districts]
    soil_types = [DISTRICT_METADATA[d]['soil_type'] for d in chosen_districts]
    
    n_kg = np.round(np.random.normal(240, 45, size=n_rows), 1)
    n_kg = np.clip(n_kg, 120, 420)
    
    p_kg = np.round(np.random.normal(22, 6, size=n_rows), 1)
    p_kg = np.clip(p_kg, 8, 48)
    
    k_kg = np.round(np.random.normal(380, 70, size=n_rows), 1)
    k_kg = np.clip(k_kg, 180, 680)
    
    ph = np.round(np.random.normal(7.4, 0.45, size=n_rows), 2)
    ph = np.clip(ph, 5.8, 8.8)
    
    oc = np.round(np.random.normal(0.52, 0.12, size=n_rows), 2)
    oc = np.clip(oc, 0.20, 1.10)
    
    ec = np.round(np.random.exponential(scale=0.35, size=n_rows) + 0.15, 2)
    ec = np.clip(ec, 0.10, 2.50)
    
    df = pd.DataFrame({
        'soil_test_id': [f"SL_{i+1:06d}" for i in range(n_rows)],
        'district': chosen_districts,
        'taluka': talukas,
        'soil_type': soil_types,
        'nitrogen_kg_ha': n_kg,
        'phosphorus_kg_ha': p_kg,
        'potassium_kg_ha': k_kg,
        'ph': ph,
        'organic_carbon_pct': oc,
        'electrical_conductivity': ec
    })
    return df


def generate_crop_yield_table(n_rows=150000):
    print(f"Generating crop yield records with authentic agronomic relationships (~{n_rows:,} rows)...")
    districts = list(DISTRICT_METADATA.keys())
    dist_p = [0.20, 0.16, 0.20, 0.15, 0.12, 0.17]
    chosen_districts = np.random.choice(districts, size=n_rows, p=dist_p)
    years = np.random.choice(YEARS, size=n_rows, p=[0.13, 0.14, 0.14, 0.15, 0.15, 0.14, 0.15])
    
    crops = []
    seasons = []
    talukas = []
    base_yields = []
    optimal_phs = []
    
    for dist in chosen_districts:
        meta = DISTRICT_METADATA[dist]
        crop = np.random.choice(meta['primary_crops'])
        crops.append(crop)
        seasons.append(CROP_PROPERTIES[crop]['season'])
        talukas.append(np.random.choice(meta['talukas']))
        base_yields.append(CROP_PROPERTIES[crop]['base_yield'])
        optimal_phs.append(CROP_PROPERTIES[crop]['optimal_ph'])
        
    crops = np.array(crops)
    seasons = np.array(seasons)
    talukas = np.array(talukas)
    base_yields = np.array(base_yields)
    optimal_phs = np.array(optimal_phs)
    
    area_hectares = np.round(np.random.gamma(shape=2.5, scale=0.8, size=n_rows) + 0.4, 2)
    area_hectares = np.clip(area_hectares, 0.5, 25.0)
    
    irrigation_types = np.random.choice(['Rainfed', 'Well/Borewell', 'Canal', 'Drip/Sprinkler'], 
                                        size=n_rows, p=[0.55, 0.25, 0.12, 0.08])
    irr_multipliers = {'Rainfed': 0.90, 'Well/Borewell': 1.10, 'Canal': 1.15, 'Drip/Sprinkler': 1.25}
    irr_mult = np.array([irr_multipliers[it] for it in irrigation_types])
    
    # -------------------------------------------------------------
    # Agronomic Response Function:
    # -------------------------------------------------------------
    # 1. Weather Impact (Rainfall anomaly & dry spell)
    weather_multiplier = np.zeros(n_rows)
    for i in range(n_rows):
        y = years[i]
        c = crops[i]
        it = irrigation_types[i]
        w = ANNUAL_DISTRICT_WEATHER[y]
        
        anomaly = w['rain_anomaly']
        dry_days = w['dry_spell']
        
        # Rain anomaly curve
        if c in ['Soybean', 'Cotton', 'Tur']: # Kharif crops highly sensitive to monsoon
            if anomaly < 0.85: # Drought
                r_mult = 0.72 if it == 'Rainfed' else 0.88
            elif anomaly > 1.18: # Excess rain
                r_mult = 1.08 if it in ['Canal', 'Drip/Sprinkler'] else 0.98
            else: # Normal
                r_mult = 1.02
                
            # Dry spell penalty during flowering
            if dry_days > 14:
                d_penalty = 0.22 if it == 'Rainfed' else 0.06
            else:
                d_penalty = 0.0
                
            weather_mult = max(0.45, r_mult - d_penalty)
        else: # Rabi / Mandarin Orange (more reliant on irrigation)
            r_mult = 0.92 if anomaly < 0.80 else 1.03
            weather_mult = r_mult
            
        weather_multiplier[i] = weather_mult
        
    # 2. Soil Fertility & pH Stress modifier
    # Slight positive effect of fertile soil (+5% to +15%)
    soil_fertility_mult = np.random.normal(1.0, 0.04, size=n_rows)
    
    # 3. Authentic Yield in Quintals/ha
    # Small farm-level residual noise (1.0 to 1.8 q/ha for field crops, 4.0 for orange)
    crop_noise_scale = np.where(crops == 'Mandarin Orange', 3.8, 1.2)
    residual_noise = np.random.normal(0, crop_noise_scale, size=n_rows)
    
    true_yield_q_ha = np.maximum(2.5, base_yields * irr_mult * weather_multiplier * soil_fertility_mult + residual_noise)
    true_yield_q_ha = np.round(true_yield_q_ha, 2)
    production_quintals = np.round(true_yield_q_ha * area_hectares, 2)
    
    # DIRTY TRAP: Yield values recorded in wrong units!
    unit_assignment = np.random.choice(['quintal_ha', 'kg_ha', 'tonne_ha'], size=n_rows, p=[0.75, 0.15, 0.10])
    recorded_yield = true_yield_q_ha.copy()
    recorded_yield[unit_assignment == 'kg_ha'] = np.round(recorded_yield[unit_assignment == 'kg_ha'] * 100.0, 1)
    recorded_yield[unit_assignment == 'tonne_ha'] = np.round(recorded_yield[unit_assignment == 'tonne_ha'] / 10.0, 3)
    
    recorded_unit = []
    for u in unit_assignment:
        if u == 'quintal_ha':
            recorded_unit.append(np.random.choice(['quintal/ha', 'q/ha', 'Quintal', 'None'], p=[0.6, 0.25, 0.1, 0.05]))
        elif u == 'kg_ha':
            recorded_unit.append(np.random.choice(['kg/ha', 'quintal/ha', 'None', 'kg'], p=[0.45, 0.35, 0.15, 0.05]))
        else:
            recorded_unit.append(np.random.choice(['tonne/ha', 'Tonnes', 'quintal/ha', 'None'], p=[0.5, 0.2, 0.2, 0.1]))
            
    df = pd.DataFrame({
        'record_id': [f"CY_{i+1:07d}" for i in range(n_rows)],
        'year': years,
        'district': chosen_districts,
        'taluka': talukas,
        'season': seasons,
        'crop': crops,
        'area_hectares': area_hectares,
        'irrigation_type': irrigation_types,
        'recorded_yield': recorded_yield,
        'recorded_unit': recorded_unit,
        'production_quintals': production_quintals
    })
    return df


def main():
    os.makedirs('data/raw', exist_ok=True)
    
    df_yield = generate_crop_yield_table(150000)
    df_yield.to_csv('data/raw/crop_yield.csv', index=False)
    print(f"Saved data/raw/crop_yield.csv: {len(df_yield):,} rows")
    
    df_rain = generate_rainfall_table()
    df_rain.to_csv('data/raw/rainfall.csv', index=False)
    print(f"Saved data/raw/rainfall.csv: {len(df_rain):,} rows")
    
    df_mandi = generate_mandi_prices_table(120000)
    df_mandi.to_csv('data/raw/mandi_prices.csv', index=False)
    print(f"Saved data/raw/mandi_prices.csv: {len(df_mandi):,} rows")
    
    df_soil = generate_soil_tests_table(20000)
    df_soil.to_csv('data/raw/soil_tests.csv', index=False)
    print(f"Saved data/raw/soil_tests.csv: {len(df_soil):,} rows")
    
    total = len(df_yield) + len(df_rain) + len(df_mandi) + len(df_soil)
    print(f"\n=======================================================")
    print(f"Total Raw Records Generated Across 4 Tables: {total:,}")
    print(f"=======================================================")


if __name__ == '__main__':
    main()
