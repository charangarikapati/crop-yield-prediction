"""
Business Procurement Strategy & District-Crop Prioritization Engine
Translates model yield forecasts, mandi price dynamics, and shortfall risk into
an actionable procurement playbook for the Nagpur Agritech procurement leadership.
"""

import os
import json
import numpy as np
import pandas as pd

# Set Matplotlib cache dir to workspace
os.environ['MPLCONFIGDIR'] = os.path.abspath('docs/.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_procurement_strategy():
    print("--- Generating Procurement Prioritization Matrix ---")
    os.makedirs('docs', exist_ok=True)
    
    test_df = pd.read_parquet('data/processed/test_with_predictions.parquet')
    
    # Aggregate test year (2023) by District and Crop
    summary = test_df.groupby(['district', 'crop']).agg(
        surveyed_farms=('record_id', 'count'),
        total_area_ha=('area_hectares', 'sum'),
        mean_predicted_yield_q_ha=('predicted_yield_q_ha', 'mean'),
        mean_baseline_yield_q_ha=('yield_rolling_mean_3y', 'mean'),
        mean_mandi_price_rs_q=('pre_sowing_mandi_price', 'mean'),
        mandi_price_spread=('district_mandi_price_spread', 'mean'),
        max_consecutive_dry_days=('max_consecutive_dry_days', 'mean')
    ).reset_index()
    
    # Expected production in thousand tonnes (1 Quintal = 0.1 Tonnes)
    summary['expected_production_k_tonnes'] = (summary['mean_predicted_yield_q_ha'] * summary['total_area_ha'] * 0.1) / 1000.0
    summary['baseline_production_k_tonnes'] = (summary['mean_baseline_yield_q_ha'] * summary['total_area_ha'] * 0.1) / 1000.0
    
    # Surplus / Deficit Ratio vs Normal
    summary['yield_surplus_ratio'] = summary['mean_predicted_yield_q_ha'] / summary['mean_baseline_yield_q_ha']
    summary['surplus_pct'] = (summary['yield_surplus_ratio'] - 1.0) * 100.0
    
    # Priority Scoring Formula:
    # 50% Surplus bonus + 25% Price Cost Advantage + 25% Low Dry-Spell Risk
    surplus_score = np.clip((summary['yield_surplus_ratio'] - 0.8) / 0.4, 0.0, 1.0) * 50.0
    price_advantage = np.clip((1.05 - summary['mandi_price_spread']) / 0.1, 0.0, 1.0) * 25.0
    weather_safety = np.clip((20.0 - summary['max_consecutive_dry_days']) / 15.0, 0.0, 1.0) * 25.0
    
    summary['procurement_priority_score'] = np.round(surplus_score + price_advantage + weather_safety, 1)
    
    # Categorize into Strategic Action Tiers
    tiers = []
    actions = []
    for _, r in summary.iterrows():
        score = r['procurement_priority_score']
        surplus = r['surplus_pct']
        if score >= 70.0 and surplus >= 0.0:
            tiers.append('Tier 1: Aggressive Forward Procurement')
            actions.append('Lock in 60-day advance pre-harvest contracts with FPOs at discount; reserve local warehouse space.')
        elif score >= 50.0:
            tiers.append('Tier 2: Balanced Dynamic Buying')
            actions.append('Monitor weekly mandi arrivals; execute spot purchasing during peak post-harvest harvest arrivals.')
        else:
            tiers.append('Tier 3: Defensive / Deficit Hedging')
            actions.append('Yield shortfall alert: Divert logistics to surplus districts; avoid forward volume commitments.')
            
    summary['procurement_tier'] = tiers
    summary['strategic_action'] = actions
    
    # Sort by priority score
    summary = summary.sort_values('procurement_priority_score', ascending=False).reset_index(drop=True)
    
    # Save CSV
    out_csv = 'docs/procurement_prioritization_matrix.csv'
    summary.to_csv(out_csv, index=False)
    print(f"Saved procurement matrix to: {out_csv}")
    
    # Create Executive Visualization Chart
    plt.figure(figsize=(12, 7))
    colors = {'Tier 1: Aggressive Forward Procurement': '#1b7837', 
              'Tier 2: Balanced Dynamic Buying': '#2b83ba', 
              'Tier 3: Defensive / Deficit Hedging': '#d7191c'}
    
    scatter_colors = [colors[t] for t in summary['procurement_tier']]
    
    plt.scatter(
        summary['surplus_pct'],
        summary['mean_mandi_price_rs_q'],
        s=summary['expected_production_k_tonnes'] * 30.0 + 60,
        c=scatter_colors,
        alpha=0.8,
        edgecolors='black',
        linewidth=1.2
    )
    
    # Annotate key district-crop combinations
    for _, r in summary.iterrows():
        plt.annotate(
            f"{r['district']}-{r['crop']}",
            (r['surplus_pct'], r['mean_mandi_price_rs_q']),
            textcoords="offset points",
            xytext=(0, 7),
            ha='center',
            fontsize=8,
            weight='bold'
        )
        
    plt.axvline(0, color='gray', linestyle='--', alpha=0.6, label='Normal Baseline Yield (0% Surplus)')
    plt.xlabel('Projected Yield Surplus / Deficit vs 3-Yr Baseline (%)', fontsize=11)
    plt.ylabel('Expected Pre-Sowing Mandi Price (Rs / Quintal)', fontsize=11)
    plt.title('Procurement Prioritization Grid: Vidarbha Agritech (Bubble Size = Projected Volume)', fontsize=13, weight='bold')
    
    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1b7837', markersize=10, label='Tier 1: Forward Buying (Surplus & Low Risk)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2b83ba', markersize=10, label='Tier 2: Dynamic Mandi Spot Purchasing'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d7191c', markersize=10, label='Tier 3: Deficit Hedging & Logistics Diversion')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/procurement_matrix.png', dpi=200)
    plt.close()
    print("Saved docs/procurement_matrix.png")
    
    return summary


if __name__ == '__main__':
    generate_procurement_strategy()

