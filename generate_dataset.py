"""
Synthetic EV Telemetry & Battery Degradation Dataset Generator
Combines Electrochemical Battery Aging Models (Arrhenius degradation, cycle life)
with Vehicle Dynamics (aerodynamic drag, ambient temperature HVAC penalty, payload).
"""

import numpy as np
import pandas as pd
import os

def generate_ev_dataset(n_samples=6000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Vehicle and Battery Specs
    battery_types = np.random.choice(["NMC", "LFP"], size=n_samples, p=[0.65, 0.35])
    nominal_capacity_kwh = np.random.choice([45.0, 58.0, 64.0, 75.0, 82.0, 100.0], size=n_samples)
    
    # 2. Historical Usage & Degradation Factors
    charge_cycles = np.random.randint(20, 1400, size=n_samples)
    fast_charge_ratio = np.random.uniform(0.05, 0.85, size=n_samples) # DC Fast charging ratio
    avg_operating_temp_c = np.random.normal(27.0, 6.0, size=n_samples)
    avg_operating_temp_c = np.clip(avg_operating_temp_c, 12.0, 48.0)
    avg_dod_pct = np.random.uniform(40.0, 95.0, size=n_samples) # Depth of Discharge
    
    # Calculate Internal Resistance (Mohm) - scales with cycles and heat
    internal_resistance_mohm = 15.0 + (charge_cycles * 0.02) + (fast_charge_ratio * 6.0) + np.random.normal(0, 1.2, size=n_samples)
    internal_resistance_mohm = np.clip(internal_resistance_mohm, 14.0, 65.0)

    # State of Health (SOH %) calculation based on electrochemistry:
    # LFP has higher cycle life (~3000 cycles to 80%) compared to NMC (~1500 cycles to 80%)
    type_factor = np.where(battery_types == "LFP", 0.65, 1.0)
    cycle_loss = 0.45 * np.sqrt(charge_cycles) * type_factor
    fast_charge_loss = fast_charge_ratio * 4.5 * (charge_cycles / 500.0)
    temp_stress = np.maximum(0, avg_operating_temp_c - 28.0) * 0.25 * (charge_cycles / 600.0)
    dod_stress = (avg_dod_pct / 100.0) * 1.8 * (charge_cycles / 700.0)
    
    soh_pct = 100.0 - (cycle_loss + fast_charge_loss + temp_stress + dod_stress) + np.random.normal(0, 0.7, size=n_samples)
    soh_pct = np.clip(soh_pct, 68.0, 100.0)
    
    # 3. Real-World Driving Conditions (for Range Prediction)
    ambient_temp_c = np.random.uniform(-10.0, 42.0, size=n_samples)
    avg_speed_kmh = np.random.uniform(25.0, 125.0, size=n_samples)
    driving_style = np.random.choice(["Eco", "Moderate", "Aggressive"], size=n_samples, p=[0.3, 0.5, 0.2])
    hvac_level = np.random.choice([0, 1, 2], size=n_samples, p=[0.25, 0.45, 0.30]) # 0=Off, 1=Comfort, 2=Max
    elevation_gain_m = np.random.uniform(0.0, 1200.0, size=n_samples)
    payload_kg = np.random.uniform(70.0, 420.0, size=n_samples) # Driver + passengers + luggage
    tire_pressure_psi = np.random.normal(35.0, 2.5, size=n_samples)
    tire_pressure_psi = np.clip(tire_pressure_psi, 26.0, 40.0)

    # 4. Energy Consumption Physics Model (kWh / 100 km)
    # Base baseline consumption: ~14.5 kWh/100km for medium EV
    base_consumption = 13.5 + (nominal_capacity_kwh * 0.02)
    
    # Aerodynamic drag penalty: Drag force ~ v^2, power ~ v^3
    aero_penalty = np.where(avg_speed_kmh > 75.0, ((avg_speed_kmh - 75.0) / 10.0) ** 1.45 * 1.1, 0.0)
    
    # Temperature & HVAC consumption penalty
    # Batteries suffer below 15C (internal resistance + cabin heating via PTC/Heat Pump)
    # Above 28C, battery cooling + cabin AC requires extra power
    cold_penalty = np.where(ambient_temp_c < 18.0, (18.0 - ambient_temp_c) * 0.22 * (1.0 + 0.4 * hvac_level), 0.0)
    hot_penalty = np.where(ambient_temp_c > 28.0, (ambient_temp_c - 28.0) * 0.16 * (1.0 + 0.35 * hvac_level), 0.0)
    
    # Driving style penalty
    style_multiplier = np.where(driving_style == "Aggressive", 1.22, np.where(driving_style == "Moderate", 1.0, 0.88))
    
    # Payload & Elevation penalty
    mass_penalty = (payload_kg - 75.0) * 0.007
    elevation_penalty = (elevation_gain_m / 100.0) * 0.35
    
    # Rolling resistance penalty (low tire pressure increases rolling friction)
    tire_penalty = np.maximum(0, 35.0 - tire_pressure_psi) * 0.18
    
    total_consumption_per_100km = (base_consumption + aero_penalty + cold_penalty + hot_penalty + mass_penalty + elevation_penalty + tire_penalty) * style_multiplier
    total_consumption_per_100km = np.clip(total_consumption_per_100km, 11.0, 36.0)
    
    # Usable energy in battery right now
    usable_energy_kwh = nominal_capacity_kwh * (soh_pct / 100.0)
    
    # Real-world range calculation
    real_world_range_km = (usable_energy_kwh / total_consumption_per_100km) * 100.0 + np.random.normal(0, 3.5, size=n_samples)
    real_world_range_km = np.clip(real_world_range_km, 60.0, 750.0)
    
    # Maintenance Status Classification Target
    # Normal (>88% SOH), Moderate Degradation (80-88%), Warning (75-80%), Critical Replacement (<75%)
    conditions = [
        soh_pct >= 88.0,
        (soh_pct >= 80.0) & (soh_pct < 88.0),
        (soh_pct >= 75.0) & (soh_pct < 80.0),
        soh_pct < 75.0
    ]
    status_labels = ["Healthy", "Normal Wear", "Degradation Warning", "Replacement Required"]
    battery_status = np.select(conditions, status_labels, default="Healthy")
    
    df = pd.DataFrame({
        "battery_type": battery_types,
        "nominal_capacity_kwh": np.round(nominal_capacity_kwh, 1),
        "charge_cycles": charge_cycles,
        "fast_charge_ratio": np.round(fast_charge_ratio, 3),
        "avg_operating_temp_c": np.round(avg_operating_temp_c, 1),
        "avg_dod_pct": np.round(avg_dod_pct, 1),
        "internal_resistance_mohm": np.round(internal_resistance_mohm, 2),
        "ambient_temp_c": np.round(ambient_temp_c, 1),
        "avg_speed_kmh": np.round(avg_speed_kmh, 1),
        "driving_style": driving_style,
        "hvac_level": hvac_level,
        "elevation_gain_m": np.round(elevation_gain_m, 1),
        "payload_kg": np.round(payload_kg, 1),
        "tire_pressure_psi": np.round(tire_pressure_psi, 1),
        "energy_consumption_kwh_per_100km": np.round(total_consumption_per_100km, 2),
        # Target Variables:
        "soh_pct": np.round(soh_pct, 2),
        "real_world_range_km": np.round(real_world_range_km, 1),
        "battery_health_status": battery_status
    })
    
    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(out_dir, "ev_battery_telemetry.csv")
    print(f"Generating EV Battery Telemetry dataset with 6,000 samples...")
    df = generate_ev_dataset(n_samples=6000)
    df.to_csv(data_path, index=False)
    print(f"Dataset successfully saved to: {data_path}")
    print(f"Features preview:\n{df.head(3).T}")
    print(f"\nTarget Summary:")
    print(f"SOH % Mean: {df['soh_pct'].mean():.2f}% (Min: {df['soh_pct'].min():.1f}%, Max: {df['soh_pct'].max():.1f}%)")
    print(f"Range Mean: {df['real_world_range_km'].mean():.1f} km")
    print(df['battery_health_status'].value_counts())
