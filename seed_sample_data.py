#!/usr/bin/env python3
"""
Seed sample data for testing the RaceSim Analyzer.

This creates:
- 1 car with 2 versions (baseline v1.0 and candidate v1.1)
- 1 experiment with 2 scenarios (dry track, wet track)
- Baseline runs (Monte Carlo, 10 runs per scenario)
- Candidate runs (Monte Carlo, 10 runs per scenario)
- Telemetry data (Parquet files) for each run

The candidate version has:
- Improved lap time and max speed (aerodynamics upgrade)
- Worse fuel consumption and tire degradation (trade-off)
"""

import random
import json
import numpy as np
from pathlib import Path
from db import RaceSimDB

# Try to import telemetry module
try:
    from telemetry import TelemetryStore, CHANNELS, PARQUET_AVAILABLE
except ImportError:
    PARQUET_AVAILABLE = False


# Monza track parameters
TRACK_LENGTH_M = 5793  # Monza circuit length
LAP_TIME_BASE = 84.0   # ~84 seconds per lap
SAMPLE_RATE = 10       # 10 Hz
N_SAMPLES = int(LAP_TIME_BASE * SAMPLE_RATE)  # ~840 samples


def add_noise(base_value: float, std_percent: float = 0.02) -> float:
    """Add Gaussian noise to a value."""
    std = base_value * std_percent
    return base_value + random.gauss(0, std)


def generate_run_metrics(
    base_metrics: dict[str, float],
    noise_percent: float = 0.02
) -> dict[str, float]:
    """Generate metrics with noise for a single run."""
    return {
        name: add_noise(value, noise_percent)
        for name, value in base_metrics.items()
    }


def generate_track_profile(n_samples: int) -> dict[str, np.ndarray]:
    """
    Generate a realistic track profile with corners and straights.
    Returns arrays for curvature, elevation, etc.
    """
    positions = np.linspace(0, TRACK_LENGTH_M, n_samples)
    
    # Simplified Monza-like profile
    # Main straight: 0-1000m
    # Turns 1-2 (chicane): 1000-1300m
    # Curva Grande: 1300-2000m
    # Turns 4-5 (chicane): 2000-2400m
    # Lesmo turns: 2400-3200m
    # Ascari: 3500-4000m
    # Parabolica: 4800-5500m
    # Start/finish straight: 5500-5793m
    
    curvature = np.zeros(n_samples)
    
    for i, pos in enumerate(positions):
        if 1000 <= pos < 1300:  # Chicane 1
            curvature[i] = 0.008 * np.sin((pos - 1000) / 300 * np.pi * 2)
        elif 1500 <= pos < 1900:  # Curva Grande
            curvature[i] = 0.003
        elif 2000 <= pos < 2400:  # Chicane 2
            curvature[i] = 0.006 * np.sin((pos - 2000) / 400 * np.pi * 2)
        elif 2500 <= pos < 2800:  # Lesmo 1
            curvature[i] = 0.005
        elif 2900 <= pos < 3200:  # Lesmo 2
            curvature[i] = 0.004
        elif 3600 <= pos < 4000:  # Ascari
            curvature[i] = 0.005 * np.sin((pos - 3600) / 400 * np.pi)
        elif 4900 <= pos < 5400:  # Parabolica
            curvature[i] = 0.004
    
    return {"positions": positions, "curvature": np.abs(curvature)}


def generate_telemetry(
    run_id: int,
    base_speed: float,
    base_tire_wear_rate: float,
    base_brake_temp: float,
    base_fuel_consumption: float,
    curvature_factor: float = 80.0,
    noise_level: float = 0.03,
    profile: dict = None
) -> dict[str, np.ndarray]:
    """
    Generate realistic telemetry data for a single run.

    Args:
        run_id: Run identifier
        base_speed: Straight-line speed in m/s (speed at zero curvature)
        base_tire_wear_rate: Tire degradation rate per lap (%)
        base_brake_temp: Base brake temperature (Celsius)
        base_fuel_consumption: Fuel used per lap (kg)
        curvature_factor: How much curvature reduces speed. Lower = better
                          cornering grip (e.g. more downforce). Default 80.
        noise_level: Noise standard deviation as fraction
        profile: Track profile dict with positions and curvature

    Returns:
        Dictionary of telemetry channel arrays
    """
    np.random.seed(run_id)  # Reproducible but different per run
    
    if profile is None:
        profile = generate_track_profile(N_SAMPLES)
    
    positions = profile["positions"]
    curvature = profile["curvature"]
    n = len(positions)
    
    # Generate speed based on curvature (slow in corners)
    # curvature_factor controls how much grip the car has:
    #   high = more speed lost in corners (less grip)
    #   low  = less speed lost in corners (more downforce/grip)
    speed_factor = 1.0 - curvature * curvature_factor
    speed_factor = np.clip(speed_factor, 0.5, 1.1)
    speed = base_speed * speed_factor
    speed += np.random.normal(0, base_speed * noise_level * 0.3, n)
    speed = np.clip(speed, 20, 100)  # Reasonable bounds

    # Throttle and brake from speed changes
    speed_diff = np.diff(speed, prepend=speed[0])
    throttle = np.clip(speed_diff / 5 + 0.5, 0, 1)
    brake = np.clip(-speed_diff / 5, 0, 1)

    # G-forces from curvature and speed
    g_lateral = curvature * speed**2 / 9.81  # v^2/r approximation
    g_lateral += np.random.normal(0, 0.1, n)
    g_lateral = np.clip(g_lateral, 0, 5)

    g_longitudinal = speed_diff / 9.81 * SAMPLE_RATE  # Acceleration in g
    g_longitudinal += np.random.normal(0, 0.1, n)
    g_longitudinal = np.clip(g_longitudinal, -5, 3)

    # Tire wear accumulates over lap, faster in corners.
    # Cars with better cornering (lower curvature_factor) carry more speed
    # through corners, which increases mechanical load and tire wear there.
    grip_ratio = 80.0 / curvature_factor  # >1 for high-grip setups
    wear_rate = base_tire_wear_rate / n
    corner_wear = 1 + curvature * 100 * grip_ratio  # More grip → more corner wear
    straight_wear = np.where(curvature < 0.001, 1.0 / grip_ratio, 1.0)  # Less wear on straights
    tire_wear_base = np.cumsum(wear_rate * corner_wear * straight_wear)

    # Individual wheel wear with some variation
    tire_wear_fl = tire_wear_base * (1 + np.random.normal(0, 0.1, n).cumsum() * 0.01)
    tire_wear_fr = tire_wear_base * (1 + np.random.normal(0, 0.1, n).cumsum() * 0.01)
    tire_wear_rl = tire_wear_base * (1 + np.random.normal(0, 0.1, n).cumsum() * 0.01) * 0.9
    tire_wear_rr = tire_wear_base * (1 + np.random.normal(0, 0.1, n).cumsum() * 0.01) * 0.9

    # Brake temps - higher during braking, cool on straights.
    # Cars with more grip brake later and harder into corners (higher peak temps)
    # but spend less time braking on straights (lower baseline temps).
    brake_heat = brake * (250 + 80 * grip_ratio)  # More grip → hotter corner braking
    brake_cool = (1 - brake) * (40 + 15 * grip_ratio)  # More grip → faster cooling
    brake_temp_base = base_brake_temp + brake_heat - brake_cool
    # Smooth with exponential moving average
    alpha = 0.1
    brake_temp_smooth = np.zeros(n)
    brake_temp_smooth[0] = base_brake_temp
    for i in range(1, n):
        brake_temp_smooth[i] = alpha * brake_temp_base[i] + (1 - alpha) * brake_temp_smooth[i-1]

    brake_temp_fl = brake_temp_smooth + np.random.normal(0, 10, n)
    brake_temp_fr = brake_temp_smooth + np.random.normal(0, 10, n)
    brake_temp_rl = brake_temp_smooth * 0.85 + np.random.normal(0, 8, n)
    brake_temp_rr = brake_temp_smooth * 0.85 + np.random.normal(0, 8, n)
    
    # Fuel decreases over lap
    fuel_start = 5.0  # kg
    fuel_rate = base_fuel_consumption / n
    fuel_remaining = fuel_start - np.cumsum(np.full(n, fuel_rate))
    fuel_remaining += np.random.normal(0, 0.01, n)  # Small noise
    
    return {
        "position_m": positions,
        "speed": speed,
        "tire_wear_fl": np.clip(tire_wear_fl, 0, 100),
        "tire_wear_fr": np.clip(tire_wear_fr, 0, 100),
        "tire_wear_rl": np.clip(tire_wear_rl, 0, 100),
        "tire_wear_rr": np.clip(tire_wear_rr, 0, 100),
        "brake_temp_fl": np.clip(brake_temp_fl, 100, 800),
        "brake_temp_fr": np.clip(brake_temp_fr, 100, 800),
        "brake_temp_rl": np.clip(brake_temp_rl, 100, 700),
        "brake_temp_rr": np.clip(brake_temp_rr, 100, 700),
        "throttle": throttle,
        "brake": brake,
        "g_lateral": g_lateral,
        "g_longitudinal": g_longitudinal,
        "fuel_remaining": np.clip(fuel_remaining, 0, fuel_start),
    }


def seed_data(db_path: str = "racesim.db", with_telemetry: bool = True):
    """Seed the database with sample data."""
    
    random.seed(42)  # Reproducible results
    np.random.seed(42)
    
    db = RaceSimDB(db_path)
    db.init_schema()
    
    # Initialize telemetry store if available
    telemetry_store = None
    if with_telemetry and PARQUET_AVAILABLE:
        telemetry_store = TelemetryStore("telemetry")
        print("Telemetry storage enabled (Parquet)")
    elif with_telemetry:
        print("Warning: pyarrow not installed, skipping telemetry generation")
    
    # -------------------------------------------------------------------------
    # Create car
    # -------------------------------------------------------------------------
    car_id = db.create_car(
        name="Phoenix RS-7",
        description="High-performance racing prototype"
    )
    print(f"Created car: Phoenix RS-7 (id={car_id})")
    
    # -------------------------------------------------------------------------
    # Create car versions
    # -------------------------------------------------------------------------
    version_baseline = db.create_car_version(
        car_id=car_id,
        software_version="1.0.0",
        hardware_version="A",
        notes="Baseline configuration"
    )
    print(f"Created baseline version: SW 1.0.0 / HW A (id={version_baseline})")
    
    version_candidate = db.create_car_version(
        car_id=car_id,
        software_version="1.1.0",
        hardware_version="A",
        notes="Aerodynamics upgrade: new front wing, revised diffuser"
    )
    print(f"Created candidate version: SW 1.1.0 / HW A (id={version_candidate})")
    
    # -------------------------------------------------------------------------
    # Create experiment and scenarios
    # -------------------------------------------------------------------------
    experiment_id = db.create_experiment(
        name="Aero Package Validation",
        description="Testing new aerodynamics package across track conditions"
    )
    print(f"Created experiment: Aero Package Validation (id={experiment_id})")
    
    scenario_dry = db.create_scenario(
        experiment_id=experiment_id,
        name="Monza - Dry",
        parameters={
            "track": "Monza",
            "weather": "dry",
            "ambient_temp_c": 25,
            "track_temp_c": 35,
            "laps": 1
        }
    )
    print(f"Created scenario: Monza - Dry (id={scenario_dry})")
    
    scenario_wet = db.create_scenario(
        experiment_id=experiment_id,
        name="Monza - Wet",
        parameters={
            "track": "Monza",
            "weather": "wet",
            "ambient_temp_c": 18,
            "track_temp_c": 20,
            "laps": 1
        }
    )
    print(f"Created scenario: Monza - Wet (id={scenario_wet})")
    
    # -------------------------------------------------------------------------
    # Define base metrics for each configuration
    # -------------------------------------------------------------------------
    
    # Baseline metrics - Dry track
    baseline_dry = {
        "lap_time": 84.5,
        "fuel_consumption": 2.1,
        "max_speed": 91.0,
        "avg_speed": 52.0,
        "tire_degradation": 2.8,
        "brake_temp_max": 450.0,
        "cornering_g_max": 3.2,
        "energy_recovered": 480.0,
    }
    
    # Candidate metrics - Dry track (improved aero)
    candidate_dry = {
        "lap_time": 82.8,
        "fuel_consumption": 2.3,
        "max_speed": 93.5,
        "avg_speed": 53.5,
        "tire_degradation": 3.2,
        "brake_temp_max": 480.0,
        "cornering_g_max": 3.6,
        "energy_recovered": 510.0,
    }
    
    # Baseline metrics - Wet track
    baseline_wet = {
        "lap_time": 92.0,
        "fuel_consumption": 1.9,
        "max_speed": 82.0,
        "avg_speed": 45.0,
        "tire_degradation": 1.8,
        "brake_temp_max": 380.0,
        "cornering_g_max": 2.4,
        "energy_recovered": 420.0,
    }
    
    # Candidate metrics - Wet track
    candidate_wet = {
        "lap_time": 90.5,
        "fuel_consumption": 2.0,
        "max_speed": 83.5,
        "avg_speed": 46.0,
        "tire_degradation": 2.1,
        "brake_temp_max": 400.0,
        "cornering_g_max": 2.6,
        "energy_recovered": 440.0,
    }
    
    # Telemetry generation parameters
    telemetry_params_baseline_dry = {
        "base_speed": 68.8,  # TRACK_LENGTH / lap_time
        "base_tire_wear_rate": 2.8,
        "base_brake_temp": 380,
        "base_fuel_consumption": 2.1,
        "curvature_factor": 80,  # Reference cornering grip
    }

    telemetry_params_candidate_dry = {
        "base_speed": 67.0,  # Slower on straights (more aero drag)
        "base_tire_wear_rate": 3.2,  # More wear
        "base_brake_temp": 400,  # Hotter brakes
        "base_fuel_consumption": 2.3,
        "curvature_factor": 62,  # Much better cornering (more downforce)
    }

    telemetry_params_baseline_wet = {
        "base_speed": 63.0,  # TRACK_LENGTH / lap_time (wet)
        "base_tire_wear_rate": 1.8,
        "base_brake_temp": 320,
        "base_fuel_consumption": 1.9,
        "curvature_factor": 95,  # Much more speed loss in wet corners
    }

    telemetry_params_candidate_wet = {
        "base_speed": 61.5,  # Aero drag penalty in wet too
        "base_tire_wear_rate": 2.1,
        "base_brake_temp": 340,
        "base_fuel_consumption": 2.0,
        "curvature_factor": 76,  # Downforce helps in wet corners
    }

    # Pre-generate track profile (shared for all runs)
    track_profile = generate_track_profile(N_SAMPLES)
    
    # -------------------------------------------------------------------------
    # Generate Monte Carlo runs
    # -------------------------------------------------------------------------
    num_runs = 10
    
    def create_runs_with_telemetry(
        version_id, scenario_id, batch_id, is_baseline,
        base_metrics, telemetry_params, num_runs
    ):
        """Create runs and optionally generate telemetry."""
        for i in range(num_runs):
            run_id = db.create_run(
                version_id=version_id,
                scenario_id=scenario_id,
                batch_id=batch_id,
                is_baseline=is_baseline
            )
            metrics = generate_run_metrics(base_metrics)
            db.add_run_metrics(run_id, metrics)
            
            # Generate and save telemetry
            if telemetry_store and telemetry_params:
                telemetry = generate_telemetry(
                    run_id=run_id + i * 1000,  # Unique seed
                    profile=track_profile,
                    **telemetry_params
                )
                file_path = telemetry_store.save(run_id, telemetry)
                db.add_telemetry_reference(
                    run_id=run_id,
                    file_path=file_path,
                    sample_count=len(telemetry["position_m"]),
                    start_position_m=float(telemetry["position_m"][0]),
                    end_position_m=float(telemetry["position_m"][-1])
                )
    
    # Baseline runs - Dry
    print(f"\nGenerating {num_runs} baseline runs for dry scenario...")
    create_runs_with_telemetry(
        version_baseline, scenario_dry, "baseline-dry-v1.0", True,
        baseline_dry, telemetry_params_baseline_dry, num_runs
    )
    
    # Baseline runs - Wet (no telemetry for simplicity)
    print(f"Generating {num_runs} baseline runs for wet scenario...")
    create_runs_with_telemetry(
        version_baseline, scenario_wet, "baseline-wet-v1.0", True,
        baseline_wet, telemetry_params_baseline_wet, num_runs
    )
    
    # Candidate runs - Dry
    print(f"Generating {num_runs} candidate runs for dry scenario...")
    create_runs_with_telemetry(
        version_candidate, scenario_dry, "candidate-dry-v1.1", False,
        candidate_dry, telemetry_params_candidate_dry, num_runs
    )
    
    # Candidate runs - Wet (no telemetry)
    print(f"Generating {num_runs} candidate runs for wet scenario...")
    create_runs_with_telemetry(
        version_candidate, scenario_wet, "candidate-wet-v1.1", False,
        candidate_wet, telemetry_params_candidate_wet, num_runs
    )
    
    # -------------------------------------------------------------------------
    # Create a "bad" candidate for comparison testing
    # -------------------------------------------------------------------------
    version_bad = db.create_car_version(
        car_id=car_id,
        software_version="1.2.0-beta",
        hardware_version="A",
        notes="Experimental aggressive setup - untested"
    )
    print(f"\nCreated experimental version: SW 1.2.0-beta / HW A (id={version_bad})")
    
    bad_candidate_dry = {
        "lap_time": 81.5,
        "fuel_consumption": 2.8,
        "max_speed": 95.0,
        "avg_speed": 54.0,
        "tire_degradation": 4.5,
        "brake_temp_max": 620.0,
        "cornering_g_max": 3.8,
        "energy_recovered": 530.0,
    }
    
    telemetry_params_bad = {
        "base_speed": 70.5,
        "base_tire_wear_rate": 4.5,  # Very high wear
        "base_brake_temp": 520,  # Overheating
        "base_fuel_consumption": 2.8,
        "curvature_factor": 58,  # Extremely aggressive cornering
    }
    
    print(f"Generating {num_runs} experimental runs for dry scenario...")
    create_runs_with_telemetry(
        version_bad, scenario_dry, "experimental-dry-v1.2", False,
        bad_candidate_dry, telemetry_params_bad, num_runs
    )
    
    # Load thresholds from requirements
    from analysis import RequirementsLoader
    req_path = Path(__file__).parent / "requirements.yaml"
    if req_path.exists():
        requirements = RequirementsLoader(str(req_path))
        for thresh in requirements.telemetry_thresholds:
            db.add_threshold(
                channel=thresh["channel"],
                name=thresh["name"],
                value=thresh["value"],
                direction=thresh["direction"],
                severity=thresh.get("severity", "warning")
            )
        print(f"Loaded {len(requirements.telemetry_thresholds)} telemetry thresholds")
    
    db.close()
    
    print("\n" + "=" * 60)
    print("Sample data seeded successfully!")
    print("=" * 60)
    print("\nTry these commands:")
    print("  python main.py list-batches")
    print("  python main.py analyze candidate-dry-v1.1")
    print("  python main.py analyze candidate-dry-v1.1 --telemetry")
    print("  python main.py analyze candidate-dry-v1.1 --telemetry --ai")
    print("  python main.py telemetry candidate-dry-v1.1")
    print("  python main.py telemetry experimental-dry-v1.2 --ai")
    print("  python main.py compare candidate-dry-v1.1 experimental-dry-v1.2 --ai")


if __name__ == "__main__":
    seed_data()
