#!/usr/bin/env python3
"""
Seed sample data for testing the RaceSim Analyzer.

This creates:
- 1 car with 2 versions (baseline v1.0 and candidate v1.1)
- 1 experiment with 2 scenarios (dry track, wet track)
- Baseline runs (Monte Carlo, 10 runs per scenario)
- Candidate runs (Monte Carlo, 10 runs per scenario)

The candidate version has:
- Improved lap time and max speed (aerodynamics upgrade)
- Worse fuel consumption and tire degradation (trade-off)
"""

import random
import json
from db import RaceSimDB


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


def seed_data(db_path: str = "racesim.db"):
    """Seed the database with sample data."""
    
    random.seed(42)  # Reproducible results
    
    db = RaceSimDB(db_path)
    db.init_schema()
    
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
        "lap_time": 84.5,           # seconds
        "fuel_consumption": 2.1,    # kg
        "max_speed": 91.0,          # m/s (~328 km/h)
        "avg_speed": 52.0,          # m/s
        "tire_degradation": 2.8,    # percent
        "brake_temp_max": 450.0,    # celsius
        "cornering_g_max": 3.2,     # g
        "energy_recovered": 480.0,  # kJ
    }
    
    # Candidate metrics - Dry track (improved aero = faster but more drag)
    candidate_dry = {
        "lap_time": 82.8,           # IMPROVED: -1.7s from better downforce
        "fuel_consumption": 2.3,    # WORSE: +0.2 kg from increased drag
        "max_speed": 93.5,          # IMPROVED: better top speed
        "avg_speed": 53.5,          # IMPROVED: faster through corners
        "tire_degradation": 3.2,    # WORSE: more grip = more wear
        "brake_temp_max": 480.0,    # WORSE: faster entry speeds
        "cornering_g_max": 3.6,     # IMPROVED: more downforce
        "energy_recovered": 510.0,  # IMPROVED: more braking events
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
    
    # -------------------------------------------------------------------------
    # Generate Monte Carlo runs
    # -------------------------------------------------------------------------
    num_runs = 10
    
    # Baseline runs - Dry
    print(f"\nGenerating {num_runs} baseline runs for dry scenario...")
    for i in range(num_runs):
        run_id = db.create_run(
            version_id=version_baseline,
            scenario_id=scenario_dry,
            batch_id="baseline-dry-v1.0",
            is_baseline=True
        )
        metrics = generate_run_metrics(baseline_dry)
        db.add_run_metrics(run_id, metrics)
    
    # Baseline runs - Wet
    print(f"Generating {num_runs} baseline runs for wet scenario...")
    for i in range(num_runs):
        run_id = db.create_run(
            version_id=version_baseline,
            scenario_id=scenario_wet,
            batch_id="baseline-wet-v1.0",
            is_baseline=True
        )
        metrics = generate_run_metrics(baseline_wet)
        db.add_run_metrics(run_id, metrics)
    
    # Candidate runs - Dry
    print(f"Generating {num_runs} candidate runs for dry scenario...")
    for i in range(num_runs):
        run_id = db.create_run(
            version_id=version_candidate,
            scenario_id=scenario_dry,
            batch_id="candidate-dry-v1.1",
            is_baseline=False
        )
        metrics = generate_run_metrics(candidate_dry)
        db.add_run_metrics(run_id, metrics)
    
    # Candidate runs - Wet
    print(f"Generating {num_runs} candidate runs for wet scenario...")
    for i in range(num_runs):
        run_id = db.create_run(
            version_id=version_candidate,
            scenario_id=scenario_wet,
            batch_id="candidate-wet-v1.1",
            is_baseline=False
        )
        metrics = generate_run_metrics(candidate_wet)
        db.add_run_metrics(run_id, metrics)
    
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
    
    # Bad candidate - fails some requirements
    bad_candidate_dry = {
        "lap_time": 81.5,           # Very fast but...
        "fuel_consumption": 2.8,    # WAY over limit (max 2.5)
        "max_speed": 95.0,
        "avg_speed": 54.0,
        "tire_degradation": 4.5,    # WAY over limit (max 4.0)
        "brake_temp_max": 620.0,    # OVER limit (max 600)
        "cornering_g_max": 3.8,
        "energy_recovered": 530.0,
    }
    
    print(f"Generating {num_runs} experimental runs for dry scenario...")
    for i in range(num_runs):
        run_id = db.create_run(
            version_id=version_bad,
            scenario_id=scenario_dry,
            batch_id="experimental-dry-v1.2",
            is_baseline=False
        )
        metrics = generate_run_metrics(bad_candidate_dry, noise_percent=0.03)
        db.add_run_metrics(run_id, metrics)
    
    db.close()
    
    print("\n" + "=" * 60)
    print("Sample data seeded successfully!")
    print("=" * 60)
    print("\nTry these commands:")
    print("  python main.py list-batches")
    print("  python main.py analyze candidate-dry-v1.1")
    print("  python main.py analyze candidate-dry-v1.1 --ai")
    print("  python main.py analyze experimental-dry-v1.2 --ai")
    print("  python main.py compare candidate-dry-v1.1 experimental-dry-v1.2 --ai")


if __name__ == "__main__":
    seed_data()
