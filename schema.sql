-- RaceSim Analyzer Database Schema
-- All units are SI (metric): seconds, meters, kg, m/s, etc.

CREATE TABLE IF NOT EXISTS cars (
    car_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS car_versions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    car_id INTEGER NOT NULL,
    software_version TEXT NOT NULL,
    hardware_version TEXT NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (car_id) REFERENCES cars(car_id),
    UNIQUE(car_id, software_version, hardware_version)
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    parameters TEXT,  -- JSON blob for input conditions (track, weather, etc.)
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id INTEGER NOT NULL,
    scenario_id INTEGER NOT NULL,
    batch_id TEXT NOT NULL,  -- Groups Monte Carlo runs together
    is_baseline INTEGER DEFAULT 0,  -- 1 = baseline run, 0 = candidate run
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (version_id) REFERENCES car_versions(version_id),
    FOREIGN KEY (scenario_id) REFERENCES scenarios(scenario_id)
);

CREATE TABLE IF NOT EXISTS run_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    UNIQUE(run_id, metric_name)
);

CREATE TABLE IF NOT EXISTS run_telemetry (
    telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL UNIQUE,
    file_path TEXT NOT NULL,  -- Path to Parquet file
    sample_count INTEGER,
    start_position_m REAL,
    end_position_m REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Threshold definitions for time-series analysis
CREATE TABLE IF NOT EXISTS thresholds (
    threshold_id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    direction TEXT NOT NULL,  -- "above" or "below"
    severity TEXT DEFAULT 'warning',  -- "info", "warning", "critical"
    UNIQUE(channel, name)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_runs_version ON runs(version_id);
CREATE INDEX IF NOT EXISTS idx_runs_scenario ON runs(scenario_id);
CREATE INDEX IF NOT EXISTS idx_runs_batch ON runs(batch_id);
CREATE INDEX IF NOT EXISTS idx_runs_baseline ON runs(is_baseline);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON run_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON run_metrics(metric_name);
