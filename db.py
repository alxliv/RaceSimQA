"""
Database helpers for RaceSim Analyzer.
"""

import sqlite3
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Car:
    car_id: int
    name: str
    description: Optional[str] = None


@dataclass
class CarVersion:
    version_id: int
    car_id: int
    software_version: str
    hardware_version: str
    notes: Optional[str] = None


@dataclass
class Scenario:
    scenario_id: int
    experiment_id: int
    name: str
    parameters: Optional[dict] = None


@dataclass
class Run:
    run_id: int
    version_id: int
    scenario_id: int
    batch_id: str
    is_baseline: bool


@dataclass
class RunMetrics:
    run_id: int
    metrics: dict[str, float]


class RaceSimDB:
    def __init__(self, db_path: str = "racesim.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def init_schema(self):
        """Initialize database schema from schema.sql."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, "r") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    def close(self):
        self.conn.close()

    # -------------------------------------------------------------------------
    # Car operations
    # -------------------------------------------------------------------------
    def create_car(self, name: str, description: str = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO cars (name, description) VALUES (?, ?)",
            (name, description)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_car(self, car_id: int) -> Optional[Car]:
        row = self.conn.execute(
            "SELECT * FROM cars WHERE car_id = ?", (car_id,)
        ).fetchone()
        if row:
            return Car(row["car_id"], row["name"], row["description"])
        return None

    # -------------------------------------------------------------------------
    # Car version operations
    # -------------------------------------------------------------------------
    def create_car_version(
        self,
        car_id: int,
        software_version: str,
        hardware_version: str,
        notes: str = None
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO car_versions (car_id, software_version, hardware_version, notes)
               VALUES (?, ?, ?, ?)""",
            (car_id, software_version, hardware_version, notes)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_car_version(self, version_id: int) -> Optional[CarVersion]:
        row = self.conn.execute(
            "SELECT * FROM car_versions WHERE version_id = ?", (version_id,)
        ).fetchone()
        if row:
            return CarVersion(
                row["version_id"], row["car_id"],
                row["software_version"], row["hardware_version"],
                row["notes"]
            )
        return None

    def get_version_display_name(self, version_id: int) -> str:
        """Get a human-readable name for a car version."""
        row = self.conn.execute(
            """SELECT c.name, cv.software_version, cv.hardware_version
               FROM car_versions cv
               JOIN cars c ON cv.car_id = c.car_id
               WHERE cv.version_id = ?""",
            (version_id,)
        ).fetchone()
        if row:
            return f"{row['name']} (SW:{row['software_version']}, HW:{row['hardware_version']})"
        return f"Unknown version {version_id}"

    # -------------------------------------------------------------------------
    # Experiment and scenario operations
    # -------------------------------------------------------------------------
    def create_experiment(self, name: str, description: str = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO experiments (name, description) VALUES (?, ?)",
            (name, description)
        )
        self.conn.commit()
        return cur.lastrowid

    def create_scenario(
        self,
        experiment_id: int,
        name: str,
        parameters: dict = None
    ) -> int:
        params_json = json.dumps(parameters) if parameters else None
        cur = self.conn.execute(
            "INSERT INTO scenarios (experiment_id, name, parameters) VALUES (?, ?, ?)",
            (experiment_id, name, params_json)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_scenario(self, scenario_id: int) -> Optional[Scenario]:
        row = self.conn.execute(
            "SELECT * FROM scenarios WHERE scenario_id = ?", (scenario_id,)
        ).fetchone()
        if row:
            params = json.loads(row["parameters"]) if row["parameters"] else None
            return Scenario(
                row["scenario_id"], row["experiment_id"],
                row["name"], params
            )
        return None

    # -------------------------------------------------------------------------
    # Run operations
    # -------------------------------------------------------------------------
    def create_run(
        self,
        version_id: int,
        scenario_id: int,
        batch_id: str,
        is_baseline: bool = False
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO runs (version_id, scenario_id, batch_id, is_baseline)
               VALUES (?, ?, ?, ?)""",
            (version_id, scenario_id, batch_id, 1 if is_baseline else 0)
        )
        self.conn.commit()
        return cur.lastrowid

    def add_run_metric(self, run_id: int, metric_name: str, value: float):
        self.conn.execute(
            """INSERT OR REPLACE INTO run_metrics (run_id, metric_name, value)
               VALUES (?, ?, ?)""",
            (run_id, metric_name, value)
        )
        self.conn.commit()

    def add_run_metrics(self, run_id: int, metrics: dict[str, float]):
        for name, value in metrics.items():
            self.add_run_metric(run_id, name, value)

    def get_run_metrics(self, run_id: int) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT metric_name, value FROM run_metrics WHERE run_id = ?",
            (run_id,)
        ).fetchall()
        return {row["metric_name"]: row["value"] for row in rows}

    # -------------------------------------------------------------------------
    # Batch queries
    # -------------------------------------------------------------------------
    def get_batch_run_ids(self, batch_id: str) -> list[int]:
        """Get all run IDs in a batch."""
        rows = self.conn.execute(
            "SELECT run_id FROM runs WHERE batch_id = ?", (batch_id,)
        ).fetchall()
        return [row["run_id"] for row in rows]

    def get_baseline_runs_for_scenario(self, scenario_id: int) -> list[int]:
        """Get all baseline run IDs for a given scenario."""
        rows = self.conn.execute(
            "SELECT run_id FROM runs WHERE scenario_id = ? AND is_baseline = 1",
            (scenario_id,)
        ).fetchall()
        return [row["run_id"] for row in rows]

    def get_runs_info(self, run_ids: list[int]) -> list[dict]:
        """Get run info for multiple runs."""
        if not run_ids:
            return []
        placeholders = ",".join("?" * len(run_ids))
        rows = self.conn.execute(
            f"""SELECT r.run_id, r.version_id, r.scenario_id, r.batch_id, r.is_baseline,
                       cv.software_version, cv.hardware_version, c.name as car_name
                FROM runs r
                JOIN car_versions cv ON r.version_id = cv.version_id
                JOIN cars c ON cv.car_id = c.car_id
                WHERE r.run_id IN ({placeholders})""",
            run_ids
        ).fetchall()
        return [dict(row) for row in rows]

    def get_all_metrics_for_runs(self, run_ids: list[int]) -> dict[int, dict[str, float]]:
        """Get metrics for multiple runs at once."""
        if not run_ids:
            return {}
        placeholders = ",".join("?" * len(run_ids))
        rows = self.conn.execute(
            f"SELECT run_id, metric_name, value FROM run_metrics WHERE run_id IN ({placeholders})",
            run_ids
        ).fetchall()

        result = {rid: {} for rid in run_ids}
        for row in rows:
            result[row["run_id"]][row["metric_name"]] = row["value"]
        return result

    def get_scenario_id_for_batch(self, batch_id: str) -> Optional[int]:
        """Get the scenario ID for a batch (assumes all runs in batch use same scenario)."""
        row = self.conn.execute(
            "SELECT DISTINCT scenario_id FROM runs WHERE batch_id = ?",
            (batch_id,)
        ).fetchone()
        return row["scenario_id"] if row else None

    def list_batches(self) -> list[dict]:
        """List all batches with summary info."""
        rows = self.conn.execute(
            """SELECT batch_id,
                      COUNT(*) as run_count,
                      MAX(is_baseline) as has_baseline,
                      MIN(created_at) as created_at
               FROM runs
               GROUP BY batch_id
               ORDER BY created_at DESC"""
        ).fetchall()
        return [dict(row) for row in rows]

    # -------------------------------------------------------------------------
    # Telemetry operations
    # -------------------------------------------------------------------------
    def add_telemetry_reference(
        self,
        run_id: int,
        file_path: str,
        sample_count: int,
        start_position_m: float,
        end_position_m: float
    ):
        """Add reference to a telemetry Parquet file."""
        self.conn.execute(
            """INSERT OR REPLACE INTO run_telemetry
               (run_id, file_path, sample_count, start_position_m, end_position_m)
               VALUES (?, ?, ?, ?, ?)""",
            (run_id, file_path, sample_count, start_position_m, end_position_m)
        )
        self.conn.commit()

    def get_telemetry_path(self, run_id: int) -> Optional[str]:
        """Get the telemetry file path for a run."""
        row = self.conn.execute(
            "SELECT file_path FROM run_telemetry WHERE run_id = ?",
            (run_id,)
        ).fetchone()
        return row["file_path"] if row else None

    def get_telemetry_paths_for_runs(self, run_ids: list[int]) -> dict[int, str]:
        """Get telemetry file paths for multiple runs."""
        if not run_ids:
            return {}
        placeholders = ",".join("?" * len(run_ids))
        rows = self.conn.execute(
            f"SELECT run_id, file_path FROM run_telemetry WHERE run_id IN ({placeholders})",
            run_ids
        ).fetchall()
        return {row["run_id"]: row["file_path"] for row in rows}

    def has_telemetry(self, run_id: int) -> bool:
        """Check if a run has telemetry data."""
        row = self.conn.execute(
            "SELECT 1 FROM run_telemetry WHERE run_id = ?",
            (run_id,)
        ).fetchone()
        return row is not None

    # -------------------------------------------------------------------------
    # Threshold operations
    # -------------------------------------------------------------------------
    def add_threshold(
        self,
        channel: str,
        name: str,
        value: float,
        direction: str,
        severity: str = "warning"
    ):
        """Add or update a threshold definition."""
        self.conn.execute(
            """INSERT OR REPLACE INTO thresholds
               (channel, name, value, direction, severity)
               VALUES (?, ?, ?, ?, ?)""",
            (channel, name, value, direction, severity)
        )
        self.conn.commit()

    def get_thresholds(self) -> list[dict]:
        """Get all threshold definitions."""
        rows = self.conn.execute(
            "SELECT channel, name, value, direction, severity FROM thresholds"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_thresholds_for_channel(self, channel: str) -> list[dict]:
        """Get thresholds for a specific channel."""
        rows = self.conn.execute(
            "SELECT name, value, direction, severity FROM thresholds WHERE channel = ?",
            (channel,)
        ).fetchall()
        return [dict(row) for row in rows]
