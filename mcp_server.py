"""
MCP Server for RaceSim QA database.

Exposes the racesim.db data through Model Context Protocol tools,
allowing LLMs to query batches, runs, metrics, and comparisons.

Run:
    python mcp_server.py
    # or via MCP inspector:
    mcp dev mcp_server.py
"""

import json
import statistics
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from db import RaceSimDB

DB_PATH = str(Path(__file__).parent / "racesim.db")

mcp = FastMCP(
    "RaceSim QA",
    instructions=(
        "Racing simulation QA database. Use these tools to explore "
        "batch results, compare candidate vs baseline metrics, inspect "
        "individual runs, and check threshold definitions."
    ),
)


def _get_db() -> RaceSimDB:
    return RaceSimDB(DB_PATH)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_batches() -> str:
    """List all simulation batches with run counts and type (baseline/candidate)."""
    db = _get_db()
    try:
        batches = db.list_batches()
        if not batches:
            return "No batches found in the database."
        return json.dumps(batches, indent=2, default=str)
    finally:
        db.close()


@mcp.tool()
def list_cars() -> str:
    """List all car versions with their software/hardware configs and associated batches."""
    db = _get_db()
    try:
        rows = db.conn.execute(
            """SELECT cv.version_id, c.name AS car,
                      cv.software_version, cv.hardware_version, cv.notes,
                      GROUP_CONCAT(DISTINCT r.batch_id) AS batches,
                      COUNT(DISTINCT r.run_id) AS total_runs
               FROM car_versions cv
               JOIN cars c ON cv.car_id = c.car_id
               LEFT JOIN runs r ON r.version_id = cv.version_id
               GROUP BY cv.version_id
               ORDER BY cv.version_id"""
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["batches"] = d["batches"].split(",") if d["batches"] else []
            results.append(d)
        return json.dumps(results, indent=2, default=str)
    finally:
        db.close()


@mcp.tool()
def get_versions_for_scenario(scenario_id: int) -> str:
    """Get car version details for all versions that have runs in a given scenario.

    Args:
        scenario_id: The scenario ID
    """
    db = _get_db()
    try:
        rows = db.conn.execute(
            """SELECT cv.version_id, c.name AS car,
                      cv.software_version, cv.hardware_version, cv.notes,
                      r.batch_id, r.is_baseline,
                      COUNT(*) AS run_count
               FROM runs r
               JOIN car_versions cv ON r.version_id = cv.version_id
               JOIN cars c ON cv.car_id = c.car_id
               WHERE r.scenario_id = ?
               GROUP BY cv.version_id, r.batch_id
               ORDER BY r.is_baseline DESC, cv.version_id""",
            (scenario_id,),
        ).fetchall()
        if not rows:
            return f"No runs found for scenario_id {scenario_id}."

        scenario = db.get_scenario(scenario_id)
        results = {
            "scenario_id": scenario_id,
            "scenario_name": scenario.name if scenario else "unknown",
            "versions": [dict(r) for r in rows],
        }
        return json.dumps(results, indent=2, default=str)
    finally:
        db.close()


@mcp.tool()
def list_scenarios() -> str:
    """List all experiment scenarios with their parameters."""
    db = _get_db()
    try:
        rows = db.conn.execute(
            """SELECT s.scenario_id, e.name AS experiment, s.name AS scenario,
                      s.parameters
               FROM scenarios s
               JOIN experiments e ON s.experiment_id = e.experiment_id"""
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d["parameters"]:
                d["parameters"] = json.loads(d["parameters"])
            results.append(d)
        return json.dumps(results, indent=2)
    finally:
        db.close()


@mcp.tool()
def get_batch_details(batch_id: str) -> str:
    """Get all runs and their metrics for a given batch.

    Args:
        batch_id: The batch identifier (e.g. "candidate-dry-v1.1")
    """
    db = _get_db()
    try:
        run_ids = db.get_batch_run_ids(batch_id)
        if not run_ids:
            return f"No runs found for batch '{batch_id}'."

        runs_info = db.get_runs_info(run_ids)
        all_metrics = db.get_all_metrics_for_runs(run_ids)
        telemetry_paths = db.get_telemetry_paths_for_runs(run_ids)

        results = []
        for info in runs_info:
            rid = info["run_id"]
            results.append({
                "run_id": rid,
                "car": info["car_name"],
                "sw_version": info["software_version"],
                "hw_version": info["hardware_version"],
                "is_baseline": bool(info["is_baseline"]),
                "metrics": all_metrics.get(rid, {}),
                "has_telemetry": rid in telemetry_paths,
            })

        return json.dumps(results, indent=2)
    finally:
        db.close()


@mcp.tool()
def get_batch_summary(batch_id: str) -> str:
    """Get aggregate statistics (mean, std, min, max) for each metric in a batch.

    Args:
        batch_id: The batch identifier
    """
    db = _get_db()
    try:
        run_ids = db.get_batch_run_ids(batch_id)
        if not run_ids:
            return f"No runs found for batch '{batch_id}'."

        all_metrics = db.get_all_metrics_for_runs(run_ids)

        # Collect values per metric name
        by_metric: dict[str, list[float]] = {}
        for metrics in all_metrics.values():
            for name, val in metrics.items():
                by_metric.setdefault(name, []).append(val)

        summary = {}
        for name, values in sorted(by_metric.items()):
            summary[name] = {
                "mean": round(statistics.mean(values), 4),
                "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "count": len(values),
            }

        return json.dumps(
            {"batch_id": batch_id, "num_runs": len(run_ids), "metrics": summary},
            indent=2,
        )
    finally:
        db.close()


@mcp.tool()
def compare_batches(candidate_batch_id: str, baseline_batch_id: str) -> str:
    """Compare metric means between candidate and baseline batches.

    Shows the delta (candidate - baseline) and percentage change for each metric.

    Args:
        candidate_batch_id: Candidate batch to evaluate
        baseline_batch_id: Baseline batch to compare against
    """
    db = _get_db()
    try:
        cand_ids = db.get_batch_run_ids(candidate_batch_id)
        base_ids = db.get_batch_run_ids(baseline_batch_id)
        if not cand_ids:
            return f"No runs found for candidate batch '{candidate_batch_id}'."
        if not base_ids:
            return f"No runs found for baseline batch '{baseline_batch_id}'."

        cand_metrics = db.get_all_metrics_for_runs(cand_ids)
        base_metrics = db.get_all_metrics_for_runs(base_ids)

        def _means(metrics_map):
            by_name: dict[str, list[float]] = {}
            for m in metrics_map.values():
                for n, v in m.items():
                    by_name.setdefault(n, []).append(v)
            return {n: statistics.mean(vs) for n, vs in by_name.items()}

        cand_means = _means(cand_metrics)
        base_means = _means(base_metrics)
        all_names = sorted(set(cand_means) | set(base_means))

        comparison = []
        for name in all_names:
            c = cand_means.get(name)
            b = base_means.get(name)
            entry = {"metric": name, "candidate_mean": None, "baseline_mean": None}
            if c is not None:
                entry["candidate_mean"] = round(c, 4)
            if b is not None:
                entry["baseline_mean"] = round(b, 4)
            if c is not None and b is not None:
                delta = c - b
                pct = (delta / abs(b) * 100) if b != 0 else None
                entry["delta"] = round(delta, 4)
                entry["pct_change"] = round(pct, 2) if pct is not None else None
            comparison.append(entry)

        return json.dumps(
            {
                "candidate": candidate_batch_id,
                "baseline": baseline_batch_id,
                "candidate_runs": len(cand_ids),
                "baseline_runs": len(base_ids),
                "comparison": comparison,
            },
            indent=2,
        )
    finally:
        db.close()


@mcp.tool()
def get_run_metrics(run_id: int) -> str:
    """Get all metrics for a specific run.

    Args:
        run_id: The run ID
    """
    db = _get_db()
    try:
        metrics = db.get_run_metrics(run_id)
        if not metrics:
            return f"No metrics found for run {run_id}."
        info = db.get_runs_info([run_id])
        result = {"run_id": run_id, "metrics": metrics}
        if info:
            result["batch_id"] = info[0]["batch_id"]
            result["car"] = info[0]["car_name"]
            result["is_baseline"] = bool(info[0]["is_baseline"])
        return json.dumps(result, indent=2)
    finally:
        db.close()


@mcp.tool()
def get_thresholds(channel: str = "") -> str:
    """Get threshold definitions, optionally filtered by telemetry channel.

    Args:
        channel: Optional channel name to filter (e.g. "speed", "tire_wear_fl").
                 Leave empty to get all thresholds.
    """
    db = _get_db()
    try:
        if channel:
            thresholds = db.get_thresholds_for_channel(channel)
        else:
            thresholds = db.get_thresholds()
        if not thresholds:
            return "No thresholds found." if not channel else f"No thresholds for channel '{channel}'."
        return json.dumps(thresholds, indent=2)
    finally:
        db.close()


@mcp.tool()
def find_batches_for_scenario(scenario_name: str) -> str:
    """Find all batches that ran under a given scenario name.

    Args:
        scenario_name: Scenario name to search for (e.g. "Monza Dry", "Monza Wet")
    """
    db = _get_db()
    try:
        rows = db.conn.execute(
            """SELECT DISTINCT r.batch_id, s.name AS scenario,
                      COUNT(*) AS run_count,
                      MAX(r.is_baseline) AS has_baseline
               FROM runs r
               JOIN scenarios s ON r.scenario_id = s.scenario_id
               WHERE s.name LIKE ?
               GROUP BY r.batch_id
               ORDER BY r.batch_id""",
            (f"%{scenario_name}%",),
        ).fetchall()
        if not rows:
            return f"No batches found for scenario matching '{scenario_name}'."
        return json.dumps([dict(r) for r in rows], indent=2, default=str)
    finally:
        db.close()


@mcp.tool()
def query_metric_across_batches(metric_name: str) -> str:
    """Query a single metric across all batches, showing per-batch averages.

    Useful for tracking how a metric evolves across different software versions.

    Args:
        metric_name: Metric to query (e.g. "lap_time", "fuel_consumption",
                     "tire_degradation", "max_speed")
    """
    db = _get_db()
    try:
        rows = db.conn.execute(
            """SELECT r.batch_id,
                      AVG(rm.value) AS mean,
                      MIN(rm.value) AS min,
                      MAX(rm.value) AS max,
                      COUNT(*) AS num_runs,
                      MAX(r.is_baseline) AS is_baseline
               FROM run_metrics rm
               JOIN runs r ON rm.run_id = r.run_id
               WHERE rm.metric_name = ?
               GROUP BY r.batch_id
               ORDER BY mean""",
            (metric_name,),
        ).fetchall()
        if not rows:
            return f"No data found for metric '{metric_name}'."
        results = []
        for r in rows:
            results.append({
                "batch_id": r["batch_id"],
                "mean": round(r["mean"], 4),
                "min": round(r["min"], 4),
                "max": round(r["max"], 4),
                "num_runs": r["num_runs"],
                "is_baseline": bool(r["is_baseline"]),
            })
        return json.dumps(results, indent=2)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("racesim://schema")
def get_schema() -> str:
    """The database schema (DDL) for the racesim.db database."""
    schema_path = Path(__file__).parent / "schema.sql"
    return schema_path.read_text()


@mcp.resource("racesim://batches")
def get_batches_resource() -> str:
    """Summary of all available batches."""
    return list_batches()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
