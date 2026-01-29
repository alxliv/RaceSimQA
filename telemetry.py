"""
Telemetry module for time-series data handling and analysis.

Handles:
- Parquet file I/O for telemetry data
- Position-based resampling and alignment
- Threshold crossing detection
- Envelope computation across Monte Carlo runs
- Curve comparison between baseline and candidate
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


# Standard telemetry channels
CHANNELS = [
    "position_m",       # Track position (meters from start)
    "speed",            # m/s
    "tire_wear_fl",     # Front-left tire wear (0-100%)
    "tire_wear_fr",     # Front-right
    "tire_wear_rl",     # Rear-left
    "tire_wear_rr",     # Rear-right
    "brake_temp_fl",    # Brake temperature front-left (Celsius)
    "brake_temp_fr",
    "brake_temp_rl",
    "brake_temp_rr",
    "throttle",         # Throttle position (0-1)
    "brake",            # Brake position (0-1)
    "g_lateral",        # Lateral acceleration (g)
    "g_longitudinal",   # Longitudinal acceleration (g)
    "fuel_remaining",   # Fuel remaining (kg)
]


@dataclass
class ThresholdCrossing:
    """Represents a single threshold crossing event."""
    channel: str
    threshold_name: str
    threshold_value: float
    direction: str  # "above" or "below"
    severity: str
    position_m: float  # Where crossing occurred
    value_at_crossing: float
    duration_m: float  # How long threshold was violated (in meters)
    peak_value: float  # Most extreme value during violation


@dataclass
class ChannelStats:
    """Statistics for a single channel across position."""
    channel: str
    mean: np.ndarray  # Mean at each position
    std: np.ndarray   # Std at each position
    min: np.ndarray
    max: np.ndarray
    positions: np.ndarray  # Position array (common x-axis)


@dataclass
class ChannelComparison:
    """Comparison of a channel between baseline and candidate."""
    channel: str
    baseline_stats: ChannelStats
    candidate_stats: ChannelStats
    mean_delta: np.ndarray  # candidate_mean - baseline_mean
    rms_delta: float  # RMS of the difference
    max_abs_delta: float
    max_delta_position: float  # Position where max delta occurs


@dataclass 
class ThresholdAnalysis:
    """Analysis of threshold crossings for a batch."""
    batch_id: str
    run_count: int
    crossings_by_run: dict[int, list[ThresholdCrossing]]
    crossing_summary: dict[str, dict]  # channel -> {count, avg_duration, positions}


@dataclass
class TelemetryComparison:
    """Complete telemetry comparison between baseline and candidate."""
    baseline_run_count: int
    candidate_run_count: int
    channel_comparisons: dict[str, ChannelComparison]
    baseline_threshold_analysis: ThresholdAnalysis
    candidate_threshold_analysis: ThresholdAnalysis
    # New crossings in candidate that weren't in baseline
    new_violations: list[str]
    # Crossings resolved in candidate that were in baseline
    resolved_violations: list[str]


class TelemetryStore:
    """Handles reading and writing telemetry Parquet files."""
    
    def __init__(self, base_dir: str = "telemetry"):
        if not PARQUET_AVAILABLE:
            raise ImportError("pyarrow is required for telemetry. Install with: pip install pyarrow")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_file_path(self, run_id: int) -> Path:
        """Get the standard file path for a run's telemetry."""
        return self.base_dir / f"run_{run_id:06d}.parquet"
    
    def save(self, run_id: int, data: dict[str, np.ndarray]) -> str:
        """
        Save telemetry data to Parquet file.
        
        Args:
            run_id: The run ID
            data: Dictionary mapping channel names to numpy arrays
                  Must include 'position_m' as the index
        
        Returns:
            Path to the saved file
        """
        if "position_m" not in data:
            raise ValueError("Telemetry data must include 'position_m' channel")
        
        # Verify all arrays have same length
        n_samples = len(data["position_m"])
        for name, arr in data.items():
            if len(arr) != n_samples:
                raise ValueError(f"Channel '{name}' has {len(arr)} samples, expected {n_samples}")
        
        # Create PyArrow table
        table = pa.table(data)
        
        # Save to Parquet
        file_path = self.get_file_path(run_id)
        pq.write_table(table, file_path, compression='snappy')
        
        return str(file_path)
    
    def load(self, file_path: str) -> dict[str, np.ndarray]:
        """Load telemetry data from Parquet file."""
        table = pq.read_table(file_path)
        return {col: table[col].to_numpy() for col in table.column_names}
    
    def load_channel(self, file_path: str, channel: str) -> tuple[np.ndarray, np.ndarray]:
        """Load a single channel with its position array."""
        table = pq.read_table(file_path, columns=["position_m", channel])
        return table["position_m"].to_numpy(), table[channel].to_numpy()


class TelemetryAnalyzer:
    """Analyzes telemetry data across runs."""
    
    def __init__(self, store: TelemetryStore, thresholds: list[dict] = None):
        """
        Args:
            store: TelemetryStore instance
            thresholds: List of threshold definitions from DB
                        [{channel, name, value, direction, severity}, ...]
        """
        self.store = store
        self.thresholds = thresholds or []
        self._threshold_lookup = self._build_threshold_lookup()
    
    def _build_threshold_lookup(self) -> dict[str, list[dict]]:
        """Build channel -> thresholds lookup."""
        lookup = {}
        for t in self.thresholds:
            channel = t["channel"]
            if channel not in lookup:
                lookup[channel] = []
            lookup[channel].append(t)
        return lookup
    
    def resample_to_positions(
        self,
        data: dict[str, np.ndarray],
        target_positions: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Resample telemetry data to common position grid.
        Uses linear interpolation.
        """
        source_positions = data["position_m"]
        resampled = {"position_m": target_positions}
        
        for channel, values in data.items():
            if channel == "position_m":
                continue
            resampled[channel] = np.interp(target_positions, source_positions, values)
        
        return resampled
    
    def compute_common_positions(
        self,
        telemetry_paths: list[str],
        resolution_m: float = 10.0
    ) -> np.ndarray:
        """
        Compute a common position grid that covers all runs.
        
        Args:
            telemetry_paths: Paths to telemetry files
            resolution_m: Position resolution in meters
        
        Returns:
            Array of position values
        """
        min_pos = float('inf')
        max_pos = float('-inf')
        
        for path in telemetry_paths:
            data = self.store.load(path)
            positions = data["position_m"]
            min_pos = min(min_pos, positions.min())
            max_pos = max(max_pos, positions.max())
        
        return np.arange(min_pos, max_pos + resolution_m, resolution_m)
    
    def compute_envelope(
        self,
        telemetry_paths: list[str],
        common_positions: np.ndarray = None,
        resolution_m: float = 10.0
    ) -> dict[str, ChannelStats]:
        """
        Compute mean/std envelope across multiple runs.
        
        Args:
            telemetry_paths: List of Parquet file paths
            common_positions: Optional pre-computed position grid
            resolution_m: Position resolution if computing grid
        
        Returns:
            Dictionary mapping channel names to ChannelStats
        """
        if not telemetry_paths:
            return {}
        
        # Compute common position grid if not provided
        if common_positions is None:
            common_positions = self.compute_common_positions(telemetry_paths, resolution_m)
        
        # Load and resample all runs
        all_data = []
        for path in telemetry_paths:
            data = self.store.load(path)
            resampled = self.resample_to_positions(data, common_positions)
            all_data.append(resampled)
        
        # Get all channels (excluding position)
        channels = [c for c in all_data[0].keys() if c != "position_m"]
        
        # Compute statistics for each channel
        stats = {}
        for channel in channels:
            # Stack arrays: shape (n_runs, n_positions)
            stacked = np.stack([d[channel] for d in all_data], axis=0)
            
            stats[channel] = ChannelStats(
                channel=channel,
                mean=np.mean(stacked, axis=0),
                std=np.std(stacked, axis=0),
                min=np.min(stacked, axis=0),
                max=np.max(stacked, axis=0),
                positions=common_positions
            )
        
        return stats
    
    def detect_threshold_crossings(
        self,
        data: dict[str, np.ndarray],
        run_id: int = None
    ) -> list[ThresholdCrossing]:
        """
        Detect threshold crossings in telemetry data.
        
        Returns list of ThresholdCrossing events.
        """
        crossings = []
        positions = data["position_m"]
        
        for channel, thresholds in self._threshold_lookup.items():
            if channel not in data:
                continue
            
            values = data[channel]
            
            for thresh in thresholds:
                t_value = thresh["value"]
                direction = thresh["direction"]
                
                # Find violation regions
                if direction == "above":
                    violated = values > t_value
                else:  # below
                    violated = values < t_value
                
                # Find contiguous violation regions
                # Detect edges
                diff = np.diff(violated.astype(int))
                starts = np.where(diff == 1)[0] + 1
                ends = np.where(diff == -1)[0] + 1
                
                # Handle edge cases
                if violated[0]:
                    starts = np.concatenate([[0], starts])
                if violated[-1]:
                    ends = np.concatenate([ends, [len(violated)]])
                
                # Create crossing events
                for start, end in zip(starts, ends):
                    region_values = values[start:end]
                    region_positions = positions[start:end]
                    
                    if direction == "above":
                        peak_idx = np.argmax(region_values)
                    else:
                        peak_idx = np.argmin(region_values)
                    
                    crossings.append(ThresholdCrossing(
                        channel=channel,
                        threshold_name=thresh["name"],
                        threshold_value=t_value,
                        direction=direction,
                        severity=thresh.get("severity", "warning"),
                        position_m=region_positions[0],
                        value_at_crossing=region_values[0],
                        duration_m=region_positions[-1] - region_positions[0],
                        peak_value=region_values[peak_idx]
                    ))
        
        return crossings
    
    def analyze_batch_thresholds(
        self,
        telemetry_paths: dict[int, str],
        batch_id: str
    ) -> ThresholdAnalysis:
        """
        Analyze threshold crossings across a batch of runs.
        
        Args:
            telemetry_paths: {run_id: file_path} mapping
            batch_id: Identifier for the batch
        
        Returns:
            ThresholdAnalysis with per-run and aggregate results
        """
        crossings_by_run = {}
        all_crossings = []
        
        for run_id, path in telemetry_paths.items():
            data = self.store.load(path)
            run_crossings = self.detect_threshold_crossings(data, run_id)
            crossings_by_run[run_id] = run_crossings
            all_crossings.extend(run_crossings)
        
        # Build summary by channel/threshold
        summary = {}
        for crossing in all_crossings:
            key = f"{crossing.channel}:{crossing.threshold_name}"
            if key not in summary:
                summary[key] = {
                    "channel": crossing.channel,
                    "threshold_name": crossing.threshold_name,
                    "threshold_value": crossing.threshold_value,
                    "direction": crossing.direction,
                    "severity": crossing.severity,
                    "count": 0,
                    "total_duration_m": 0,
                    "positions": [],
                    "peak_values": []
                }
            summary[key]["count"] += 1
            summary[key]["total_duration_m"] += crossing.duration_m
            summary[key]["positions"].append(crossing.position_m)
            summary[key]["peak_values"].append(crossing.peak_value)
        
        # Compute averages
        for key in summary:
            s = summary[key]
            s["avg_duration_m"] = s["total_duration_m"] / s["count"] if s["count"] > 0 else 0
            s["avg_position_m"] = np.mean(s["positions"]) if s["positions"] else 0
            s["worst_peak"] = max(s["peak_values"]) if s["direction"] == "above" else min(s["peak_values"])
        
        return ThresholdAnalysis(
            batch_id=batch_id,
            run_count=len(telemetry_paths),
            crossings_by_run=crossings_by_run,
            crossing_summary=summary
        )
    
    def compare_channels(
        self,
        baseline_stats: dict[str, ChannelStats],
        candidate_stats: dict[str, ChannelStats]
    ) -> dict[str, ChannelComparison]:
        """Compare channel statistics between baseline and candidate."""
        comparisons = {}
        
        common_channels = set(baseline_stats.keys()) & set(candidate_stats.keys())
        
        for channel in common_channels:
            bs = baseline_stats[channel]
            cs = candidate_stats[channel]
            
            # Ensure same position grid (should be if computed together)
            if len(bs.positions) != len(cs.positions):
                continue
            
            mean_delta = cs.mean - bs.mean
            rms_delta = np.sqrt(np.mean(mean_delta ** 2))
            max_abs_idx = np.argmax(np.abs(mean_delta))
            
            comparisons[channel] = ChannelComparison(
                channel=channel,
                baseline_stats=bs,
                candidate_stats=cs,
                mean_delta=mean_delta,
                rms_delta=rms_delta,
                max_abs_delta=float(np.abs(mean_delta[max_abs_idx])),
                max_delta_position=float(bs.positions[max_abs_idx])
            )
        
        return comparisons
    
    def compare_telemetry(
        self,
        baseline_paths: dict[int, str],
        candidate_paths: dict[int, str],
        baseline_batch_id: str = "baseline",
        candidate_batch_id: str = "candidate",
        resolution_m: float = 10.0
    ) -> TelemetryComparison:
        """
        Full telemetry comparison between baseline and candidate batches.
        
        Args:
            baseline_paths: {run_id: file_path} for baseline runs
            candidate_paths: {run_id: file_path} for candidate runs
            baseline_batch_id: ID for baseline batch
            candidate_batch_id: ID for candidate batch
            resolution_m: Position resolution for resampling
        
        Returns:
            TelemetryComparison with full analysis
        """
        # Compute common position grid across all runs
        all_paths = list(baseline_paths.values()) + list(candidate_paths.values())
        common_positions = self.compute_common_positions(all_paths, resolution_m)
        
        # Compute envelopes
        baseline_stats = self.compute_envelope(
            list(baseline_paths.values()),
            common_positions
        )
        candidate_stats = self.compute_envelope(
            list(candidate_paths.values()),
            common_positions
        )
        
        # Compare channels
        channel_comparisons = self.compare_channels(baseline_stats, candidate_stats)
        
        # Analyze thresholds
        baseline_threshold = self.analyze_batch_thresholds(baseline_paths, baseline_batch_id)
        candidate_threshold = self.analyze_batch_thresholds(candidate_paths, candidate_batch_id)
        
        # Find new and resolved violations
        baseline_violations = set(baseline_threshold.crossing_summary.keys())
        candidate_violations = set(candidate_threshold.crossing_summary.keys())
        
        new_violations = list(candidate_violations - baseline_violations)
        resolved_violations = list(baseline_violations - candidate_violations)
        
        return TelemetryComparison(
            baseline_run_count=len(baseline_paths),
            candidate_run_count=len(candidate_paths),
            channel_comparisons=channel_comparisons,
            baseline_threshold_analysis=baseline_threshold,
            candidate_threshold_analysis=candidate_threshold,
            new_violations=new_violations,
            resolved_violations=resolved_violations
        )
    
    def format_threshold_report(self, analysis: ThresholdAnalysis) -> str:
        """Format threshold analysis as text report."""
        lines = []
        lines.append(f"Threshold Analysis: {analysis.batch_id}")
        lines.append(f"Runs analyzed: {analysis.run_count}")
        lines.append("")
        
        if not analysis.crossing_summary:
            lines.append("No threshold crossings detected.")
            return "\n".join(lines)
        
        lines.append("Crossings detected:")
        lines.append("-" * 60)
        
        # Sort by severity (critical first)
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        sorted_items = sorted(
            analysis.crossing_summary.items(),
            key=lambda x: (severity_order.get(x[1]["severity"], 99), -x[1]["count"])
        )
        
        for key, summary in sorted_items:
            sev = summary["severity"].upper()
            lines.append(f"\n[{sev}] {summary['channel']} - {summary['threshold_name']}")
            lines.append(f"  Threshold: {summary['direction']} {summary['threshold_value']}")
            lines.append(f"  Occurrences: {summary['count']}")
            lines.append(f"  Avg duration: {summary['avg_duration_m']:.1f} m")
            lines.append(f"  Avg position: {summary['avg_position_m']:.0f} m")
            lines.append(f"  Worst peak: {summary['worst_peak']:.2f}")
        
        return "\n".join(lines)
    
    def to_dict(self, comparison: TelemetryComparison) -> dict:
        """Convert TelemetryComparison to dictionary for AI consumption."""
        channel_data = {}
        for name, comp in comparison.channel_comparisons.items():
            channel_data[name] = {
                "baseline_mean_range": [
                    float(comp.baseline_stats.mean.min()),
                    float(comp.baseline_stats.mean.max())
                ],
                "candidate_mean_range": [
                    float(comp.candidate_stats.mean.min()),
                    float(comp.candidate_stats.mean.max())
                ],
                "rms_delta": comp.rms_delta,
                "max_abs_delta": comp.max_abs_delta,
                "max_delta_position_m": comp.max_delta_position
            }
        
        def summarize_thresholds(analysis: ThresholdAnalysis) -> list[dict]:
            return [
                {
                    "channel": s["channel"],
                    "threshold": s["threshold_name"],
                    "severity": s["severity"],
                    "count": s["count"],
                    "avg_duration_m": s["avg_duration_m"],
                    "worst_peak": s["worst_peak"]
                }
                for s in analysis.crossing_summary.values()
            ]
        
        return {
            "baseline_run_count": comparison.baseline_run_count,
            "candidate_run_count": comparison.candidate_run_count,
            "channels": channel_data,
            "baseline_threshold_crossings": summarize_thresholds(comparison.baseline_threshold_analysis),
            "candidate_threshold_crossings": summarize_thresholds(comparison.candidate_threshold_analysis),
            "new_violations": comparison.new_violations,
            "resolved_violations": comparison.resolved_violations
        }
