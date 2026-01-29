"""
Visualization module for telemetry data.

Provides:
- Single channel plots with baseline envelope vs candidate
- Multi-channel dashboard views
- Threshold crossing highlighting
- Export to PNG/PDF
"""

import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for file output
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from telemetry import (
    TelemetryStore, TelemetryAnalyzer, ChannelStats, 
    ChannelComparison, TelemetryComparison, ThresholdAnalysis
)


# Channel display configuration
CHANNEL_CONFIG = {
    "speed": {"label": "Speed", "unit": "m/s", "color": "#2563eb"},
    "tire_wear_fl": {"label": "Tire Wear FL", "unit": "%", "color": "#dc2626"},
    "tire_wear_fr": {"label": "Tire Wear FR", "unit": "%", "color": "#ea580c"},
    "tire_wear_rl": {"label": "Tire Wear RL", "unit": "%", "color": "#d97706"},
    "tire_wear_rr": {"label": "Tire Wear RR", "unit": "%", "color": "#ca8a04"},
    "brake_temp_fl": {"label": "Brake Temp FL", "unit": "°C", "color": "#dc2626"},
    "brake_temp_fr": {"label": "Brake Temp FR", "unit": "°C", "color": "#ea580c"},
    "brake_temp_rl": {"label": "Brake Temp RL", "unit": "°C", "color": "#d97706"},
    "brake_temp_rr": {"label": "Brake Temp RR", "unit": "°C", "color": "#ca8a04"},
    "throttle": {"label": "Throttle", "unit": "", "color": "#16a34a"},
    "brake": {"label": "Brake", "unit": "", "color": "#dc2626"},
    "g_lateral": {"label": "Lateral G", "unit": "g", "color": "#7c3aed"},
    "g_longitudinal": {"label": "Longitudinal G", "unit": "g", "color": "#2563eb"},
    "fuel_remaining": {"label": "Fuel Remaining", "unit": "kg", "color": "#0891b2"},
}

# Default style settings
STYLE = {
    "baseline_color": "#3b82f6",      # Blue
    "baseline_fill_alpha": 0.25,
    "candidate_color": "#ef4444",      # Red
    "candidate_fill_alpha": 0.25,
    "threshold_warning_color": "#f59e0b",   # Amber
    "threshold_critical_color": "#dc2626",  # Red
    "threshold_linestyle": "--",
    "grid_alpha": 0.3,
    "figure_dpi": 150,
}


def check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


class TelemetryVisualizer:
    """Creates visualizations for telemetry comparison data."""
    
    def __init__(
        self,
        style: dict = None,
        figsize_single: tuple = (12, 5),
        figsize_dashboard: tuple = (16, 12),
    ):
        check_matplotlib()
        self.style = {**STYLE, **(style or {})}
        self.figsize_single = figsize_single
        self.figsize_dashboard = figsize_dashboard
    
    def plot_channel(
        self,
        ax: plt.Axes,
        baseline_stats: ChannelStats,
        candidate_stats: ChannelStats = None,
        channel_name: str = None,
        thresholds: list[dict] = None,
        show_legend: bool = True,
        title: str = None,
    ):
        """
        Plot a single channel with baseline envelope and optional candidate overlay.
        
        Args:
            ax: Matplotlib axes to plot on
            baseline_stats: ChannelStats for baseline runs
            candidate_stats: Optional ChannelStats for candidate runs
            channel_name: Channel name for labeling
            thresholds: List of threshold definitions for this channel
            show_legend: Whether to show legend
            title: Optional title override
        """
        positions = baseline_stats.positions
        config = CHANNEL_CONFIG.get(channel_name, {})
        label = config.get("label", channel_name)
        unit = config.get("unit", "")
        
        # Plot baseline envelope (mean ± std)
        ax.fill_between(
            positions,
            baseline_stats.mean - baseline_stats.std,
            baseline_stats.mean + baseline_stats.std,
            alpha=self.style["baseline_fill_alpha"],
            color=self.style["baseline_color"],
            label="Baseline ±1σ"
        )
        ax.plot(
            positions,
            baseline_stats.mean,
            color=self.style["baseline_color"],
            linewidth=1.5,
            label="Baseline mean"
        )
        
        # Plot candidate envelope if provided
        if candidate_stats is not None:
            ax.fill_between(
                positions,
                candidate_stats.mean - candidate_stats.std,
                candidate_stats.mean + candidate_stats.std,
                alpha=self.style["candidate_fill_alpha"],
                color=self.style["candidate_color"],
                label="Candidate ±1σ"
            )
            ax.plot(
                positions,
                candidate_stats.mean,
                color=self.style["candidate_color"],
                linewidth=1.5,
                label="Candidate mean"
            )
        
        # Plot threshold lines
        if thresholds:
            for thresh in thresholds:
                color = (
                    self.style["threshold_critical_color"]
                    if thresh.get("severity") == "critical"
                    else self.style["threshold_warning_color"]
                )
                ax.axhline(
                    y=thresh["value"],
                    color=color,
                    linestyle=self.style["threshold_linestyle"],
                    linewidth=1,
                    alpha=0.8,
                    label=f"{thresh['name']}: {thresh['value']}"
                )
        
        # Formatting
        ax.set_xlabel("Track Position (m)")
        ylabel = f"{label} ({unit})" if unit else label
        ax.set_ylabel(ylabel)
        ax.set_title(title or label)
        ax.grid(True, alpha=self.style["grid_alpha"])
        ax.set_xlim(positions[0], positions[-1])
        
        if show_legend:
            ax.legend(loc="best", fontsize=8)
    
    def plot_single_channel(
        self,
        baseline_stats: ChannelStats,
        candidate_stats: ChannelStats = None,
        channel_name: str = None,
        thresholds: list[dict] = None,
        title: str = None,
        save_path: str = None,
    ) -> plt.Figure:
        """
        Create a standalone figure for a single channel.
        
        Args:
            baseline_stats: ChannelStats for baseline
            candidate_stats: Optional ChannelStats for candidate
            channel_name: Channel name
            thresholds: Threshold definitions
            title: Optional title
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        self.plot_channel(
            ax, baseline_stats, candidate_stats,
            channel_name, thresholds, title=title
        )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.style["figure_dpi"], bbox_inches="tight")
        
        return fig
    
    def plot_dashboard(
        self,
        comparison: TelemetryComparison,
        channels: list[str] = None,
        thresholds_by_channel: dict[str, list[dict]] = None,
        title: str = None,
        save_path: str = None,
        cols: int = 3,
    ) -> plt.Figure:
        """
        Create a multi-channel dashboard view.
        
        Args:
            comparison: TelemetryComparison object
            channels: List of channels to plot (default: all available)
            thresholds_by_channel: {channel: [thresholds]} mapping
            title: Dashboard title
            save_path: Optional path to save figure
            cols: Number of columns in grid
        
        Returns:
            matplotlib Figure object
        """
        # Get available channels
        available = list(comparison.channel_comparisons.keys())
        if channels is None:
            # Default: prioritize key channels
            priority = [
                "speed", "tire_wear_fl", "tire_wear_fr", "tire_wear_rl", "tire_wear_rr",
                "brake_temp_fl", "brake_temp_fr", "g_lateral", "fuel_remaining"
            ]
            channels = [c for c in priority if c in available]
            # Add any remaining
            for c in available:
                if c not in channels:
                    channels.append(c)
        else:
            channels = [c for c in channels if c in available]
        
        if not channels:
            raise ValueError("No channels available to plot")
        
        # Calculate grid dimensions
        n_channels = len(channels)
        rows = (n_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(self.figsize_dashboard[0], rows * 3.5)
        )
        
        # Flatten axes array for easy iteration
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes_flat = axes.flatten()
        
        # Plot each channel
        for i, channel in enumerate(channels):
            ax = axes_flat[i]
            comp = comparison.channel_comparisons[channel]
            
            channel_thresholds = None
            if thresholds_by_channel and channel in thresholds_by_channel:
                channel_thresholds = thresholds_by_channel[channel]
            
            self.plot_channel(
                ax,
                comp.baseline_stats,
                comp.candidate_stats,
                channel,
                channel_thresholds,
                show_legend=(i == 0),  # Only show legend on first plot
            )
        
        # Hide unused subplots
        for i in range(n_channels, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Add overall title
        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.style["figure_dpi"], bbox_inches="tight")
        
        return fig
    
    def plot_threshold_violations(
        self,
        baseline_stats: dict[str, ChannelStats],
        candidate_stats: dict[str, ChannelStats],
        baseline_analysis: ThresholdAnalysis,
        candidate_analysis: ThresholdAnalysis,
        thresholds_by_channel: dict[str, list[dict]] = None,
        save_path: str = None,
    ) -> plt.Figure:
        """
        Create focused plots for channels with threshold violations.
        
        Shows only channels that have violations in either baseline or candidate,
        highlighting the violation regions.
        """
        # Find channels with violations
        violation_channels = set()
        
        for key in baseline_analysis.crossing_summary:
            channel = baseline_analysis.crossing_summary[key]["channel"]
            violation_channels.add(channel)
        
        for key in candidate_analysis.crossing_summary:
            channel = candidate_analysis.crossing_summary[key]["channel"]
            violation_channels.add(channel)
        
        if not violation_channels:
            # No violations - create a simple message figure
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(
                0.5, 0.5, "No threshold violations detected",
                ha="center", va="center", fontsize=14
            )
            ax.axis("off")
            if save_path:
                fig.savefig(save_path, dpi=self.style["figure_dpi"])
            return fig
        
        # Filter to channels we have data for
        violation_channels = [
            c for c in violation_channels
            if c in baseline_stats and c in candidate_stats
        ]
        
        n_channels = len(violation_channels)
        cols = min(2, n_channels)
        rows = (n_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(7 * cols, 4 * rows)
        )
        
        if n_channels == 1:
            axes = np.array([axes])
        axes_flat = np.array(axes).flatten()
        
        for i, channel in enumerate(violation_channels):
            ax = axes_flat[i]
            
            bs = baseline_stats[channel]
            cs = candidate_stats[channel]
            
            channel_thresholds = None
            if thresholds_by_channel and channel in thresholds_by_channel:
                channel_thresholds = thresholds_by_channel[channel]
            
            self.plot_channel(
                ax, bs, cs, channel, channel_thresholds,
                show_legend=(i == 0)
            )
            
            # Highlight violation regions for candidate
            if channel_thresholds:
                for thresh in channel_thresholds:
                    value = thresh["value"]
                    direction = thresh["direction"]
                    
                    if direction == "above":
                        # Shade region above threshold where candidate exceeds it
                        violation_mask = cs.mean > value
                        if violation_mask.any():
                            ax.fill_between(
                                cs.positions,
                                value,
                                np.where(violation_mask, cs.mean, value),
                                alpha=0.3,
                                color=self.style["threshold_critical_color"],
                                hatch="//",
                            )
                    else:  # below
                        violation_mask = cs.mean < value
                        if violation_mask.any():
                            ax.fill_between(
                                cs.positions,
                                np.where(violation_mask, cs.mean, value),
                                value,
                                alpha=0.3,
                                color=self.style["threshold_critical_color"],
                                hatch="//",
                            )
        
        # Hide unused subplots
        for i in range(n_channels, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        fig.suptitle("Threshold Violation Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.style["figure_dpi"], bbox_inches="tight")
        
        return fig
    
    def plot_delta(
        self,
        comparison: TelemetryComparison,
        channels: list[str] = None,
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot the difference (delta) between candidate and baseline.
        
        Useful for seeing where on track the biggest changes occur.
        """
        available = list(comparison.channel_comparisons.keys())
        if channels is None:
            channels = ["speed", "tire_wear_fl", "brake_temp_fl", "g_lateral"]
            channels = [c for c in channels if c in available]
        
        n_channels = len(channels)
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels))
        
        if n_channels == 1:
            axes = [axes]
        
        for i, channel in enumerate(channels):
            ax = axes[i]
            comp = comparison.channel_comparisons[channel]
            positions = comp.baseline_stats.positions
            delta = comp.mean_delta
            
            config = CHANNEL_CONFIG.get(channel, {})
            label = config.get("label", channel)
            unit = config.get("unit", "")
            
            # Color positive/negative differently
            colors = np.where(delta >= 0, "#16a34a", "#dc2626")  # Green/Red
            
            ax.fill_between(
                positions, 0, delta,
                where=(delta >= 0),
                alpha=0.5,
                color="#16a34a",
                label="Candidate better" if i == 0 else None
            )
            ax.fill_between(
                positions, 0, delta,
                where=(delta < 0),
                alpha=0.5,
                color="#dc2626",
                label="Baseline better" if i == 0 else None
            )
            ax.axhline(y=0, color="black", linewidth=0.5)
            
            ax.set_xlabel("Track Position (m)")
            ylabel = f"Δ {label} ({unit})" if unit else f"Δ {label}"
            ax.set_ylabel(ylabel)
            ax.set_title(f"{label}: Candidate - Baseline")
            ax.grid(True, alpha=self.style["grid_alpha"])
            ax.set_xlim(positions[0], positions[-1])
            
            # Add RMS annotation
            ax.annotate(
                f"RMS: {comp.rms_delta:.3f}",
                xy=(0.98, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )
        
        if n_channels > 0:
            axes[0].legend(loc="upper left")
        
        fig.suptitle("Delta Analysis (Candidate - Baseline)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.style["figure_dpi"], bbox_inches="tight")
        
        return fig
    
    def create_report(
        self,
        comparison: TelemetryComparison,
        thresholds_by_channel: dict[str, list[dict]] = None,
        output_dir: str = "plots",
        prefix: str = "telemetry",
    ) -> list[str]:
        """
        Generate a complete set of report plots.
        
        Args:
            comparison: TelemetryComparison object
            thresholds_by_channel: Threshold definitions by channel
            output_dir: Directory for output files
            prefix: Filename prefix
        
        Returns:
            List of generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 1. Main dashboard
        dashboard_path = output_path / f"{prefix}_dashboard.png"
        self.plot_dashboard(
            comparison,
            thresholds_by_channel=thresholds_by_channel,
            title="Telemetry Comparison Dashboard",
            save_path=str(dashboard_path)
        )
        generated_files.append(str(dashboard_path))
        plt.close()
        
        # 2. Delta analysis
        delta_path = output_path / f"{prefix}_delta.png"
        self.plot_delta(comparison, save_path=str(delta_path))
        generated_files.append(str(delta_path))
        plt.close()
        
        # 3. Threshold violations
        baseline_stats = {
            name: comp.baseline_stats
            for name, comp in comparison.channel_comparisons.items()
        }
        candidate_stats = {
            name: comp.candidate_stats
            for name, comp in comparison.channel_comparisons.items()
        }
        
        violations_path = output_path / f"{prefix}_violations.png"
        self.plot_threshold_violations(
            baseline_stats,
            candidate_stats,
            comparison.baseline_threshold_analysis,
            comparison.candidate_threshold_analysis,
            thresholds_by_channel,
            save_path=str(violations_path)
        )
        generated_files.append(str(violations_path))
        plt.close()
        
        # 4. Individual channel plots for key metrics
        key_channels = ["speed", "tire_wear_fl", "brake_temp_fl"]
        for channel in key_channels:
            if channel in comparison.channel_comparisons:
                comp = comparison.channel_comparisons[channel]
                channel_thresholds = (
                    thresholds_by_channel.get(channel)
                    if thresholds_by_channel else None
                )
                
                channel_path = output_path / f"{prefix}_{channel}.png"
                self.plot_single_channel(
                    comp.baseline_stats,
                    comp.candidate_stats,
                    channel,
                    channel_thresholds,
                    save_path=str(channel_path)
                )
                generated_files.append(str(channel_path))
                plt.close()
        
        return generated_files


def build_thresholds_by_channel(thresholds: list[dict]) -> dict[str, list[dict]]:
    """Convert flat threshold list to channel-grouped dictionary."""
    by_channel = {}
    for t in thresholds:
        channel = t["channel"]
        if channel not in by_channel:
            by_channel[channel] = []
        by_channel[channel].append(t)
    return by_channel
