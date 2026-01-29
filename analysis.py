"""
Analysis module for comparing simulation runs and scoring against requirements.
"""

import yaml
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricRequirement:
    name: str
    description: str
    unit: str
    target: float
    weight: float
    direction: str  # "lower_better" or "higher_better"
    min_val: Optional[float] = None
    max_val: Optional[float] = None


@dataclass
class MetricStats:
    """Statistics for a single metric across multiple runs."""
    name: str
    mean: float
    std: float
    min: float
    max: float
    count: int


@dataclass
class MetricComparison:
    """Comparison of a metric between baseline and candidate."""
    name: str
    baseline_stats: MetricStats
    candidate_stats: MetricStats
    delta_mean: float  # candidate - baseline
    delta_percent: float  # percentage change
    improved: bool  # True if candidate is better
    requirement: Optional[MetricRequirement] = None
    score: float = 0.0  # 0.0 to 1.0


@dataclass
class AnalysisResult:
    """Complete analysis result for a candidate batch vs baseline."""
    candidate_batch_id: str
    baseline_run_ids: list[int]
    candidate_run_ids: list[int]
    metric_comparisons: dict[str, MetricComparison]
    overall_score: float
    status: str  # "excellent", "good", "acceptable", "poor", "fail"
    requirement_violations: list[str] = field(default_factory=list)


class RequirementsLoader:
    """Load and parse requirements from YAML file."""
    
    def __init__(self, yaml_path: str = "requirements.yaml"):
        self.yaml_path = yaml_path
        self._requirements: dict[str, MetricRequirement] = {}
        self._scoring_config: dict = {}
        self._load()
    
    def _load(self):
        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        for name, cfg in data.get("metrics", {}).items():
            self._requirements[name] = MetricRequirement(
                name=name,
                description=cfg.get("description", ""),
                unit=cfg.get("unit", ""),
                target=cfg["target"],
                weight=cfg.get("weight", 0.1),
                direction=cfg.get("direction", "lower_better"),
                min_val=cfg.get("min"),
                max_val=cfg.get("max"),
            )
        
        self._scoring_config = data.get("scoring", {})
    
    @property
    def metrics(self) -> dict[str, MetricRequirement]:
        return self._requirements
    
    @property
    def thresholds(self) -> dict[str, float]:
        return self._scoring_config.get("thresholds", {
            "excellent": 0.90,
            "good": 0.75,
            "acceptable": 0.60,
            "poor": 0.40
        })


class Analyzer:
    """Main analysis engine for comparing runs."""
    
    def __init__(self, requirements: RequirementsLoader):
        self.requirements = requirements
    
    def compute_stats(self, run_metrics: dict[int, dict[str, float]]) -> dict[str, MetricStats]:
        """Compute statistics for each metric across runs."""
        if not run_metrics:
            return {}
        
        # Transpose: metric_name -> list of values
        metrics_by_name: dict[str, list[float]] = {}
        for run_id, metrics in run_metrics.items():
            for name, value in metrics.items():
                if name not in metrics_by_name:
                    metrics_by_name[name] = []
                metrics_by_name[name].append(value)
        
        stats = {}
        for name, values in metrics_by_name.items():
            if len(values) >= 2:
                std = statistics.stdev(values)
            else:
                std = 0.0
            
            stats[name] = MetricStats(
                name=name,
                mean=statistics.mean(values),
                std=std,
                min=min(values),
                max=max(values),
                count=len(values)
            )
        
        return stats
    
    def score_metric(
        self,
        value: float,
        requirement: MetricRequirement
    ) -> float:
        """
        Score a metric value against its requirement.
        Returns a score from 0.0 (worst) to 1.0 (perfect).
        """
        target = requirement.target
        
        if requirement.direction == "lower_better":
            # For lower_better: at target = 1.0, at max = 0.0, below target = 1.0
            if value <= target:
                return 1.0
            if requirement.max_val is not None:
                if value >= requirement.max_val:
                    return 0.0
                # Linear interpolation between target and max
                return 1.0 - (value - target) / (requirement.max_val - target)
            else:
                # No max defined, use 2x target as implicit max
                implicit_max = target * 2
                if value >= implicit_max:
                    return 0.0
                return 1.0 - (value - target) / (implicit_max - target)
        
        else:  # higher_better
            # For higher_better: at target = 1.0, at min = 0.0, above target = 1.0
            if value >= target:
                return 1.0
            if requirement.min_val is not None:
                if value <= requirement.min_val:
                    return 0.0
                # Linear interpolation between min and target
                return (value - requirement.min_val) / (target - requirement.min_val)
            else:
                # No min defined, use 0.5x target as implicit min
                implicit_min = target * 0.5
                if value <= implicit_min:
                    return 0.0
                return (value - implicit_min) / (target - implicit_min)
    
    def is_improved(
        self,
        baseline_val: float,
        candidate_val: float,
        requirement: MetricRequirement
    ) -> bool:
        """Check if candidate value is better than baseline."""
        if requirement.direction == "lower_better":
            return candidate_val < baseline_val
        else:
            return candidate_val > baseline_val
    
    def check_violations(
        self,
        stats: dict[str, MetricStats]
    ) -> list[str]:
        """Check which requirements are violated."""
        violations = []
        
        for name, req in self.requirements.metrics.items():
            if name not in stats:
                continue
            
            value = stats[name].mean
            
            if req.direction == "lower_better" and req.max_val is not None:
                if value > req.max_val:
                    violations.append(
                        f"{name}: {value:.3f} exceeds max {req.max_val} {req.unit}"
                    )
            
            elif req.direction == "higher_better" and req.min_val is not None:
                if value < req.min_val:
                    violations.append(
                        f"{name}: {value:.3f} below min {req.min_val} {req.unit}"
                    )
        
        return violations
    
    def compare(
        self,
        baseline_metrics: dict[int, dict[str, float]],
        candidate_metrics: dict[int, dict[str, float]],
        candidate_batch_id: str
    ) -> AnalysisResult:
        """
        Compare candidate runs against baseline runs.
        
        Args:
            baseline_metrics: {run_id: {metric_name: value}} for baseline runs
            candidate_metrics: {run_id: {metric_name: value}} for candidate runs
            candidate_batch_id: ID of the candidate batch
        
        Returns:
            AnalysisResult with detailed comparison
        """
        baseline_stats = self.compute_stats(baseline_metrics)
        candidate_stats = self.compute_stats(candidate_metrics)
        
        comparisons = {}
        weighted_score_sum = 0.0
        weight_sum = 0.0
        
        # Get all metric names from both sets
        all_metrics = set(baseline_stats.keys()) | set(candidate_stats.keys())
        
        for metric_name in all_metrics:
            if metric_name not in baseline_stats or metric_name not in candidate_stats:
                continue
            
            bs = baseline_stats[metric_name]
            cs = candidate_stats[metric_name]
            
            delta_mean = cs.mean - bs.mean
            delta_percent = (delta_mean / bs.mean * 100) if bs.mean != 0 else 0
            
            # Get requirement if exists
            req = self.requirements.metrics.get(metric_name)
            
            improved = False
            score = 0.5  # Default neutral score if no requirement
            
            if req:
                improved = self.is_improved(bs.mean, cs.mean, req)
                score = self.score_metric(cs.mean, req)
                weighted_score_sum += score * req.weight
                weight_sum += req.weight
            
            comparisons[metric_name] = MetricComparison(
                name=metric_name,
                baseline_stats=bs,
                candidate_stats=cs,
                delta_mean=delta_mean,
                delta_percent=delta_percent,
                improved=improved,
                requirement=req,
                score=score
            )
        
        # Compute overall score
        overall_score = weighted_score_sum / weight_sum if weight_sum > 0 else 0.0
        
        # Determine status
        thresholds = self.requirements.thresholds
        if overall_score >= thresholds["excellent"]:
            status = "excellent"
        elif overall_score >= thresholds["good"]:
            status = "good"
        elif overall_score >= thresholds["acceptable"]:
            status = "acceptable"
        elif overall_score >= thresholds["poor"]:
            status = "poor"
        else:
            status = "fail"
        
        # Check for violations
        violations = self.check_violations(candidate_stats)
        
        return AnalysisResult(
            candidate_batch_id=candidate_batch_id,
            baseline_run_ids=list(baseline_metrics.keys()),
            candidate_run_ids=list(candidate_metrics.keys()),
            metric_comparisons=comparisons,
            overall_score=overall_score,
            status=status,
            requirement_violations=violations
        )
    
    def format_report(self, result: AnalysisResult) -> str:
        """Format analysis result as a text report."""
        lines = []
        lines.append("=" * 70)
        lines.append("RACESIM ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"Candidate Batch: {result.candidate_batch_id}")
        lines.append(f"Baseline Runs: {len(result.baseline_run_ids)}")
        lines.append(f"Candidate Runs: {len(result.candidate_run_ids)}")
        lines.append("")
        lines.append(f"Overall Score: {result.overall_score:.1%}")
        lines.append(f"Status: {result.status.upper()}")
        lines.append("")
        
        if result.requirement_violations:
            lines.append("REQUIREMENT VIOLATIONS:")
            for v in result.requirement_violations:
                lines.append(f"  ⚠ {v}")
            lines.append("")
        
        lines.append("-" * 70)
        lines.append("METRIC COMPARISON:")
        lines.append("-" * 70)
        
        for name, comp in sorted(result.metric_comparisons.items()):
            indicator = "✓" if comp.improved else "✗"
            unit = comp.requirement.unit if comp.requirement else ""
            
            lines.append(f"\n{name}:")
            lines.append(f"  Baseline: {comp.baseline_stats.mean:.3f} ± {comp.baseline_stats.std:.3f} {unit}")
            lines.append(f"  Candidate: {comp.candidate_stats.mean:.3f} ± {comp.candidate_stats.std:.3f} {unit}")
            lines.append(f"  Delta: {comp.delta_mean:+.3f} ({comp.delta_percent:+.1f}%) {indicator}")
            if comp.requirement:
                lines.append(f"  Target: {comp.requirement.target} {unit}")
                lines.append(f"  Score: {comp.score:.1%}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_dict(self, result: AnalysisResult) -> dict:
        """Convert analysis result to dictionary for AI consumption."""
        return {
            "candidate_batch_id": result.candidate_batch_id,
            "baseline_run_count": len(result.baseline_run_ids),
            "candidate_run_count": len(result.candidate_run_ids),
            "overall_score": result.overall_score,
            "overall_score_percent": f"{result.overall_score:.1%}",
            "status": result.status,
            "requirement_violations": result.requirement_violations,
            "metrics": {
                name: {
                    "baseline_mean": comp.baseline_stats.mean,
                    "baseline_std": comp.baseline_stats.std,
                    "candidate_mean": comp.candidate_stats.mean,
                    "candidate_std": comp.candidate_stats.std,
                    "delta": comp.delta_mean,
                    "delta_percent": comp.delta_percent,
                    "improved": comp.improved,
                    "score": comp.score,
                    "target": comp.requirement.target if comp.requirement else None,
                    "unit": comp.requirement.unit if comp.requirement else "",
                    "direction": comp.requirement.direction if comp.requirement else None,
                }
                for name, comp in result.metric_comparisons.items()
            }
        }
