#!/usr/bin/env python3
"""
RaceSim Analyzer - CLI for analyzing racing simulation results.

Usage:
    python main.py init                      # Initialize database
    python main.py list-batches              # List all batches
    python main.py analyze <batch_id>        # Analyze a batch vs baseline
    python main.py analyze <batch_id> --ai   # Include AI insights
    python main.py analyze <batch_id> --telemetry  # Include telemetry analysis
    python main.py compare <b1> <b2> ...     # Compare multiple batches
    python main.py telemetry <batch_id>      # Telemetry-only analysis
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from db import RaceSimDB
from analysis import RequirementsLoader, Analyzer
from ai_analyzer import AIAnalyzer

# Telemetry imports (optional - may not be installed)
try:
    from telemetry import TelemetryStore, TelemetryAnalyzer, PARQUET_AVAILABLE
except ImportError:
    PARQUET_AVAILABLE = False

# Visualization imports (optional)
try:
    from visualization import TelemetryVisualizer, build_thresholds_by_channel, MATPLOTLIB_AVAILABLE
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# PDF report imports (optional)
try:
    from report import PDFReportGenerator, ReportConfig, generate_report, REPORTLAB_AVAILABLE
except ImportError:
    REPORTLAB_AVAILABLE = False


def cmd_init(args):
    """Initialize the database."""
    db = RaceSimDB(args.database)
    db.init_schema()
    print(f"Database initialized: {args.database}")
    
    # Load telemetry thresholds from requirements if available
    req_path = args.requirements or "requirements.yaml"
    if Path(req_path).exists():
        requirements = RequirementsLoader(req_path)
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


def cmd_list_batches(args):
    """List all batches in the database."""
    db = RaceSimDB(args.database)
    batches = db.list_batches()
    
    if not batches:
        print("No batches found.")
        db.close()
        return
    
    print(f"{'Batch ID':<30} {'Runs':>6} {'Baseline':>10} {'Created':<20}")
    print("-" * 70)
    for b in batches:
        baseline = "Yes" if b["has_baseline"] else "No"
        print(f"{b['batch_id']:<30} {b['run_count']:>6} {baseline:>10} {b['created_at']:<20}")
    
    db.close()


def cmd_analyze(args):
    """Analyze a candidate batch against baseline."""
    db = RaceSimDB(args.database)
    
    # Get candidate batch runs
    candidate_run_ids = db.get_batch_run_ids(args.batch_id)
    if not candidate_run_ids:
        print(f"Error: Batch '{args.batch_id}' not found.")
        db.close()
        return 1
    
    # Get scenario for this batch
    scenario_id = db.get_scenario_id_for_batch(args.batch_id)
    if not scenario_id:
        print(f"Error: Could not determine scenario for batch '{args.batch_id}'.")
        db.close()
        return 1
    
    # Get baseline runs for this scenario
    baseline_run_ids = db.get_baseline_runs_for_scenario(scenario_id)
    if not baseline_run_ids:
        print(f"Error: No baseline runs found for scenario {scenario_id}.")
        print("Mark some runs as baseline first.")
        db.close()
        return 1
    
    # Filter out any baseline runs from candidate (in case batch contains baselines)
    candidate_run_ids = [r for r in candidate_run_ids if r not in baseline_run_ids]
    
    if not candidate_run_ids:
        print(f"Error: Batch '{args.batch_id}' contains only baseline runs.")
        db.close()
        return 1
    
    # Load metrics
    baseline_metrics = db.get_all_metrics_for_runs(baseline_run_ids)
    candidate_metrics = db.get_all_metrics_for_runs(candidate_run_ids)
    
    # Run analysis
    req_path = args.requirements or "requirements.yaml"
    requirements = RequirementsLoader(req_path)
    analyzer = Analyzer(requirements)
    
    result = analyzer.compare(baseline_metrics, candidate_metrics, args.batch_id)
    
    # Print report
    report = analyzer.format_report(result)
    print(report)
    
    # Telemetry analysis if requested
    telemetry_dict = None
    if args.telemetry and PARQUET_AVAILABLE:
        print("\n" + "=" * 70)
        print("TELEMETRY ANALYSIS")
        print("=" * 70)
        
        # Get telemetry paths
        baseline_paths = db.get_telemetry_paths_for_runs(baseline_run_ids)
        candidate_paths = db.get_telemetry_paths_for_runs(candidate_run_ids)
        
        if not baseline_paths:
            print("No telemetry data for baseline runs.")
        elif not candidate_paths:
            print("No telemetry data for candidate runs.")
        else:
            store = TelemetryStore(args.telemetry_dir)
            thresholds = db.get_thresholds()
            tel_analyzer = TelemetryAnalyzer(store, thresholds)
            
            tel_comparison = tel_analyzer.compare_telemetry(
                baseline_paths, candidate_paths,
                "baseline", args.batch_id
            )
            
            # Print threshold report
            print("\n--- Baseline Thresholds ---")
            print(tel_analyzer.format_threshold_report(tel_comparison.baseline_threshold_analysis))
            print("\n--- Candidate Thresholds ---")
            print(tel_analyzer.format_threshold_report(tel_comparison.candidate_threshold_analysis))
            
            if tel_comparison.new_violations:
                print(f"\n⚠ NEW VIOLATIONS in candidate: {', '.join(tel_comparison.new_violations)}")
            if tel_comparison.resolved_violations:
                print(f"\n✓ RESOLVED from baseline: {', '.join(tel_comparison.resolved_violations)}")
            
            telemetry_dict = tel_analyzer.to_dict(tel_comparison)
    elif args.telemetry and not PARQUET_AVAILABLE:
        print("\nTelemetry analysis requires pyarrow. Install with: pip install pyarrow")
    
    # AI analysis if requested
    if args.ai:
        print("\n" + "=" * 70)
        print("AI ANALYSIS")
        print("=" * 70)
        
        ai = AIAnalyzer(
            base_url=args.ollama_url,
            model=args.model,
            api_key=args.api_key,
        )

        # Check connection
        connected, msg = ai.check_connection()
        if not connected:
            print(f"Warning: {msg}")
            print("Skipping AI analysis.")
        else:
            analysis_dict = analyzer.to_dict(result)
            
            # Combined analysis if we have telemetry
            if telemetry_dict:
                ai_response = ai.analyze_combined(analysis_dict, telemetry_dict)
            else:
                context = args.context if hasattr(args, "context") else None
                ai_response = ai.analyze(analysis_dict, context)
            
            print(ai_response)
            
            if args.suggest:
                print("\n" + "-" * 70)
                print("IMPROVEMENT SUGGESTIONS")
                print("-" * 70)
                suggestions = ai.suggest_improvements(analysis_dict)
                print(suggestions)
    
    db.close()
    return 0


def cmd_telemetry(args):
    """Standalone telemetry analysis."""
    if not PARQUET_AVAILABLE:
        print("Telemetry analysis requires pyarrow. Install with: pip install pyarrow")
        return 1
    
    db = RaceSimDB(args.database)
    
    # Get candidate batch runs
    candidate_run_ids = db.get_batch_run_ids(args.batch_id)
    if not candidate_run_ids:
        print(f"Error: Batch '{args.batch_id}' not found.")
        db.close()
        return 1
    
    # Get scenario and baseline
    scenario_id = db.get_scenario_id_for_batch(args.batch_id)
    baseline_run_ids = db.get_baseline_runs_for_scenario(scenario_id)
    
    if not baseline_run_ids:
        print(f"Error: No baseline runs found for scenario {scenario_id}.")
        db.close()
        return 1
    
    candidate_run_ids = [r for r in candidate_run_ids if r not in baseline_run_ids]
    
    # Get telemetry paths
    baseline_paths = db.get_telemetry_paths_for_runs(baseline_run_ids)
    candidate_paths = db.get_telemetry_paths_for_runs(candidate_run_ids)
    
    if not baseline_paths:
        print("Error: No telemetry data for baseline runs.")
        db.close()
        return 1
    
    if not candidate_paths:
        print("Error: No telemetry data for candidate runs.")
        db.close()
        return 1
    
    store = TelemetryStore(args.telemetry_dir)
    thresholds = db.get_thresholds()
    analyzer = TelemetryAnalyzer(store, thresholds)
    
    comparison = analyzer.compare_telemetry(
        baseline_paths, candidate_paths,
        "baseline", args.batch_id
    )
    
    # Print reports
    print("=" * 70)
    print("TELEMETRY COMPARISON REPORT")
    print("=" * 70)
    print(f"Baseline runs: {comparison.baseline_run_count}")
    print(f"Candidate runs: {comparison.candidate_run_count}")
    
    print("\n--- Channel Differences ---")
    for name, comp in sorted(comparison.channel_comparisons.items()):
        print(f"\n{name}:")
        print(f"  Baseline range: {comp.baseline_stats.mean.min():.2f} - {comp.baseline_stats.mean.max():.2f}")
        print(f"  Candidate range: {comp.candidate_stats.mean.min():.2f} - {comp.candidate_stats.mean.max():.2f}")
        print(f"  RMS delta: {comp.rms_delta:.4f}")
        print(f"  Max delta: {comp.max_abs_delta:.4f} at {comp.max_delta_position:.0f}m")
    
    print("\n" + "-" * 70)
    print("BASELINE THRESHOLD ANALYSIS")
    print("-" * 70)
    print(analyzer.format_threshold_report(comparison.baseline_threshold_analysis))
    
    print("\n" + "-" * 70)
    print("CANDIDATE THRESHOLD ANALYSIS")
    print("-" * 70)
    print(analyzer.format_threshold_report(comparison.candidate_threshold_analysis))
    
    if comparison.new_violations:
        print(f"\n⚠ NEW VIOLATIONS: {', '.join(comparison.new_violations)}")
    if comparison.resolved_violations:
        print(f"\n✓ RESOLVED: {', '.join(comparison.resolved_violations)}")
    
    # Generate plots if requested
    if args.plot:
        if not MATPLOTLIB_AVAILABLE:
            print("\nPlotting requires matplotlib. Install with: pip install matplotlib")
        else:
            print("\n" + "=" * 70)
            print("GENERATING PLOTS")
            print("=" * 70)
            
            visualizer = TelemetryVisualizer()
            thresholds_by_channel = build_thresholds_by_channel(thresholds)
            
            output_dir = args.output_dir or "plots"
            generated = visualizer.create_report(
                comparison,
                thresholds_by_channel,
                output_dir=output_dir,
                prefix=args.batch_id.replace("/", "_")
            )
            
            print(f"Generated {len(generated)} plot files in '{output_dir}/':")
            for path in generated:
                print(f"  - {path}")
    
    # AI analysis if requested
    if args.ai:
        print("\n" + "=" * 70)
        print("AI TELEMETRY ANALYSIS")
        print("=" * 70)
        
        ai = AIAnalyzer(base_url=args.ollama_url, model=args.model, api_key=args.api_key)
        connected, msg = ai.check_connection()
        
        if not connected:
            print(f"Warning: {msg}")
        else:
            telemetry_dict = analyzer.to_dict(comparison)
            ai_response = ai.analyze_telemetry(telemetry_dict)
            print(ai_response)
    
    db.close()
    return 0


def cmd_plot(args):
    """Generate telemetry visualization plots."""
    if not PARQUET_AVAILABLE:
        print("Plotting requires pyarrow. Install with: pip install pyarrow")
        return 1
    
    if not MATPLOTLIB_AVAILABLE:
        print("Plotting requires matplotlib. Install with: pip install matplotlib")
        return 1
    
    db = RaceSimDB(args.database)
    
    # Get candidate batch runs
    candidate_run_ids = db.get_batch_run_ids(args.batch_id)
    if not candidate_run_ids:
        print(f"Error: Batch '{args.batch_id}' not found.")
        db.close()
        return 1
    
    # Get scenario and baseline
    scenario_id = db.get_scenario_id_for_batch(args.batch_id)
    baseline_run_ids = db.get_baseline_runs_for_scenario(scenario_id)
    
    if not baseline_run_ids:
        print(f"Error: No baseline runs found for scenario {scenario_id}.")
        db.close()
        return 1
    
    candidate_run_ids = [r for r in candidate_run_ids if r not in baseline_run_ids]
    
    # Get telemetry paths
    baseline_paths = db.get_telemetry_paths_for_runs(baseline_run_ids)
    candidate_paths = db.get_telemetry_paths_for_runs(candidate_run_ids)
    
    if not baseline_paths:
        print("Error: No telemetry data for baseline runs.")
        db.close()
        return 1
    
    if not candidate_paths:
        print("Error: No telemetry data for candidate runs.")
        db.close()
        return 1
    
    store = TelemetryStore(args.telemetry_dir)
    thresholds = db.get_thresholds()
    analyzer = TelemetryAnalyzer(store, thresholds)
    
    print(f"Loading telemetry for {len(baseline_paths)} baseline and {len(candidate_paths)} candidate runs...")
    
    comparison = analyzer.compare_telemetry(
        baseline_paths, candidate_paths,
        "baseline", args.batch_id
    )
    
    visualizer = TelemetryVisualizer()
    thresholds_by_channel = build_thresholds_by_channel(thresholds)
    
    output_dir = args.output_dir or "plots"
    
    if args.channel:
        # Single channel plot
        if args.channel not in comparison.channel_comparisons:
            print(f"Error: Channel '{args.channel}' not found.")
            print(f"Available channels: {', '.join(comparison.channel_comparisons.keys())}")
            db.close()
            return 1
        
        comp = comparison.channel_comparisons[args.channel]
        channel_thresholds = thresholds_by_channel.get(args.channel)
        
        output_path = Path(output_dir) / f"{args.batch_id.replace('/', '_')}_{args.channel}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        visualizer.plot_single_channel(
            comp.baseline_stats,
            comp.candidate_stats,
            args.channel,
            channel_thresholds,
            save_path=str(output_path)
        )
        print(f"Saved: {output_path}")
    
    elif args.dashboard:
        # Dashboard view
        output_path = Path(output_dir) / f"{args.batch_id.replace('/', '_')}_dashboard.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        visualizer.plot_dashboard(
            comparison,
            thresholds_by_channel=thresholds_by_channel,
            title=f"Telemetry: {args.batch_id}",
            save_path=str(output_path)
        )
        print(f"Saved: {output_path}")
    
    elif args.delta:
        # Delta analysis plot
        output_path = Path(output_dir) / f"{args.batch_id.replace('/', '_')}_delta.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        visualizer.plot_delta(comparison, save_path=str(output_path))
        print(f"Saved: {output_path}")
    
    elif args.violations:
        # Violations plot
        baseline_stats = {
            name: comp.baseline_stats
            for name, comp in comparison.channel_comparisons.items()
        }
        candidate_stats = {
            name: comp.candidate_stats
            for name, comp in comparison.channel_comparisons.items()
        }
        
        output_path = Path(output_dir) / f"{args.batch_id.replace('/', '_')}_violations.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        visualizer.plot_threshold_violations(
            baseline_stats,
            candidate_stats,
            comparison.baseline_threshold_analysis,
            comparison.candidate_threshold_analysis,
            thresholds_by_channel,
            save_path=str(output_path)
        )
        print(f"Saved: {output_path}")
    
    else:
        # Full report (default)
        generated = visualizer.create_report(
            comparison,
            thresholds_by_channel,
            output_dir=output_dir,
            prefix=args.batch_id.replace("/", "_")
        )
        
        print(f"Generated {len(generated)} plot files in '{output_dir}/':")
        for path in generated:
            print(f"  - {path}")
    
    db.close()
    return 0


def cmd_report(args):
    """Generate comprehensive PDF report."""
    if not REPORTLAB_AVAILABLE:
        print("PDF report generation requires reportlab. Install with: pip install reportlab")
        return 1
    
    db = RaceSimDB(args.database)
    
    # Get candidate batch runs
    candidate_run_ids = db.get_batch_run_ids(args.batch_id)
    if not candidate_run_ids:
        print(f"Error: Batch '{args.batch_id}' not found.")
        db.close()
        return 1
    
    # Get scenario for this batch
    scenario_id = db.get_scenario_id_for_batch(args.batch_id)
    if not scenario_id:
        print(f"Error: Could not determine scenario for batch '{args.batch_id}'.")
        db.close()
        return 1
    
    # Get baseline runs for this scenario
    baseline_run_ids = db.get_baseline_runs_for_scenario(scenario_id)
    if not baseline_run_ids:
        print(f"Error: No baseline runs found for scenario {scenario_id}.")
        db.close()
        return 1
    
    # Filter out any baseline runs from candidate
    candidate_run_ids = [r for r in candidate_run_ids if r not in baseline_run_ids]
    
    if not candidate_run_ids:
        print(f"Error: Batch '{args.batch_id}' contains only baseline runs.")
        db.close()
        return 1
    
    print(f"Analyzing batch: {args.batch_id}")
    print(f"  Baseline runs: {len(baseline_run_ids)}")
    print(f"  Candidate runs: {len(candidate_run_ids)}")
    
    # Load metrics and run analysis
    baseline_metrics = db.get_all_metrics_for_runs(baseline_run_ids)
    candidate_metrics = db.get_all_metrics_for_runs(candidate_run_ids)
    
    req_path = args.requirements or "requirements.yaml"
    requirements = RequirementsLoader(req_path)
    analyzer = Analyzer(requirements)
    
    result = analyzer.compare(baseline_metrics, candidate_metrics, args.batch_id)
    print(f"  Overall score: {result.overall_score:.1%} ({result.status})")
    
    # Telemetry analysis
    telemetry_data = None
    tel_comparison = None
    if PARQUET_AVAILABLE:
        baseline_paths = db.get_telemetry_paths_for_runs(baseline_run_ids)
        candidate_paths = db.get_telemetry_paths_for_runs(candidate_run_ids)
        
        if baseline_paths and candidate_paths:
            print("  Loading telemetry data...")
            store = TelemetryStore(args.telemetry_dir)
            thresholds = db.get_thresholds()
            tel_analyzer = TelemetryAnalyzer(store, thresholds)
            
            tel_comparison = tel_analyzer.compare_telemetry(
                baseline_paths, candidate_paths,
                "baseline", args.batch_id
            )
            telemetry_data = tel_analyzer.to_dict(tel_comparison)
            print(f"  Telemetry: {len(telemetry_data.get('new_violations', []))} new violations")
    
    # Generate plots
    plot_paths = []
    if MATPLOTLIB_AVAILABLE and PARQUET_AVAILABLE and tel_comparison:
        print("  Generating plots...")
        
        plot_dir = Path(args.output_dir or ".") / "report_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer = TelemetryVisualizer()
        thresholds = db.get_thresholds()
        thresholds_by_channel = build_thresholds_by_channel(thresholds)
        
        plot_paths = visualizer.create_report(
            tel_comparison,
            thresholds_by_channel,
            output_dir=str(plot_dir),
            prefix=args.batch_id.replace("/", "_")
        )
        print(f"  Generated {len(plot_paths)} plots")
        
        # Close matplotlib figures
        import matplotlib.pyplot as plt
        plt.close('all')
    
    # AI analysis
    ai_analysis = None
    if args.ai:
        print("  Running AI analysis...")
        ai = AIAnalyzer(base_url=args.ollama_url, model=args.model, api_key=args.api_key)
        connected, msg = ai.check_connection()
        
        if not connected:
            print(f"  Warning: {msg}")
        else:
            analysis_dict = analyzer.to_dict(result)
            if telemetry_data:
                ai_analysis = ai.analyze_combined(analysis_dict, telemetry_data)
            else:
                ai_analysis = ai.analyze(analysis_dict)
            print("  AI analysis complete")
    
    # Generate PDF
    print("  Generating PDF report...")
    
    output_path = args.output
    if not output_path:
        output_dir = args.output_dir or "."
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = str(Path(output_dir) / f"{args.batch_id.replace('/', '_')}_report.pdf")
    
    config = ReportConfig(
        title=args.title or "RaceSim Analysis Report",
        subtitle=f"Batch: {args.batch_id}",
        include_plots=bool(plot_paths),
        include_ai_analysis=bool(ai_analysis),
    )
    
    generator = PDFReportGenerator(config)
    generator.generate(
        output_path,
        result,
        telemetry_data,
        plot_paths,
        ai_analysis,
    )
    
    print(f"\n✓ Report generated: {output_path}")
    
    db.close()
    return 0


def cmd_compare(args):
    """Compare multiple batches."""
    db = RaceSimDB(args.database)
    
    req_path = args.requirements or "requirements.yaml"
    requirements = RequirementsLoader(req_path)
    analyzer = Analyzer(requirements)
    
    batch_analyses = []
    
    for batch_id in args.batch_ids:
        candidate_run_ids = db.get_batch_run_ids(batch_id)
        if not candidate_run_ids:
            print(f"Warning: Batch '{batch_id}' not found, skipping.")
            continue
        
        scenario_id = db.get_scenario_id_for_batch(batch_id)
        baseline_run_ids = db.get_baseline_runs_for_scenario(scenario_id)
        
        if not baseline_run_ids:
            print(f"Warning: No baseline for batch '{batch_id}', skipping.")
            continue
        
        candidate_run_ids = [r for r in candidate_run_ids if r not in baseline_run_ids]
        
        baseline_metrics = db.get_all_metrics_for_runs(baseline_run_ids)
        candidate_metrics = db.get_all_metrics_for_runs(candidate_run_ids)
        
        result = analyzer.compare(baseline_metrics, candidate_metrics, batch_id)
        batch_analyses.append(analyzer.to_dict(result))
    
    if not batch_analyses:
        print("Error: No valid batches to compare.")
        db.close()
        return 1
    
    # Print summary table
    print(f"{'Batch ID':<30} {'Score':>10} {'Status':<12} {'Violations':<10}")
    print("-" * 65)
    for ba in batch_analyses:
        print(f"{ba['candidate_batch_id']:<30} {ba['overall_score_percent']:>10} "
              f"{ba['status']:<12} {len(ba['requirement_violations']):<10}")
    
    # AI comparison if requested
    if args.ai and len(batch_analyses) > 1:
        print("\n" + "=" * 70)
        print("AI COMPARISON")
        print("=" * 70)
        
        ai = AIAnalyzer(
            base_url=args.ollama_url,
            model=args.model,
            api_key=args.api_key,
        )

        connected, msg = ai.check_connection()
        if not connected:
            print(f"Warning: {msg}")
        else:
            comparison = ai.compare_multiple_batches(batch_analyses)
            print(comparison)
    
    db.close()
    return 0


def cmd_check_ai(args):
    """Check AI/LLM connection."""
    ai = AIAnalyzer(
        base_url=args.ollama_url,
        model=args.model,
        api_key=args.api_key,
    )
    connected, msg = ai.check_connection()
    print(msg)
    return 0 if connected else 1


def main():
    parser = argparse.ArgumentParser(
        description="RaceSim Analyzer - QA tool for racing simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--database", "-d",
        default="racesim.db",
        help="Path to SQLite database (default: racesim.db)"
    )
    parser.add_argument(
        "--requirements", "-r",
        default=None,
        help="Path to requirements YAML (default: requirements.yaml)"
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        help="LLM API base URL (default: $LLM_BASE_URL or http://localhost:11434/v1)"
    )
    parser.add_argument(
        "--model", "-m",
        default=os.environ.get("LLM_MODEL", "llama3.2"),
        help="LLM model name (default: $LLM_MODEL or llama3.2)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key for authenticated LLM providers (default: $OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--telemetry-dir",
        default="telemetry",
        help="Directory for telemetry Parquet files (default: telemetry)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # init
    sub_init = subparsers.add_parser("init", help="Initialize database")
    sub_init.set_defaults(func=cmd_init)
    
    # list-batches
    sub_list = subparsers.add_parser("list-batches", help="List all batches")
    sub_list.set_defaults(func=cmd_list_batches)
    
    # analyze
    sub_analyze = subparsers.add_parser("analyze", help="Analyze a batch vs baseline")
    sub_analyze.add_argument("batch_id", help="Batch ID to analyze")
    sub_analyze.add_argument("--ai", action="store_true", help="Include AI analysis")
    sub_analyze.add_argument("--suggest", action="store_true", help="Include AI improvement suggestions")
    sub_analyze.add_argument("--context", "-c", help="Additional context for AI")
    sub_analyze.add_argument("--telemetry", "-t", action="store_true", help="Include telemetry analysis")
    sub_analyze.set_defaults(func=cmd_analyze)
    
    # telemetry (standalone)
    sub_tel = subparsers.add_parser("telemetry", help="Telemetry-only analysis")
    sub_tel.add_argument("batch_id", help="Batch ID to analyze")
    sub_tel.add_argument("--ai", action="store_true", help="Include AI analysis")
    sub_tel.add_argument("--plot", "-p", action="store_true", help="Generate visualization plots")
    sub_tel.add_argument("--output-dir", "-o", default="plots", help="Output directory for plots")
    sub_tel.set_defaults(func=cmd_telemetry)
    
    # plot (visualization)
    sub_plot = subparsers.add_parser("plot", help="Generate telemetry plots")
    sub_plot.add_argument("batch_id", help="Batch ID to visualize")
    sub_plot.add_argument("--output-dir", "-o", default="plots", help="Output directory")
    sub_plot.add_argument("--channel", "-c", help="Plot single channel")
    sub_plot.add_argument("--dashboard", action="store_true", help="Generate dashboard view")
    sub_plot.add_argument("--delta", action="store_true", help="Generate delta analysis plot")
    sub_plot.add_argument("--violations", action="store_true", help="Generate violations plot")
    sub_plot.set_defaults(func=cmd_plot)
    
    # report (PDF generation)
    sub_report = subparsers.add_parser("report", help="Generate PDF report")
    sub_report.add_argument("batch_id", help="Batch ID to report on")
    sub_report.add_argument("--output", help="Output PDF path (default: <batch_id>_report.pdf)")
    sub_report.add_argument("--output-dir", "-o", default=".", help="Output directory for PDF and plots")
    sub_report.add_argument("--title", "-t", help="Custom report title")
    sub_report.add_argument("--ai", action="store_true", help="Include AI analysis")
    sub_report.set_defaults(func=cmd_report)
    
    # compare
    sub_compare = subparsers.add_parser("compare", help="Compare multiple batches")
    sub_compare.add_argument("batch_ids", nargs="+", help="Batch IDs to compare")
    sub_compare.add_argument("--ai", action="store_true", help="Include AI comparison")
    sub_compare.set_defaults(func=cmd_compare)
    
    # check-ai
    sub_check = subparsers.add_parser("check-ai", help="Check Ollama connection")
    sub_check.set_defaults(func=cmd_check_ai)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
