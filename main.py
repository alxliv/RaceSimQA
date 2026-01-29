#!/usr/bin/env python3
"""
RaceSim Analyzer - CLI for analyzing racing simulation results.

Usage:
    python main.py init                      # Initialize database
    python main.py list-batches              # List all batches
    python main.py analyze <batch_id>        # Analyze a batch vs baseline
    python main.py analyze <batch_id> --ai   # Include AI insights
    python main.py compare <b1> <b2> ...     # Compare multiple batches
"""

import argparse
import sys
from pathlib import Path

from db import RaceSimDB
from analysis import RequirementsLoader, Analyzer
from ai_analyzer import AIAnalyzer


def cmd_init(args):
    """Initialize the database."""
    db = RaceSimDB(args.database)
    db.init_schema()
    print(f"Database initialized: {args.database}")
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
    
    # AI analysis if requested
    if args.ai:
        print("\n" + "=" * 70)
        print("AI ANALYSIS")
        print("=" * 70)
        
        ai = AIAnalyzer(
            base_url=args.ollama_url,
            model=args.model
        )
        
        # Check connection
        connected, msg = ai.check_connection()
        if not connected:
            print(f"Warning: {msg}")
            print("Skipping AI analysis.")
        else:
            analysis_dict = analyzer.to_dict(result)
            
            # Add context if provided
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
            model=args.model
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
    """Check AI/Ollama connection."""
    ai = AIAnalyzer(
        base_url=args.ollama_url,
        model=args.model
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
        default="http://localhost:11434/v1",
        help="Ollama API URL (default: http://localhost:11434/v1)"
    )
    parser.add_argument(
        "--model", "-m",
        default="llama3.2",
        help="Ollama model to use (default: llama3.2)"
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
    sub_analyze.set_defaults(func=cmd_analyze)
    
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
