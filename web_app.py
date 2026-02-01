"""
FastAPI Web Interface for RaceSim Analyzer.

Run with:
    python web_app.py
    # or
    uvicorn web_app:app --reload --port 8000
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import analyzer modules directly
from db import RaceSimDB
from analysis import RequirementsLoader, Analyzer

# Optional imports
try:
    from telemetry import TelemetryStore, TelemetryAnalyzer
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

try:
    from visualization import TelemetryVisualizer, build_thresholds_by_channel
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from report import PDFReportGenerator, ReportConfig
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False

try:
    from ai_analyzer import AIAnalyzer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

try:
    from mcp_server import (
        list_batches as mcp_list_batches,
        list_cars as mcp_list_cars,
        list_scenarios as mcp_list_scenarios,
        get_batch_details as mcp_get_batch_details,
        get_batch_summary as mcp_get_batch_summary,
        compare_batches as mcp_compare_batches,
        get_run_metrics as mcp_get_run_metrics,
        get_thresholds as mcp_get_thresholds,
        find_batches_for_scenario as mcp_find_batches_for_scenario,
        query_metric_across_batches as mcp_query_metric_across_batches,
        get_versions_for_scenario as mcp_get_versions_for_scenario,
        get_telemetry_summary as mcp_get_telemetry_summary,
        get_telemetry_crossings as mcp_get_telemetry_crossings,
    )
    CHAT_TOOLS_AVAILABLE = True
except ImportError:
    CHAT_TOOLS_AVAILABLE = False


# Configuration
DATABASE_PATH = "racesim.db"
REQUIREMENTS_PATH = "requirements.yaml"
TELEMETRY_DIR = "telemetry"
OUTPUT_DIR = "web_output"
OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.1:8b-instruct-q8_0" # "llama3.2" # "gpt-oss:20b"
AI_CACHE_DIR = f"{OUTPUT_DIR}/ai_cache"

CHAT_SYSTEM_PROMPT = """You are RaceSim Assistant, an expert racing simulation QA engineer.
You help users understand simulation results, compare batches, and interpret telemetry data.

You have access to a racing simulation QA system that:
- Runs Monte Carlo simulations of car configurations
- Compares candidate configurations against baselines
- Tracks metrics: lap_time, fuel_consumption, max_speed, avg_speed, tire_degradation, brake_temp_max, cornering_g_max, energy_recovered
- Analyzes telemetry data with threshold crossings for speed, tire wear, brake temps, G-forces, fuel

You have tools to query the database directly. Use them to look up real data \
before answering questions about batches, runs, metrics, or comparisons. \
Do not guess values — call a tool to get the actual numbers.

Available tools (use ONLY these — do not invent tool names):
- list_batches(): List all simulation batches
- list_cars(): List all car versions with configs and associated batches
- list_scenarios(): List experiment scenarios with parameters
- get_versions_for_scenario(scenario_id): Get car versions for a specific scenario
- get_batch_details(batch_id): Get all runs and their metrics for a batch
- get_batch_summary(batch_id): Get aggregate stats (mean, std, min, max) per metric
- compare_batches(candidate_batch_id, baseline_batch_id): Compare two batches
- get_run_metrics(run_id): Get all metrics for a specific run
- get_thresholds(channel?): Get threshold definitions, optionally by channel
- find_batches_for_scenario(scenario_name): Find batches by scenario name
- query_metric_across_batches(metric_name): Query a metric across all batches
- get_telemetry_summary(batch_id): Get per-channel telemetry stats (speed, tire wear, brake temps, etc.)
- get_telemetry_crossings(batch_id): Get threshold violations for a batch's telemetry

Use racing domain knowledge to explain results:
- Lower lap times with higher fuel consumption might indicate more aggressive engine mapping
- Higher tire degradation with better cornering G might indicate softer compound or higher downforce
- Temperature increases might indicate brake or cooling issues

Be concise, technical, and helpful. Format using markdown for readability.
When given batch data, analyze it and provide actionable insights.

{context}
"""


# =============================================================================
# Chat Tool Definitions (OpenAI function-calling format)
# =============================================================================

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_batches",
            "description": "List all simulation batches with run counts and type (baseline/candidate).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_cars",
            "description": "List all car versions with their software/hardware configs and associated batches.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_scenarios",
            "description": "List all experiment scenarios with their parameters.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_versions_for_scenario",
            "description": "Get car version details for all versions that have runs in a given scenario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "integer", "description": "The scenario ID"},
                },
                "required": ["scenario_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_batch_details",
            "description": "Get all runs and their metrics for a given batch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "Batch identifier (e.g. 'candidate-dry-v1.1')"},
                },
                "required": ["batch_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_batch_summary",
            "description": "Get aggregate statistics (mean, std, min, max) for each metric in a batch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "Batch identifier"},
                },
                "required": ["batch_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_batches",
            "description": "Compare metric means between candidate and baseline batches. Shows delta and percentage change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_batch_id": {"type": "string", "description": "Candidate batch to evaluate"},
                    "baseline_batch_id": {"type": "string", "description": "Baseline batch to compare against"},
                },
                "required": ["candidate_batch_id", "baseline_batch_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_run_metrics",
            "description": "Get all metrics for a specific run by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer", "description": "The run ID"},
                },
                "required": ["run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_thresholds",
            "description": "Get threshold definitions, optionally filtered by telemetry channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Channel name to filter (e.g. 'speed', 'tire_wear_fl'). Leave empty for all."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_batches_for_scenario",
            "description": "Find all batches that ran under a given scenario name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario_name": {"type": "string", "description": "Scenario name to search (e.g. 'Monza Dry')"},
                },
                "required": ["scenario_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_metric_across_batches",
            "description": "Query a single metric across all batches, showing per-batch averages. Useful for tracking metric evolution across versions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {"type": "string", "description": "Metric name (e.g. 'lap_time', 'fuel_consumption', 'tire_degradation')"},
                },
                "required": ["metric_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_telemetry_summary",
            "description": "Get per-channel telemetry statistics (mean, min, max, std) for a batch. Covers speed, tire wear, brake temps, throttle, g-forces, fuel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "Batch identifier"},
                },
                "required": ["batch_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_telemetry_crossings",
            "description": "Get threshold crossing (violation) summary for a batch's telemetry. Shows which channels exceed limits, severity, positions, and durations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "Batch identifier"},
                },
                "required": ["batch_id"],
            },
        },
    },
]

# Dispatcher: maps tool name -> callable
_TOOL_DISPATCH: dict = {}
if CHAT_TOOLS_AVAILABLE:
    _TOOL_DISPATCH = {
        "list_batches": lambda: mcp_list_batches(),
        "list_cars": lambda: mcp_list_cars(),
        "list_scenarios": lambda: mcp_list_scenarios(),
        "get_batch_details": lambda **kw: mcp_get_batch_details(**kw),
        "get_batch_summary": lambda **kw: mcp_get_batch_summary(**kw),
        "compare_batches": lambda **kw: mcp_compare_batches(**kw),
        "get_run_metrics": lambda **kw: mcp_get_run_metrics(**kw),
        "get_thresholds": lambda **kw: mcp_get_thresholds(**kw),
        "find_batches_for_scenario": lambda **kw: mcp_find_batches_for_scenario(**kw),
        "query_metric_across_batches": lambda **kw: mcp_query_metric_across_batches(**kw),
        "get_versions_for_scenario": lambda **kw: mcp_get_versions_for_scenario(**kw),
        "get_telemetry_summary": lambda **kw: mcp_get_telemetry_summary(**kw),
        "get_telemetry_crossings": lambda **kw: mcp_get_telemetry_crossings(**kw),
    }


def execute_chat_tool(name: str, arguments: dict) -> str:
    """Execute a chat tool by name. Returns the tool's string result."""
    fn = _TOOL_DISPATCH.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    if arguments:
        return fn(**arguments)
    return fn()


class ChatRequest(BaseModel):
    messages: list[dict]
    batch_id: Optional[str] = None
    page_context: Optional[str] = None

# Ensure output directories exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/plots").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/reports").mkdir(parents=True, exist_ok=True)
Path(AI_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Create FastAPI app
app = FastAPI(title="RaceSim Analyzer", version="1.0.0")

# Setup templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")


# =============================================================================
# Core Functions (same logic as main.py but callable directly)
# =============================================================================

def get_db():
    return RaceSimDB(DATABASE_PATH)


def get_requirements():
    return RequirementsLoader(REQUIREMENTS_PATH)


def run_analysis(batch_id: str, include_telemetry: bool = True) -> dict:
    """
    Run complete analysis on a batch.
    Same logic as cmd_analyze in main.py but returns data directly.
    """
    db = get_db()

    try:
        # Get candidate runs
        candidate_run_ids = db.get_batch_run_ids(batch_id)
        if not candidate_run_ids:
            raise HTTPException(404, f"Batch '{batch_id}' not found")

        # Get scenario and baseline
        scenario_id = db.get_scenario_id_for_batch(batch_id)
        if not scenario_id:
            raise HTTPException(400, "Could not determine scenario")

        baseline_run_ids = db.get_baseline_runs_for_scenario(scenario_id)
        if not baseline_run_ids:
            raise HTTPException(400, "No baseline runs found")

        # Filter candidates
        candidate_run_ids = [r for r in candidate_run_ids if r not in baseline_run_ids]
        if not candidate_run_ids:
            raise HTTPException(400, "Batch contains only baseline runs")

        # Load metrics
        baseline_metrics = db.get_all_metrics_for_runs(baseline_run_ids)
        candidate_metrics = db.get_all_metrics_for_runs(candidate_run_ids)

        # Run analysis
        requirements = get_requirements()
        analyzer = Analyzer(requirements)
        result = analyzer.compare(baseline_metrics, candidate_metrics, batch_id)
        result_dict = analyzer.to_dict(result)

        # Telemetry analysis
        telemetry_data = None
        tel_comparison = None

        if include_telemetry and TELEMETRY_AVAILABLE:
            baseline_paths = db.get_telemetry_paths_for_runs(baseline_run_ids)
            candidate_paths = db.get_telemetry_paths_for_runs(candidate_run_ids)

            if baseline_paths and candidate_paths:
                store = TelemetryStore(TELEMETRY_DIR)
                thresholds = db.get_thresholds()
                tel_analyzer = TelemetryAnalyzer(store, thresholds)

                tel_comparison = tel_analyzer.compare_telemetry(
                    baseline_paths, candidate_paths, "baseline", batch_id
                )
                telemetry_data = tel_analyzer.to_dict(tel_comparison)

        return {
            "result": result,
            "result_dict": result_dict,
            "telemetry_data": telemetry_data,
            "tel_comparison": tel_comparison,
        }
    finally:
        db.close()


def run_comparison(candidate_batch_id: str, baseline_batch_id: str) -> dict:
    """
    Run comparison between two specific batches.
    """
    db = get_db()

    try:
        # Get run IDs
        candidate_run_ids = db.get_batch_run_ids(candidate_batch_id)
        if not candidate_run_ids:
            raise HTTPException(404, f"Candidate batch '{candidate_batch_id}' not found")

        baseline_run_ids = db.get_batch_run_ids(baseline_batch_id)
        if not baseline_run_ids:
            raise HTTPException(404, f"Baseline batch '{baseline_batch_id}' not found")

        # Load metrics
        baseline_metrics = db.get_all_metrics_for_runs(baseline_run_ids)
        candidate_metrics = db.get_all_metrics_for_runs(candidate_run_ids)

        # Run analysis (Candidate vs Baseline)
        requirements = get_requirements()
        analyzer = Analyzer(requirements)
        result = analyzer.compare(baseline_metrics, candidate_metrics, candidate_batch_id)
        result_dict = analyzer.to_dict(result)

        # Telemetry analysis
        telemetry_data = None
        tel_comparison = None

        if TELEMETRY_AVAILABLE:
            baseline_paths = db.get_telemetry_paths_for_runs(baseline_run_ids)
            candidate_paths = db.get_telemetry_paths_for_runs(candidate_run_ids)

            if baseline_paths and candidate_paths:
                store = TelemetryStore(TELEMETRY_DIR)
                thresholds = db.get_thresholds()
                tel_analyzer = TelemetryAnalyzer(store, thresholds)

                # Using baseline batch ID as the "baseline name" in the comparison logic
                tel_comparison = tel_analyzer.compare_telemetry(
                    baseline_paths, candidate_paths, baseline_batch_id, candidate_batch_id
                )
                telemetry_data = tel_analyzer.to_dict(tel_comparison)

        return {
            "result": result,
            "result_dict": result_dict,
            "telemetry_data": telemetry_data,
            "tel_comparison": tel_comparison,
        }
    finally:
        db.close()


def create_plots(batch_id: str, tel_comparison) -> list:
    """Generate plots and return URLs."""
    if not VISUALIZATION_AVAILABLE or not tel_comparison:
        return []

    db = get_db()
    try:
        thresholds = db.get_thresholds()
    finally:
        db.close()

    visualizer = TelemetryVisualizer()
    thresholds_by_channel = build_thresholds_by_channel(thresholds)

    plot_dir = f"{OUTPUT_DIR}/plots"
    prefix = batch_id.replace("/", "_").replace(" ", "_")

    # Clear old plots
    for f in Path(plot_dir).glob(f"{prefix}_*.png"):
        f.unlink()

    # Generate plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    paths = visualizer.create_report(
        tel_comparison, thresholds_by_channel,
        output_dir=plot_dir, prefix=prefix
    )
    plt.close('all')

    return [f"/static/plots/{Path(p).name}" for p in paths]


def create_pdf_report(batch_id: str, result, telemetry_data, plot_urls, ai_text=None) -> str:
    """Generate PDF report and return URL."""
    if not REPORT_AVAILABLE:
        raise HTTPException(500, "reportlab not installed")

    prefix = batch_id.replace("/", "_").replace(" ", "_")
    output_path = f"{OUTPUT_DIR}/reports/{prefix}_report.pdf"

    # Convert plot URLs to file paths
    plot_paths = None
    if plot_urls:
        plot_paths = [f"{OUTPUT_DIR}/plots/{Path(u).name}" for u in plot_urls]

    config = ReportConfig(
        title="RaceSim Analysis Report",
        subtitle=f"Batch: {batch_id}",
    )
    print("Doing PDF report!")
    generator = PDFReportGenerator(config)
    generator.generate(
        output_path,
        result,
        telemetry_data,
        plot_paths,
        ai_text,
        ai_model_name=OLLAMA_MODEL if ai_text else None
    )

    return f"/static/reports/{prefix}_report.pdf"


def run_ai_analysis(result_dict: dict, telemetry_data: dict = None) -> str:
    """Run AI analysis and return text."""
    if not AI_AVAILABLE:
        raise HTTPException(500, "AI not available")

    ai = AIAnalyzer(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
    connected, msg = ai.check_connection()

    if not connected:
        raise HTTPException(503, f"Ollama not available: {msg}")

    if telemetry_data:
        return ai.analyze_combined(result_dict, telemetry_data)
    return ai.analyze(result_dict)


def get_cached_ai_analysis(batch_id: str) -> str:
    """Get cached AI analysis text if available."""
    prefix = batch_id.replace("/", "_").replace(" ", "_")
    cache_path = Path(AI_CACHE_DIR) / f"{prefix}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return None


def save_cached_ai_analysis(batch_id: str, text: str):
    """Save AI analysis text to cache."""
    prefix = batch_id.replace("/", "_").replace(" ", "_")
    cache_path = Path(AI_CACHE_DIR) / f"{prefix}.txt"
    cache_path.write_text(text, encoding="utf-8")


def build_chat_context(batch_id: Optional[str] = None, page_context: Optional[str] = None) -> str:
    """Build context string for chat system prompt."""
    parts = []

    # Always include available batches
    db = get_db()
    try:
        batches = db.list_batches()
        if batches:
            batch_lines = []
            for b in batches:
                btype = "baseline" if b.get("has_baseline") else "candidate"
                batch_lines.append(f"  - {b['batch_id']} ({btype}, {b['run_count']} runs)")
            parts.append("Available batches:\n" + "\n".join(batch_lines))
    finally:
        db.close()

    # If a batch_id is provided, include its analysis data
    if batch_id:
        try:
            data = get_analysis_data(batch_id)
            result_dict = data["result_dict"]
            summary = json.dumps(result_dict, indent=2)
            if len(summary) > 8000:
                summary = summary[:8000] + "\n... (truncated)"
            parts.append(f"Currently viewing batch '{batch_id}'. Analysis data:\n```json\n{summary}\n```")
        except Exception:
            parts.append(f"User is viewing batch '{batch_id}' but analysis data could not be loaded.")

    if page_context:
        page_names = {
            "index": "the home page (batch list)",
            "analyze": f"the analysis page for batch '{batch_id}'",
            "compare": f"the comparison page for '{batch_id}'",
            "report": f"the PDF report page for batch '{batch_id}'",
        }
        parts.append(f"The user is currently on {page_names.get(page_context, page_context)}.")

    return "\n\n".join(parts) if parts else "No specific context available."


def get_analysis_data(batch_id: str) -> dict:
    """Get analysis data for a batch or comparison."""
    if " vs " in batch_id:
        parts = batch_id.split(" vs ")
        if len(parts) == 2:
            try:
                return run_comparison(parts[0], parts[1])
            except HTTPException:
                # If comparison fails (e.g. individual batches don't exist),
                # fall back to treating it as a single batch ID
                pass

    return run_analysis(batch_id, include_telemetry=True)


def generate_full_report(batch_id: str) -> tuple[str, dict]:
    """Helper to run analysis, plots, AI, and generate PDF. Returns (pdf_url, analysis_data)."""
    data = get_analysis_data(batch_id)

    # Generate plots
    plot_urls = []
    if data["tel_comparison"]:
        plot_urls = create_plots(batch_id, data["tel_comparison"])

    # Check for cached AI analysis or generate if missing
    ai_text = get_cached_ai_analysis(batch_id)
    if not ai_text and AI_AVAILABLE:
        try:
            ai_text = run_ai_analysis(data["result_dict"], data["telemetry_data"])
            save_cached_ai_analysis(batch_id, ai_text)
        except Exception as e:
            print(f"Error generating AI analysis for report: {e}")

    # Generate PDF
    pdf_url = create_pdf_report(
        batch_id, data["result"], data["telemetry_data"], plot_urls, ai_text
    )

    return pdf_url, data


# =============================================================================
# Web Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with batch list."""
    db = get_db()
    try:
        batches = db.list_batches()

        # Sort: First by Type (candidate first, baseline last), then by Batch ID
        if batches:
            batches.sort(key=lambda b: (
                1 if b.get("has_baseline") else 0, # Candidate (0) before Baseline (1)
                b.get("batch_id", "")
            ))
    finally:
        db.close()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "batches": batches,
        "batch_id": None,
        "page_context": "index",
        "features": {
            "telemetry": TELEMETRY_AVAILABLE,
            "plots": VISUALIZATION_AVAILABLE,
            "pdf": REPORT_AVAILABLE,
            "ai": AI_AVAILABLE,
        }
    })


@app.get("/analyze/{batch_id:path}", response_class=HTMLResponse)
async def analyze_page(request: Request, batch_id: str):
    """Analyze a batch and show results."""
    data = run_analysis(batch_id, include_telemetry=True)

    # Generate plots
    plot_urls = []
    if data["tel_comparison"]:
        print("Creating plots..")
        plot_urls = create_plots(batch_id, data["tel_comparison"])
        print("Creating plots done.")

    return templates.TemplateResponse("analyze.html", {
        "request": request,
        "batch_id": batch_id,
        "page_context": "analyze",
        "result": data["result"],
        "analysis": data["result_dict"],
        "telemetry": data["telemetry_data"],
        "plot_urls": plot_urls,
        "features": {
            "pdf": REPORT_AVAILABLE,
            "ai": AI_AVAILABLE,
        }
    })


@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request, candidate: str, baseline: str):
    """Compare two batches."""
    data = run_comparison(candidate, baseline)

    # Generate plots
    plot_urls = []
    if data["tel_comparison"]:
        combined_id = f"{candidate}_vs_{baseline}"
        plot_urls = create_plots(combined_id, data["tel_comparison"])

    return templates.TemplateResponse("analyze.html", {
        "request": request,
        "batch_id": f"{candidate} vs {baseline}",
        "page_context": "compare",
        "result": data["result"],
        "analysis": data["result_dict"],
        "telemetry": data["telemetry_data"],
        "plot_urls": plot_urls,
        "features": {
            "pdf": REPORT_AVAILABLE,
            "ai": AI_AVAILABLE,
        }
    })


@app.post("/api/report/{batch_id:path}")
async def api_generate_report(batch_id: str, include_ai: bool = Form(False)):
    """Generate PDF report via API."""
    pdf_url, _ = generate_full_report(batch_id)
    return JSONResponse({"pdf_url": pdf_url})


@app.get("/api/ai_analysis/{batch_id:path}")
async def get_ai_analysis(batch_id: str):
    """Get AI analysis for a batch."""
    if not AI_AVAILABLE:
        raise HTTPException(500, "AI not available")

    # Check cache first
    cached = get_cached_ai_analysis(batch_id)
    if cached:
        return JSONResponse({"analysis": cached})

    data = get_analysis_data(batch_id)

    try:
        ai_text = run_ai_analysis(data["result_dict"], data["telemetry_data"])
        save_cached_ai_analysis(batch_id, ai_text)
        return JSONResponse({"analysis": ai_text})
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/report/{batch_id:path}", response_class=HTMLResponse)
async def report_page(request: Request, batch_id: str, ai: bool = False):
    """Generate and display PDF report."""
    pdf_url, data = generate_full_report(batch_id)

    return templates.TemplateResponse("report.html", {
        "request": request,
        "batch_id": batch_id,
        "page_context": "report",
        "pdf_url": pdf_url,
        "result": data["result"],
    })


@app.post("/api/ai/{batch_id:path}")
async def api_ai_analysis(batch_id: str):
    """Get AI analysis via API."""
    data = run_analysis(batch_id, include_telemetry=True)
    ai_text = run_ai_analysis(data["result_dict"], data["telemetry_data"])
    return JSONResponse({"analysis": ai_text})


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Chat with AI assistant about RaceSim data (with tool calling)."""
    if not AI_AVAILABLE:
        raise HTTPException(500, "AI not available - Ollama module not loaded")

    # Build context-aware system prompt
    context = build_chat_context(req.batch_id, req.page_context)
    system_prompt = CHAT_SYSTEM_PROMPT.format(context=context)

    # Build messages: system prompt + conversation history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(req.messages)

    ai = AIAnalyzer(base_url=OLLAMA_URL, model=OLLAMA_MODEL)

    try:
        tools = CHAT_TOOLS if CHAT_TOOLS_AVAILABLE else None
        executor = execute_chat_tool if CHAT_TOOLS_AVAILABLE else None

        response_text = ai.chat_with_tools(
            messages, tools=tools, tool_executor=executor,
        )

        if response_text.startswith("ERROR:"):
            raise HTTPException(503, response_text)

        return JSONResponse({
            "role": "assistant",
            "content": response_text
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Chat error: {str(e)}")


@app.get("/api/batches")
async def api_batches():
    """List all batches via API."""
    db = get_db()
    try:
        batches = db.list_batches()
    finally:
        db.close()
    return {"batches": batches}


@app.get("/api/analyze/{batch_id:path}")
async def api_analyze(batch_id: str, telemetry: bool = True):
    """Analyze batch via API."""
    data = run_analysis(batch_id, include_telemetry=telemetry)
    return {
        "analysis": data["result_dict"],
        "telemetry": data["telemetry_data"],
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "telemetry": TELEMETRY_AVAILABLE,
            "plots": VISUALIZATION_AVAILABLE,
            "pdf": REPORT_AVAILABLE,
            "ai": AI_AVAILABLE,
        }
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("RaceSim Analyzer Web Interface")
    print("=" * 60)
    print(f"Open http://localhost:8000 in your browser")
    print(f"Features: telemetry={TELEMETRY_AVAILABLE}, plots={VISUALIZATION_AVAILABLE}, pdf={REPORT_AVAILABLE}, ai={AI_AVAILABLE}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
