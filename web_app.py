"""
FastAPI Web Interface for RaceSim Analyzer.

Run with:
    python web_app.py
    # or
    uvicorn web_app:app --reload --port 8000
"""

import os
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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


# Configuration
DATABASE_PATH = "racesim.db"
REQUIREMENTS_PATH = "requirements.yaml"
TELEMETRY_DIR = "telemetry"
OUTPUT_DIR = "web_output"
OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.1:8b-instruct-q8_0" # "llama3.2" # "gpt-oss:20b"

# Ensure output directories exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/plots").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/reports").mkdir(parents=True, exist_ok=True)

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


# =============================================================================
# Web Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with batch list."""
    db = get_db()
    try:
        batches = db.list_batches()
    finally:
        db.close()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "batches": batches,
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
    data = run_analysis(batch_id, include_telemetry=True)

    # Generate plots
    plot_urls = []
    if data["tel_comparison"]:
        plot_urls = create_plots(batch_id, data["tel_comparison"])

    # AI analysis
    ai_text = None
    if include_ai and AI_AVAILABLE:
        try:
            ai_text = run_ai_analysis(data["result_dict"], data["telemetry_data"])
        except:
            pass

    # Generate PDF
    pdf_url = create_pdf_report(
        batch_id, data["result"], data["telemetry_data"], plot_urls, ai_text
    )

    return JSONResponse({"pdf_url": pdf_url})


@app.get("/api/ai_analysis/{batch_id:path}")
async def get_ai_analysis(batch_id: str):
    """Get AI analysis for a batch."""
    if not AI_AVAILABLE:
        raise HTTPException(500, "AI not available")

    data = run_analysis(batch_id, include_telemetry=True)

    try:
        ai_text = run_ai_analysis(data["result_dict"], data["telemetry_data"])
        return JSONResponse({"analysis": ai_text})
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/report/{batch_id:path}", response_class=HTMLResponse)
async def report_page(request: Request, batch_id: str, ai: bool = False):
    """Generate and display PDF report."""
    data = run_analysis(batch_id, include_telemetry=True)

    # Generate plots
    plot_urls = []
    if data["tel_comparison"]:
        plot_urls = create_plots(batch_id, data["tel_comparison"])

    # AI analysis
    ai_text = None
    if ai and AI_AVAILABLE:
        try:
            ai_text = run_ai_analysis(data["result_dict"], data["telemetry_data"])
        except:
            pass

    # Generate PDF
    pdf_url = create_pdf_report(
        batch_id, data["result"], data["telemetry_data"], plot_urls, ai_text
    )

    return templates.TemplateResponse("report.html", {
        "request": request,
        "batch_id": batch_id,
        "pdf_url": pdf_url,
        "result": data["result"],
    })


@app.post("/api/ai/{batch_id:path}")
async def api_ai_analysis(batch_id: str):
    """Get AI analysis via API."""
    data = run_analysis(batch_id, include_telemetry=True)
    ai_text = run_ai_analysis(data["result_dict"], data["telemetry_data"])
    return JSONResponse({"analysis": ai_text})


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
