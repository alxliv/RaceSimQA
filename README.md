# RaceSim Analyzer

A QA regression testing tool for racing car simulations. Compares Monte Carlo simulation results between car versions and provides AI-powered insights.

## Features

- **Database-backed storage**: SQLite schema for cars, versions, experiments, scenarios, runs, and metrics
- **Statistical comparison**: Compares candidate runs against baseline with mean, std, delta calculations
- **Requirements validation**: YAML-based requirements with pass/fail thresholds
- **Scoring system**: Weighted scoring with configurable thresholds (excellent/good/acceptable/poor/fail)
- **Time-series telemetry**: Position-based telemetry analysis stored in Parquet files
- **Threshold crossing detection**: Configurable alerts for tire wear, brake temps, G-forces, etc.
- **AI analysis**: LLM integration via Ollama (local) or OpenAI API (cloud, GPT-4o) for natural language insights
- **LLM chat with tool calling**: Interactive chat in the web UI where the LLM can query the database via tools
- **MCP server**: Model Context Protocol server exposing database tools for external LLM clients

## Project Structure

```
racesim_analyzer/
├── schema.sql          # Database schema
├── db.py               # Database helpers and data classes
├── requirements.yaml   # Performance requirements + telemetry thresholds
├── analysis.py         # Statistical comparison and scoring engine
├── telemetry.py        # Time-series data handling and threshold detection
├── visualization.py    # Matplotlib plotting for telemetry curves
├── report.py           # PDF report generation with reportlab
├── ai_analyzer.py      # LLM integration: Ollama or OpenAI (chat with tool calling)
├── mcp_server.py       # MCP server exposing DB tools for LLM clients
├── main.py             # CLI entry point
├── web_app.py          # FastAPI web interface (includes chat endpoint)
├── templates/          # HTML templates for web UI
│   ├── index.html      # Home page with batch list
│   ├── analyze.html    # Analysis results page
│   ├── report.html     # PDF viewer page
│   └── chat_panel.html # Floating chat panel component
├── seed_sample_data.py # Sample data generator for testing
├── python_dependencies.txt # Python dependencies (I used python 3.14)
├── telemetry/          # Parquet files (generated)
├── plots/              # CLI visualization output (generated)
└── web_output/         # Web interface output (generated)
```

## Requirements

- Python 3.10+ (I used python 3.14)
- Dependencies listed in `python_dependencies.txt`
- Ollama (optional, for local AI features) **or** an OpenAI API key (optional, for cloud AI features)

```bash
pip install -r python_dependencies.txt
```

Or install individually:
```bash
pip install pyyaml requests numpy pyarrow matplotlib reportlab mcp python-dotenv
```

## Quick Start

1. **Initialize and seed sample data**:
   ```bash
   python seed_sample_data.py
   ```

2. **List available batches**:
   ```bash
   python main.py list-batches
   ```

3. **Analyze a candidate batch (summary metrics)**:
   ```bash
   python main.py analyze candidate-dry-v1.1
   ```

4. **Analyze with telemetry (threshold crossings)**:
   ```bash
   python main.py analyze candidate-dry-v1.1 --telemetry
   ```

5. **Generate visualization plots**:
   ```bash
   python main.py plot candidate-dry-v1.1              # Full report (all plots)
   python main.py plot candidate-dry-v1.1 --dashboard  # Dashboard only
   python main.py plot candidate-dry-v1.1 --delta      # Delta analysis
   python main.py plot candidate-dry-v1.1 --violations # Threshold violations
   python main.py plot candidate-dry-v1.1 -c speed     # Single channel
   ```

6. **Telemetry-only analysis with plots**:
   ```bash
   python main.py telemetry candidate-dry-v1.1 --plot
   python main.py telemetry experimental-dry-v1.2 --plot  # Shows more violations
   ```

7. **With AI analysis** (requires Ollama **or** OpenAI API key):
   ```bash
   # Option A: Local LLM via Ollama
   ollama serve  # In another terminal
   ollama pull llama3.2
   python main.py analyze candidate-dry-v1.1 --telemetry --ai

   # Option B: OpenAI API (GPT-4o) — set key in .env file or environment
   echo OPENAI_API_KEY=sk-... > .env
   python main.py analyze candidate-dry-v1.1 --telemetry --ai \
     --ollama-url https://api.openai.com/v1 --model gpt-4o
   ```

8. **Compare multiple batches**:
   ```bash
   python main.py compare candidate-dry-v1.1 experimental-dry-v1.2 --ai
   ```

## Database Schema

```
cars ─────────────────┐
                      │
car_versions ─────────┼──► runs ──► run_metrics
                      │      │
experiments ──► scenarios ───┘
                             │
                      run_telemetry ──► Parquet files
```

## Telemetry Channels

Position-based telemetry (10 Hz, ~600 samples/lap):

| Channel | Unit | Description |
|---------|------|-------------|
| position_m | m | Track position from start |
| speed | m/s | Vehicle speed |
| tire_wear_fl/fr/rl/rr | % | Tire wear per wheel |
| brake_temp_fl/fr/rl/rr | °C | Brake temperature per wheel |
| throttle | 0-1 | Throttle position |
| brake | 0-1 | Brake position |
| g_lateral | g | Lateral acceleration |
| g_longitudinal | g | Longitudinal acceleration |
| fuel_remaining | kg | Fuel remaining |

## Threshold Configuration

Define crossing alerts in `requirements.yaml`:

```yaml
telemetry_thresholds:
  - channel: tire_wear_fl
    name: high_wear_fl
    value: 3.5
    direction: above
    severity: warning

  - channel: brake_temp_fl
    name: brake_overheat_fl
    value: 550.0
    direction: above
    severity: warning

  - channel: brake_temp_fl
    name: brake_critical_fl
    value: 650.0
    direction: above
    severity: critical
```

## Visualization

The `plot` command generates matplotlib visualizations:

| Plot Type | Description |
|-----------|-------------|
| **Dashboard** | Multi-channel grid showing all telemetry with baseline (blue) vs candidate (red) envelopes |
| **Delta** | Shows difference (candidate - baseline) at each track position. Green = candidate better, Red = baseline better |
| **Violations** | Focused view of channels with threshold crossings, hatched regions show violations |
| **Single channel** | Detailed view of one channel with mean ± std envelopes |

**Example output:**

```bash
$ python main.py plot experimental-dry-v1.2

Generated 6 plot files in 'plots/':
  - plots/experimental-dry-v1.2_dashboard.png
  - plots/experimental-dry-v1.2_delta.png
  - plots/experimental-dry-v1.2_violations.png
  - plots/experimental-dry-v1.2_speed.png
  - plots/experimental-dry-v1.2_tire_wear_fl.png
  - plots/experimental-dry-v1.2_brake_temp_fl.png
```

**Programmatic usage:**

```python
from telemetry import TelemetryStore, TelemetryAnalyzer
from visualization import TelemetryVisualizer, build_thresholds_by_channel

store = TelemetryStore("telemetry")
analyzer = TelemetryAnalyzer(store, thresholds)
comparison = analyzer.compare_telemetry(baseline_paths, candidate_paths)

visualizer = TelemetryVisualizer()
thresholds_by_channel = build_thresholds_by_channel(thresholds)

# Generate all plots
visualizer.create_report(comparison, thresholds_by_channel, output_dir="plots")

# Or individual plots
visualizer.plot_dashboard(comparison, save_path="dashboard.png")
visualizer.plot_delta(comparison, save_path="delta.png")
```

## PDF Report Generation

Generate comprehensive PDF reports combining metrics, telemetry, plots, and AI analysis:

```bash
# Basic report
python main.py report candidate-dry-v1.1

# With custom output location and title
python main.py report candidate-dry-v1.1 -o reports --title "Aero Package V1.1 Analysis"

# With AI analysis (requires Ollama)
python main.py report candidate-dry-v1.1 --ai
```

**Report contents:**
- Executive summary with overall score and status
- Requirement violations (if any)
- Detailed metrics comparison table
- Telemetry threshold analysis
- Embedded visualization plots (dashboard, delta, violations)
- AI-generated insights (optional)

## Web Interface

A FastAPI-based web interface provides a graphical way to interact with the analyzer.

### Starting the Web Server

```bash
# Run the web server
python web_app.py

# Or with uvicorn for development (auto-reload)
uvicorn web_app:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

### Features

- **Home Page**: Lists all batches with quick access links
- **Analysis View**: Shows score, metrics table, telemetry crossings, and embedded plots
- **PDF Reports**: Generate and view reports directly in the browser
- **Tabbed Interface**: Switch between Summary, Metrics, Telemetry, and Plots
- **LLM Chat Panel**: Floating chat panel on every page for asking questions about the data

### Chat with LLM

The web interface includes a chat panel where you can ask natural language questions about the simulation data. The LLM has access to database tools and can query real data to answer questions.

**How it works:**
1. The chat panel is available on every page (bottom-right corner)
2. When you ask a question, the LLM receives the available tool definitions
3. The LLM decides which tools to call (e.g., `get_batch_summary`, `compare_batches`)
4. Tools execute against the database and return results to the LLM
5. The LLM interprets the data and responds in natural language

**Example questions:**
- "List all batches and their run counts"
- "Compare candidate-dry-v1.1 against baseline-dry-v1.0"
- "Which batch has the best lap time?"
- "Show telemetry threshold violations for experimental-dry-v1.2"
- "What car version has the highest max speed?"
- "Show me run metrics for run 2"
- "Compare lap time across all batches"
- "List all runs in this batch"
- "Analyze run id 24"
- "What run is the fastest?"
- "Show front tires wear"
- "Which run in the current batch has the highest brake temp?"
- "how thresholds values"
- "Which run has best lap time for this batch?
- "And the second best?"
- "List 3 best runs sorted by top speed"
- "How tire degradation depends on max speed if at all?"
  
**Robustness features:**
- **Fallback text parsing**: Handles models that emit tool calls as JSON text instead of structured `tool_calls`
- **Unknown tool correction**: Detects when the LLM hallucinates a non-existent tool name and nudges it to use a valid one
- **Multi-round tool calling**: Supports up to 5 rounds of tool calls per question

Requires either Ollama running locally or an OpenAI API key. Configure via environment variables or a `.env` file:

```bash
# Option A: Ollama (default, no configuration needed)
# Just start Ollama: ollama serve

# Option B: OpenAI API — create a .env file in the project root:
OPENAI_API_KEY=sk-...          # Required for OpenAI
LLM_MODEL=gpt-4o               # Optional (defaults to gpt-4o when key is set)
LLM_BASE_URL=https://api.openai.com/v1  # Optional (auto-detected when key is set)
```

When `OPENAI_API_KEY` is set, the web app automatically switches to OpenAI with `gpt-4o`.

### API Endpoints

The web app also exposes REST API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/batches` | GET | List all batches |
| `/api/analyze/{batch_id}` | GET | Get analysis JSON |
| `/api/report/{batch_id}` | POST | Generate PDF report |
| `/api/ai/{batch_id}` | POST | Get AI analysis |
| `/api/chat` | POST | Chat with LLM (tool calling) |
| `/health` | GET | Health check |

Example API usage:
```bash
curl http://localhost:8000/api/batches
curl http://localhost:8000/api/analyze/candidate-dry-v1.1?telemetry=true
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "List all batches"}]}'
```

**Programmatic usage:**

```python
from report import PDFReportGenerator, ReportConfig

config = ReportConfig(
    title="My Analysis Report",
    subtitle="Candidate v1.1 vs Baseline",
)

generator = PDFReportGenerator(config)
generator.generate(
    "report.pdf",
    analysis_result,           # From Analyzer.compare()
    telemetry_data=tel_dict,   # From TelemetryAnalyzer.to_dict()
    plot_paths=plot_files,     # List of PNG paths
    ai_analysis=ai_text,       # Optional AI text
)
```

## MCP Server

The project includes an MCP (Model Context Protocol) server that exposes the database through standardized tools. This allows external LLM clients (Claude Desktop, MCP Inspector, or any MCP-compatible client) to query the simulation data.

### Running the MCP Server

```bash
# Standalone
python mcp_server.py

# With MCP Inspector (for debugging)
mcp dev mcp_server.py
```

### Available Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `list_batches` | — | List all simulation batches with run counts |
| `list_cars` | — | List car versions with SW/HW configs and associated batches |
| `list_scenarios` | — | List experiment scenarios with parameters |
| `get_versions_for_scenario` | `scenario_id` | Get car versions for a specific scenario |
| `get_batch_details` | `batch_id` | Get all runs and their metrics for a batch |
| `get_batch_summary` | `batch_id` | Get aggregate stats (mean, std, min, max) per metric |
| `compare_batches` | `candidate_batch_id`, `baseline_batch_id` | Compare two batches with delta and % change |
| `get_run_metrics` | `run_id` | Get all metrics for a specific run |
| `get_thresholds` | `channel?` | Get threshold definitions, optionally by channel |
| `find_batches_for_scenario` | `scenario_name` | Find batches by scenario name (fuzzy match) |
| `query_metric_across_batches` | `metric_name` | Track a metric across all batches |
| `get_telemetry_summary` | `batch_id` | Per-channel telemetry stats (speed, tire wear, brake temps, etc.) |
| `get_telemetry_crossings` | `batch_id` | Threshold violation summary for a batch's telemetry |

### Resources

| URI | Description |
|-----|-------------|
| `racesim://schema` | Database schema (DDL) |
| `racesim://batches` | Summary of all available batches |

### Integration with the Web App

The same tool functions from `mcp_server.py` are imported directly into `web_app.py` for the chat endpoint. This means the chat LLM and external MCP clients share the same underlying query logic.

## CLI Reference

```
python main.py [options] <command> [args]

Options:
  -d, --database PATH      Database file (default: racesim.db)
  -r, --requirements PATH  Requirements YAML (default: requirements.yaml)
  --ollama-url URL         LLM API base URL (default: $LLM_BASE_URL or http://localhost:11434/v1)
  -m, --model NAME         LLM model name (default: $LLM_MODEL or llama3.2)
  --api-key KEY            API key for OpenAI or other authenticated providers (default: $OPENAI_API_KEY)
  --telemetry-dir PATH     Telemetry directory (default: telemetry)

Commands:
  init                     Initialize database and load thresholds
  list-batches             List all batches
  analyze BATCH_ID         Analyze batch vs baseline
    --ai                   Include AI analysis
    --suggest              Include AI improvement suggestions
    -c, --context TEXT     Additional context for AI
    -t, --telemetry        Include telemetry analysis
  telemetry BATCH_ID       Telemetry-only analysis
    --ai                   Include AI analysis
    -p, --plot             Generate visualization plots
    -o, --output-dir PATH  Output directory for plots
  plot BATCH_ID            Generate telemetry visualizations
    -o, --output-dir PATH  Output directory (default: plots)
    -c, --channel NAME     Plot single channel
    --dashboard            Generate dashboard view only
    --delta                Generate delta analysis only
    --violations           Generate violations plot only
  report BATCH_ID          Generate comprehensive PDF report
    --output PATH          Output PDF path
    -o, --output-dir PATH  Output directory for PDF and plots
    -t, --title TEXT       Custom report title
    --ai                   Include AI analysis in report
  compare BATCH_ID...      Compare multiple batches
    --ai                   Include AI comparison
  check-ai                 Test LLM API connection (Ollama or OpenAI)
```

## Integrating with Your Simulation

### Adding run data:

```python
from db import RaceSimDB

db = RaceSimDB("racesim.db")

run_id = db.create_run(
    version_id=your_version_id,
    scenario_id=your_scenario_id,
    batch_id="my-experiment-2024-01",
    is_baseline=False
)

db.add_run_metrics(run_id, {
    "lap_time": 83.2,
    "fuel_consumption": 2.1,
    # ... other metrics
})
```

### Adding telemetry:

```python
from telemetry import TelemetryStore
import numpy as np

store = TelemetryStore("telemetry")

# Your simulation output arrays
telemetry = {
    "position_m": positions,     # Track position array
    "speed": speeds,             # Speed at each position
    "tire_wear_fl": tire_fl,     # Tire wear arrays
    # ... other channels
}

file_path = store.save(run_id, telemetry)

db.add_telemetry_reference(
    run_id=run_id,
    file_path=file_path,
    sample_count=len(positions),
    start_position_m=positions[0],
    end_position_m=positions[-1]
)
```

### Programmatic analysis:

```python
from analysis import RequirementsLoader, Analyzer
from telemetry import TelemetryStore, TelemetryAnalyzer

# Summary analysis
requirements = RequirementsLoader("requirements.yaml")
analyzer = Analyzer(requirements)
result = analyzer.compare(baseline_metrics, candidate_metrics, "batch-id")
print(analyzer.format_report(result))

# Telemetry analysis
store = TelemetryStore("telemetry")
thresholds = db.get_thresholds()
tel_analyzer = TelemetryAnalyzer(store, thresholds)
comparison = tel_analyzer.compare_telemetry(baseline_paths, candidate_paths)
print(tel_analyzer.format_threshold_report(comparison.candidate_threshold_analysis))
```

## Units Convention

All metrics use SI units:
- Time: seconds (s)
- Distance: meters (m)
- Speed: meters per second (m/s)
- Mass: kilograms (kg)
- Energy: kilojoules (kJ)
- Temperature: Celsius (°C)
- Acceleration: g-force (g)

## License

MIT
