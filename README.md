# RaceSim Analyzer

A QA regression testing tool for racing car simulations. Compares Monte Carlo simulation results between car versions and provides AI-powered insights.

## Features

- **Database-backed storage**: SQLite schema for cars, versions, experiments, scenarios, runs, and metrics
- **Statistical comparison**: Compares candidate runs against baseline with mean, std, delta calculations
- **Requirements validation**: YAML-based requirements with pass/fail thresholds
- **Scoring system**: Weighted scoring with configurable thresholds (excellent/good/acceptable/poor/fail)
- **AI analysis**: Local LLM integration via Ollama for natural language insights

## Project Structure

```
racesim_analyzer/
├── schema.sql          # Database schema
├── db.py               # Database helpers and data classes
├── requirements.yaml   # Performance requirements configuration
├── analysis.py         # Statistical comparison and scoring engine
├── ai_analyzer.py      # Ollama/LLM integration
├── main.py             # CLI entry point
└── seed_sample_data.py # Sample data generator for testing
```

## Requirements

- Python 3.10+
- PyYAML
- requests
- Ollama (for AI features)

```bash
pip install pyyaml requests
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

3. **Analyze a candidate batch**:
   ```bash
   python main.py analyze candidate-dry-v1.1
   ```

4. **With AI analysis** (requires Ollama running):
   ```bash
   ollama serve  # In another terminal
   ollama pull llama3.2
   python main.py analyze candidate-dry-v1.1 --ai
   ```

5. **Compare multiple batches**:
   ```bash
   python main.py compare candidate-dry-v1.1 experimental-dry-v1.2 --ai
   ```

## Database Schema

```
cars ─────────────────┐
                      │
car_versions ─────────┼──► runs ──► run_metrics
                      │      ▲
experiments ──► scenarios ───┘
```

- **cars**: Car definitions (name, description)
- **car_versions**: SW/HW version combinations per car
- **experiments**: Test campaigns grouping scenarios
- **scenarios**: Input conditions (track, weather, etc.)
- **runs**: Individual simulation runs with batch grouping
- **run_metrics**: Key-value metric storage per run

## Requirements Configuration

Edit `requirements.yaml` to define your performance targets:

```yaml
metrics:
  lap_time:
    target: 82.0        # Ideal value
    max: 85.0           # Failure threshold
    weight: 0.25        # Importance (0-1)
    direction: lower_better
    unit: "seconds"
```

## CLI Reference

```
python main.py [options] <command> [args]

Options:
  -d, --database PATH      Database file (default: racesim.db)
  -r, --requirements PATH  Requirements YAML (default: requirements.yaml)
  --ollama-url URL         Ollama API URL (default: http://localhost:11434/v1)
  -m, --model NAME         Ollama model (default: llama3.2)

Commands:
  init                     Initialize database
  list-batches             List all batches
  analyze BATCH_ID         Analyze batch vs baseline
    --ai                   Include AI analysis
    --suggest              Include improvement suggestions
    -c, --context TEXT     Additional context for AI
  compare BATCH_ID...      Compare multiple batches
    --ai                   Include AI comparison
  check-ai                 Test Ollama connection
```

## Integrating with Your Simulation

To use with your actual simulation:

1. **After simulation runs**, insert data:
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

2. **Mark baseline runs** by setting `is_baseline=True` for your reference configuration.

3. **Run analysis** via CLI or programmatically:
   ```python
   from analysis import RequirementsLoader, Analyzer
   
   requirements = RequirementsLoader("requirements.yaml")
   analyzer = Analyzer(requirements)
   result = analyzer.compare(baseline_metrics, candidate_metrics, "batch-id")
   print(analyzer.format_report(result))
   ```

## Units Convention

All metrics use SI units:
- Time: seconds (s)
- Speed: meters per second (m/s)
- Mass: kilograms (kg)
- Energy: kilojoules (kJ)
- Temperature: Celsius (°C)
- Acceleration: g-force (g)

## License

MIT
