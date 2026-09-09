"""
AI-Powered Automatic Business Intelligence Dashboard Generator
==============================================================

Transform raw CSV/Excel data into a complete interactive, Power BI-inspired
dashboard using automated data engineering, statistics, machine learning,
and OpenAI-powered dashboard planning.

## Overview

This is **not** a manual chart builder.

```
USER DATA + USER INTENT
        ↓
AUTOMATED DATA ENGINEERING
        ↓
STATISTICAL ANALYSIS
        ↓
MACHINE LEARNING (when appropriate)
        ↓
AI REASONING / DASHBOARD PLAN
        ↓
DYNAMIC VISUALIZATION GENERATION
        ↓
INTERACTIVE BI DASHBOARD + EXPORTS
```

The user only needs to:

1. Upload a CSV / XLSX / XLS file  
2. Describe what they want in plain language  
3. Click **Generate Complete AI Dashboard**

## Features

- Automatic file validation & loading (CSV, XLSX, XLS)
- Intelligent cleaning with structured cleaning report
- Column type detection (numerical, categorical, date, boolean, text, identifier)
- Full profiling, statistics, correlations, outliers, trends
- Data quality score with documented penalties
- Real ML when useful: classification, regression, clustering, anomaly detection, forecasting
- OpenAI dashboard planner (structured JSON) with validation & repair
- **Dynamic visualization count** — no fixed 4/6/12 chart limit
- Duplicate chart prevention
- Interactive filters that update KPIs, charts, tables, and insights
- Plotly interactive charts
- Exports: cleaned/filtered CSV, analysis JSON, plan JSON, Excel, optional PDF
- Streamlit Cloud friendly

## Architecture

| Module | Responsibility |
|--------|----------------|
| `app.py` | Streamlit UI & workflow orchestration |
| `data_engine.py` | Load, validate, clean, profile, type detection |
| `data_analyzer.py` | Statistics, correlations, outliers, trends |
| `insight_engine.py` | Fact-based business insights |
| `ml_engine.py` | Real ML training & evaluation |
| `ai_planner.py` | OpenAI dashboard planning |
| `validation_engine.py` | AI plan / chart validation |
| `chart_engine.py` | Plotly charts + rule-based recommendations |
| `dashboard_engine.py` | Filters, KPIs, tables, layout organization |
| `dashboard_layout.py` | UI components / styling |
| `export_engine.py` | Downloads & reports |
| `utils.py` | Shared helpers & performance limits |

## Requirements

- Python **3.10+** (3.11 recommended)
- An OpenAI API key (optional fallback planner works without it, but AI planning requires a key)

## Installation

```bash
cd Deshbord
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## OpenAI API setup

### Option A — Streamlit secrets (recommended)

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Set:

```toml
OPENAI_API_KEY = "sk-your-real-key"
```

### Option B — Environment variable

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-your-real-key"

# macOS / Linux
export OPENAI_API_KEY=sk-your-real-key
```

**Never commit real keys. Never hard-code keys in source files.**

## How to run

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

## Example workflow

1. Upload `sample_data/sales_sample.csv`
2. Enter a prompt such as:

```text
Create a complete sales and profitability dashboard.
Show revenue, profit, monthly trends, regional performance,
top products, customer segments, important KPIs,
relationships between sales and profit, and any other
meaningful analysis you can find.
```

3. Click **Generate Complete AI Dashboard**
4. Use filters; export cleaned data / plan / Excel report

### Other example prompts

```text
Analyze employee attrition. Show KPIs, department comparison,
salary distribution, age analysis, attrition trends and related factors.
```

```text
Give me every meaningful visualization available for this dataset
and highlight the most important business insights.
```

## Sample data

Generated samples live in `sample_data/`:

- `sales_sample.csv` — sales / commerce style
- `hr_attrition_sample.csv` — HR attrition style

## Testing

```bash
pip install pytest
pytest tests -q
```

Tests cover loading, cleaning, typing, profiling, correlations, insights,
ML opportunity paths, chart generation, and AI plan validation — including edge cases.

## Performance safeguards

These are **not** arbitrary dashboard chart limits:

| Limit | Purpose |
|-------|---------|
| Max ~30 categories per chart | Readable axes |
| Scatter sampling (5,000) | Browser performance |
| Correlation columns (40) | Pairwise cost control |
| Pie/donut max 8 categories | Avoid unreadable pies |
| Filter cardinality 100 | Useful filters only |
| ML minimum 30 rows | Statistical reliability |

Visualization **count** remains dynamic based on analytical usefulness.

## Data quality score

Starts at **100** and applies penalties for:

- Missing values before cleaning
- Duplicate row share
- Constant / low-information columns
- Remaining missing values after cleaning

The explanation is shown in the cleaning report expander.

## Streamlit Cloud deployment

1. Push this project to GitHub (without real secrets)
2. Create a new app at https://share.streamlit.io
3. Set main file path to `app.py`
4. Add secret `OPENAI_API_KEY` in the Streamlit Cloud secrets UI
5. Deploy

`requirements.txt` uses pure-Python-friendly packages (no GPU).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `OPENAI_API_KEY is not configured` | Add key to secrets or environment |
| AI timeout / API error | App falls back to analytical planner |
| Excel fails to open | Ensure `openpyxl` / `xlrd` installed; file not password-protected |
| Empty dashboard charts | Check filters; verify columns after cleaning |
| ML skipped | Expected when no reliable target / too few rows |
| Import errors | Run from project root; `pip install -r requirements.txt` |

## Security

- API keys never displayed in the UI
- Uploaded files are read as data only (no macro/code execution)
- AI output is validated before rendering
- No arbitrary user code execution

## License

MIT — use and extend freely for your organization.
