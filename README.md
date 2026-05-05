# ✈️ Flight Delay Predictor

**CMPS344 Applied Data Science — Spring 2026**
Cairo University, Faculty of Engineering

---

## Team

| Name |
|------|
| Jumana Timor |
| Youssef El-sawy |
| Lujina Shawki |
| George Ayman |

---

## Project Description

Flight delays create significant operational, financial, and customer-service challenges for airlines and airports. This project builds a **supervised binary classification model** that predicts whether a flight will be delayed, enabling airline and airport operations teams to act proactively and reduce downstream disruption.

**Stakeholders:** Airline operations teams, airport operations teams, and travel platforms.

**Classification target:** Whether a given flight is delayed (binary: delayed / on-time).

---

## Data Sources

- **Flight performance data** — historical on-time performance records (e.g., BTS/FAA)
- **Weather data** — meteorological conditions at origin/destination airports
- **Airport / route metadata** — airport characteristics, route statistics

---

## Repository Structure

```text
flight-delay-predictor/
├── .github/                 # GitHub Actions CI workflows
├── backend/                 # REST API / serving layer
│   ├── __pycache__/         # Python bytecode cache (git-ignored)
│   ├── main.py              # Entry point for the backend server
│   └── requirements.txt     # Backend-specific dependencies
├── dashboards/              # Interactive dashboard outputs (Streamlit / Dash)
├── data/
│   ├── raw/                 # Raw input data (git-ignored, kept local)
│   ├── interim/             # Intermediate data (git-ignored, kept local)
│   └── processed/           # Final processed data (git-ignored, kept local)
├── frontend/                # User-facing web interface
│   └── index.html           # Main HTML page
├── models/                  # Saved trained model artifacts
├── notebooks/               # Jupyter notebooks for exploration and EDA
├── reports/
│   ├── figures/             # Charts and plots
│   └── tables/              # Generated tables and evaluation reports
├── src/
│   ├── data/                # Data acquisition, validation, and merging
│   ├── features/            # Feature engineering and transformation
│   ├── notebooks/           # Jupyter notebooks for exploration and EDA
│   ├── models/              # Model training and prediction
│   ├── evaluation/          # Evaluation metrics (standard + business-oriented)
│   └── utils/               # Utility/helper functions
├── tests/                   # Unit and integration tests (pytest)
├── .env.example             # Example environment variables (copy to .env)
├── .gitignore               # Git ignore rules
├── Makefile                 # Shortcut commands for common tasks
├── pyproject.toml           # Poetry project configuration
├── poetry.lock              # Dependency lock file
└── README.md                # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/flight-delay-predictor.git
cd flight-delay-predictor
```

### 2. Install dependencies

```bash
poetry install
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in any required API keys or credentials
```

---

## Running the Project

All common tasks are available via `make`. Run `make help` to see all targets.

```bash
# Acquire and validate data
make data

# Run preprocessing and feature engineering
make features

# Train all models
make train

# Evaluate models
make evaluate

# Launch the MLflow experiment tracking UI
make mlflow

# Run the interactive dashboard
make dashboard

# Run all tests with coverage report
make test
```

To run the full pipeline end-to-end:

```bash
make all
```

---

## Experiment Tracking

This project uses **MLflow** to track all model runs. Each run logs:
- Model name and hyperparameters
- Standard metrics (e.g., accuracy, F1-score)
- Business-oriented metrics (e.g., cost of missed delays, precision on high-traffic routes)
- Trained model artifacts

To view the MLflow UI locally:

```bash
make mlflow
# Then open http://localhost:5000
```

---

## Testing

Tests are written with **pytest** and cover unit and integration tests across all critical modules.

```bash
# Run tests with coverage
make test

# Or directly
poetry run pytest --cov=src tests/
```

---
