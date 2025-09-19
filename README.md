# Adverse Event Prediction in Clinical Trial Data

**Author:** Bhavana Vemula  
**Goal:** Build a reproducible ML pipeline that predicts **adverse events (AEs)** from clinical trial–style data and ships with clean ETL, training, evaluation, and lightweight deployment.

## 📦 What’s in here
- `ae_predictor/`: Python package with ETL, training, evaluation, and inference modules
- `configs/`: YAML config(s) for paths & experiment settings
- `data/`: `raw/` for source CSVs, `processed/` for cleaned/split data
- `models/`: serialized model artifacts (joblib)
- `reports/figures/`: ROC/PR curves, confusion matrices, etc.
- `dashboards/`: optional Streamlit app for interactive risk scoring
- `notebooks/`: EDA notebooks (optional)
- `scripts/`: helper scripts (e.g., schema check)
- `tests/`: unit tests (minimal)

## 🧠 Problem
In clinical research, an **adverse event (AE)** is any unwanted medical occurrence during a study (e.g., side effects). Predicting AE risk helps with **patient safety** and **trial monitoring**.

## 🗂️ Data
This repo assumes a **public clinical dataset** exported to CSV (e.g., FDA FAERS extracts or trial-level records from ClinicalTrials.gov).  
Place your source CSV(s) in `data/raw/` as `ae.csv` with columns similar to:

```
age, sex, weight_kg, drug_name, indication, concomitant_drugs, serious
63, F, 68.0, DRUG_A, HYPERTENSION, DRUG_X|DRUG_Y, 1
```

- **Target**: `serious` (1 = serious AE, 0 = not serious)
- You can customize column names in `configs/default.yaml`.

> ⚠️ Data privacy: use only **public** or **synthetic** data. Do not commit PHI/PII.

## 🚀 Quickstart

### 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Configure paths
Edit `configs/default.yaml` if your filenames/columns differ.

### 3) Run ETL → Train → Evaluate
```bash
# Clean, validate, split
python -m ae_predictor.etl --config configs/default.yaml

# Train models (logistic + XGBoost) and save best
python -m ae_predictor.train --config configs/default.yaml

# Evaluate on the holdout set and produce figures
python -m ae_predictor.evaluate --config configs/default.yaml
```


## 🧪 Models & Metrics
- Baselines: **Logistic Regression** (with class weights) and **XGBoost**
- Metrics: **ROC-AUC**, **PR-AUC**, **F1**, **Balanced Accuracy**
- Outputs saved under `reports/` and `reports/figures/`

## 🐳 Docker
```bash
docker build -t ae-predictor .
docker run --rm -it -v $(pwd):/app ae-predictor python -m ae_predictor.train --config configs/default.yaml
```

## ☁️ Cloud
The code is compatible with AWS batch/EC2 workflows. Use S3 paths in `configs/default.yaml` if you sync data/models externally (optional).

## 📜 License
MIT — see `LICENSE`.
