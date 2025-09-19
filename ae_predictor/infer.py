import argparse
import yaml
import joblib
import json
import pandas as pd
from pathlib import Path

def infer(cfg_path: str, json_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    # Choose a model
    model_dir = Path(paths["model_dir"])
    model_path = None
    for name in ["xgboost_model.joblib", "logreg_model.joblib"]:
        p = model_dir / name
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError("No trained model found in models/")

    model = joblib.load(model_path)

    with open(json_path, "r") as f:
        row = json.load(f)

    df = pd.DataFrame([row])
    proba = model.predict_proba(df)[:, 1][0]
    print(json.dumps({"probability_serious_ae": float(proba)}, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--input_json", type=str, required=True)
    args = parser.parse_args()
    infer(args.config, args.input_json)
