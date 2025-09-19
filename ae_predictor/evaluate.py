import argparse
import yaml
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
from .utils import ensure_dir

def evaluate(cfg):
    paths = cfg["paths"]
    data_cfg = cfg["data"]

    # Find a model file (pick xgboost if exists else logreg)
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
    test = pd.read_csv(paths["test_csv"])
    y = test[data_cfg["target"]].astype(int).values
    X = test.drop(columns=[data_cfg["target"]])

    proba = model.predict_proba(X)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(y, proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    ensure_dir(paths["figures_dir"])
    roc_path = str(Path(paths["figures_dir"]) / "roc_curve.png")
    plt.savefig(roc_path, dpi=160, bbox_inches="tight")
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y, proba)
    plt.figure()
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    pr_path = str(Path(paths["figures_dir"]) / "pr_curve.png")
    plt.savefig(pr_path, dpi=160, bbox_inches="tight")
    plt.close()

    # Confusion at 0.5
    preds = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y, preds).tolist()
    with open(Path(paths["reports_dir"]) / "confusion_matrix.json", "w") as f:
        json.dump({"threshold": 0.5, "matrix": cm}, f, indent=2)

    print(f"[EVAL] Saved figures: {roc_path}, {pr_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg)
