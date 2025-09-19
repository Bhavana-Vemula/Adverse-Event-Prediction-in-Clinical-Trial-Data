import argparse
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from .utils import ensure_dir, save_json

def compute_pos_weight(y):
    # For imbalance: scale_pos_weight = (neg / pos)
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return float(neg / max(pos, 1))

def train_models(cfg):
    paths = cfg["paths"]
    cols = cfg["columns"]
    data_cfg = cfg["data"]
    mdl_cfg = cfg["modeling"]

    train = pd.read_csv(paths["train_csv"])
    target = data_cfg["target"]
    y = train[target].astype(int)
    X = train.drop(columns=[target])

    numeric = cols["numeric"]
    categorical = cols["categorical"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical),
        ],
        remainder="drop"
    )

    # Logistic Regression
    logreg = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            C=mdl_cfg["logistic_regression"]["C"],
            max_iter=mdl_cfg["logistic_regression"]["max_iter"],
            class_weight="balanced",
            n_jobs=None
        ))
    ])

    # XGBoost
    spw = compute_pos_weight(y.values)
    xgb = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=mdl_cfg["xgboost"]["n_estimators"],
            max_depth=mdl_cfg["xgboost"]["max_depth"],
            learning_rate=mdl_cfg["xgboost"]["learning_rate"],
            subsample=mdl_cfg["xgboost"]["subsample"],
            colsample_bytree=mdl_cfg["xgboost"]["colsample_bytree"],
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=4,
            scale_pos_weight=spw
        ))
    ])

    models = {"logreg": logreg, "xgboost": xgb}
    metrics = {}

    # Simple holdout evaluation on the test set
    test = pd.read_csv(paths["test_csv"])
    y_test = test[target].astype(int)
    X_test = test.drop(columns=[target])

    best_name, best_auc = None, -1.0
    for name, pipe in models.items():
        pipe.fit(X, y)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        f1 = f1_score(y_test, preds)
        bal = balanced_accuracy_score(y_test, preds)

        metrics[name] = {"roc_auc": float(auc), "pr_auc": float(ap), "f1": float(f1), "balanced_acc": float(bal)}

        if auc > best_auc:
            best_auc = auc
            best_name = name

    # Save best model
    best_model = models[best_name]
    ensure_dir(paths["model_dir"])
    model_path = str(Path(paths["model_dir"]) / f"{best_name}_model.joblib")
    joblib.dump(best_model, model_path)

    # Save metrics
    ensure_dir(paths["reports_dir"])
    save_json(metrics, str(Path(paths["reports_dir"]) / "metrics.json"))

    print(f"[TRAIN] Saved best model: {model_path} (by ROC-AUC). Metrics: {json.dumps(metrics, indent=2)}")
    return best_name, model_path, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_models(cfg)
