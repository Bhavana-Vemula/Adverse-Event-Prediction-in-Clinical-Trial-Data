import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import ensure_dir

def run(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    data_cfg = cfg["data"]

    # Load raw
    df = pd.read_csv(paths["raw_csv"])
    # Basic cleaning examples (customize as needed)
    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop rows with missing target
    target = data_cfg["target"]
    df = df.dropna(subset=[target])

    # Save a cleaned copy
    ensure_dir(paths["processed_csv"].rsplit("/", 1)[0])
    df.to_csv(paths["processed_csv"], index=False)

    # Train/test split
    train_df, test_df = train_test_split(
        df, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"], stratify=df[target]
    )

    train_df.to_csv(paths["train_csv"], index=False)
    test_df.to_csv(paths["test_csv"], index=False)
    print(f"[ETL] Wrote: {paths['processed_csv']}, {paths['train_csv']}, {paths['test_csv']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    run(args.config)
