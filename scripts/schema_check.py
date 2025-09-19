import argparse
import pandas as pd

REQUIRED = ["age", "sex", "weight_kg", "drug_name", "indication", "concomitant_drugs", "serious"]

def main(csv_path: str):
    df = pd.read_csv(csv_path, nrows=5)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    print("Schema OK. Sample:")
    print(df.head())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    args = p.parse_args()
    main(args.csv)
