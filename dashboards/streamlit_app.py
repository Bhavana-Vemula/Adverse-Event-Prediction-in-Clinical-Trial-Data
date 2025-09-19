import streamlit as st
import yaml
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Clinical AE Risk Scoring", layout="centered")

st.title("Clinical Adverse Event (AE) Risk Scoring")

cfg_path = "configs/default.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
paths = cfg["paths"]

# Load trained model if available
model = None
model_name = None
for name in ["xgboost_model.joblib", "logreg_model.joblib"]:
    p = Path(paths["model_dir"]) / name
    if p.exists():
        model = joblib.load(p)
        model_name = name.split("_")[0].upper()
        break

st.sidebar.header("Model")
st.sidebar.write(f"Loaded: **{model_name or 'None'}**")

st.header("Enter Patient Features")
age = st.number_input("Age", min_value=0, max_value=120, value=63)
sex = st.selectbox("Sex", options=["M", "F"])
weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=68.0)
drug_name = st.text_input("Drug name", value="DRUG_A")
indication = st.text_input("Indication", value="HYPERTENSION")
concom = st.text_input("Concomitant drugs (pipe-separated)", value="DRUG_X|DRUG_Y")

if st.button("Predict AE Risk"):
    if model is None:
        st.error("No trained model found. Train a model first.")
    else:
        df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "weight_kg": weight,
            "drug_name": drug_name,
            "indication": indication,
            "concomitant_drugs": concom
        }])
        proba = model.predict_proba(df)[:, 1][0]
        st.metric("Probability of Serious Adverse Event", f"{proba:.2%}")
        st.info("This is a demo tool and not a medical device. For educational purposes only.")
