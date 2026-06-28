"""
Day 4: FastAPI endpoint that returns a predicted completion probability
plus the most similar historical trials.
Run: uvicorn src.api:app --reload   ->  http://localhost:8000/docs
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Clinical Trial Outcome Predictor")

model_clf = joblib.load("src/model.joblib")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("src/trials.faiss")
meta = pd.read_parquet("src/trials_meta.parquet")
meta["label"] = (meta["overall_status"].str.upper() == "COMPLETED").astype(int)


class Trial(BaseModel):
    description: str
    phase: str = "PHASE2"
    enrollment: int = 100
    number_of_arms: int = 2
    start_year: int = 2024
    duration_months: float = 24
    n_conditions: int = 1
    n_interventions: int = 1
    sponsor_class: str = "INDUSTRY"
    allocation: str = "Randomized"
    intervention_model: str = "Parallel Assignment"
    masking: str = "Double"
    primary_purpose: str = "Treatment"


@app.get("/")
def health():
    return {"status": "ok", "service": "clinical-trial-outcome-predictor"}


@app.post("/analyze")
def analyze(t: Trial, k: int = 5):
    # 1) retrieval
    q = embedder.encode([t.description], normalize_embeddings=True)
    scores, idx = index.search(q, k)
    similar = meta.iloc[idx[0]][
        ["nct_id", "phase", "overall_status", "label"]
    ].to_dict("records")
    # 2) prediction
    row = pd.DataFrame([t.dict()]).drop(columns=["description"])
    prob = float(model_clf.predict_proba(row)[:, 1][0])
    # 3) synthesis
    hist_rate = float(np.mean([s["label"] for s in similar])) if similar else None
    return {
        "predicted_completion_probability": round(prob, 3),
        "historical_completion_rate_of_similar": hist_rate,
        "similar_trials": similar,
    }
