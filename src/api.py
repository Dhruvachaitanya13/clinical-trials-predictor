"""
Day 4: FastAPI service combining the Day 2 classifier with the Day 3 retrieval index.
Run from the project root:  uvicorn src.api:app --reload
Then open:                  http://localhost:8000/docs

Endpoints:
  GET  /         -> health check
  GET  /ready    -> readiness (are all artifacts loaded?)
  POST /analyze  -> predicted completion probability + similar historical trials
"""
# Import faiss FIRST, before torch/sentence-transformers, to avoid the OpenMP
# library conflict that causes a segmentation fault on macOS.
import faiss

import logging
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trials-api")

# Artifacts are loaded once at startup and held in this dict.
ARTIFACTS: dict = {}

VALID_PHASES = {
    "EARLY_PHASE1", "PHASE1", "PHASE1/PHASE2", "PHASE2",
    "PHASE2/PHASE3", "PHASE3", "PHASE4",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, embedder, and index once when the server starts."""
    try:
        logger.info("Loading artifacts...")
        ARTIFACTS["model"] = joblib.load("src/model.joblib")
        ARTIFACTS["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
        ARTIFACTS["index"] = faiss.read_index("src/trials.faiss")
        meta = pd.read_parquet("src/trials_meta.parquet")
        meta["label"] = (meta["overall_status"].str.upper() == "COMPLETED").astype(int)
        ARTIFACTS["meta"] = meta
        logger.info("Artifacts loaded: %d trials in index.", ARTIFACTS["index"].ntotal)
    except Exception as e:
        logger.error("Failed to load artifacts: %s", e)
        # Leave ARTIFACTS partially empty; /ready will report not-ready.
    yield
    ARTIFACTS.clear()


app = FastAPI(
    title="Clinical Trial Outcome Predictor",
    description="Predicts trial completion and retrieves similar historical trials.",
    version="1.0.0",
    lifespan=lifespan,
)


class Trial(BaseModel):
    description: str = Field(..., min_length=10, max_length=5000,
                             description="Free-text description of the trial.")
    phase: str = "PHASE2"
    enrollment: int = Field(100, ge=0, le=1_000_000)
    number_of_arms: int = Field(2, ge=1, le=100)
    start_year: int = Field(2024, ge=1990, le=2035)
    duration_months: float = Field(24, ge=0, le=600)
    n_conditions: int = Field(1, ge=0, le=100)
    n_interventions: int = Field(1, ge=0, le=100)
    sponsor_class: str = "INDUSTRY"
    allocation: str = "Randomized"
    intervention_model: str = "Parallel Assignment"
    masking: str = "Double"
    primary_purpose: str = "Treatment"

    @field_validator("phase")
    @classmethod
    def phase_known(cls, v):
        if v.upper() not in VALID_PHASES:
            raise ValueError(f"phase must be one of {sorted(VALID_PHASES)}")
        return v.upper()


class SimilarTrial(BaseModel):
    nct_id: str
    phase: Optional[str]
    overall_status: str
    completed: int


class AnalyzeResponse(BaseModel):
    predicted_completion_probability: float
    historical_completion_rate_of_similar: Optional[float]
    similar_trials: list[SimilarTrial]


@app.get("/")
def health():
    return {"status": "ok", "service": "clinical-trial-outcome-predictor"}


@app.get("/ready")
def ready():
    needed = {"model", "embedder", "index", "meta"}
    missing = needed - ARTIFACTS.keys()
    if missing:
        raise HTTPException(status_code=503,
                            detail=f"Not ready, missing artifacts: {sorted(missing)}")
    return {"status": "ready", "trials_indexed": ARTIFACTS["index"].ntotal}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(t: Trial, k: int = 5):
    if not ARTIFACTS.get("model"):
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if k < 1 or k > 50:
        raise HTTPException(status_code=422, detail="k must be between 1 and 50.")
    try:
        # 1) retrieval
        q = ARTIFACTS["embedder"].encode([t.description], normalize_embeddings=True)
        scores, idx = ARTIFACTS["index"].search(q, k)
        rows = ARTIFACTS["meta"].iloc[idx[0]]
        similar = [
            SimilarTrial(
                nct_id=r["nct_id"], phase=r.get("phase"),
                overall_status=r["overall_status"], completed=int(r["label"]),
            )
            for _, r in rows.iterrows()
        ]
        # 2) prediction
        features = t.model_dump()
        features.pop("description")
        prob = float(ARTIFACTS["model"].predict_proba(pd.DataFrame([features]))[:, 1][0])
        # 3) synthesis
        hist = float(np.mean([s.completed for s in similar])) if similar else None
        return AnalyzeResponse(
            predicted_completion_probability=round(prob, 3),
            historical_completion_rate_of_similar=hist,
            similar_trials=similar,
        )
    except Exception as e:
        logger.exception("analyze failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
