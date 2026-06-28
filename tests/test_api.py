"""
Day 6: API test suite. Run from the project root:  python -m pytest tests/ -v
Tests that the service starts, loads artifacts, validates input, and predicts.
Requires the Day 2/3 artifacts (model.joblib, trials.faiss, trials_meta.parquet)
to exist in src/.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    # Context manager runs the lifespan startup, which loads the artifacts.
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "description": "Phase 2 randomized trial of a GLP-1 agonist for type 2 diabetes in adults",
    "phase": "PHASE2",
    "enrollment": 120,
    "number_of_arms": 2,
    "start_year": 2023,
    "duration_months": 24,
    "n_conditions": 1,
    "n_interventions": 1,
    "sponsor_class": "INDUSTRY",
    "allocation": "Randomized",
    "intervention_model": "Parallel Assignment",
    "masking": "Double",
    "primary_purpose": "Treatment",
}


def test_health_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_index(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["trials_indexed"] > 0


def test_analyze_returns_prediction(client):
    r = client.post("/analyze", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    p = body["predicted_completion_probability"]
    assert 0.0 <= p <= 1.0
    assert len(body["similar_trials"]) == 5
    first = body["similar_trials"][0]
    assert set(first) >= {"nct_id", "phase", "overall_status", "completed"}


def test_analyze_respects_k(client):
    r = client.post("/analyze?k=3", json=VALID_PAYLOAD)
    assert r.status_code == 200
    assert len(r.json()["similar_trials"]) == 3


def test_historical_rate_in_range(client):
    r = client.post("/analyze", json=VALID_PAYLOAD)
    assert r.status_code == 200
    rate = r.json()["historical_completion_rate_of_similar"]
    assert rate is None or 0.0 <= rate <= 1.0


def test_rejects_short_description(client):
    bad = dict(VALID_PAYLOAD, description="too short")
    assert client.post("/analyze", json=bad).status_code == 422


def test_rejects_unknown_phase(client):
    bad = dict(VALID_PAYLOAD, phase="PHASE9")
    assert client.post("/analyze", json=bad).status_code == 422


def test_rejects_negative_enrollment(client):
    bad = dict(VALID_PAYLOAD, enrollment=-10)
    assert client.post("/analyze", json=bad).status_code == 422


def test_rejects_out_of_range_k(client):
    r = client.post("/analyze?k=999", json=VALID_PAYLOAD)
    assert r.status_code == 422


def test_missing_description_field(client):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "description"}
    assert client.post("/analyze", json=bad).status_code == 422
