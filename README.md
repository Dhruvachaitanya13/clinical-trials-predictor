# Clinical Trial Outcome Predictor

Predicts whether an interventional clinical trial will **complete** vs. be **terminated/withdrawn/suspended**, and retrieves similar historical trials by semantic similarity. Built on the AACT / ClinicalTrials.gov registry.

> Status: in progress. Fill in the bracketed `[...]` placeholders with your real results as you go.

## What it does
- **Structured model:** gradient-boosted classifier predicting trial completion from sponsor, phase, enrollment, design, and other registry features.
- **Semantic retrieval:** sentence-transformer embeddings + FAISS index to find the most similar historical trials to a free-text trial description.
- **Combined API:** one FastAPI endpoint returns a predicted completion probability *and* the most similar past trials with their real outcomes.

## Data
- Source: [AACT database](https://aact.ctti-clinicaltrials.org/) (cleaned relational mirror of ClinicalTrials.gov), refreshed daily.
- Scope: interventional trials with a defined phase and a resolved status.
- Size: [N] trials after filtering.
- Label: `1` = Completed; `0` = Terminated / Withdrawn / Suspended. In-progress trials excluded (outcome unknown → would leak the future).

## Methodology
- Baseline: logistic regression.
- Main model: XGBoost.
- Validation: stratified train/test split [+ temporal split if done].
- Features: phase, enrollment, number of arms, sponsor class, allocation, masking, primary purpose, condition/intervention counts, start year, duration.

## Results
- Logistic regression AUC: [X]
- XGBoost AUC: [X]
- Precision / Recall at chosen threshold: [X] / [X]
- Top features (SHAP): [list]

## Limitations
- Label is coarse — completion ≠ clinical success.
- Registry data is self-reported and inconsistent.
- `duration_months` is borderline-leaky and treated with care.
- AUC in the 0.7x range reflects a genuinely hard problem.

## Next steps
Temporal validation · per-therapeutic-area models · use of outcome-measure text · LLM-generated trial briefings.

## How to run
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# set AACT credentials, then:
python test_connection.py          # verify DB access
# work through notebooks/ in order, then:
uvicorn src.api:app --reload       # serve the API
streamlit run dashboard/app.py     # run the dashboard
```
