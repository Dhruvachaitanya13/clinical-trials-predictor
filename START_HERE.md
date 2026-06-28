# START HERE

This is your project scaffold. Every file is in place — you fill in credentials and real results as you go.

## First 10 minutes
1. Open this folder in VS Code: `File → Open Folder` → select `clinical-trials-predictor`.
2. Open a terminal in VS Code (`Ctrl+` `) and create + activate the venv:
   ```
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
   ```
3. Select the interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick the `venv` one.
4. Install everything:
   ```
   pip install -r requirements.txt
   ```
5. Put your AACT username/password into `test_connection.py`, then run it.
   If it prints a study count + status table, you're ready for Day 1.

## What's where
- `test_connection.py` — Day 0 gate. Run first.
- `sql/build_features.sql` — Day 1 query that builds the dataset.
- `notebooks/01_data_labels.ipynb` — Day 1. Produces `data/trials_features.csv`.
- `notebooks/02_baseline_model.ipynb` / `src/train.py` — Day 2. Trains + saves the model.
- `notebooks/03_retrieval.ipynb` / `src/retrieval.py` — Day 3. Builds the FAISS index.
- `src/api.py` — Day 4. `uvicorn src.api:app --reload`
- `dashboard/app.py` — Day 5. `streamlit run dashboard/app.py`
- `Dockerfile` — Day 6. Containerize + deploy.
- `README.md` — Day 7. Fill in the bracketed results.

## Notebooks use ../ paths
The notebooks live in `notebooks/`, so they reference `../data/` and `../sql/`.
The standalone scripts in `src/` use paths relative to the project root — run them
from the project root (e.g. `python src/train.py`).

## Daily habit
Commit at the end of every day:
```
git add . && git commit -m "Day N: <what you did>" && git push
```

Refer to the 7-day playbook and the learning curriculum (separate files) as you go.
