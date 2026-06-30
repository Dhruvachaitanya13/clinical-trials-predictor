"""
Build demo-sized artifacts for free-tier deployment.
Samples the dataset down so the FAISS index fits comfortably on Hugging Face Spaces.
Run from the project root:  python build_demo_artifacts.py

Outputs (into a deploy/ folder):
  deploy/model.joblib          (copied as-is; already tiny)
  deploy/trials_demo.faiss     (sampled index, ~25MB instead of 228MB)
  deploy/trials_meta_demo.parquet
"""
import shutil
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "deploy"
OUT.mkdir(exist_ok=True)

N_SAMPLE = 20_000          # demo size; ~25MB index. Raise/lower as you like.
SEED = 42

print("Loading full data + metadata…")
df = pd.read_csv(ROOT / "data" / "trials_features.csv").dropna(subset=["brief_summary"])
df = df.reset_index(drop=True)

# Stratify the sample by completion label so the demo keeps the real balance.
df["label"] = (df["overall_status"].str.upper() == "COMPLETED").astype(int)
if len(df) > N_SAMPLE:
    frac = N_SAMPLE / len(df)
    sample = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=SEED))
        .reset_index(drop=True)
    )
else:
    sample = df
print(f"Sampled {len(sample):,} of {len(df):,} trials.")

print("Re-embedding the sample…")
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(
    sample["brief_summary"].tolist(),
    show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
)

print("Building + writing demo index…")
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(OUT / "trials_demo.faiss"))
sample.to_parquet(OUT / "trials_meta_demo.parquet")
shutil.copy(SRC / "model.joblib", OUT / "model.joblib")

# Report sizes
for f in ["trials_demo.faiss", "trials_meta_demo.parquet", "model.joblib"]:
    mb = (OUT / f).stat().st_size / 1e6
    print(f"  {f}: {mb:.1f} MB")

print("\nDone. Demo artifacts are in deploy/ — these are what you upload to the Space.")
