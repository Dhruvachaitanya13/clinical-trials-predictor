"""
Day 3: build a FAISS index over trial summaries and provide a search function.
Run as a script to build the index; import `search` elsewhere.
"""
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

EMB_MODEL = "all-MiniLM-L6-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL)
    return _model


def build_index():
    df = pd.read_csv("data/trials_features.csv").dropna(subset=["brief_summary"])
    df = df.reset_index(drop=True)
    emb = get_model().encode(
        df["brief_summary"].tolist(),
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )
    index = faiss.IndexFlatIP(emb.shape[1])  # cosine sim on normalized vectors
    index.add(emb)
    faiss.write_index(index, "src/trials.faiss")
    df.to_parquet("src/trials_meta.parquet")
    print(f"Indexed {len(df)} trials.")


def search(query, k=5):
    index = faiss.read_index("src/trials.faiss")
    meta = pd.read_parquet("src/trials_meta.parquet")
    q = get_model().encode([query], normalize_embeddings=True)
    scores, idx = index.search(q, k)
    out = meta.iloc[idx[0]][
        ["nct_id", "phase", "overall_status", "brief_summary"]
    ].copy()
    out["label"] = (out["overall_status"].str.upper() == "COMPLETED").astype(int)
    out["similarity"] = scores[0]
    return out


if __name__ == "__main__":
    build_index()
    print(search("Phase 2 trial of a GLP-1 agonist for type 2 diabetes in adults"))
