import os
import json
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
import faiss


DATA_PATH = "data/abstracts.csv"
OUT_DIR = "vectorstore"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "text" not in df.columns:
        df["text"] = df["title"].fillna("") + "\n\n" + df["abstract"].fillna("")

    rows = []

    for _, row in df.iterrows():

        chunks = chunk_text(row["text"])

        for i, chunk in enumerate(chunks):
            rows.append({
                "id": row["id"],
                "year": row["year"],
                "title": row["title"],
                "journal": row["journal"],
                "source": row["source"],
                "chunk_id": i,
                "text": chunk
            })
    
    chunk_df = pd.DataFrame(rows)

    return chunk_df


def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks


def embed_texts(model: SentenceTransformer, texts: list, batch_size: int = 64) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype("float32")


def build_faiss_index(emb: np.ndarray) -> faiss.Index:
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index


def save_metadata(df: pd.DataFrame, out_dir: str) -> None:
    meta_cols = ["id", "year", "title", "journal", "source", "text"]
    meta = df[meta_cols].copy()
    meta.to_parquet(os.path.join(out_dir, "metadata.parquet"), index=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_corpus(DATA_PATH)
    texts = df["text"].tolist()

    print(f"Loaded {len(texts)} documents")

    model = SentenceTransformer(MODEL_NAME)

    emb = embed_texts(model, texts, batch_size=64)
    print("Embeddings shape:", emb.shape)

    index = build_faiss_index(emb)

    index_path = os.path.join(OUT_DIR, "index.faiss")
    faiss.write_index(index, index_path)
    print("Saved:", index_path)

    save_metadata(df, OUT_DIR)
    print("Saved:", os.path.join(OUT_DIR, "metadata.parquet"))

    cfg = {
        "model_name": MODEL_NAME,
        "n_docs": int(len(df)),
        "dim": int(emb.shape[1]),
        "index_type": "IndexFlatIP",
        "similarity": "cosine via normalized embeddings",
        "data_path": DATA_PATH,
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("Saved:", os.path.join(OUT_DIR, "config.json"))


if __name__ == "__main__":
    main()