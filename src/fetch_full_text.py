import os
import time
import requests
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss


INPUT_PATH = "data/abstracts.csv"
OUT_PATH = "data/fulltext_corpus.csv"
TEXT_DIR = "data/fulltext_txt"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BIOC_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"


QUERY = "LRP1 mediated transport blood brain barrier receptor mediated transcytosis"


# --------------------------------------------------
# Embedding utilities
# --------------------------------------------------

def embed_texts(model, texts):
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype("float32")


def rank_papers(df, query, model):

    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

    doc_emb = embed_texts(model, texts)
    q_emb = embed_texts(model, [query])

    scores = np.dot(doc_emb, q_emb.T).squeeze()

    df = df.copy()
    df["score"] = scores

    df = df.sort_values("score", ascending=False)

    return df


# --------------------------------------------------
# Full text download
# --------------------------------------------------

def fetch_bioc_json(pmcid):

    url = BIOC_URL.format(pmcid=pmcid)
    try:
        r = requests.get(url, timeout=30)
        
        if r.status_code != 200:
            return None
        
        return r.json()

    except Exception:
        return None


def extract_text_from_bioc(payload):
    parts = []

    def collect(passages):
        for p in passages:
            txt = p.get('text', '')
            if isinstance(txt, str) and txt.strip():
                parts.append(txt.strip())

    if isinstance(payload, dict):

        if "documents" in payload:

            for doc in payload["documents"]:

                passages = doc.get("passages", [])
                collect(passages)

    elif isinstance(payload, list):

        for doc in payload:

            passages = doc.get("passages", [])
            collect(passages)

    return "\n\n".join(parts)


# --------------------------------------------------
# Full pipeline
# --------------------------------------------------

def build_fulltext_corpus(df, limit=50, sleep_s=0.3):

    os.makedirs(TEXT_DIR, exist_ok=True)

    rows = []

    success = 0

    for _, row in df.iterrows():

        if success >= limit:
            break

        pmcid = row.get("pmcid")

        if pd.isna(pmcid):
            continue

        pmcid = str(pmcid).strip()

        print(f"Fetching {pmcid}")

        payload = fetch_bioc_json(pmcid)
        
        if payload is None:
            print("  failed")
            continue
        
        text = extract_text_from_bioc(payload[0])

        if not text.strip():
            print("  empty text")
            continue

        txt_path = os.path.join(TEXT_DIR, f"{pmcid}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        rows.append({
            "id": row.get("id"),
            "pmcid": pmcid,
            "title": row.get("title"),
            "year": row.get("year"),
            "journal": row.get("journal"),
            "source": row.get("source"),
            "abstract": row.get("abstract"),
            "text": text
        })

        success += 1

        print(f"  saved ({success}/50)")

        time.sleep(sleep_s)

    return pd.DataFrame(rows)


# --------------------------------------------------
# Main
# --------------------------------------------------
            

def main():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("data/abstracts.csv not found")

    df = pd.read_csv(INPUT_PATH)

    print("Abstract corpus:", len(df))

    model = SentenceTransformer(MODEL_NAME)

    print("Ranking papers...")

    ranked = rank_papers(df, QUERY, model)

    candidates = ranked[
        ranked["pmcid"].notna()
    ]

    print("Candidates with PMCID:", len(candidates))

    top = candidates.head(50)

    full_df = build_fulltext_corpus(top, limit=50)

    full_df.to_csv(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)
    print("Full papers:", len(full_df))

    if len(full_df) > 0:
        print(full_df.head(3)[["pmcid", "year", "title"]])


if __name__ == "__main__":
    main()