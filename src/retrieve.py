import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


INDEX_PATH = "vectorstore/index.faiss"
META_PATH = "vectorstore/metadata.parquet"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_index():
    index = faiss.read_index(INDEX_PATH)
    meta = pd.read_parquet(META_PATH)
    return index, meta


def embed_query(model, query):
    emb = model.encode(
        [query],
        normalize_embeddings=True
    )
    return emb


def search(index, query_vector, k=5):
    scores, ids = index.search(query_vector, k)
    return scores[0], ids[0]


def main():

    index, meta = load_index()

    model = SentenceTransformer(MODEL_NAME)

    query = input("Enter query: ")

    q_emb = embed_query(model, query)

    scores, ids = search(index, q_emb, k=5)

    print("\nTop results:\n")

    seen_titles = set()
    unique_results = []

    for score, idx in zip(scores, ids):
        row = meta.iloc[idx]
        title = row["title"]

        if title in seen_titles:
            continue

        seen_titles.add(title)
        unique_results.append((score, row))

        if len(unique_results) == 5:
            break

    print("\nTop results:\n")

    for rank, (score, row) in enumerate(unique_results, start=1):
        print(f"{rank}. {row['title']}")
        print(f"Score: {score:.3f}")
        print(row["text"][:300])
        print()


if __name__ == "__main__":
    main()