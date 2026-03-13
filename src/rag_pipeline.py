import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INDEX_PATH = "vectorstore/index.faiss"
META_PATH = "vectorstore/metadata.parquet"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_index():
    index = faiss.read_index(INDEX_PATH)
    meta = pd.read_parquet(META_PATH)
    return index, meta


def embed_query(model, query):
    emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return emb


def search(index, query_vector, k=10):
    scores, ids = index.search(query_vector, k)
    return scores[0], ids[0]


def build_bm25(meta):
    corpus = meta["text"].tolist()
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def bm25_search(bm25, tokenized_corpus, query, k=10):
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(
        enumerate(scores),
        key = lambda x: x[1],
        reverse = True
    )
    ids = [i for i, _ in ranked[:k]]
    scores = [s for _, s in ranked[:k]]
    return scores, ids


def hybrid_search(index, bm25, tokenized_corpus, meta, query, model, k=10):

    q_emb = embed_query(model, query)

    faiss_scores, faiss_ids = search(index, q_emb, k)

    bm25_scores, bm25_ids = bm25_search(bm25, tokenized_corpus, query, k)

    combined = {}

    for score, idx in zip(faiss_scores, faiss_ids):
        combined[int(idx)] = combined.get(int(idx), 0) + float(score)

    for score, idx in zip(bm25_scores, bm25_ids):
        combined[int(idx)] = combined.get(int(idx), 0) + float(score)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    final_ids = [idx for idx, _ in ranked[:k]]
    final_scores = [score for _, score in ranked[:k]]

    return final_scores, final_ids


def rerank_results(query, results, reranker, top_n=5):
    """
    Rerank retrieved results using a cross-encoder.

    Inputs:
    - query: user question
    - results: list of (score, row) pairs
    - reranker: CrossEncoder model
    - top_n: number of final results to keep

    Output:
    - reranked list of (rerank_score, row)
    """
    pairs = [(query, row["text"]) for _, row in results]

    rerank_scores = reranker.predict(pairs)

    reranked = list(zip(rerank_scores, [row for _, row in results]))
    reranked = sorted(reranked, key=lambda x: x[0], reverse=True)

    return reranked[:top_n]


def synthesize_answer(query, reranked_results, tokenizer, gen_model):
    docs = {}

    for score, row in reranked_results:
        doc_id = row["id"]

        if doc_id not in docs:
            docs[doc_id] = {
                "title": row["title"],
                "year": row["year"],
                "chunks": []
            }

        docs[doc_id]["chunks"].append(row["text"][:250])

    context_blocks = []

    for i, (_, doc) in enumerate(docs.items(), start=1):
        merged_chunks = "\n".join(doc["chunks"])
        block = (
            f"Source {i}\n"
            f"Title: {doc['title']}\n"
            f"Year: {doc['year']}\n"
            f"Evidence:\n{merged_chunks}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)
    #print(context)
    
    prompt = f"""
You are a biomedical research assistant.

Answer the question using only the evidence below.

If the evidence is incomplete, say so explicitly.


Question:
{query}

Evidence:
{context}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    output = gen_model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


def group_chunks_by_document(scores, ids, meta, max_docs=5, max_chunks_per_doc=3):

    doc_chunks = {}

    for score, idx in zip(scores, ids):

        row = meta.iloc[idx]
        doc_id = row["id"]

        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []

        doc_chunks[doc_id].append((score, row))

    # rank documents by best chunk score
    ranked_docs = sorted(
        doc_chunks.items(),
        key=lambda x: max(s for s, _ in x[1]),
        reverse=True
    )

    selected = []

    for doc_id, chunks in ranked_docs[:max_docs]:

        chunks = sorted(chunks, key=lambda x: x[0], reverse=True)

        for score, row in chunks[:max_chunks_per_doc]:
            selected.append((score, row))

    return selected


def expand_with_neighbors(selected_results, meta, window=1):
    expanded = []
    seen = set()

    for score, row in selected_results:
        doc_id = row["id"]
        chunk_id = row["chunk_id"]

        mask = (
            (meta["id"] == doc_id) &
            (meta["chunk_id"] >= chunk_id - window) &
            (meta["chunk_id"] <= chunk_id + window)
        )

        neighbors = meta[mask].sort_values("chunk_id")

        for _, nrow in neighbors.iterrows():

            key = (nrow["id"], nrow["chunk_id"])

            if key in seen:
                continue

            seen.add(key)
            expanded.append((score, nrow))

    return expanded


def main():
    index, meta = load_index()
    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    bm25, tokenized_corpus = build_bm25(meta)

    query = input("Enter question: ")

    scores, ids = hybrid_search(
        index,
        bm25,
        tokenized_corpus,
        meta,
        query,
        model, 
        k=10
    )

    results = group_chunks_by_document(
        scores,
        ids,
        meta,
        max_docs=3,
        max_chunks_per_doc=3
    )

    reranked_results = rerank_results(query, results, reranker, top_n=3)
    expanded_results = expand_with_neighbors(reranked_results, meta, window=1)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

    answer = synthesize_answer(query, expanded_results, tokenizer, gen_model)

    print("\n=================================")
    print("GENERATED ANSWER")
    print("=================================\n")

    print(answer)

    print("\nSources:\n")
    for i, (_, row) in enumerate(reranked_results, start = 1):
        print(f"{i}. {row['title']} ({row['year']})")

    
if __name__ == "__main__":
    main()