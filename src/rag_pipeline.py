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


def deduplicate_results(scores, ids, meta, top_n):
    seen_titles = set()
    unique_results = []

    for score, idx in zip(scores, ids):
        row = meta.iloc[idx]
        title = row["title"]

        if title in seen_titles:
            continue

        seen_titles.add(title)
        unique_results.append((score,row))

        if len(unique_results) == top_n:
            break

    return unique_results


def build_context(results):
    context_parts = []

    for i, (score, row) in enumerate(results, start=1):
        block = (
            f"Source: {i}\n"
            f"Title: {row['title']}\n"
            f"Year: {row['year']}\n"
            f"Journal: {row['journal']}\n"
            f"Snippet: {row['text']}\n"
        )
        context_parts.append(block)

    return "\n\n".join(context_parts)


def generate_answer_template(query, results):
    lines = []
    lines.append("Answer")
    lines.append("")
    lines.append(f"Question: {query}")
    lines.append("")
    lines.append("Retrieved evidence suggests the following:")
    lines.append("")

    for i, (score, row) in enumerate(results, start=1):
        snippet = row["text"][:400].replace("\n", " ")
        lines.append(f"{i}. {row['title']} ({row['year']})")
        lines.append(f"   Relevance score: {score:.3f}")
        lines.append(f"   Evidence: {snippet}...")
        lines.append("")

    lines.append("Sources")
    lines.append("")
    for i, (_, row) in enumerate(results, start=1):
        lines.append(f"{i}. {row['title']}")

    return "\n".join(lines)


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

    context_blocks = []

    for score, row in reranked_results:
        context_blocks.append(row["text"][:600])

    context = "\n\n".join(context_blocks)

    prompt = f"""
Use the evidence below to answer the biomedical question.

Focus on:
- the mechanism
- what the retrieved studies suggest

If the evidence is incomplete, say so.

Question:
{query}

Evidence:
{context}

Write a concise scientific answer (4-6 sentences).
"""
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024
    )

    output = gen_model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer


def main():
    index, meta = load_index()
    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    bm25, tokenized_corpus = build_bm25(meta)

    query = input("Enter question: ")

    q_emb = embed_query(model, query)
    scores, ids = hybrid_search(
        index,
        bm25,
        tokenized_corpus,
        meta,
        query,
        model, 
        k=25
    )

    results = deduplicate_results(scores, ids, meta, top_n=15)

    reranked_results = rerank_results(query, results, reranker, top_n=4)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    answer = synthesize_answer(query, reranked_results, tokenizer, gen_model)

    print("\n=================================")
    print("GENERATED ANSWER")
    print("=================================\n")

    print(answer)

    print("\nSources:\n")

    for i, (_, row) in enumerate(reranked_results, start = 1):
        print(f"{i}. {row['title']} ({row['year']})")

    
if __name__ == "__main__":
    main()