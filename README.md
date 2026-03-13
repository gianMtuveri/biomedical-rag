# Biomedical RAG: Literature Retrieval and Question Answering

A lightweight Retrieval-Augmented Generation (RAG) system for biomedical literature.  
The system retrieves relevant scientific papers from Europe PMC, builds a semantic search index, and answers research questions using retrieved evidence.

This project demonstrates a **full RAG pipeline built from scratch**, including hybrid retrieval, reranking, and local answer generation.

---

# Overview

The system allows users to ask biomedical questions such as:

```
How does LRP1 mediate transport across the blood-brain barrier?
```

The pipeline:

1. retrieves relevant literature
2. ranks evidence passages
3. assembles contextual evidence
4. generates an answer grounded in retrieved papers

---

# Architecture

```
User Question
      │
      ▼
Query Embedding (SentenceTransformers)
      │
      ▼
Hybrid Retrieval
   ├─ FAISS semantic search
   └─ BM25 lexical search
      │
      ▼
Candidate Pool
      │
      ▼
Cross-Encoder Reranking
      │
      ▼
Document-aware Chunk Grouping
      │
      ▼
Context Assembly
      │
      ▼
Answer Generation (FLAN-T5)
```

---

# Features

- Biomedical literature retrieval from **Europe PMC**
- Semantic search using **SentenceTransformers**
- Fast vector search with **FAISS**
- Lexical search with **BM25**
- Hybrid retrieval (semantic + lexical)
- Cross-encoder reranking
- Document-aware chunk grouping
- Local LLM answer synthesis

---

# Project Structure

```
biomedical-rag
│
├── data
│   └── abstracts.csv
│
├── src
│   ├── fetch_corpus.py
│   ├── fetch_full_text.py
│   ├── embed_index.py
│   └── rag_pipeline.py
│
├── vectorstore
│   ├── index.faiss
│   ├── metadata.parquet
│   └── config.json
│
├── requirements.txt
└── README.md
```

---

# Pipeline Components

## 1. Literature Retrieval

`fetch_corpus.py`

Queries Europe PMC to build a biomedical literature corpus.

Features:

- keyword-based search
- filtering for review articles
- abstract extraction
- metadata collection

Output:

```
data/abstracts.csv
```

---

## 2. Embedding and Index Construction

`embed_index.py`

The corpus is split into sentence-based chunks and embedded using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are stored in a **FAISS vector index**.

Output:

```
vectorstore/index.faiss
vectorstore/metadata.parquet
vectorstore/config.json
```

---

## 3. Hybrid Retrieval

`rag_pipeline.py`

Retrieval combines:

### Semantic search
FAISS similarity search over embeddings.

### Lexical search
BM25 ranking over text tokens.

### Hybrid ranking
Scores from both methods are combined to produce candidate documents.

---

## 4. Cross-Encoder Reranking

Candidate passages are reranked using:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This improves precision by scoring each passage with the query.

---

## 5. Context Assembly

Relevant chunks are grouped by document to preserve context.

Multiple chunks per document are merged to produce coherent evidence blocks.

---

## 6. Answer Generation

The final context is passed to a local LLM:

```
google/flan-t5-base
```

The model generates a short answer using only the retrieved evidence.

---

# Example Query

```
Enter question: How does LRP1 mediate transport across the blood-brain barrier?
```

Output:

```
GENERATED ANSWER

LRP1 is a member of the LDL receptor family and functions as a large
endocytic receptor. It binds multiple ligands and mediates their
internalization and trafficking across endothelial cells. In the
blood-brain barrier, LRP1 participates in receptor-mediated
transcytosis, enabling transport of molecules such as amyloid-beta
and therapeutic cargo across brain endothelial cells.
```

Sources are also displayed for transparency.

---

# Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Build the Corpus

Fetch biomedical abstracts:

```bash
python src/fetch_corpus.py
```

---

# Build the Vector Index

```bash
python src/embed_index.py
```

This creates the FAISS index used for retrieval.

---

# Run the RAG Pipeline

```bash
python src/rag_pipeline.py
```

Example interaction:

```
Enter question: What is LRP1?
```

---

# Limitations

- Answer quality depends on the capability of the local LLM.
- Scientific literature often contains complex sections not optimized for question answering.
- Chunking may separate related information across sections.
- Larger instruction-tuned models could improve answer synthesis.

---

# Future Improvements

Possible extensions:

- larger LLMs for answer generation
- citation-aware answer synthesis
- multi-hop retrieval
- domain-specific biomedical embeddings
- improved document chunking strategies

---

# Technologies Used

- Python
- SentenceTransformers
- FAISS
- BM25
- HuggingFace Transformers
- Europe PMC API
- Pandas / NumPy

---

# Purpose

This project demonstrates a **complete Retrieval-Augmented Generation system built from scratch** for biomedical literature exploration.

It is intended as a portfolio project showcasing:

- NLP
- information retrieval
- vector databases
- hybrid search
- LLM-based question answering

---

# License

MIT License
