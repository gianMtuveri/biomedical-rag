# Biomedical Literature RAG

A Retrieval-Augmented Generation (RAG) pipeline for answering biomedical questions using scientific literature.

This project builds a local knowledge base from biomedical abstracts and implements a hybrid retrieval system combining semantic embeddings, keyword search, reranking, and answer synthesis.

The goal is to demonstrate a full modern RAG architecture applied to biomedical research questions.

---

# Project Overview

The system retrieves relevant scientific evidence from a corpus of biomedical papers and synthesizes a concise answer using a lightweight language model.

Pipeline stages:

1. Corpus ingestion from Europe PMC  
2. Document chunking and embedding  
3. Vector indexing with FAISS  
4. Hybrid retrieval (semantic + keyword)  
5. Cross-encoder reranking  
6. Answer synthesis using an instruction-tuned model  

---

# Architecture

```
fetch_corpus.py
        │
        ▼
abstracts.csv
        │
        ▼
embed_index.py
        │
        ▼
vectorstore/
    index.faiss
    metadata.parquet
        │
        ▼
rag_pipeline.py
        │
        ▼
Hybrid retrieval
(FAISS + BM25)
        │
        ▼
Cross-encoder reranking
        │
        ▼
Answer synthesis
```

---

# Repository Structure

```
biomedical-rag
│
├── data
│   └── abstracts.csv
│
├── vectorstore
│   ├── index.faiss
│   ├── metadata.parquet
│   └── config.json
│
├── src
│   ├── fetch_corpus.py
│   ├── embed_index.py
│   └── rag_pipeline.py
│
├── requirements.txt
└── README.md
```

---

# Installation

Create a Python virtual environment and install dependencies.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required libraries include:

- sentence-transformers  
- faiss  
- rank-bm25  
- transformers  
- pandas  
- numpy  

---

# Building the Corpus

Fetch biomedical abstracts from Europe PMC.

```
python src/fetch_corpus.py
```

This generates:

```
data/abstracts.csv
```

---

# Building the Vector Index

Create document chunks, compute embeddings, and build the FAISS index.

```
python src/embed_index.py
```

Outputs:

```
vectorstore/index.faiss
vectorstore/metadata.parquet
vectorstore/config.json
```

---

# Running the RAG System

Launch the question-answering pipeline:

```
python src/rag_pipeline.py
```

Example:

```
Enter question: How does LRP1 mediate transport across the blood-brain barrier?
```

The system will:

1. retrieve candidate chunks  
2. perform hybrid search (FAISS + BM25)  
3. rerank results using a cross-encoder  
4. synthesize a final answer from the retrieved evidence  

---

# Retrieval Pipeline

The retrieval system combines multiple techniques.

## Semantic Search

Embeddings generated using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Indexed with FAISS using cosine similarity.

---

## Keyword Retrieval

BM25 retrieval is used to capture exact keyword matches in documents.

---

## Hybrid Retrieval

Semantic and BM25 scores are combined to improve recall and robustness.

---

## Cross-Encoder Reranking

Top candidate passages are reranked using:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This improves ranking precision before answer generation.

---

# Answer Generation

Answer synthesis currently uses the instruction-tuned model:

```
google/flan-t5-base
```

The model receives the top reranked evidence chunks and produces a concise biomedical explanation.

---

# Example Output

Question:

```
How does LRP1 mediate transport across the blood-brain barrier?
```

Generated answer:

```
LRP1 mediates receptor-mediated transcytosis across the blood–brain barrier by binding ligands on endothelial cells, internalizing them through endocytosis, and transporting them across the cell in vesicular compartments. Experimental studies suggest that proteins such as amyloid-β and receptor-associated protein can utilize this pathway for transport. This mechanism has also been explored for targeted drug delivery to the central nervous system.
```

---

# Current Limitations

This implementation demonstrates the architecture of a modern RAG pipeline but still has several limitations.

## Chunk-level retrieval

The system retrieves fixed-size text chunks. In many scientific papers, important information is distributed across distant sections of the document. A single chunk may therefore contain incomplete context.

## Document deduplication

Current deduplication keeps only one chunk per document, which may discard relevant evidence from the same paper.

## Lightweight generation model

Answer synthesis currently uses a small model (`flan-t5-base`). While efficient, it may struggle with complex scientific reasoning.

## Abstract-only corpus

The corpus currently contains abstracts rather than full papers, limiting the available context.

---

# Planned Improvements

Future improvements include:

- multi-chunk aggregation per document  
- parent–child retrieval architecture  
- context window expansion around retrieved chunks  
- stronger instruction-tuned generation models  
- evaluation with biomedical QA benchmarks  

---

# Purpose

This project was developed as a hands-on exploration of modern Retrieval-Augmented Generation architectures applied to biomedical literature.

It demonstrates practical implementation of:

- semantic retrieval  
- hybrid search  
- cross-encoder reranking  
- answer synthesis  

within a compact and reproducible pipeline.

---

# License

MIT License
