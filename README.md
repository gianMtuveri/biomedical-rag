# Biomedical Literature RAG

A lightweight **Retrieval-Augmented Generation (RAG)** system built on biomedical literature.  
The project retrieves scientific abstracts from Europe PMC, builds a semantic search index, and answers user questions using retrieved evidence from the literature.

The system combines **dense vector retrieval, sparse keyword retrieval, cross-encoder reranking, and grounded answer synthesis** to produce responses supported by relevant scientific publications.

---

# Project Overview

Modern information retrieval systems increasingly rely on **retrieval-augmented pipelines** that combine machine learning with structured document retrieval.

This project implements a small but complete RAG architecture designed for biomedical literature exploration.

The pipeline performs the following steps:

1. Retrieve biomedical abstracts from the Europe PMC API.
2. Split documents into smaller chunks.
3. Generate semantic embeddings using SentenceTransformers.
4. Store embeddings in a FAISS vector index.
5. Retrieve relevant passages using both dense and sparse retrieval.
6. Rerank candidate passages using a cross-encoder model.
7. Produce a grounded answer template based on retrieved evidence.

The system runs **entirely locally** and does not require external LLM APIs.

---

# Architecture

The retrieval pipeline follows this structure:


Europe PMC API  
↓  
Corpus construction  
↓  
Document chunking  
↓  
SentenceTransformer embeddings  
↓  
FAISS vector index  
↓  
Hybrid retrieval (FAISS + BM25)  
↓  
Cross-encoder reranking  
↓  
Evidence synthesis  


This architecture is representative of many modern RAG systems used in production search and knowledge systems.

---

# Repository Structure


biomedical-rag  
│  
├── src  
│ ├── fetch_corpus.py  
│ ├── embed_index.py  
│ ├── retrieve.py  
│ └── rag_pipeline.py  
│  
├── data  
│ └── .gitkeep  
│  
├── vectorstore  
│ └── .gitkeep  
│  
├── README.md  
└── .gitignore  
  

---

# Installation

Create a Python environment:


python -m venv .venv
source .venv/bin/activate


Install dependencies:


pip install pandas numpy requests
pip install sentence-transformers
pip install faiss-cpu
pip install rank-bm25


---

# Building the Corpus

Download biomedical abstracts from Europe PMC.


python src/fetch_corpus.py


This creates:


data/abstracts.csv


The dataset contains scientific abstracts related to topics such as:

- blood-brain barrier transport
- receptor-mediated transcytosis
- CNS drug delivery
- nanoparticle-based delivery systems

---

# Building the Vector Index

Generate embeddings and create the FAISS index.


python src/embed_index.py


Outputs:


vectorstore/index.faiss
vectorstore/metadata.parquet
vectorstore/config.json


Documents are chunked before embedding to improve retrieval granularity.

---

# Running the RAG Pipeline

Launch the interactive question-answer system.


python src/rag_pipeline.py


Example query:


How does LRP1 mediate transport across the blood-brain barrier?


Example output:


GENERATED ANSWER

Question: How does LRP1 mediate transport across the blood-brain barrier?

Answer based on retrieved literature:

LRP1 mediates bidirectional transcytosis of amyloid-β across the blood-brain barrier (2011)

Efficient transfer of receptor-associated protein across the blood-brain barrier (2004)

Interaction between intravenous immunoglobulin and LRP1 (2012)


The system reports evidence extracted from the most relevant publications.

---

# Retrieval Strategy

The retriever combines multiple ranking approaches.

## Dense Retrieval

Semantic similarity search using:


sentence-transformers/all-MiniLM-L6-v2


Embeddings are indexed with FAISS.

---

## Sparse Retrieval

Keyword-based retrieval using **BM25**.

This improves recall for queries containing specific biomedical terms.

---

## Hybrid Retrieval

Dense and sparse results are merged to improve coverage and precision.

---

## Cross-Encoder Reranking

Candidate passages are reranked using:


cross-encoder/ms-marco-MiniLM-L-6-v2


This model evaluates each query-document pair to improve ranking quality.

---

# Answer Generation

The final answer is generated using a **grounded synthesis template**.

Instead of hallucinating information, the system:

1. extracts relevant evidence passages
2. summarizes them into a structured response
3. cites the source publications

This approach ensures answers remain tied to the retrieved literature.

---

# Models Used

Embedding model:


sentence-transformers/all-MiniLM-L6-v2


Reranking model:


cross-encoder/ms-marco-MiniLM-L-6-v2


Both models are lightweight and run locally.

---

# Limitations

This project is intended as a **lightweight research prototype**.

Limitations include:

- answer synthesis uses template summarization rather than a generative LLM
- retrieval corpus is limited to several thousand abstracts
- the system does not perform advanced query expansion or citation grounding

Nevertheless, the architecture reflects modern RAG pipelines used in production retrieval systems.

---

# Possible Improvements

Future extensions could include:

- larger biomedical corpora
- query expansion techniques
- LLM-based answer synthesis
- citation-aware generation
- document-level retrieval and ranking
- evaluation benchmarks for retrieval quality

---

# Purpose of the Project

This repository demonstrates how to build a **complete retrieval pipeline** combining:

- API-based corpus collection
- vector embedding models
- FAISS similarity search
- BM25 keyword retrieval
- cross-encoder reranking
- grounded answer generation
# Biomedical Literature RAG

A lightweight **Retrieval-Augmented Generation (RAG)** system built on biomedical literature.  
The project retrieves scientific abstracts from Europe PMC, builds a semantic search index, and answers user questions using retrieved evidence from the literature.

The system combines **dense vector retrieval, sparse keyword retrieval, cross-encoder reranking, and grounded answer synthesis** to produce responses supported by relevant scientific publications.

---

# Project Overview

Modern information retrieval systems increasingly rely on **retrieval-augmented pipelines** that combine machine learning with structured document retrieval.

This project implements a small but complete RAG architecture designed for biomedical literature exploration.

The pipeline performs the following steps:

1. Retrieve biomedical abstracts from the Europe PMC API.
2. Split documents into smaller chunks.
3. Generate semantic embeddings using SentenceTransformers.
4. Store embeddings in a FAISS vector index.
5. Retrieve relevant passages using both dense and sparse retrieval.
6. Rerank candidate passages using a cross-encoder model.
7. Produce a grounded answer template based on retrieved evidence.

The system runs **entirely locally** and does not require external LLM APIs.

---

# Architecture

The retrieval pipeline follows this structure:


Europe PMC API
↓
Corpus construction
↓
Document chunking
↓
SentenceTransformer embeddings
↓
FAISS vector index
↓
Hybrid retrieval (FAISS + BM25)
↓
Cross-encoder reranking
↓
Evidence synthesis


This architecture is representative of many modern RAG systems used in production search and knowledge systems.

---

# Repository Structure


biomedical-rag
│
├── src
│ ├── fetch_corpus.py
│ ├── embed_index.py
│ ├── retrieve.py
│ └── rag_pipeline.py
│
├── data
│ └── .gitkeep
│
├── vectorstore
│ └── .gitkeep
│
├── README.md
└── .gitignore


---

# Installation

Create a Python environment:


python -m venv .venv
source .venv/bin/activate


Install dependencies:


pip install pandas numpy requests
pip install sentence-transformers
pip install faiss-cpu
pip install rank-bm25


---

# Building the Corpus

Download biomedical abstracts from Europe PMC.


python src/fetch_corpus.py


This creates:


data/abstracts.csv


The dataset contains scientific abstracts related to topics such as:

- blood-brain barrier transport
- receptor-mediated transcytosis
- CNS drug delivery
- nanoparticle-based delivery systems

---

# Building the Vector Index

Generate embeddings and create the FAISS index.


python src/embed_index.py


Outputs:


vectorstore/index.faiss
vectorstore/metadata.parquet
vectorstore/config.json


Documents are chunked before embedding to improve retrieval granularity.

---

# Running the RAG Pipeline

Launch the interactive question-answer system.


python src/rag_pipeline.py


Example query:


How does LRP1 mediate transport across the blood-brain barrier?


Example output:


GENERATED ANSWER

Question: How does LRP1 mediate transport across the blood-brain barrier?

Answer based on retrieved literature:

LRP1 mediates bidirectional transcytosis of amyloid-β across the blood-brain barrier (2011)

Efficient transfer of receptor-associated protein across the blood-brain barrier (2004)

Interaction between intravenous immunoglobulin and LRP1 (2012)


The system reports evidence extracted from the most relevant publications.

---

# Retrieval Strategy

The retriever combines multiple ranking approaches.

## Dense Retrieval

Semantic similarity search using:


sentence-transformers/all-MiniLM-L6-v2


Embeddings are indexed with FAISS.

---

## Sparse Retrieval

Keyword-based retrieval using **BM25**.

This improves recall for queries containing specific biomedical terms.

---

## Hybrid Retrieval

Dense and sparse results are merged to improve coverage and precision.

---

## Cross-Encoder Reranking

Candidate passages are reranked using:


cross-encoder/ms-marco-MiniLM-L-6-v2


This model evaluates each query-document pair to improve ranking quality.

---

# Answer Generation

The final answer is generated using a **grounded synthesis template**.

Instead of hallucinating information, the system:

1. extracts relevant evidence passages
2. summarizes them into a structured response
3. cites the source publications

This approach ensures answers remain tied to the retrieved literature.

---

# Models Used

Embedding model:


sentence-transformers/all-MiniLM-L6-v2


Reranking model:


cross-encoder/ms-marco-MiniLM-L-6-v2


Both models are lightweight and run locally.

---

# Limitations

This project is intended as a **lightweight research prototype**.

Limitations include:

- answer synthesis uses template summarization rather than a generative LLM
- retrieval corpus is limited to several thousand abstracts
- the system does not perform advanced query expansion or citation grounding

Nevertheless, the architecture reflects modern RAG pipelines used in production retrieval systems.

---

# Possible Improvements

Future extensions could include:

- larger biomedical corpora
- query expansion techniques
- LLM-based answer synthesis
- citation-aware generation
- document-level retrieval and ranking
- evaluation benchmarks for retrieval quality

---

# Purpose of the Project

This repository demonstrates how to build a **complete retrieval pipeline** combining:

- API-based corpus collection
- vector embedding models
- FAISS similarity search
- BM25 keyword retrieval
- cross-encoder reranking
- grounded answer generation
# Biomedical Literature RAG

A lightweight **Retrieval-Augmented Generation (RAG)** system built on biomedical literature.  
The project retrieves scientific abstracts from Europe PMC, builds a semantic search index, and answers user questions using retrieved evidence from the literature.

The system combines **dense vector retrieval, sparse keyword retrieval, cross-encoder reranking, and grounded answer synthesis** to produce responses supported by relevant scientific publications.

---

# Project Overview

Modern information retrieval systems increasingly rely on **retrieval-augmented pipelines** that combine machine learning with structured document retrieval.

This project implements a small but complete RAG architecture designed for biomedical literature exploration.

The pipeline performs the following steps:

1. Retrieve biomedical abstracts from the Europe PMC API.
2. Split documents into smaller chunks.
3. Generate semantic embeddings using SentenceTransformers.
4. Store embeddings in a FAISS vector index.
5. Retrieve relevant passages using both dense and sparse retrieval.
6. Rerank candidate passages using a cross-encoder model.
7. Produce a grounded answer template based on retrieved evidence.

The system runs **entirely locally** and does not require external LLM APIs.

---

# Architecture

The retrieval pipeline follows this structure:


Europe PMC API
↓
Corpus construction
↓
Document chunking
↓
SentenceTransformer embeddings
↓
FAISS vector index
↓
Hybrid retrieval (FAISS + BM25)
↓
Cross-encoder reranking
↓
Evidence synthesis


This architecture is representative of many modern RAG systems used in production search and knowledge systems.

---

# Repository Structure


biomedical-rag
│
├── src
│ ├── fetch_corpus.py
│ ├── embed_index.py
│ ├── retrieve.py
│ └── rag_pipeline.py
│
├── data
│ └── .gitkeep
│
├── vectorstore
│ └── .gitkeep
│
├── README.md
└── .gitignore


---

# Installation

Create a Python environment:


python -m venv .venv
source .venv/bin/activate


Install dependencies:


pip install pandas numpy requests
pip install sentence-transformers
pip install faiss-cpu
pip install rank-bm25


---

# Building the Corpus

Download biomedical abstracts from Europe PMC.


python src/fetch_corpus.py


This creates:


data/abstracts.csv


The dataset contains scientific abstracts related to topics such as:

- blood-brain barrier transport
- receptor-mediated transcytosis
- CNS drug delivery
- nanoparticle-based delivery systems

---

# Building the Vector Index

Generate embeddings and create the FAISS index.


python src/embed_index.py


Outputs:


vectorstore/index.faiss
vectorstore/metadata.parquet
vectorstore/config.json


Documents are chunked before embedding to improve retrieval granularity.

---

# Running the RAG Pipeline

Launch the interactive question-answer system.


python src/rag_pipeline.py


Example query:


How does LRP1 mediate transport across the blood-brain barrier?


Example output:


GENERATED ANSWER

Question: How does LRP1 mediate transport across the blood-brain barrier?

Answer based on retrieved literature:

LRP1 mediates bidirectional transcytosis of amyloid-β across the blood-brain barrier (2011)

Efficient transfer of receptor-associated protein across the blood-brain barrier (2004)

Interaction between intravenous immunoglobulin and LRP1 (2012)


The system reports evidence extracted from the most relevant publications.

---

# Retrieval Strategy

The retriever combines multiple ranking approaches.

## Dense Retrieval

Semantic similarity search using:


sentence-transformers/all-MiniLM-L6-v2


Embeddings are indexed with FAISS.

---

## Sparse Retrieval

Keyword-based retrieval using **BM25**.

This improves recall for queries containing specific biomedical terms.

---

## Hybrid Retrieval

Dense and sparse results are merged to improve coverage and precision.

---

## Cross-Encoder Reranking

Candidate passages are reranked using:


cross-encoder/ms-marco-MiniLM-L-6-v2


This model evaluates each query-document pair to improve ranking quality.

---

# Answer Generation

The final answer is generated using a **grounded synthesis template**.

Instead of hallucinating information, the system:

1. extracts relevant evidence passages
2. summarizes them into a structured response
3. cites the source publications

This approach ensures answers remain tied to the retrieved literature.

---

# Models Used

Embedding model:


sentence-transformers/all-MiniLM-L6-v2


Reranking model:


cross-encoder/ms-marco-MiniLM-L-6-v2


Both models are lightweight and run locally.

---

# Limitations

This project is intended as a **lightweight research prototype**.

Limitations include:

- answer synthesis uses template summarization rather than a generative LLM
- retrieval corpus is limited to several thousand abstracts
- the system does not perform advanced query expansion or citation grounding

Nevertheless, the architecture reflects modern RAG pipelines used in production retrieval systems.

---

# Possible Improvements

Future extensions could include:

- larger biomedical corpora
- query expansion techniques
- LLM-based answer synthesis
- citation-aware generation
- document-level retrieval and ranking
- evaluation benchmarks for retrieval quality

---

# Purpose of the Project

This repository demonstrates how to build a **complete retrieval pipeline** combining:

- API-based corpus collection
- vector embedding models
- FAISS similarity search
- BM25 keyword retrieval
- cross-encoder reranking
- grounded answer generation

The project is designed as a **learning exercise in modern information retrieval and RAG architectures**.
