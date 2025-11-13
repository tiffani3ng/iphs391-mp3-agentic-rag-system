### Planner RAG — Mini-Project #3
A Retrieval-Augmented Generation (RAG) assistant that ingests syllabi, task lists, and calendar exports to act as a personalized scheduler. The agent answers questions like “What should I work on next?” or “What must be done before my vacation?”, surfaces all relevant deadlines, and proposes feasible, balanced plans with transparent citations to the user’s own documents.

---

## Features

* Grounded Q&A over syllabi, task lists, and calendar exports
* Personalized “what to work on next” and pre-vacation planning
* Task- and deadline-aware suggestions with basic time-window reasoning
* Semantic retrieval over small, user-owned corpus (≈10–20 docs, ~100 tasks)
* Local embeddings + lightweight vector store (e.g., ChromaDB)
* Hosted LLM for fluent planning recommendations (GPT-3.5 in MVP)
* Source-aware answers that reference courses/files for traceability
* Privacy-conscious design with option to keep embeddings local

---

## Table of Contents

|                Section | Description                                               | Type       |
| ---------------------: | --------------------------------------------------------- | ---------- |
|     1. Project Context | Define domain, use case, users, success metrics           | Core       |
|  2. Data & Constraints | Specify corpus, formats, budget, privacy limits           | Core       |
|    3. RAG Architecture | Ingestion → chunking → embeddings → retrieval → LLM       | Core       |
|   4. Component Bakeoff | Compare vector DBs, embeddings, LLMs, retrieval           | Annex      |
|          5. Evaluation | Test questions, accuracy, latency, qualitative notes      | Annex      |
| 6. Risks & Future Work | Hallucinations, stale data, scaling, calendar integration | Annex      |
|          7. References | Key RAG/GraphRAG/LightRAG resources and docs              | Supporting |
