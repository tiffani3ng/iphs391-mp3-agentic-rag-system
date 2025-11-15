### Planner RAG — Mini-Project #3
A Retrieval-Augmented Generation (RAG) assistant that ingests syllabi, task lists, and calendar exports to act as a dynamic personalized scheduler. The agent answers questions like “I have 3 hours between classes today, how should I allocate my time?” or “What do I have to get done before my vacation in two weeks?”, surfaces all relevant deadlines, and proposes feasible, balanced plans that adjust according to task completion.

---

## Features

* Grounded Q&A over syllabi, task lists, and calendar exports
* Personalized hour-by-hour/day-by-day planning
* Task- and deadline-aware suggestions with basic time-window reasoning
* Semantic retrieval over small, user-owned corpus (≈10–20 docs, ~100 tasks)
* Local embeddings + lightweight vector store (e.g., ChromaDB)
* Hosted LLM for fluent planning recommendations (GPT-3.5 in MVP)

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
