# Tesla-RAG: Evaluation-Driven RAG Optimization on Financial Filings

A question-answering RAG system over Tesla's IR reports (earnings filings), improved incrementally across 8 versions. Each version changes exactly one improvement lever, and the effect is measured and recorded.

> **Core principles**: One change per version. Measure everything. Data quality before algorithms.

---

## Accuracy Progression

| Version | Change | Exact Match | Delta | Relevancy | Faithfulness | Completeness |
|---------|--------|-------------|-------|-----------|--------------|--------------|
| **V1** | Vector search baseline | 10% (1/10) | — | 4.7 | 5.0 | 3.3 |
| **V2** | + BM25 hybrid search (RRF) | 30% (3/10) | +20pp | 4.9 | 5.0 | 4.2 |
| **V3** | + Table-aware chunking | 60% (6/10) | +30pp | 5.0 | 5.0 | 4.5 |
| **V4** | + Cross-Encoder reranking | 80% (8/10) | +20pp | 5.0 | 5.0 | 4.8 |
| **V5** | + CRAG (self-correction loop) | 80% (8/10) | +0pp | 5.0 | 5.0 | 4.7 |
| **V6** | FastAPI service | — | — | — | — | — |
| **V7** | pytest (21 tests, bug found) | — | — | — | — | — |
| **V8** | Dockerization | — | — | — | — | — |

**Evaluation method**: A 10-question financial Q&A dataset. Exact-match scoring plus LLM-as-a-Judge on three metrics (Relevancy / Faithfulness / Completeness, each scored 1–5), fully automated.

---

## Architecture (V8, final)

```
┌──────────────────────────────────────────────────────────────┐
│                       Docker container                        │
│                                                               │
│  PDF ──→ table/text detection ──→ chunking ──→ ChromaDB      │
│          (is_table_page())         tables: 2,500 chars       │
│                                    text:   1,000 chars       │
│                                                               │
│  Query ─┬─→ vector search (top_k×2) ──┐                      │
│         └─→ BM25 search (top_k×2) ────┤                      │
│                                        ▼                      │
│                             RRF fusion (k=60)                 │
│                          + table boost (×1.2)                 │
│                                        │                      │
│                                        ▼                      │
│                        Cross-Encoder reranker                 │
│                       (15 candidates → top 5)                 │
│                                        │                      │
│                                        ▼                      │
│                          CRAG quality gate                    │
│                  (CORRECT / AMBIGUOUS / INCORRECT)            │
│                        ↓                  ↓                   │
│                      pass       query rewrite → re-search     │
│                        ↓              (max 1 retry)           │
│                                        │                      │
│                                        ▼                      │
│                        Claude API ──→ answer                  │
│                                                               │
│  FastAPI: POST /query, GET /health                            │
│  pytest: 21 tests (5 mocked, 16 integration)                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Version History

### V1: Baseline — vector search only (10%)
**Goal**: Build the RAG pipeline from scratch, without LangChain, to understand every layer deeply enough to debug it.
**Result**: Faithfulness was a perfect 5.0 — the LLM produced zero hallucinations. But exact match was 1/10. Vector search failed to retrieve financial-data chunks, returning "Supercharger statistics" for questions about revenue.
**Insight**: The bottleneck was retrieval, not generation.

### V2: Hybrid search — BM25 + RRF (30%)
**Problem**: Vector search alone misses keyword-dependent financial terms ("revenue", etc.).
**Solution**: Added BM25 keyword search and fused scores with Reciprocal Rank Fusion (k=60).
**Why RRF**: RRF fuses by rank rather than raw score, so the scale mismatch between BM25 scores and cosine similarity never needs normalizing. Three lines of code, zero hyperparameters.
**Result**: +20pp. BM25's keyword matching reliably surfaced the revenue chunks.

### V3: Table-aware chunking (60%) ⭐
**Problem**: Of the 7 remaining failures, 3 had the answer present in ChromaDB but split across chunk boundaries — table labels separated from their values.
**Original plan**: V3 was supposed to be CRAG (Active RAG). Failure analysis changed the priority.
**Solution**: Rule-based table-page detection (`is_table_page()`), with a larger chunk size for tables (2,500 vs. 1,000 chars) to keep each table in a single chunk.
**Result**: +30pp — even though chunk count *dropped* 22% (144 → 112). **Proof that data quality beats algorithms**: the largest single accuracy gain in the project came from fixing the data, not the pipeline.

### V4: Cross-Encoder reranking (80%)
**Problem**: Even with hybrid search, noisy chunks (semantically close but irrelevant) leaked into context.
**Solution**: Two-stage retrieval. RRF collects 15 candidates → `cross-encoder/ms-marco-MiniLM-L-6-v2` narrows to 5. A cross-encoder scores query and chunk jointly, giving far more precise relevance judgments than a bi-encoder.
**Result**: +20pp. Also found and fixed a hardcoded bug in the debugging tool itself (`search_chunks.py`).

### V5: CRAG — Corrective RAG (80%)
**Problem**: The pipeline had no self-correction for retrieval quality.
**Solution**: An LLM grades retrieval quality (CORRECT / AMBIGUOUS / INCORRECT); on failure, it rewrites the query and re-searches.
**Result**: All 10 questions graded CORRECT, zero retries triggered. **This is evidence that V4's reranker was already returning sufficient-quality chunks** — a null result that validates the previous stage. CRAG stays in as a safety net for unseen queries.
**Design decision**: On parse failure, default to INCORRECT (trigger the retry). One extra API call costs less than answering from insufficient context.

### V6: FastAPI service
Exposed the pipeline as `POST /query` (4 retrieval modes) and `GET /health`. Zero changes to core logic — the modular design from V2 paid off.

### V7: pytest — 21 automated tests
- **Found a critical bug**: the CRAG quality check used substring matching, so "CORRECT" inside "INCORRECT" caused misclassification. Fixed the evaluation order.
- CRAG tests mock the Claude API (no key required, fast, reproducible).
- API tests run the real pipeline through FastAPI's `TestClient`.

### V8: Dockerization
The full pipeline runs with just `docker build && docker run`. The clean Docker build exposed a missing `rank-bm25` entry in requirements.txt — a dependency gap that the local conda environment had been silently hiding.

---

## Key Design Decisions

### 1. No LangChain
LangChain's `EnsembleRetriever` would have delivered hybrid search in minutes. Building it by hand was a deliberate choice: to understand each layer well enough to debug it and explain every design trade-off. Slower to build, far easier to diagnose.

### 2. Reprioritizing V3 — data quality over algorithms
The original roadmap had V3 = CRAG. Failure analysis showed the root cause was chunk boundaries splitting tables — and re-searching with CRAG would only re-retrieve the same broken chunks. Fixing data quality first was the single highest-impact decision in the project (+30pp).

### 3. Independent modules
Every major capability (hybrid_search.py, reranker.py, crag.py) is a standalone module, switched by `query.py` via a `--mode` flag:
```bash
python src/query.py --mode vector   # V1
python src/query.py --mode hybrid   # V2-V3
python src/query.py --mode rerank   # V4
python src/query.py --mode crag     # V5
```
Every version remains A/B-testable with zero code changes.

### 4. Fail-safe defaults
When the CRAG quality check fails to parse, it defaults to INCORRECT (retry). In production RAG, a wrong answer is far more dangerous than a slow one.

---

## Lessons Learned

### Technical
- **Two-stage retrieval is the industry standard**: fast recall (BM25 + vector) → precise reranking (cross-encoder). Azure AI Search's Semantic Ranker uses the same structure internally.
- **Evaluation metrics locate the bottleneck**: low Faithfulness → generation problem; low Completeness → retrieval problem. V1's profile (Faithfulness 5.0, Completeness 3.3) pinpointed retrieval immediately.
- **A null result is still a result**: V5's zero retries doesn't mean CRAG was pointless — it quantifies how good V4's pipeline already was.

### Process
- **One change per version**: every accuracy gain is attributable to a specific improvement.
- **Debugging tools have bugs too**: V4 uncovered a hardcoded keyword in the debug script. The tool for finding bugs was itself buggy.
- **Docker is a dependency auditor**: the V8 clean build surfaced a requirements.txt gap that the local environment had masked.

---

## Limitations & Next Steps

Honest evaluation includes evaluating the evaluation itself:

- **The evaluation set is small (10 questions).** It covers the core financial query types (revenue, margins, EPS, EBITDA, FCF, segment breakdowns) but is too small to make strong claims about generalization. Scores on a set this size carry wide confidence intervals.
- **Exact-match scoring penalized formatting, not correctness.** One V4/V5 failure (Q8) contained the correct figures ($28,095M → $24,901M, decreased) but failed the string match due to formatting variance. Rather than adjusting the score upward, I'm treating this as a finding about metric design: exact match is a poor fit for numeric financial answers, and a normalization layer or judge-based scoring is the right fix.
- **One failure is a structural limit of text-based RAG.** Q6's answer sits inside a 12-quarter reconciliation table — a case that likely needs table-native retrieval rather than better text chunking.
- **In progress**: this project is the first case study for [kaibo](https://github.com/), my open-source RAG/agent evaluation framework. The evaluation set is being expanded (target: ~38 questions) and the system will be re-scored against it — I expect headline numbers to drop, and that's the point.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Vector DB | ChromaDB (persistent mode) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| BM25 | rank_bm25 (BM25Okapi) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Claude API (claude-sonnet-4-20250514) |
| API | FastAPI + Uvicorn |
| Testing | pytest + unittest.mock |
| Container | Docker (python:3.11-slim) |
| Score fusion | Reciprocal Rank Fusion (k=60) |

---

## Quick Start

### Docker (recommended)
```bash
docker build -t tesla-rag .
docker run -d -p 8000:8000 --env-file .env tesla-rag

# Health check
curl http://localhost:8000/health

# Run a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Tesla total revenue in Q4 2025?"}'
```

### Local
```bash
pip install -r requirements.txt

# Ingest PDFs (first run only)
python src/ingest.py

# Run a query (retrieval mode selectable)
python src/query.py --mode crag "What was Tesla's Adjusted EBITDA in 2025?"

# Run the evaluation pipeline
python src/evaluate.py --mode crag

# Run tests
PYTHONPATH=src pytest tests/ -v
```

---

## Project Structure

```
Tesla-RAG/
├── src/
│   ├── api.py              # FastAPI server (POST /query, GET /health)
│   ├── config.py           # All hyperparameters in one place
│   ├── ingest.py           # PDF → chunks → ChromaDB
│   ├── query.py            # Orchestrator (4 retrieval modes)
│   ├── hybrid_search.py    # BM25 + vector + RRF fusion
│   ├── reranker.py         # Cross-Encoder reranking
│   ├── crag.py             # Quality gate + query rewriting
│   ├── evaluate.py         # 10-question benchmark
│   └── search_chunks.py    # Keyword search for debugging
├── tests/
│   ├── conftest.py         # Shared fixtures (session-scoped model loading)
│   ├── test_ingest.py      # Chunk counts, metadata, table detection
│   ├── test_search.py      # Hybrid search, reranker, table boost
│   ├── test_crag.py        # Quality-grade parsing (mocked, no API key)
│   └── test_api.py         # Endpoint validation (TestClient)
├── data/                   # Tesla IR report PDFs (Q3, Q4 2025)
├── results/                # Evaluation result JSON
├── Dockerfile
├── requirements.txt
└── CLAUDE.md               # AI assistant instructions
```

---

## Evaluation Details

**Dataset**: 10 financial questions built from Tesla's Q3/Q4 2025 IR reports, covering revenue, margins, EPS, EBITDA, free cash flow, and segment breakdowns.

**Metrics**:
- **Exact Match**: string match against the expected answer
- **Relevancy** (1–5): does the answer address the question?
- **Faithfulness** (1–5): is the answer grounded in the retrieved context (no hallucination)?
- **Completeness** (1–5): does the answer contain all required information?

**On the two remaining V4/V5 failures**: see [Limitations & Next Steps](#limitations--next-steps) — one is a structural limit of text-based RAG (answer buried in a 12-quarter reconciliation table), the other is a metric-design finding (correct figures, failed string match).
