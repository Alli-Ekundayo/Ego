# Ego — Nigerian-Centric Recommendation System

> An agentic recommendation engine grounded in authentic Nigerian e-commerce language, built on LangGraph, Turbovec, and Google Gemini.

---

## Overview

**Ego** is a dual-task recommendation system designed for the Nigerian e-commerce context. It combines a user modelling pipeline (Task A) with a contextual recommendation pipeline (Task B), fusing dense semantic search, sparse BM25 keyword search, collaborative filtering, cross-encoder reranking, and LLM personalisation into a single coherent agentic workflow.

The system is grounded in real Jumia product reviews scraped from the Nigerian market, giving the language models authentic vocabulary, slang, and cultural tone to draw on when generating reviews and recommendations.

---

## Architecture

```
                        ┌──────────────────────────────────────┐
                        │             FastAPI (Port 8000)       │
                        │  POST /simulate-review  POST /recommend│
                        └──────────┬───────────────────┬────────┘
                                   │                   │
                          Task A Graph           Task B Graph
                        (User Modelling)    (Recommendation)
                               │                   │
               ┌───────────────▼───┐   ┌───────────▼──────────────────┐
               │ profile_retrieval │   │ load_profile_node            │
               │ rating_prediction │   │ context_extraction_node      │
               │ style_analysis    │   │ aspect_extraction_node       │
               │ review_generation │   │ cold_start_node (if needed)  │
               └───────────────────┘   │ hybrid_retrieval_node        │
                                       │   ├─ Dense ANN (Turbovec)    │
                                       │   ├─ Collaborative Filtering  │
                                       │   └─ Sparse BM25 (RRF fusion) │
                                       │ reranking_node               │
                                       │   ├─ Blended local scorer    │
                                       │   └─ LLM reason generation   │
                                       │ multiturn_node               │
                                       └──────────────────────────────┘
```

### Task A — User Modelling

Simulates how a given user would rate and review a product.

| Node | Purpose |
|---|---|
| `profile_retrieval_node` | Load the user's historical reviews from Turbovec; compute per-aspect exemplars via embedding cosine |
| `rating_prediction_node` | Predict a 1–5 star rating using persona similarity |
| `style_analysis_node` | Derive a writing-style profile from review statistics (length, TTR, tone) — no LLM call |
| `review_generation_node` | Generate a review with authentic Naija voice in a single LLM call (RAG over past reviews + Jumia examples) |

### Task B — Contextual Recommendation

Returns ranked product recommendations for a user given a conversational context.

| Node | Purpose |
|---|---|
| `load_profile_node` | Load the user profile; detect cold-start |
| `context_extraction_node` | Parse the user's request into structured signals (preferences, domain, mood) |
| `aspect_extraction_node` | Extract product aspects and generate BM25 keyword tokens |
| `cold_start_node` | Build a proxy embedding from the nearest user cluster (new users only) |
| `hybrid_retrieval_node` | Dense ANN + Collaborative Filtering + BM25, fused via Reciprocal Rank Fusion |
| `reranking_node` | Blended local scorer (CE + aspect cosine + category preference + retrieval) → LLM reason generation |
| `multiturn_node` | Conversational refinement: re-retrieves on detected preference shifts |

---

## Technical Stack

| Component | Technology |
|---|---|
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | Google Gemini (`gemini-flash-latest` by default, configurable) |
| Embeddings | `all-MiniLM-L6-v2` via [SentenceTransformers](https://www.sbert.net/) |
| Vector store | [Turbovec](https://pypi.org/project/turbovec/) |
| Sparse retrieval | BM25 via `rank-bm25` |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| LLM response cache | SQLite (`langchain_community.cache.SQLiteCache`) |
| Embedding cache | `diskcache` (persistent, disk-backed) |
| Python | 3.10+ |

---

## Project Structure

```
Ego/
├── api/
│   ├── main.py              # FastAPI app, startup preloading, route handlers
│   └── schemas.py           # Pydantic request/response models
│
├── agents/
│   ├── rerank_agent.py      # Blended local scorer + LLM reason generation
│   └── retrieval_agent.py   # Hybrid retrieval: Dense ANN + CF + BM25 (RRF)
│
├── core/
│   ├── aspect_extractor.py  # Rule-based aspect extraction + sparse keyword tokens
│   ├── config.py            # Pydantic settings (reads .env)
│   ├── cross_encoder.py     # Cross-encoder + aspect alignment + category preference scoring
│   ├── embeddings.py        # SentenceTransformer wrapper with diskcache
│   ├── hybrid_search.py     # BM25 corpus builder + Reciprocal Rank Fusion
│   ├── llm.py               # Gemini singleton with SQLite response caching
│   ├── math_utils.py        # Cosine similarity, dot product
│   ├── profiles.py          # Shared profile store (single JSON load, mtime-invalidated)
│   ├── user_profile.py      # UserProfile dataclass + builder
│   ├── utils.py             # tokenize, normalise_category, to_vector_id, clean_review_text
│   └── vector_store.py      # Turbovec client wrapper (search, upsert)
│
├── graphs/
│   ├── task_a.py            # LangGraph pipeline: User Modelling
│   └── task_b.py            # LangGraph pipeline: Recommendation
│
├── scripts/
│   ├── ingest.py            # Data pipeline: download → parse → build profiles
│   ├── build_index.py       # Embed items.json and index into Turbovec
│   ├── build_user_profiles.py # Aggregate reviews into per-user profile JSON
│   ├── scrape_jumia.py      # Jumia review scraper (source of Naija training data)
│   ├── seed_naija_examples.py # Index Jumia reviews into naija_style_examples collection
│   ├── evaluate.py          # End-to-end eval harness (ROUGE, BERTScore, RMSE, NDCG)
│   └── run_ablations.py     # Ablation study runner (no-BM25, no-cross-encoder variants)
│
├── tests/                   # Pytest test suite
├── data/                    # user_profiles.json, items.json, jumia_reviews.json
├── scratch/cache/           # SQLite LLM cache + diskcache embedding store
├── Dockerfile               # Multi-stage build (builder → runtime, non-root user)
├── docker-compose.yml       # Services: frontend, api, indexer (tools profile)
├── Makefile                 # Developer workflow shortcuts
└── requirements.txt         # Pinned Python dependencies
```

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- A [Google AI Studio](https://aistudio.google.com/) API key

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your API key:

```dotenv
GOOGLE_API_KEY=your_google_api_key_here
LLM_MODEL=gemini-flash-latest          # or any Gemini model you have access to
EMBEDDING_MODEL=all-MiniLM-L6-v2
DATASET_BASE_URL=https://huggingface.co/datasets/DreamerX/Ego-Jumia-Review/resolve/main
```

### 2. Build and start services

```bash
make up
```

This starts three containers:

| Container | Port | Purpose |
|---|---|---|
| `ego-api` | `8000` | FastAPI recommendation API |
| `ego-frontend` | `3000` | Static evaluation dashboard |

### 3. Run the data pipeline

The indexer downloads the dataset, builds user profiles, and seeds Turbovec:

```bash
make index
```

> **Note:** This only needs to run once. Data is persisted in `./data/`. Use `make reindex` to force a full rebuild.

### 4. Verify the API

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## API Reference

### `POST /simulate-review`

Simulates how a specific user would rate and review a product.

**Request:**
```json
{
  "user_id": "user_042",
  "item": {
    "name": "Infinix Hot 40 Pro",
    "category": "Smartphones",
    "description": "6.78-inch display, 108MP camera, 5000mAh battery"
  }
}
```

**Response:**
```json
{
  "rating": 4.2,
  "review": "This phone is quite solid for the price. Battery lasts all day, camera sharp well well.",
  "naija_review": "Chai, this phone don cast o! Battery life strong well well, camera clear like HD..."
}
```

---

### `POST /recommend`

Returns personalised product recommendations for a user.

**Request:**
```json
{
  "user_id": "user_042",
  "context": "I need a good pair of wireless earbuds for the gym",
  "n": 5,
  "persona_description": "budget-conscious tech lover",
  "session_history": [],
  "domain_filter": "electronics"
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "item_id": "3f8a21bc",
      "name": "JBL Tune 230NC TWS",
      "reason": "Based on your preference for budget-friendly electronics, this offers excellent noise cancellation at a competitive price point."
    }
  ]
}
```

**Parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `user_id` | `string` | required | Stable user identifier |
| `context` | `string` | required | Natural language request or query |
| `n` | `int` | `10` | Number of recommendations (max 10) |
| `persona_description` | `string` | `""` | Free-text persona for cold-start users |
| `session_history` | `list[dict]` | `[]` | Prior conversation turns `[{role, content}]` |
| `domain_filter` | `string` | `null` | Category constraint (e.g. `"electronics"`, `"fashion"`) |

---

## Key Design Decisions

### Hybrid Retrieval with RRF

The retrieval stage fuses three signals using Reciprocal Rank Fusion (k=60):

1. **Dense ANN** — Turbovec cosine-similarity search over user history embeddings
2. **Collaborative Filtering** — Cosine similarity over cross-domain projected user vectors
3. **Sparse BM25** — Keyword search over the full review corpus

Items appearing in multiple ranked lists receive a compounded RRF boost.

### Embedding-Based Reranking

After retrieval, a two-stage reranker narrows from ~100 candidates to the final top-N:

1. **Blended local scorer** — Combines four signals entirely on-device with no API cost:
   - Cross-encoder score (`ms-marco-MiniLM-L-6-v2`) — pairwise relevance
   - Aspect alignment — cosine similarity between candidate embeddings and the user's target aspect query embeddings
   - Category preference weight — from the user's historical category distribution
   - Retrieval score — upstream RRF signal

   A persona-conditioned emotional intensity multiplier (derived from rating variance) is applied across the blend. Prunes to top-N deterministically.

2. **LLM reason generation** — Gemini receives only the top-N pre-ranked items (~300 tokens) and writes a short personalised reason per item. It does not re-rank; if it fails, templated reasons are used and the ranking is preserved.

### Task A: LLM Call Reduction

The Task A pipeline runs **one LLM call** per invocation:

- **Style analysis** is computed statistically from review text (avg word length, type-token ratio, rating distribution, recurring bigrams) — no LLM required.
- **Naija voice injection** is merged into the generation prompt rather than running as a separate rewrite pass, eliminating voice drift and one full round-trip.
- **Aspect exemplars** are extracted by embedding cosine similarity over the user's own reviews, giving the generation LLM targeted evidence (e.g. what the user wrote about battery life) instead of bare aspect labels.

### Cold-Start Handling

New users with no Turbovec profile are routed through `cold_start_node`, which:
- Parses the `persona_description` into explicit/implicit signals via the context extraction node
- Maps the persona to the nearest user cluster centroid using cosine similarity
- Uses the centroid as a proxy embedding for the retrieval stage

### Caching Strategy

| Layer | Technology | Scope |
|---|---|---|
| LLM responses | SQLite (`scratch/cache/llm_cache.db`) | Persistent across restarts |
| Embeddings | `diskcache` (`scratch/cache/embeddings/`) | Persistent across restarts |
| BM25 corpus | In-memory, mtime-invalidated | Per process, rebuilt on file change |
| User profiles | `core.profiles` shared store, mtime-invalidated | Per process |
| User profile payloads | `lru_cache(maxsize=2048)` on DB read | Per process |

---

## Evaluation

Run the full evaluation harness:

```bash
# Task A only (user modelling)
PYTHONPATH=. python scripts/evaluate.py --task a --limit 20

# Task B only (recommendations)
PYTHONPATH=. python scripts/evaluate.py --task b --limit 20

# Both tasks
PYTHONPATH=. python scripts/evaluate.py --task both
```

**Metrics and targets:**

| Metric | Target | Task |
|---|---|---|
| ROUGE-L | ≥ 0.35 | Task A (review generation) |
| BERTScore F1 | ≥ 0.82 | Task A (review generation) |
| RMSE | ≤ 0.80 | Task A (rating prediction) |
| NDCG@10 | ≥ 0.15 | Task B (recommendation ranking) |

### Ablation Studies

```bash
PYTHONPATH=. python scripts/run_ablations.py --limit 5
```

Runs four ablation conditions:
- **Baseline** — Full pipeline
- **No Jumia Context** — Task A without Naija style examples
- **No BM25** — Task B with dense-only retrieval
- **No Cross-Encoder** — Task B without the local blended scorer

---

## Development

All developer workflows are available via `make`:

```bash
make help          # Show all available targets

make up            # Start all services
make down          # Stop all services
make build         # Rebuild the API Docker image
make fresh         # Tear down → rebuild → bring back up

make logs          # Tail API container logs
make shell         # Open a bash shell inside the API container
make restart       # Restart only the API container

make index         # Run the full data pipeline
make reindex       # Force re-download and re-index

make test          # Run pytest test suite
make lint          # Lint with ruff
```

### Running locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the API
uvicorn api.main:app --reload --port 8000
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *(required)* | Google AI Studio or Vertex AI key for Gemini |
| `LLM_MODEL` | `gemini-flash-latest` | Gemini model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `DATASET_BASE_URL` | *(see `.env.example`)* | Base URL for dataset files (HuggingFace or custom host) |

---

## Turbovec Collections

| Collection | Content | Indexed by |
|---|---|---|
| `user_profiles` | One point per user — their history embedding + profile payload | `scripts/ingest.py` |
| `naija_style_examples` | Individual Jumia review bodies for RAG | `scripts/seed_naija_examples.py` |

---

## Data Pipeline

```
scripts/scrape_jumia.py          → data/jumia_reviews.json
scripts/ingest.py                → data/user_profiles.json
                                   data/items.json
scripts/build_index.py           → Turbovec: user_profiles collection
scripts/seed_naija_examples.py   → Turbovec: naija_style_examples collection
```

The dataset is also available on Hugging Face at `DreamerX/Ego-Jumia-Review` and is downloaded automatically by `make index`.
