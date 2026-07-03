import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    MemoryIngestRequest,
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemorySnapshotResponse,
    PaginatedProducts,
    PaginatedUsers,
    ProductSummary,
    RecalledMemory,
    RecommendRequest,
    RecommendResponse,
    SimulateReviewRequest,
    SimulateReviewResponse,
    UserSummary,
)
from agents.memory_agent import memory_agent_graph, recall_for_user
from graphs.task_a import user_modeling_agent
from graphs.task_b import task_b_graph

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    Eagerly initialise all heavy singletons on startup.

    This prevents slow first requests and thread-safety races caused by
    concurrent on-demand initialisation under asyncio.to_thread.
    """
    def _load() -> None:
        # Embedding model (SentenceTransformer + diskcache)
        from core.embeddings import embedding_model
        _ = embedding_model.model

        # Shared profile store — single load warms Task A, Task B, and RetrievalAgent
        from core.profiles import profiles_list
        profiles_list()

        # BM25 corpus (reads user_profiles.json + builds BM25 index)
        from core.hybrid_search import _get_corpus
        _get_corpus()

        # Cross-encoder (only loads if model is cached locally)
        from core.cross_encoder import _get_cross_encoder
        _get_cross_encoder()

        # Warm up RetrievalAgent lazy cached properties
        from agents.retrieval_agent import retrieval_agent
        _ = retrieval_agent.user_vectors
        _ = retrieval_agent.cross_domain_projection
        _ = retrieval_agent.cluster_centroids

        # Warm up product catalog cache
        from core.products import load_products_by_id
        load_products_by_id()

        # Check API key configuration
        from core.config import settings
        if not settings.DASHSCOPE_API_KEY or not settings.DASHSCOPE_API_KEY.get_secret_value():
            log.warning("DASHSCOPE_API_KEY is not set — LLM calls (consolidate, rerank) will fail at runtime.")

        log.info("Startup: all singletons pre-loaded.")

    await asyncio.to_thread(_load)
    yield  # application runs here


app = FastAPI(title="Ego Gateway", lifespan=_lifespan)
api_app = FastAPI(title="Ego User Modelling Agent API")

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _invoke_graph(
    invoker: Callable[[dict[str, Any]], dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(invoker, state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _normalise_recommendations(result: dict[str, Any]) -> list[dict]:
    recommendations = (
        result.get("ranked_recommendations") or result.get("recommendations") or []
    )

    if not result.get("ranked_recommendations") and not result.get("recommendations"):
        log.warning("Task B graph returned empty or invalid recommendations key in result state: %s", result)

    if recommendations and isinstance(recommendations[0], list):
        recommendations = [item for sublist in recommendations for item in sublist]

    normalised: list[dict] = []
    for rec in recommendations:
        if isinstance(rec, dict):
            normalised.append(
                {
                    "item_id": str(rec.get("item_id", "")),
                    "name": str(rec.get("name", "Unknown")),
                    "reason": str(rec.get("reason", rec.get("reasoning", ""))),
                    # ── Price + rating metadata (zero/empty when not yet enriched) ──
                    "price_raw": str(rec.get("price_raw", "")),
                    "price_value": float(rec.get("price_value", 0.0)),
                    "old_price_raw": str(rec.get("old_price_raw", "")),
                    "old_price_value": float(rec.get("old_price_value", 0.0)),
                    "discount_percent": float(rec.get("discount_percent", 0.0)),
                    "currency": str(rec.get("currency", "NGN")),
                    "rating_stats": rec.get("rating_stats") or {},
                }
            )
    return normalised


@api_app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe used by Docker healthcheck and load-balancers."""
    return {"status": "ok"}


@app.get("/health", tags=["ops"])
async def gateway_health() -> dict[str, str]:
    """Liveness probe on the gateway app itself."""
    return {"status": "ok"}


@api_app.post("/simulate-review", response_model=SimulateReviewResponse)
async def simulate_review(request: SimulateReviewRequest) -> SimulateReviewResponse:
    """
    Run the user modelling pipeline (Task A) to predict a rating
    and generate a culturally-grounded simulated review.
    """
    initial_state = {
        "user_persona": request.user_id,
        "item_metadata": request.item.model_dump(),
    }

    result = await _invoke_graph(user_modeling_agent.invoke, initial_state)

    return SimulateReviewResponse(
        rating=result["predicted_rating"],
        review=result.get("final_review", result.get("simulated_review", "")),
    )


@api_app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Run the retrieval + rerank pipeline (Task B) to return N recommendations.
    After returning the response, persists the interaction to the MemoryAgent
    so that future sessions benefit from accumulated experience.
    """
    initial_state = {
        "user_id": request.user_id,
        "context_text": request.context,
        "persona_description": request.persona_description,
        "session_history": request.session_history,
        "domain_filter": request.domain_filter,
        "n": request.n,
    }

    result = await _invoke_graph(task_b_graph.invoke, initial_state)
    response = RecommendResponse(recommendations=_normalise_recommendations(result))

    # ── Async memory ingestion (fire-and-forget) ───────────────────────────
    # Persist the interaction context and inferred domain so the MemoryAgent
    # accumulates cross-session experience without blocking the response.
    async def _persist_memory() -> None:
        try:
            domain = result.get("extracted_domain") or request.domain_filter or ""
            recs = [
                r.get("name", "") for r in (result.get("ranked_recommendations") or [])[:3]
            ]
            rec_summary = ", ".join(filter(None, recs))
            events = [
                {
                    "type": "interaction",
                    "content": f"User searched: {request.context!r}. Top results: {rec_summary}.",
                    "importance": 0.4,
                    "metadata": {"domain": domain, "n": request.n},
                }
            ]
            if request.persona_description:
                events.append({
                    "type": "preference",
                    "content": f"User persona: {request.persona_description}",
                    "importance": 0.65,
                })
            if domain:
                events.append({
                    "type": "preference",
                    "content": f"User browsed category: {domain}",
                    "importance": 0.55,
                })
            await asyncio.to_thread(
                memory_agent_graph.invoke,
                {
                    "user_id": request.user_id,
                    "session_id": "",
                    "events": events,
                    "memories_before": 0,
                    "memories_after": 0,
                    "pruned": 0,
                    "evicted": 0,
                    "summary": "",
                    "preferences": {},
                    "error": None,
                },
            )
        except Exception as exc:
            log.warning("Background memory ingestion failed: %s", exc)

    asyncio.create_task(_persist_memory())
    return response


# ---------------------------------------------------------------------------
# Memory Agent endpoints (Track 1 — persistent cross-session memory)
# ---------------------------------------------------------------------------


@api_app.post("/memory/ingest", tags=["memory"])
async def memory_ingest(request: MemoryIngestRequest) -> dict:
    """
    Persist one or more interaction events to the MemoryAgent store and
    optionally run a Qwen-powered consolidation pass.

    Use this to explicitly record user feedback, explicit preferences, or
    any interaction signal you want the agent to remember long-term.
    """
    events = [{"type": ev.type, "content": ev.content, "importance": ev.importance, "metadata": ev.metadata}
              for ev in request.events]
    state = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "events": events,
        "memories_before": 0,
        "memories_after": 0,
        "pruned": 0,
        "evicted": 0,
        "summary": "",
        "preferences": {},
        "error": None,
    }
    result = await asyncio.to_thread(memory_agent_graph.invoke, state)
    return {
        "user_id": request.user_id,
        "memories_before": result.get("memories_before", 0),
        "memories_after": result.get("memories_after", 0),
        "pruned": result.get("pruned", 0),
        "evicted": result.get("evicted", 0),
        "summary_updated": bool(result.get("summary")),
    }


@api_app.post("/memory/recall", response_model=MemoryRecallResponse, tags=["memory"])
async def memory_recall(request: MemoryRecallRequest) -> MemoryRecallResponse:
    """
    Retrieve the most relevant memories, preferences, and long-term summary
    for a user given a query context. Results are budget-capped to fit within
    any LLM context window.
    """
    data = await asyncio.to_thread(
        recall_for_user,
        request.user_id,
        request.query,
        request.max_results,
        request.max_tokens,
    )
    return MemoryRecallResponse(
        user_id=request.user_id,
        summary=data["summary"],
        preferences=data["preferences"],
        recent_memories=[
            RecalledMemory(content=m["content"], type=m["type"], score=m["score"])
            for m in data["recent_memories"]
        ],
    )


@api_app.get("/memory/snapshot/{user_id}", response_model=MemorySnapshotResponse, tags=["memory"])
async def memory_snapshot(user_id: str) -> MemorySnapshotResponse:
    """
    Return a lightweight snapshot of the user's memory state:
    total memory count, named preferences, and the latest long-term summary.
    """
    from core.memory import MemoryStore
    snap = await asyncio.to_thread(MemoryStore(user_id).snapshot)
    return MemorySnapshotResponse(**snap)


# ---------------------------------------------------------------------------
# Catalogue browsing
# ---------------------------------------------------------------------------

def _load_items() -> list[dict]:
    from core.products import load_products_list
    return load_products_list()


@api_app.get("/users", response_model=PaginatedUsers, tags=["catalogue"])
async def list_users(
    page: int = 1,
    page_size: int = 20,
    search: str = "",
) -> PaginatedUsers:
    """
    Paginated list of unique users with display names and top category.
    Supports ?search= to filter by name or user_id.
    """
    from core.profiles import profiles_by_user_id

    by_uid = await asyncio.to_thread(profiles_by_user_id)

    summaries: list[UserSummary] = []
    for uid, p in by_uid.items():
        name = str(p.get("name") or uid)
        cat_pref: dict = p.get("category_pref") or {}
        top_cat = max(cat_pref, key=cat_pref.get) if cat_pref else "Unknown"
        rating_stats: dict = p.get("rating_stats") or {}
        mean_rating = float(rating_stats.get("mean") or 0.0)
        summaries.append(
            UserSummary(
                user_id=uid,
                name=name,
                review_count=int(p.get("review_count") or 0),
                top_category=top_cat,
                mean_rating=round(mean_rating, 2),
            )
        )

    if search:
        q = search.lower()
        summaries = [
            u for u in summaries if q in u.name.lower() or q in u.user_id.lower()
        ]

    # Sort by review_count desc so the richest profiles come first
    summaries.sort(key=lambda u: u.review_count, reverse=True)

    total = len(summaries)
    start = (page - 1) * page_size
    return PaginatedUsers(
        total=total,
        page=page,
        page_size=page_size,
        items=summaries[start : start + page_size],
    )


@api_app.get("/products", response_model=PaginatedProducts, tags=["catalogue"])
async def list_products(
    page: int = 1,
    page_size: int = 20,
    search: str = "",
    category: str = "",
) -> PaginatedProducts:
    """
    Paginated list of products from the Jumia catalogue.
    Supports ?search= (name/description) and ?category= filters.
    """

    items = await asyncio.to_thread(_load_items)

    summaries: list[ProductSummary] = []
    for item in items:
        cat = str(item.get("category") or "")
        if category and category.lower() not in cat.lower():
            continue
        rating_stats: dict = item.get("rating_stats") or {}
        mean_rating = float(rating_stats.get("mean") or 0.0)
        summaries.append(
            ProductSummary(
                id=str(item.get("id") or ""),
                name=str(item.get("name") or "Unknown"),
                category=cat,
                description=str(item.get("description") or "") or None,
                mean_rating=round(mean_rating, 2),
                review_count=int(item.get("review_count") or 0),
                price_raw=str(item.get("price_raw") or ""),
                price_value=float(item.get("price_value") or 0.0),
                old_price_raw=str(item.get("old_price_raw") or ""),
                old_price_value=float(item.get("old_price_value") or 0.0),
                discount_percent=float(item.get("discount_percent") or 0.0),
                currency=str(item.get("currency") or "NGN"),
                rating_stats=rating_stats,
            )
        )

    if search:
        q = search.lower()
        summaries = [
            p
            for p in summaries
            if q in p.name.lower() or (p.description and q in p.description.lower())
        ]

    summaries.sort(key=lambda p: p.review_count, reverse=True)

    total = len(summaries)
    start = (page - 1) * page_size
    return PaginatedProducts(
        total=total,
        page=page,
        page_size=page_size,
        items=summaries[start : start + page_size],
    )


# Mount the API sub-app under /api
app.mount("/api", api_app)

# Serve Vite static files
dist_path = Path(__file__).parent.parent / "frontend" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
