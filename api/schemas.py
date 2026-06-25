from pydantic import BaseModel, Field


class Item(BaseModel):
    name: str
    category: str
    description: str | None = None


class SimulateReviewRequest(BaseModel):
    user_id: str
    item: Item


class SimulateReviewResponse(BaseModel):
    rating: float  # was int — predicted_rating is always a float
    review: str


class RecommendRequest(BaseModel):
    user_id: str
    context: str
    n: int = 10
    persona_description: str = ""
    session_history: list[dict] = Field(default_factory=list)
    domain_filter: str | None = None


class Recommendation(BaseModel):
    item_id: str
    name: str
    reason: str
    # ── Price metadata (populated after enrich_prices.py run) ─────────────
    price_raw: str = ""
    price_value: float = 0.0
    old_price_raw: str = ""
    old_price_value: float = 0.0
    discount_percent: float = 0.0
    currency: str = "NGN"
    rating_stats: dict = Field(default_factory=dict)


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]


# ---------------------------------------------------------------------------
# Catalogue browsing
# ---------------------------------------------------------------------------


class UserSummary(BaseModel):
    user_id: str
    name: str
    review_count: int
    top_category: str
    mean_rating: float


class ProductSummary(BaseModel):
    id: str
    name: str
    category: str
    description: str | None = None
    mean_rating: float
    review_count: int
    # ── Price metadata (populated after enrich_prices.py run) ─────────────
    price_raw: str = ""
    price_value: float = 0.0
    old_price_raw: str = ""
    old_price_value: float = 0.0
    discount_percent: float = 0.0
    currency: str = "NGN"
    rating_stats: dict = Field(default_factory=dict)


class PaginatedUsers(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[UserSummary]


class PaginatedProducts(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[ProductSummary]


# ---------------------------------------------------------------------------
# Memory Agent (Track 1 — MemoryAgent)
# ---------------------------------------------------------------------------


class MemoryEvent(BaseModel):
    """A single interaction event to persist in the MemoryStore."""
    type: str = "interaction"   # "preference" | "interaction" | "feedback" | "context"
    content: str
    importance: float = 0.5     # initial salience in [0, 1]
    metadata: dict = Field(default_factory=dict)


class MemoryIngestRequest(BaseModel):
    user_id: str
    session_id: str = ""
    events: list[MemoryEvent]
    run_consolidation: bool = True  # if True, run full Qwen consolidation pass


class MemoryRecallRequest(BaseModel):
    user_id: str
    query: str
    max_results: int = 10
    max_tokens: int = 600


class RecalledMemory(BaseModel):
    content: str
    type: str
    score: float


class MemoryRecallResponse(BaseModel):
    user_id: str
    summary: str
    preferences: dict
    recent_memories: list[RecalledMemory]


class MemorySnapshotResponse(BaseModel):
    user_id: str
    memory_count: int
    preferences: dict
    summary: str

