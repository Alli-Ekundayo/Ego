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
