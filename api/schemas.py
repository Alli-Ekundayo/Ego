from pydantic import BaseModel
from typing import List, Optional


class Item(BaseModel):
    name: str
    category: str
    description: Optional[str] = None


class SimulateReviewRequest(BaseModel):
    user_id: str
    item: Item


class SimulateReviewResponse(BaseModel):
    rating: float          # was int — predicted_rating is always a float
    review: str
    naija_review: Optional[str] = None


class RecommendRequest(BaseModel):
    user_id: str
    context: str
    n: int = 10


class Recommendation(BaseModel):
    item_id: str
    name: str
    reason: str


class RecommendResponse(BaseModel):
    recommendations: List[Recommendation]
