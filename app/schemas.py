from pydantic import BaseModel
from typing import List, Literal, Optional

class RecommendationItem(BaseModel):
    id: str
    title: str
    score: float

class RecommendationResponse(BaseModel):
    mode: Literal["personalized", "popular"]
    user_id: Optional[str] = None
    items: List[RecommendationItem]
