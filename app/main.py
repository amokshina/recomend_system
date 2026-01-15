from fastapi import FastAPI, Depends, Query
from sqlalchemy.orm import Session
from .db import SessionLocal, db_ping
from .schemas import RecommendationResponse
from .services.recommend import recommend_restaurants, recommend_dishes
from .services.train import train_stub

app = FastAPI(title="Food Recommender Service", version="0.1.0")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health")
def health():
    return {"status": "ok", "db": "up" if db_ping() else "down"}

@app.post("/train")
def train(db: Session = Depends(get_db)):
    return train_stub(db)

@app.get("/recommend/restaurants", response_model=RecommendationResponse)
def recommend_restaurants_api(
    user_id: str | None = Query(default=None),
    k: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    mode, items = recommend_restaurants(db, user_id, k)
    return RecommendationResponse(mode="personalized" if mode == "personalized" else "popular", user_id=user_id, items=items)

@app.get("/recommend/dishes", response_model=RecommendationResponse)
def recommend_dishes_api(
    user_id: str | None = Query(default=None),
    k: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    mode, items = recommend_dishes(db, user_id, k)
    return RecommendationResponse(mode="personalized" if mode == "personalized" else "popular", user_id=user_id, items=items)
