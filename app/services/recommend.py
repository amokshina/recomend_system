from typing import List, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from ..schemas import RecommendationItem


def popular_restaurants(db: Session, k: int) -> List[RecommendationItem]:
    q = text("""
        SELECT
            uri.restaurant_id,
            r.restaurant_name,
            SUM(uri.weight) AS score
        FROM user_restaurant_interactions uri
        JOIN restaurants r ON r.restaurant_id = uri.restaurant_id
        GROUP BY uri.restaurant_id, r.restaurant_name
        ORDER BY score DESC
        LIMIT :k
    """)
    rows = db.execute(q, {"k": k}).fetchall()
    return [
        RecommendationItem(id=str(r.restaurant_id), title=str(r.restaurant_name), score=float(r.score))
        for r in rows
    ]


def popular_dishes(db: Session, k: int) -> List[RecommendationItem]:
    q = text("""
        SELECT
            dish_id,
            dish_name,
            SUM(qty) AS score
        FROM order_items
        GROUP BY dish_id, dish_name
        ORDER BY score DESC
        LIMIT :k
    """)
    rows = db.execute(q, {"k": k}).fetchall()
    return [
        RecommendationItem(id=str(r.dish_id), title=str(r.dish_name), score=float(r.score))
        for r in rows
    ]


def user_has_history(db: Session, user_id: str) -> bool:
    q = text("SELECT 1 FROM orders WHERE customer_id = :uid LIMIT 1")
    return db.execute(q, {"uid": user_id}).first() is not None


def recommend_restaurants(db: Session, user_id: Optional[str], k: int) -> tuple[str, List[RecommendationItem]]:
    if not user_id or not user_has_history(db, user_id):
        return "popular", popular_restaurants(db, k)

    # Пока baseline: персональные = популярные среди ресторанов, где был пользователь (частотный профиль)
    q = text("""
        SELECT
            uri.restaurant_id,
            r.restaurant_name,
            SUM(uri.weight) AS score
        FROM user_restaurant_interactions uri
        JOIN restaurants r ON r.restaurant_id = uri.restaurant_id
        WHERE uri.customer_id = :uid
        GROUP BY uri.restaurant_id, r.restaurant_name
        ORDER BY score DESC
        LIMIT :k
    """)
    rows = db.execute(q, {"uid": user_id, "k": k}).fetchall()

    if not rows:
        return "popular", popular_restaurants(db, k)

    items = [RecommendationItem(id=str(r.restaurant_id), title=str(r.restaurant_name), score=float(r.score)) for r in rows]
    return "personalized", items


def recommend_dishes(db: Session, user_id: Optional[str], k: int) -> tuple[str, List[RecommendationItem]]:
    if not user_id or not user_has_history(db, user_id):
        return "popular", popular_dishes(db, k)

    q = text("""
        SELECT
            oi.dish_id,
            oi.dish_name,
            SUM(oi.qty) AS score
        FROM order_items oi
        JOIN orders o ON o.order_id = oi.order_id
        WHERE o.customer_id = :uid
        GROUP BY oi.dish_id, oi.dish_name
        ORDER BY score DESC
        LIMIT :k
    """)
    rows = db.execute(q, {"uid": user_id, "k": k}).fetchall()

    if not rows:
        return "popular", popular_dishes(db, k)

    items = [RecommendationItem(id=str(r.dish_id), title=str(r.dish_name), score=float(r.score)) for r in rows]
    return "personalized", items
