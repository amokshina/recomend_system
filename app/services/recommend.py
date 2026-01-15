import os
import numpy as np
import joblib
from sqlalchemy.orm import Session
from sqlalchemy import text
from ..config import settings
from ..schemas import RecommendationItem


def _load_model(name: str):
    path = os.path.join(settings.artifacts_dir, f"als_{name}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def _popular_restaurants(db: Session, k: int):
    q = text("""
        SELECT r.restaurant_id, r.restaurant_name, COUNT(*)::float AS score
        FROM orders o
        JOIN restaurants r ON r.restaurant_id = o.restaurant_id
        GROUP BY 1,2
        ORDER BY score DESC
        LIMIT :k
    """)
    rows = db.execute(q, {"k": k}).fetchall()
    return [RecommendationItem(id=str(r.restaurant_id), title=str(r.restaurant_name), score=float(r.score)) for r in rows]

def _popular_dishes(db: Session, k: int):
    q = text("""
        SELECT dish_id, dish_name, SUM(qty)::float AS score
        FROM order_items
        GROUP BY 1,2
        ORDER BY score DESC
        LIMIT :k
    """)
    rows = db.execute(q, {"k": k}).fetchall()
    return [RecommendationItem(id=str(r.dish_id), title=str(r.dish_name), score=float(r.score)) for r in rows]

def _user_has_history(db: Session, user_id: str) -> bool:
    return db.execute(text("SELECT 1 FROM orders WHERE customer_id=:u LIMIT 1"), {"u": user_id}).first() is not None

def _als_recommend(model_blob, user_id: str, k: int):
    u2i = model_blob["user_to_idx"]
    if user_id not in u2i:
        return None

    u_idx = u2i[user_id]
    U = model_blob["user_factors"]     # (n_users, f)
    V = model_blob["item_factors"]     # (n_items, f)
    scores = V @ U[u_idx]              # (n_items,)

    # исключим то, что пользователь уже заказывал (просто по train-истории из БД, быстро)
    # Для простоты: не исключаем — тоже допустимо, но лучше исключить.
    # Здесь сделаем исключение через запрос на seen items по типу модели.
    return scores

def recommend_restaurants(db: Session, user_id: str | None, k: int):
    if not user_id or not _user_has_history(db, user_id):
        return "popular", _popular_restaurants(db, k)

    model = _load_model("restaurants")
    if model is None:
        return "popular", _popular_restaurants(db, k)

    scores = _als_recommend(model, user_id, k)
    if scores is None:
        return "popular", _popular_restaurants(db, k)

    # исключаем рестораны, которые пользователь уже заказывал
    # seen_rows = db.execute(text("""
    #     SELECT DISTINCT restaurant_id::int AS rid
    #     FROM orders
    #     WHERE customer_id = :u
    # """), {"u": user_id}).fetchall()
    # seen = {model["item_to_idx"][r.rid] for r in seen_rows if r.rid in model["item_to_idx"]}
    # if seen:
    #     scores[list(seen)] = -np.inf

    top = np.argpartition(-scores, min(k, len(scores)-1))[:k]
    top = top[np.argsort(-scores[top])]

    items = []
    for idx in top:
        item_id = model["idx_to_item"][int(idx)]
        title = model["titles"].get(item_id, str(item_id))
        items.append(RecommendationItem(id=str(item_id), title=str(title), score=float(scores[idx])))

    return "personalized", items

def recommend_dishes(db: Session, user_id: str | None, k: int):
    if not user_id or not _user_has_history(db, user_id):
        return "popular", _popular_dishes(db, k)

    model = _load_model("dishes")
    if model is None:
        return "popular", _popular_dishes(db, k)

    scores = _als_recommend(model, user_id, k)
    if scores is None:
        return "popular", _popular_dishes(db, k)

    # исключаем блюда, которые пользователь уже заказывал
    # seen_rows = db.execute(text("""
    #     SELECT DISTINCT oi.dish_id::text AS did
    #     FROM order_items oi
    #     JOIN orders o ON o.order_id = oi.order_id
    #     WHERE o.customer_id = :u
    # """), {"u": user_id}).fetchall()
    # seen = {model["item_to_idx"][r.did] for r in seen_rows if r.did in model["item_to_idx"]}
    # if seen:
    #     scores[list(seen)] = -np.inf

    top = np.argpartition(-scores, min(k, len(scores)-1))[:k]
    top = top[np.argsort(-scores[top])]

    items = []
    for idx in top:
        item_id = model["idx_to_item"][int(idx)]
        title = model["titles"].get(item_id, str(item_id))
        items.append(RecommendationItem(id=str(item_id), title=str(title), score=float(scores[idx])))

    return "personalized", items
