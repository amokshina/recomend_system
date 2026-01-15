import os
import numpy as np
import joblib
from scipy.sparse import coo_matrix, csr_matrix
from sqlalchemy import text
from implicit.als import AlternatingLeastSquares
from ..config import settings


K_EVAL = 5

def _ndcg_at_k(recommended_ids, relevant_set, k):
    dcg = 0.0
    for i, item in enumerate(recommended_ids[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 1)
    # идеальный DCG: все релевантные в начале
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0

def _recall_at_k(recommended_ids, relevant_set, k):
    if not relevant_set:
        return 0.0
    hits = sum(1 for x in recommended_ids[:k] if x in relevant_set)
    return hits / len(relevant_set)

# def _topk_scores(user_factors, item_factors, user_idx, seen_item_idx_set, k):
#     # scores = u dot V
#     u = user_factors[user_idx]  # (f,)
#     scores = item_factors @ u   # (n_items,)

#     if seen_item_idx_set:
#         scores[list(seen_item_idx_set)] = -np.inf

#     if k >= len(scores):
#         top = np.argsort(-scores)
#     else:
#         top = np.argpartition(-scores, k)[:k]
#         top = top[np.argsort(-scores[top])]
#     return top
def _topk_scores(user_factors, item_factors, user_idx, seen_item_idx_set, k):
    """
    user_factors: факторы пользователей (n_users, f)
    item_factors: факторы айтемов (n_items, f)
    """
    u = user_factors[user_idx]      # (f,)
    scores = item_factors @ u       # (n_items,)

    if seen_item_idx_set:
        scores[list(seen_item_idx_set)] = -np.inf

    k_eff = min(k, len(scores))
    if k_eff <= 0:
        return np.array([], dtype=int)

    top = np.argpartition(-scores, k_eff - 1)[:k_eff]
    top = top[np.argsort(-scores[top])]
    return top

def _fit_als(interactions_csr: csr_matrix, factors=64, reg=0.01, iterations=20, seed=42):
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iterations,
        random_state=seed
    )
    # implicit ожидает item-user матрицу для fit
    model.fit(interactions_csr.T)
    return model

def _ensure_artifacts_dir():
    os.makedirs(settings.artifacts_dir, exist_ok=True)

def train_stub(db):
    """
    Обучает ALS для ресторанов и блюд.
    Возвращает метрики и пути к артефактам.
    """
    _ensure_artifacts_dir()

    # 1) Сплит: для каждого customer_id последний order_id по времени -> test_orders
    test_orders = db.execute(text("""
        WITH ranked AS (
          SELECT
            order_id,
            customer_id,
            order_placed_at,
            ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_placed_at DESC) AS rn
          FROM orders
          WHERE order_placed_at IS NOT NULL
        )
        SELECT order_id, customer_id
        FROM ranked
        WHERE rn = 1
    """)).fetchall()

    test_order_ids = set(int(r.order_id) for r in test_orders)

    # -------------------- Restaurants interactions --------------------
    # train: все заказы кроме test_order_ids
    rest_rows = db.execute(text("""
        SELECT customer_id::text AS customer_id, restaurant_id::int AS restaurant_id, COUNT(*)::float AS w
        FROM orders
        WHERE order_id <> ALL(:test_ids)
        GROUP BY 1,2
    """), {"test_ids": list(test_order_ids)}).fetchall()

    # test ground truth: ресторан из последнего заказа
    rest_test_rows = db.execute(text("""
        SELECT o.customer_id::text AS customer_id, o.restaurant_id::int AS restaurant_id
        FROM orders o
        WHERE o.order_id = ANY(:test_ids)
    """), {"test_ids": list(test_order_ids)}).fetchall()

    rest_metrics, rest_art = _train_one(
        name="restaurants",
        train_rows=rest_rows,
        test_rows=rest_test_rows,
        item_id_field="restaurant_id",
        item_title_query="""
            SELECT restaurant_id::int AS item_id, restaurant_name::text AS title
            FROM restaurants
        """,
        db=db
    )

    # -------------------- Dishes interactions --------------------
    # train: все order_items, где order_id не в test
    dish_rows = db.execute(text("""
        SELECT o.customer_id::text AS customer_id, oi.dish_id::text AS dish_id, SUM(oi.qty)::float AS w
        FROM order_items oi
        JOIN orders o ON o.order_id = oi.order_id
        WHERE oi.order_id <> ALL(:test_ids)
        GROUP BY 1,2
    """), {"test_ids": list(test_order_ids)}).fetchall()

    # test ground truth: блюда из последнего заказа пользователя
    dish_test_rows = db.execute(text("""
        SELECT o.customer_id::text AS customer_id, oi.dish_id::text AS dish_id
        FROM order_items oi
        JOIN orders o ON o.order_id = oi.order_id
        WHERE oi.order_id = ANY(:test_ids)
    """), {"test_ids": list(test_order_ids)}).fetchall()

    dish_metrics, dish_art = _train_one(
        name="dishes",
        train_rows=dish_rows,
        test_rows=dish_test_rows,
        item_id_field="dish_id",
        item_title_query="""
            SELECT dish_id::text AS item_id, dish_name::text AS title
            FROM order_items
            GROUP BY 1,2
        """,
        db=db
    )

    return {
        "status": "ok",
        "restaurants": {"metrics@5": rest_metrics, "artifacts": rest_art},
        "dishes": {"metrics@5": dish_metrics, "artifacts": dish_art},
        "note": "Split: last order per user in test; ALS trained on remaining orders."
    }

def _train_one(name, train_rows, test_rows, item_id_field, item_title_query, db):
    # train_rows: (customer_id, item_id, w)
    users = sorted({r.customer_id for r in train_rows})
    items = sorted({getattr(r, item_id_field) for r in train_rows})

    if len(users) == 0 or len(items) == 0:
        raise RuntimeError(f"Not enough data to train ALS for {name}")

    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: i for i, it in enumerate(items)}
    idx_to_user = users
    idx_to_item = items

    row_idx = []
    col_idx = []
    data = []
    for r in train_rows:
        u = user_to_idx[r.customer_id]
        it = item_to_idx[getattr(r, item_id_field)]
        row_idx.append(u)
        col_idx.append(it)
        data.append(float(r.w))

    mat = coo_matrix((data, (row_idx, col_idx)), shape=(len(users), len(items))).tocsr()

    model = _fit_als(mat, factors=64, reg=0.01, iterations=20)

    # titles для красивого ответа (id -> title)
    titles = {}
    for r in db.execute(text(item_title_query)).fetchall():
        titles[r.item_id] = r.title

    # seen items (train)
    # seen = {}
    # mat_csr = mat.tocsr()
    # for u_idx in range(mat_csr.shape[0]):
    #     start = mat_csr.indptr[u_idx]
    #     end = mat_csr.indptr[u_idx + 1]
        # seen[u_idx] = set(mat_csr.indices[start:end].tolist())

    # test ground truth: customer_id -> set(item_id)
    gt = {}
    for r in test_rows:
        uid = r.customer_id
        it = getattr(r, item_id_field)
        gt.setdefault(uid, set()).add(it)

    # eval по пользователям, которые есть в train
    recalls = []
    ndcgs = []

    for uid, rel_items in gt.items():
        if uid not in user_to_idx:
            continue
        u_idx = user_to_idx[uid]

        # релевантные item_idx (только те, что есть в train items)
        rel_idx = {item_to_idx[x] for x in rel_items if x in item_to_idx}
        if not rel_idx:
            continue

        # top_idx = _topk_scores(model.user_factors, model.item_factors, u_idx, seen[u_idx], K_EVAL)
        # top_idx = _topk_scores(model.item_factors, model.user_factors, u_idx, seen[u_idx], K_EVAL)
        top_idx = _topk_scores(model.item_factors, model.user_factors, u_idx, set(), K_EVAL)
        recalls.append(_recall_at_k(top_idx, rel_idx, K_EVAL))
        ndcgs.append(_ndcg_at_k(top_idx, rel_idx, K_EVAL))

    metrics = {
        "users_in_train": len(users),
        "items_in_train": len(items),
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }

    # save artifacts
    art_path = os.path.join(settings.artifacts_dir, f"als_{name}.joblib")
    # joblib.dump(
    #     {
    #         "name": name,
    #         "user_to_idx": user_to_idx,
    #         "item_to_idx": item_to_idx,
    #         "idx_to_user": idx_to_user,
    #         "idx_to_item": idx_to_item,
    #         "user_factors": model.user_factors.astype(np.float32),
    #         "item_factors": model.item_factors.astype(np.float32),
    #         "titles": titles,
    #     },
    #     art_path
    # )
    joblib.dump(
        {
            "name": name,
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
            "idx_to_user": idx_to_user,
            "idx_to_item": idx_to_item,
            # сохраняем в понятном виде:
            # users -> model.item_factors, items -> model.user_factors
            "user_factors": model.item_factors.astype(np.float32),
            "item_factors": model.user_factors.astype(np.float32),
            "titles": titles,
        },
        art_path
    )

    return metrics, {"model_path": art_path}
