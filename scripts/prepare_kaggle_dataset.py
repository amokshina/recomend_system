import re
import hashlib
import pandas as pd
from sqlalchemy import create_engine
from app.config import settings


def snake_case(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def parse_distance_km(val) -> float | None:
    if pd.isna(val):
        return None
    s = str(val)
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


_item_pat = re.compile(r"^\s*(\d+)\s*x\s*(.+?)\s*$", re.IGNORECASE)


def dish_id_for(restaurant_id: int, dish_name: str) -> str:
    # стабильный id: sha1(restaurant_id|dish_name)
    raw = f"{restaurant_id}|{dish_name}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def parse_items(items_str: str, restaurant_id: int, order_id: int):
    """
    Возвращает список dict'ов: {order_id, restaurant_id, dish_id, dish_name, qty}
    """
    if pd.isna(items_str):
        return []
    s = str(items_str).strip()
    if not s:
        return []

    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        m = _item_pat.match(p)
        if not m:
            # если не совпало (редко) — запишем qty=1 и всё строкой
            dish_name = p
            qty = 1
        else:
            qty = int(m.group(1))
            dish_name = m.group(2)

        did = dish_id_for(int(restaurant_id), dish_name)
        out.append(
            {
                "order_id": int(order_id),
                "restaurant_id": int(restaurant_id),
                "dish_id": did,
                "dish_name": dish_name,
                "qty": qty,
            }
        )
    return out


def main(csv_path: str):
    df = pd.read_csv(csv_path)

    # 0) удаляем лишнее
    cols = [
        'Restaurant penalty (Rejection)',
        'Restaurant compensation (Cancellation)',
        'Cancellation / Rejection reason'
    ]

    df = df[~df[cols].notna().any(axis=1)]
    df = df.drop(columns=['Restaurant penalty (Rejection)', 
                 'Restaurant compensation (Cancellation)', 'Cancellation / Rejection reason', 'Instructions'])

    # 1) нормализуем имена колонок
    df.columns = [snake_case(c) for c in df.columns]

    # 2) обязательные колонки (после snake_case)
    # restaurant_id, restaurant_name, city, subzone, order_id, order_placed_at, order_status, items_in_order, customer_id
    # (проверим, что они есть)
    needed = [
        "restaurant_id",
        "restaurant_name",
        "city",
        "subzone",
        "order_id",
        "order_placed_at",
        "order_status",
        "items_in_order",
        "customer_id",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}")

    # 3) типы
    df["restaurant_id"] = df["restaurant_id"].astype("int64")
    df["order_id"] = df["order_id"].astype("int64")
    df["customer_id"] = df["customer_id"].astype(str)

    # 4) datetime
    df["order_placed_at_ts"] = pd.to_datetime(df["order_placed_at"], errors="coerce")
    # дополнительные фичи времени (вдруг пригодятся позже)
    df["order_hour"] = df["order_placed_at_ts"].dt.hour
    df["order_dow"] = df["order_placed_at_ts"].dt.dayofweek
    df["is_weekend"] = df["order_dow"].isin([5, 6])

    # 5) distance
    if "distance" in df.columns:
        df["distance_km"] = df["distance"].apply(parse_distance_km)
    else:
        df["distance_km"] = None

    # 6) restaurants
    restaurants = (
        df[["restaurant_id", "restaurant_name", "city", "subzone"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # 7) orders
    orders_cols = {
        "order_id": "order_id",
        "customer_id": "customer_id",
        "restaurant_id": "restaurant_id",
        "order_placed_at_ts": "order_placed_at",
        "order_status": "order_status",
        "delivery": "delivery",
        "distance_km": "distance_km",
        "bill_subtotal": "bill_subtotal",
        "packaging_charges": "packaging_charges",
        "restaurant_discount_promo": "restaurant_discount_promo",
        "restaurant_discount_flat_offs_freebies_others": "restaurant_discount_flat_offs_freebies_others",
        "gold_discount": "gold_discount",
        "brand_pack_discount": "brand_pack_discount",
        "total": "total",
        "rating": "rating",
        "discount_construct": "discount_construct",
        "kpt_duration_minutes": "kpt_duration_minutes",
        "rider_wait_time_minutes": "rider_wait_time_minutes",
        "order_ready_marked": "order_ready_marked",
        "customer_complaint_tag": "customer_complaint_tag",
    }

    # оставляем только те, что реально есть (у тебя они есть почти все, но делаем безопасно)
    present = {k: v for k, v in orders_cols.items() if k in df.columns}
    orders = df[list(present.keys())].rename(columns=present).copy()

    # 8) order_items
    items_rows = []
    for row in df[["order_id", "restaurant_id", "items_in_order"]].itertuples(index=False):
        items_rows.extend(parse_items(row.items_in_order, row.restaurant_id, row.order_id))
    order_items = pd.DataFrame(items_rows)

    # 9) interactions
    # базовый вес = 1 за заказ (потом легко поменяем)
    user_restaurant_interactions = (
        orders[["customer_id", "restaurant_id"]]
        .dropna()
        .assign(weight=1.0)
        .groupby(["customer_id", "restaurant_id"], as_index=False)["weight"]
        .sum()
    )

    # user-dish вес = сумма qty
    if not order_items.empty:
        # добавляем customer_id через orders
        order_to_user = orders[["order_id", "customer_id"]]
        odi = order_items.merge(order_to_user, on="order_id", how="left")
        user_dish_interactions = (
            odi.assign(weight=odi["qty"].astype(float))
            .groupby(["customer_id", "dish_id"], as_index=False)["weight"]
            .sum()
        )
    else:
        user_dish_interactions = pd.DataFrame(columns=["customer_id", "dish_id", "weight"])

    # 10) грузим в Postgres
    engine = create_engine(settings.db_url)

    restaurants.to_sql("restaurants", engine, if_exists="replace", index=False)
    orders.to_sql("orders", engine, if_exists="replace", index=False)
    order_items.to_sql("order_items", engine, if_exists="replace", index=False)
    user_restaurant_interactions.to_sql("user_restaurant_interactions", engine, if_exists="replace", index=False)
    user_dish_interactions.to_sql("user_dish_interactions", engine, if_exists="replace", index=False)

    print("Loaded tables:")
    print(f"  restaurants: {len(restaurants)}")
    print(f"  orders: {len(orders)}")
    print(f"  order_items: {len(order_items)}")
    print(f"  user_restaurant_interactions: {len(user_restaurant_interactions)}")
    print(f"  user_dish_interactions: {len(user_dish_interactions)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/prepare_kaggle_dataset.py data/orders.csv")
        raise SystemExit(1)
    main(sys.argv[1])
