from sqlalchemy import text
from app.db import SessionLocal


def test_required_tables_exist_and_not_empty():
    db = SessionLocal()
    try:
        # проверяем, что таблицы есть и в них что-то загружено
        tables = ["orders", "restaurants", "order_items", "user_restaurant_interactions", "user_dish_interactions"]
        for t in tables:
            cnt = db.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar_one()
            assert cnt > 0, f"Table {t} is empty"
    finally:
        db.close()
