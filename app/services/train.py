from sqlalchemy.orm import Session
from sqlalchemy import text

def train_stub(db: Session) -> dict:
    # Минимально: проверим, что данные есть, и посчитаем базовую статистику.
    total = db.execute(text("SELECT COUNT(*) FROM orders_raw")).scalar_one()
    return {"status": "ok", "rows_in_orders_raw": int(total)}
