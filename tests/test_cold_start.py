from fastapi.testclient import TestClient
from app.main import app


def test_recommend_restaurants_cold_start_without_user_id_returns_valid_response():
    client = TestClient(app)

    r = client.get("/recommend/restaurants?k=10")
    assert r.status_code == 200

    data = r.json()

    # базовая структура ответа
    assert "mode" in data
    assert "items" in data
    assert isinstance(data["items"], list)

    # cold start режим (без user_id)
    assert data["mode"] in ("popular", "cold_start")

    # k=10 -> не больше 10 рекомендаций
    assert len(data["items"]) <= 10

    # структура элементов рекомендаций
    for item in data["items"]:
        assert "id" in item
        assert "title" in item
        assert "score" in item
        assert isinstance(item["id"], str)
        assert isinstance(item["title"], str)
        assert isinstance(item["score"], (int, float))
