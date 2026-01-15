import os
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

def test_train_creates_artifacts():
    client = TestClient(app)
    r = client.post("/train")
    assert r.status_code == 200
    assert os.path.exists(os.path.join(settings.artifacts_dir, "als_restaurants.joblib"))
    assert os.path.exists(os.path.join(settings.artifacts_dir, "als_dishes.joblib"))
