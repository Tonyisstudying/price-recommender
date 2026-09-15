from fastapi.testclient import TestClient
from api.main import app


def test_health():
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
