from fastapi.testclient import TestClient

from api.server import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


def test_reset_endpoint_all_tasks() -> None:
    client = TestClient(app)
    for tid in ("easy", "medium", "hard"):
        r = client.post("/reset", json={"task_id": tid})
        assert r.status_code == 200
        data = r.json()
        assert data["observation"]["step"] == 0
        assert data["observation"]["max_steps"] > 0


def test_step_endpoint_valid_action() -> None:
    client = TestClient(app)
    client.post("/reset", json={"task_id": "easy"})
    r = client.post(
        "/step",
        json={"action": {"action_type": "check_metrics", "service": "api-gateway"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert "observation" in body
    assert "reward" in body
    assert body["done"] is False


def test_step_endpoint_invalid_action_returns_error() -> None:
    client = TestClient(app)
    client.post("/reset", json={"task_id": "easy"})
    r = client.post(
        "/step",
        json={"action": {"action_type": "check_metrics"}},
    )
    assert r.status_code == 200
    obs = r.json()["observation"]
    assert obs.get("last_action_error")


def test_state_endpoint() -> None:
    client = TestClient(app)
    client.post("/reset", json={"task_id": "easy"})
    r = client.get("/state")
    assert r.status_code == 200
    assert "service_metrics" in r.json()
