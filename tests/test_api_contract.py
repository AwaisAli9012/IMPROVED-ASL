"""One happy-path call per route in the API contract, the JSON error shape, and the size and routing limits."""
import pytest

from signbridge import config

HAND = [[0.5, 0.5, 0.0]] * 21


@pytest.mark.parametrize("path,keys", [
    ("/api/config", {"POSE_IDX", "thresholds", "words", "gemini"}),
    ("/api/health", {"status", "models", "vocabulary_matches_model", "words", "gemini"}),
    ("/api/reports", set()),
    ("/api/phrases", None),
    ("/api/curriculum", None),
    ("/api/profiles", None),
])
def test_get_routes(client, path, keys):
    r = client.get(path)
    assert r.status_code == 200
    if keys:
        assert keys <= set(r.json())


def test_post_routes(client):
    if (config.ARTIFACTS / "static.joblib").exists():
        r = client.post("/api/predict/static", json={"hand": HAND, "aspect": 1.0, "mode": "letters"})
        assert r.status_code == 200 and len(r.json()["top"]) == 3 and "accept" in r.json()
    if (config.ARTIFACTS / "emotion.joblib").exists():
        r = client.post("/api/emotion", json={"bs": [0.1] * 52, "baseline": None})
        assert r.status_code == 200 and set(r.json()) == {"probs", "label"}
    assert client.post("/api/sentence", json={"glosses": ["hello"], "tone": 5}).json()["source"] == "offline"  # a bad tone becomes neutral
    assert "tokens" in client.post("/api/gloss", json={"text": "hello"}).json()
    pid = client.post("/api/profiles", json={"name": "Kim"}).json()["id"]
    assert client.get(f"/api/learn/{pid}/review").status_code == 200
    assert client.get(f"/api/progress/{pid}").status_code == 200
    assert client.post("/api/learn/unit_pass", json={"profile_id": pid, "unit_id": "start", "score": 1}).status_code == 200
    assert client.delete(f"/api/profiles/{pid}").json() == {"deleted": True}


def test_errors_are_json_with_an_error_key(client):
    r = client.post("/api/sentence", json={"glosses": []})
    assert r.status_code == 400 and "error" in r.json()
    r = client.post("/api/gloss", json=[1])  # not an object
    assert r.status_code == 400 and "error" in r.json()
    r = client.post("/api/learn/unit_pass", json={"profile_id": 1, "unit_id": "start", "score": "high"})
    assert r.status_code == 400 and "error" in r.json()


def test_too_large_is_413(client):
    r = client.post("/api/gloss", content=b"x" * (config.MAX_BODY_BYTES + 1), headers={"Content-Type": "application/json"})
    assert r.status_code == 413 and r.json() == {"error": "request too large"}


def test_unknown_api_path_is_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and "error" in r.json()
    assert client.get("/api/learn/abc").status_code == 404
