import pytest

from signbridge import config
from signbridge.server import app

pytestmark = pytest.mark.skipif(not (config.ARTIFACTS / "vocab_words.json").exists(), reason="vocab not built yet")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return app.test_client()


def test_config_has_everything_the_browser_needs(client):
    cfg = client.get("/api/config").get_json()
    for key in ("POSE_IDX", "LIP_IDX", "thresholds", "words", "SEG_END_FRAMES", "LETTERS", "gemini"):
        assert key in cfg
    assert len(cfg["words"]) == 100 and cfg["gemini"] is False


def test_sentence_and_gloss_work_offline(client):
    r = client.post("/api/sentence", json={"glosses": ["you", "hungry"], "question": "yesno"})
    assert r.get_json() == {"english": "You hungry?", "source": "offline"}
    r = client.post("/api/gloss", json={"text": "thank you"})
    assert r.get_json()["tokens"] == [{"type": "sign", "label": "thankyou"}]


@pytest.mark.parametrize("path,payload", [
    ("/api/sentence", {"glosses": []}),
    ("/api/sentence", {"glosses": ["a"], "question": "maybe"}),
    ("/api/sentence", {"glosses": ["a"], "question": []}),
    ("/api/learn/attempt", {"profile_id": 1, "item": [], "correct": True}),
    ("/api/learn/unit_pass", {"profile_id": 1, "unit_id": {}, "score": 1}),
    ("/api/gloss", {"text": ""}),
    ("/api/predict/word", {"frames": []}),
    ("/api/predict/static", {"hand": [[0.5, 0.5, 0]] * 20, "aspect": 1, "mode": "letters"}),
    ("/api/predict/static", {"hand": [[0.5, 0.5, 0]] * 21, "aspect": 1, "mode": "emoji"}),
    ("/api/emotion", {"bs": [0.1] * 51}),
    ("/api/profiles", {"name": ""}),
    ("/api/learn/attempt", {"profile_id": 1, "item": "hello", "correct": "yes"}),
])
def test_bad_input_is_400(client, path, payload):
    assert client.post(path, json=payload).status_code == 400
    assert client.post(path, data="not json").status_code == 400


def test_word_prediction_returns_top5(client):
    if not (config.ARTIFACTS / "words.pt").exists():
        pytest.skip("words model not trained yet")
    point = [0.5, 0.5]
    frames = [{"rh": [point] * 21, "lh": None, "pose": [point] * 7, "lips": [point] * 40} for _ in range(20)]
    r = client.post("/api/predict/word", json={"frames": frames})
    assert r.status_code == 200 and len(r.get_json()["top"]) == 5


def test_learning_round_trip(client):
    pid = client.post("/api/profiles", json={"name": "Sam"}).get_json()["id"]
    r = client.post("/api/learn/attempt", json={"profile_id": pid, "item": "hello", "correct": True})
    assert r.get_json()["box"] == 2
    progress = client.get(f"/api/learn/{pid}").get_json()
    assert progress["profile"]["xp"] == 10 and progress["units"][0]["state"] == "open"
    assert client.post("/api/learn/attempt", json={"profile_id": 999, "item": "hello", "correct": True}).status_code == 400
