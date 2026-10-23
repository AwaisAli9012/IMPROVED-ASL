import json

import pytest
from fastapi.testclient import TestClient

from signbridge import config, learn, nlg
from signbridge.app import app


@pytest.fixture
def pid(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    return learn.create_profile("Sam")["id"]


def test_progress_report_counts_and_badges(pid):
    learn.record_attempt(pid, "hello", True)
    learn.record_attempt(pid, "bye", False)
    learn.record_attempt(pid, "bye", False)
    r = learn.progress_report(pid)
    assert r["totals"]["attempts"] == 3 and r["totals"]["correct"] == 1 and r["totals"]["accuracy"] == pytest.approx(1 / 3)
    assert len(r["activity"]) == 14 and r["activity"][-1]["n"] == 3
    assert [w["item"] for w in r["weak"]] == ["bye"]
    earned = {b["id"] for b in r["badges"] if b["earned"]}
    assert earned == {"first_sign"}  # nothing else is earned yet


def test_unit_badges_and_perfect_quiz(pid):
    learn.record_unit_pass(pid, "start", 1)
    for u in ("abc1", "abc2", "abc3"):
        learn.record_unit_pass(pid, u, 1.0)
    earned = {b["id"] for b in learn.progress_report(pid)["badges"] if b["earned"]}
    assert {"alphabet", "perfect"} <= earned and "numbers" not in earned


def test_longest_run():
    assert learn._longest_run({"2026-10-01", "2026-10-02", "2026-10-04"}) == 2
    assert learn._longest_run(set()) == 0


def test_every_phrase_is_mostly_real_signs(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for group in json.loads((learn.Path(__file__).parent.parent / "signbridge" / "phrases.json").read_text()):
        for text in group["items"]:
            tokens = nlg.english_to_gloss(text)["tokens"]
            signs = [t for t in tokens if t["type"] == "sign"]
            assert tokens and len(signs) >= len(tokens) / 2, f"{text!r} is mostly fingerspelled: {tokens}"


def test_health_and_progress_endpoints(pid, monkeypatch):
    c = TestClient(app)
    h = c.get("/api/health").json()
    assert h["status"] in ("ok", "degraded") and set(h["models"]) == {"words", "static", "emotion"}
    assert h["status"] == "ok" or not (h["vocabulary_matches_model"] and all(h["models"].values()))  # degraded only for a real reason
    assert c.get(f"/api/progress/{pid}").json()["profile"]["name"] == "Sam"
    assert c.get("/api/progress/999").status_code == 400
    assert len(c.get("/api/phrases").json()) == 4


def test_delete_profile_removes_everything(pid):
    learn.record_attempt(pid, "hello", True)
    learn.record_unit_pass(pid, "start", 1)
    c = TestClient(app)
    assert c.delete(f"/api/profiles/{pid}").json() == {"deleted": True}
    assert learn.list_profiles() == []
    assert c.delete(f"/api/profiles/{pid}").status_code == 400  # already gone
