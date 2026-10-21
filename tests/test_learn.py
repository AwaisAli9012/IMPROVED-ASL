import sqlite3
from datetime import date, timedelta

import pytest

from signbridge import config, learn
from signbridge.config import DIGITS, LETTERS
from signbridge.vocab import word_labels


@pytest.fixture
def pid(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    return learn.create_profile("Sam")["id"]


def test_curriculum_content_is_consistent():
    words = [i for u in learn.CURRICULUM if u["kind"] == "word" for i in u["items"]]
    assert sorted(words) == sorted(word_labels())  # every word exactly once
    hand = [i for u in learn.CURRICULUM if u["kind"] == "hand" for i in u["items"]]
    assert sorted(hand) == sorted(LETTERS + DIGITS)
    vocab = set(word_labels())
    for s in learn.UNITS["sentences"]["sentences"]:
        assert 2 <= len(s["gloss"]) <= 4 and set(s["gloss"]) <= vocab


def test_profile_names(pid):
    assert learn.create_profile("Sam")["id"] == pid  # same name, same profile
    with pytest.raises(ValueError):
        learn.create_profile("  ")


def test_leitner_boxes_and_due_dates(pid):
    assert learn.record_attempt(pid, "hello", True)["box"] == 2
    assert learn.due_items(pid) == []  # box 2 is due tomorrow
    assert learn.record_attempt(pid, "hello", False)["box"] == 1
    assert [d["item"] for d in learn.due_items(pid)] == ["hello"]  # box 1 is due now
    for _ in range(6):
        out = learn.record_attempt(pid, "hello", True)
    assert out["box"] == 5 and out["mastered"]
    with pytest.raises(ValueError):
        learn.record_attempt(pid, "not-a-sign", True)


def test_units_unlock_in_order(pid):
    states = lambda: [u["state"] for u in learn.curriculum_with_progress(pid)["units"][:3]]  # noqa: E731
    assert states() == ["open", "locked", "locked"]
    assert learn.record_unit_pass(pid, "start", 1)["passed"]
    assert states() == ["passed", "open", "locked"]
    assert learn.record_unit_pass(pid, "abc1", 0.7) == {"passed": False}
    assert states() == ["passed", "open", "locked"]
    xp = learn.curriculum_with_progress(pid)["profile"]["xp"]
    learn.record_unit_pass(pid, "start", 1)  # repeat pass gives no extra XP
    assert learn.curriculum_with_progress(pid)["profile"]["xp"] == xp


def test_streak(pid, tmp_path):
    learn.record_attempt(pid, "hello", True)
    assert learn.curriculum_with_progress(pid)["profile"]["streak"] == 1
    learn.record_attempt(pid, "bye", True)  # same day: unchanged
    assert learn.curriculum_with_progress(pid)["profile"]["streak"] == 1
    db = sqlite3.connect(config.DB_PATH)
    db.execute("UPDATE profiles SET last_active = ?", ((date.today() - timedelta(days=1)).isoformat(),))
    db.commit()
    learn.record_attempt(pid, "hello", True)
    assert learn.curriculum_with_progress(pid)["profile"]["streak"] == 2
    db.execute("UPDATE profiles SET last_active = ?", ((date.today() - timedelta(days=5)).isoformat(),))
    db.commit()
    db.close()
    learn.record_attempt(pid, "hello", True)
    assert learn.curriculum_with_progress(pid)["profile"]["streak"] == 1
