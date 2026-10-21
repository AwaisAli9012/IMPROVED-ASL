"""Learning engine: course content, SQLite progress, Leitner spaced repetition, XP and streaks."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from . import config
from .vocab import display

CURRICULUM = json.loads((Path(__file__).parent / "curriculum.json").read_text())["units"]
UNITS = {u["id"]: u for u in CURRICULUM}
ITEMS = {i for u in CURRICULUM for i in u["items"]}
BOX_DAYS = {1: 0, 2: 1, 3: 3, 4: 7, 5: 16}  # days until an item is due again, by Leitner box
MASTERED_BOX, PASS_SCORE = 4, 0.8
XP_CORRECT, XP_WRONG, XP_UNIT = 10, 2, 50

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, xp INTEGER DEFAULT 0,
  streak INTEGER DEFAULT 0, last_active TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS progress (profile_id INTEGER, item TEXT, box INTEGER DEFAULT 1, correct INTEGER DEFAULT 0,
  attempts INTEGER DEFAULT 0, due_at TEXT, PRIMARY KEY (profile_id, item));
CREATE TABLE IF NOT EXISTS unit_pass (profile_id INTEGER, unit_id TEXT, score REAL, passed_at TEXT, PRIMARY KEY (profile_id, unit_id));
"""


@contextmanager
def _db():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    try:
        yield db
        db.commit()
    finally:
        db.close()


def _profile(db, profile_id):
    row = db.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    if row is None:
        raise ValueError("unknown profile")
    return row


def create_profile(name):
    name = (name or "").strip()
    if not 1 <= len(name) <= 40:
        raise ValueError("name must be 1 to 40 characters")
    with _db() as db:
        db.execute("INSERT OR IGNORE INTO profiles (name) VALUES (?)", (name,))
        return dict(db.execute("SELECT * FROM profiles WHERE name = ?", (name,)).fetchone())


def list_profiles():
    with _db() as db:
        return [dict(r) for r in db.execute("SELECT id, name, xp, streak FROM profiles ORDER BY name")]


def _add_xp(db, profile_id, xp):
    """Adds XP and moves the daily streak: +1 if last active yesterday, reset if older, unchanged if already today."""
    p = _profile(db, profile_id)
    today = date.today()
    last = date.fromisoformat(p["last_active"]) if p["last_active"] else None
    streak = p["streak"] if last == today else p["streak"] + 1 if last == today - timedelta(days=1) else 1
    db.execute("UPDATE profiles SET xp = xp + ?, streak = ?, last_active = ? WHERE id = ?", (xp, streak, today.isoformat(), profile_id))


def record_attempt(profile_id, item, correct):
    if item not in ITEMS:
        raise ValueError("unknown item")
    with _db() as db:
        _profile(db, profile_id)
        row = db.execute("SELECT box FROM progress WHERE profile_id = ? AND item = ?", (profile_id, item)).fetchone()
        box = min((row["box"] if row else 1) + 1, 5) if correct else 1
        due = (datetime.now() + timedelta(days=BOX_DAYS[box])).isoformat(timespec="seconds")
        db.execute("INSERT INTO progress (profile_id, item, box, correct, attempts, due_at) VALUES (?, ?, ?, ?, 1, ?) "
                   "ON CONFLICT(profile_id, item) DO UPDATE SET box = ?, correct = correct + ?, attempts = attempts + 1, due_at = ?",
                   (profile_id, item, box, int(correct), due, box, int(correct), due))
        _add_xp(db, profile_id, XP_CORRECT if correct else XP_WRONG)
        return {"item": item, "box": box, "mastered": box >= MASTERED_BOX, "xp": _profile(db, profile_id)["xp"]}


def record_unit_pass(profile_id, unit_id, score):
    """Records a pass at score >= 0.8 (the intro posts 1). XP is awarded the first time only."""
    if unit_id not in UNITS:
        raise ValueError("unknown unit")
    if not 0 <= score <= 1:
        raise ValueError("score must be between 0 and 1")
    with _db() as db:
        _profile(db, profile_id)
        if score < PASS_SCORE:
            return {"passed": False}
        new = db.execute("SELECT 1 FROM unit_pass WHERE profile_id = ? AND unit_id = ?", (profile_id, unit_id)).fetchone() is None
        db.execute("INSERT INTO unit_pass (profile_id, unit_id, score, passed_at) VALUES (?, ?, ?, ?) "
                   "ON CONFLICT(profile_id, unit_id) DO UPDATE SET score = MAX(score, excluded.score)",
                   (profile_id, unit_id, score, datetime.now().isoformat(timespec="seconds")))
        if new:
            _add_xp(db, profile_id, XP_UNIT)
        return {"passed": True, "first_time": new}


def due_items(profile_id, limit=20):
    with _db() as db:
        rows = db.execute("SELECT item FROM progress WHERE profile_id = ? AND due_at <= ? ORDER BY due_at LIMIT ?",
                          (profile_id, datetime.now().isoformat(timespec="seconds"), limit))
        return [{"item": r["item"], "display": display(r["item"])} for r in rows]


def curriculum_with_progress(profile_id):
    with _db() as db:
        p = _profile(db, profile_id)
        boxes = {r["item"]: r["box"] for r in db.execute("SELECT item, box FROM progress WHERE profile_id = ?", (profile_id,))}
        passed = {r["unit_id"] for r in db.execute("SELECT unit_id FROM unit_pass WHERE profile_id = ?", (profile_id,))}
    units, prev_passed = [], True  # the first unit is always open
    for u in CURRICULUM:
        state = "passed" if u["id"] in passed else "open" if prev_passed else "locked"
        items = [{"item": i, "display": display(i), "box": boxes.get(i, 0), "mastered": boxes.get(i, 0) >= MASTERED_BOX} for i in u["items"]]
        units.append({**u, "items": items, "state": state, "mastered": sum(i["mastered"] for i in items)})
        prev_passed = u["id"] in passed
    return {"profile": {"id": p["id"], "name": p["name"], "xp": p["xp"], "streak": p["streak"]}, "units": units,
            "due": len(due_items(profile_id)), "complete": all(u["state"] == "passed" for u in units)}
