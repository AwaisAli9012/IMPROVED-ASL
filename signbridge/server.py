"""Flask app: pages plus a small stateless JSON API. Run: .venv/bin/python -m signbridge.server"""
import json
import os
from functools import cache

import joblib
import numpy as np
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory

from . import config, learn, nlg
from .landmarks import emotion_features, frames_to_array, hand_features, prepare_sequence, sequence_features
from .models import WordPredictor
from .vocab import display, word_labels

load_dotenv(config.ROOT / ".env")
app = Flask(__name__, template_folder=str(config.WEB / "templates"), static_folder=str(config.WEB / "static"))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
MAX_FRAMES, MAX_GLOSSES = 120, 30


@app.errorhandler(ValueError)
def bad_request(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(FileNotFoundError)
def model_missing(e):
    return jsonify(error="model not trained yet"), 503


@app.errorhandler(413)
def too_large(e):
    return jsonify(error="request too large"), 413


@cache
def get_words():
    return WordPredictor()


@cache
def get_static():
    return joblib.load(config.ARTIFACTS / "static.joblib")


@cache
def get_emotion():
    return joblib.load(config.ARTIFACTS / "emotion.joblib")


@cache
def thresholds():
    """Acceptance thresholds: the ones tuned on validation data when a report exists, else the config defaults."""
    out = {"word": config.WORD_ACCEPT, "static": config.STATIC_ACCEPT}
    for key, task in (("word", "words"), ("static", "static")):
        f = config.REPORTS / f"{task}.json"
        if f.exists() and json.loads(f.read_text()).get("accept_threshold") is not None:
            out[key] = json.loads(f.read_text())["accept_threshold"]
    return out


def body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def array(value, shape, name):
    try:
        a = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not numeric") from None
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite numbers of shape {list(shape)}")
    return a


def top(p, labels, k):
    return [{"label": labels[i], "display": display(labels[i]), "prob": float(p[i])} for i in np.argsort(p)[::-1][:k]]


# ---- pages ----------------------------------------------------------------

LANDING = config.WEB / "scrollcraft" / "builds" / "signbridge"
LANDING_FILES = {"scrollcraft.js", "scrollcraft.css", "landing.css", "landing.js"}  # a whitelist: BRIEF.md stays private


@app.get("/")
def landing():
    if not (LANDING / "index.html").exists():
        return redirect("/app")
    return send_from_directory(LANDING, "index.html")


@app.get("/<name>")
def landing_file(name):
    if name not in LANDING_FILES:
        abort(404)
    return send_from_directory(LANDING, name)


@app.get("/app")
def converse_page():
    return render_template("converse.html")


@app.get("/learn")
def learn_page():
    return render_template("learn.html")


@app.get("/learn/<unit_id>")
def lesson_page(unit_id):
    if unit_id not in learn.UNITS and unit_id != "review":  # "review" is the spaced-repetition session
        return "unknown unit", 404
    return render_template("lesson.html", unit_id=unit_id)


@app.get("/dictionary")
def dictionary_page():
    return render_template("dictionary.html")


@app.get("/metrics")
def metrics_page():
    return render_template("metrics.html")


@app.get("/certificate")
def certificate_page():
    return render_template("certificate.html")


# ---- recognition and language API -------------------------------------------

@app.get("/api/config")
def api_config():
    names = ("POSE_IDX", "LIP_IDX", "LETTERS", "DIGITS", "EMOTIONS", "STATIC_STREAK", "SPELL_GAP_FRAMES", "SEG_START_FRAMES",
             "SEG_END_FRAMES", "SEG_MIN_FRAMES", "SEG_MAX_FRAMES", "BROW_RAISE_DELTA", "BROW_DOWN_DELTA",
             "EMOTION_INTERVAL_MS", "EMOTION_EMA")
    return jsonify({n: getattr(config, n) for n in names} | {
        "thresholds": thresholds(), "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "words": [{"label": w, "display": display(w)} for w in word_labels()]})


@app.post("/api/predict/word")
def predict_word():
    frames = body().get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError(f"frames must be a list of 1 to {MAX_FRAMES} frames")
    feats = sequence_features(prepare_sequence(frames_to_array(frames)))
    model = get_words()
    result = top(model.proba(feats[None])[0], model.labels, 5)
    return jsonify(top=result, accept=result[0]["prob"] >= thresholds()["word"])


@app.post("/api/predict/static")
def predict_static():
    data = body()
    hand = array(data.get("hand"), (config.HAND_N, 3), "hand")
    aspect = data.get("aspect")
    if not isinstance(aspect, (int, float)) or not 0.2 <= aspect <= 5:
        raise ValueError("aspect must be a number between 0.2 and 5")
    if data.get("mode") not in ("letters", "digits"):
        raise ValueError("mode must be letters or digits")
    art = get_static()
    p = art["model"].predict_proba(hand_features(hand, aspect)[None])[0].copy()
    n = len(config.LETTERS)
    if data["mode"] == "letters":
        p[n:] = 0
    else:
        p[:n] = 0
    result = top(p / max(p.sum(), 1e-9), art["labels"], 3)
    return jsonify(top=result, accept=result[0]["prob"] >= thresholds()["static"])


@app.post("/api/emotion")
def predict_emotion():
    data = body()
    bs = array(data.get("bs"), (52,), "bs")
    baseline = None if data.get("baseline") is None else array(data["baseline"], (52,), "baseline")
    art = get_emotion()
    p = art["model"].predict_proba(emotion_features(bs, baseline, art["neutral_mean"])[None])[0]
    probs = {label: float(v) for label, v in zip(art["labels"], p)}
    return jsonify(probs=probs, label=max(probs, key=probs.get))


@app.post("/api/sentence")
def make_sentence():
    data = body()
    glosses = data.get("glosses")
    if not isinstance(glosses, list) or not 1 <= len(glosses) <= MAX_GLOSSES or not all(isinstance(g, str) and 0 < len(g) <= 40 for g in glosses):
        raise ValueError(f"glosses must be 1 to {MAX_GLOSSES} short strings")
    tone, question = data.get("tone", "neutral"), data.get("question", "none")
    if not isinstance(question, str) or question not in nlg.END:
        raise ValueError("question must be none, yesno or wh")
    return jsonify(nlg.gloss_to_english(glosses, tone if isinstance(tone, str) and tone in nlg.TONE_STYLE else "neutral", question))


@app.post("/api/gloss")
def make_gloss():
    text = body().get("text")
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 500:
        raise ValueError("text must be 1 to 500 characters")
    return jsonify(nlg.english_to_gloss(text))


# ---- learning API -----------------------------------------------------------

@app.get("/api/profiles")
def get_profiles():
    return jsonify(learn.list_profiles())


@app.post("/api/profiles")
def post_profile():
    return jsonify(learn.create_profile(body().get("name")))


@app.get("/api/curriculum")
def get_curriculum():
    return jsonify(learn.CURRICULUM)


@app.get("/api/learn/<int:profile_id>")
def get_progress(profile_id):
    return jsonify(learn.curriculum_with_progress(profile_id))


@app.get("/api/learn/<int:profile_id>/review")
def get_review(profile_id):
    return jsonify(learn.due_items(profile_id))


@app.post("/api/learn/attempt")
def post_attempt():
    data = body()
    if not isinstance(data.get("correct"), bool):
        raise ValueError("correct must be true or false")
    return jsonify(learn.record_attempt(data.get("profile_id"), data.get("item"), data["correct"]))


@app.post("/api/learn/unit_pass")
def post_unit_pass():
    data = body()
    score = data.get("score")
    if not isinstance(score, (int, float)):
        raise ValueError("score must be a number")
    return jsonify(learn.record_unit_pass(data.get("profile_id"), data.get("unit_id"), score))


@app.get("/reports/<path:name>")
def report_file(name):
    return send_from_directory(config.REPORTS, name)


@app.get("/api/reports")
def get_reports():
    return jsonify({f.stem: json.loads(f.read_text()) for f in sorted(config.REPORTS.glob("*.json"))})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
