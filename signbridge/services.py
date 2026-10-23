"""Model loading, input validation and prediction. Routes stay thin and call these."""
import json
import os
from functools import cache

import joblib
import numpy as np

from . import config, nlg
from .landmarks import emotion_features, frames_to_array, hand_features, prepare_sequence, sequence_features
from .models import WordPredictor
from .vocab import display, word_labels


@cache
def get_words() -> WordPredictor:
    return WordPredictor()


@cache
def get_static() -> dict:
    return joblib.load(config.ARTIFACTS / "static.joblib")


@cache
def get_emotion() -> dict:
    return joblib.load(config.ARTIFACTS / "emotion.joblib")


def warm_up() -> None:
    for load in (get_words, get_static, get_emotion):
        try:
            load()
        except (FileNotFoundError, OSError):
            pass  # /api/health reports the missing artifact


@cache
def thresholds() -> dict:
    """Acceptance thresholds: the ones tuned on validation data when a report exists, else the config defaults."""
    out = {"word": config.WORD_ACCEPT, "static": config.STATIC_ACCEPT}
    for key, task in (("word", "words"), ("static", "static")):
        f = config.REPORTS / f"{task}.json"
        if f.exists() and json.loads(f.read_text()).get("accept_threshold") is not None:
            out[key] = json.loads(f.read_text())["accept_threshold"]
    return out


def array(value, shape: tuple, name: str) -> np.ndarray:
    try:
        a = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not numeric") from None
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite numbers of shape {list(shape)}")
    return a


def top(p, labels: list, k: int) -> list[dict]:
    return [{"label": labels[i], "display": display(labels[i]), "prob": float(p[i])} for i in np.argsort(p)[::-1][:k]]


def gemini_enabled() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def app_config() -> dict:
    names = ("POSE_IDX", "LIP_IDX", "LETTERS", "DIGITS", "EMOTIONS", "STATIC_STREAK", "SPELL_GAP_FRAMES", "SEG_START_FRAMES",
             "SEG_END_FRAMES", "SEG_MIN_FRAMES", "SEG_MAX_FRAMES", "BROW_RAISE_DELTA", "BROW_DOWN_DELTA",
             "EMOTION_INTERVAL_MS", "EMOTION_EMA")
    return {n: getattr(config, n) for n in names} | {
        "thresholds": thresholds(), "gemini": gemini_enabled(),
        "words": [{"label": w, "display": display(w)} for w in word_labels()]}


def health() -> dict:
    a = config.ARTIFACTS
    models = {"words": (a / "words.pt").exists(), "static": (a / "static.joblib").exists(), "emotion": (a / "emotion.joblib").exists()}
    try:  # the vocabulary file and the trained model must list the same words in the same order
        consistent = get_words().labels == word_labels()
    except (FileNotFoundError, OSError):
        consistent = False
    return {"status": "ok" if all(models.values()) and consistent else "degraded", "models": models,
            "vocabulary_matches_model": consistent, "words": len(word_labels()), "gemini": gemini_enabled()}


def predict_word(frames) -> dict:
    if not isinstance(frames, list) or not 1 <= len(frames) <= config.MAX_FRAMES:
        raise ValueError(f"frames must be a list of 1 to {config.MAX_FRAMES} frames")
    feats = sequence_features(prepare_sequence(frames_to_array(frames)))
    model = get_words()
    result = top(model.proba(feats[None])[0], model.labels, 5)
    return {"top": result, "accept": result[0]["prob"] >= thresholds()["word"]}


def predict_static(hand, aspect, mode) -> dict:
    hand = array(hand, (config.HAND_N, 3), "hand")
    if not isinstance(aspect, (int, float)) or not 0.2 <= aspect <= 5:
        raise ValueError("aspect must be a number between 0.2 and 5")
    if mode not in ("letters", "digits"):
        raise ValueError("mode must be letters or digits")
    art = get_static()
    p = art["model"].predict_proba(hand_features(hand, aspect)[None])[0].copy()
    n = len(config.LETTERS)
    if mode == "letters":
        p[n:] = 0
    else:
        p[:n] = 0
    result = top(p / max(p.sum(), 1e-9), art["labels"], 3)
    return {"top": result, "accept": result[0]["prob"] >= thresholds()["static"]}


def predict_emotion(bs, baseline) -> dict:
    bs = array(bs, (52,), "bs")
    baseline = None if baseline is None else array(baseline, (52,), "baseline")
    art = get_emotion()
    p = art["model"].predict_proba(emotion_features(bs, baseline, art["neutral_mean"])[None])[0]
    probs = {label: float(v) for label, v in zip(art["labels"], p)}
    return {"probs": probs, "label": max(probs, key=probs.get)}


def make_sentence(glosses, tone, question) -> dict:
    if not isinstance(glosses, list) or not 1 <= len(glosses) <= config.MAX_GLOSSES or not all(isinstance(g, str) and 0 < len(g) <= 40 for g in glosses):
        raise ValueError(f"glosses must be 1 to {config.MAX_GLOSSES} short strings")
    if not isinstance(question, str) or question not in nlg.END:
        raise ValueError("question must be none, yesno or wh")
    return nlg.gloss_to_english(glosses, tone if isinstance(tone, str) and tone in nlg.TONE_STYLE else "neutral", question)


def make_gloss(text) -> dict:
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 500:
        raise ValueError("text must be 1 to 500 characters")
    return nlg.english_to_gloss(text)
