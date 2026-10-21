"""Export one reference animation per sign for the signing figure and the dictionary.

The clearest training example of each sign (highest model probability for its own class) is centred on the
shoulders, scaled by shoulder width, lightly smoothed and written to web/static/signs/<label>.json.
Output stays local: web/static/signs is gitignored because it is derived from ASL Citizen signers' landmarks,
which the dataset license does not let us redistribute.

Run: .venv/bin/python -m scripts.export_references
"""
import json
import warnings

import joblib
import numpy as np

from scripts.train_static import LABELS, load_static, mask_groups
from signbridge import learn
from signbridge.config import ARTIFACTS, LH, LIPS, POSE, PROCESSED, RH, SIGNS_DIR
from signbridge.landmarks import sequence_features
from signbridge.models import WordPredictor
from signbridge.vocab import display

FPS = 24


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def word_frames(seq):
    """(64, 89, 2) with NaN -> list of {lh, rh, pose, lips} in shoulder-width units, or None if the shoulders are missing."""
    shoulders = seq[:, 43:45]
    center = _nanmean(shoulders.reshape(-1, 2), 0)
    scale = _nanmean(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=-1), None)
    if not np.isfinite(center).all() or not np.isfinite(scale) or scale < 1e-3:
        return None
    pts = (seq - center) / scale
    padded = np.concatenate([pts[:1], pts, pts[-1:]])  # 3-frame moving average to calm landmark jitter
    pts = _nanmean(np.stack([padded[:-2], padded[1:-1], padded[2:]]), 0)
    frames = []
    for frame in pts:
        out = {}
        for key, block in (("lh", LH), ("rh", RH), ("pose", POSE), ("lips", LIPS)):
            b = frame[block]
            out[key] = np.round(b, 3).tolist() if np.isfinite(b).all() else None
        frames.append(out)
    return frames


def hand_frame(hand, aspect):
    """(21, 3) image-normalised hand -> one {rh} frame: wrist at the origin, middle knuckle one unit straight up."""
    p = hand[:, :2].astype(np.float64).copy()
    p[:, 0] *= aspect
    p -= p[0]
    p /= np.linalg.norm(p[9]) + 1e-9
    turn = -np.pi / 2 - np.arctan2(p[9, 1], p[9, 0])
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    return [{"rh": np.round(p @ rot.T, 3).tolist()}]


def write(label, kind, frames, extra=None):
    SIGNS_DIR.mkdir(parents=True, exist_ok=True)
    (SIGNS_DIR / f"{label}.json").write_text(json.dumps({"label": label, "display": display(label), "kind": kind, **(extra or {}), "frames": frames}, separators=(",", ":")))


def export_words():
    d = np.load(PROCESSED / "words.npz")
    X, y = d["X"].astype(np.float32), d["y"].astype(int)
    train = np.flatnonzero(d["split"] == "train")
    model = WordPredictor()
    conf = model.proba(np.stack([sequence_features(X[i]) for i in train]))[np.arange(len(train)), y[train]]
    for c, label in enumerate(model.labels):
        mine = np.flatnonzero(y[train] == c)
        for i in mine[np.argsort(-conf[mine])]:  # best first, until one has usable shoulders
            frames = word_frames(X[train[i]])
            if frames:
                write(label, "word", frames, {"fps": FPS})
                break
        else:
            print("no usable clip for", label)
    return len(model.labels)


def export_hands():
    d = load_static()
    model = joblib.load(ARTIFACTS / "static.joblib")["model"]
    train = np.flatnonzero(d["split"] == "train")
    conf = mask_groups(model.predict_proba(d["X"][train]), d["y"][train])[np.arange(len(train)), d["y"][train]]
    for c, label in enumerate(LABELS):
        mine = np.flatnonzero(d["y"][train] == c)
        best = train[mine[np.argmax(conf[mine])]]
        write(label, "hand", hand_frame(d["hands"][best], d["aspect"][best]))
    return len(LABELS)


def write_index():
    topic = {}
    for unit in learn.CURRICULUM:
        for item in unit["items"]:
            topic[item] = unit["title"] if unit["kind"] == "word" else "Numbers" if item.isdigit() else "Alphabet"
    kinds = {"word": "Words", "hand": "Letters and numbers"}
    rows = []
    for unit in learn.CURRICULUM:
        for item in unit["items"]:
            if (SIGNS_DIR / f"{item}.json").exists():
                rows.append({"label": item, "display": display(item), "kind": kinds[unit["kind"]], "topic": topic[item]})
    (SIGNS_DIR / "index.json").write_text(json.dumps(rows))
    return len(rows)


if __name__ == "__main__":
    print(export_words(), "word animations;", export_hands(), "letter and digit animations;", write_index(), "indexed")
    size = sum(f.stat().st_size for f in SIGNS_DIR.glob("*.json")) / 1e6
    print(f"web/static/signs is {size:.1f} MB")
