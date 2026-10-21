"""Train the letter and digit classifier on hand landmarks. Picks the best of ExtraTrees, MLP and their soft vote by validation macro-F1.

Run: .venv/bin/python -m scripts.train_static
Writes artifacts/static.joblib.
"""
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from signbridge.config import ARTIFACTS, DIGITS, LETTERS, PROCESSED, SEED
from signbridge.landmarks import hand_features, mirror_hand
from signbridge.models import SoftVote

LABELS = LETTERS + DIGITS
N_LETTERS = len(LETTERS)


def load_static():
    """Features, labels, sources and splits. Image sources have no signer info, so they are split 70/15/15 per label
    at random; digit clips from ASL Citizen keep their official signer-disjoint split."""
    d = np.load(PROCESSED / "static.npz")
    y = np.array([LABELS.index(label) for label in d["label"]])
    split = d["split"].astype("<U5")
    rng = np.random.default_rng(SEED)
    for src in set(d["source"]) - {"digit-video"}:
        for lab in np.unique(y[d["source"] == src]):
            idx = rng.permutation(np.flatnonzero((d["source"] == src) & (y == lab)))
            a, b = int(len(idx) * 0.7), int(len(idx) * 0.85)
            split[idx[:a]], split[idx[a:b]], split[idx[b:]] = "train", "val", "test"
    X = np.stack([hand_features(h, a) for h, a in zip(d["hands"], d["aspect"])])
    return {"X": X, "y": y, "source": d["source"], "split": split, "hands": d["hands"], "aspect": d["aspect"]}


def mask_groups(p, y):
    """Zero the other group's probabilities, as the app does once the user picks ABC or 123 mode."""
    p = p.copy()
    letters = y < N_LETTERS
    p[letters, N_LETTERS:] = 0
    p[~letters, :N_LETTERS] = 0
    return p / np.clip(p.sum(1, keepdims=True), 1e-9, None)


def main():
    d = load_static()
    tr, va = d["split"] == "train", d["split"] == "val"
    mirrored = np.stack([hand_features(mirror_hand(h), a) for h, a in zip(d["hands"][tr], d["aspect"][tr])])
    X_train, y_train = np.concatenate([d["X"][tr], mirrored]), np.concatenate([d["y"][tr]] * 2)  # mirror: left-handed signers
    print(f"train {len(X_train)} (with mirrored copies)  val {va.sum()}  test {(d['split'] == 'test').sum()}", flush=True)
    candidates = {
        # min_samples_leaf keeps the saved model small: unpruned trees came to 450 MB. XGBoost was dropped: 27 min to train and the lowest score.
        "extra_trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=24, random_state=SEED),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True,
                                                              max_iter=400, random_state=SEED)),
    }
    scores = {}
    for name, est in candidates.items():
        t0 = time.time()
        est.fit(X_train, y_train)
        scores[name] = f1_score(d["y"][va], mask_groups(est.predict_proba(d["X"][va]), d["y"][va]).argmax(1), average="macro")
        print(f"{name}: val macro-F1 {scores[name]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    candidates["soft_vote"] = SoftVote(list(candidates.values()))
    scores["soft_vote"] = f1_score(d["y"][va], mask_groups(candidates["soft_vote"].predict_proba(d["X"][va]), d["y"][va]).argmax(1), average="macro")
    print(f"soft_vote: val macro-F1 {scores['soft_vote']:.3f}")
    best = max(scores, key=scores.get)
    joblib.dump({"model": candidates[best], "labels": LABELS, "kind": best}, ARTIFACTS / "static.joblib")
    print("saved artifacts/static.joblib:", best)


if __name__ == "__main__":
    main()
