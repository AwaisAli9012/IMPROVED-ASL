"""Train the emotion classifier on face blendshapes (FER2013). Picks the best of XGBoost, ExtraTrees, MLP and their soft vote by validation macro-F1.

Run: .venv/bin/python -m scripts.train_emotion
Writes artifacts/emotion.joblib.
"""
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from signbridge.config import ARTIFACTS, EMOTIONS, PROCESSED, SEED
from signbridge.models import SoftVote


def load_emotion():
    d = np.load(PROCESSED / "emotion.npz")
    return d["bs"], np.array([EMOTIONS.index(label) for label in d["label"]]), d["split"], list(d["names"])


def main():
    bs, y, split, names = load_emotion()
    tr, va = split == "train", split == "val"
    print(f"train {tr.sum()} val {va.sum()} test {(split == 'test').sum()}", flush=True)
    weights = compute_sample_weight("balanced", y[tr])  # FER2013 is dominated by happy and neutral
    candidates = {
        "xgboost": (XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                                  tree_method="hist", n_jobs=24, random_state=SEED), {"sample_weight": weights}),
        "extra_trees": (ExtraTreesClassifier(n_estimators=500, n_jobs=24, class_weight="balanced", random_state=SEED), {}),
        "mlp": (make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True,
                                                               max_iter=400, random_state=SEED)), {}),
    }
    models, scores = {}, {}
    for name, (est, fit_args) in candidates.items():
        t0 = time.time()
        est.fit(bs[tr], y[tr], **fit_args)
        models[name] = est
        scores[name] = f1_score(y[va], est.predict(bs[va]), average="macro")
        print(f"{name}: val macro-F1 {scores[name]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    models["soft_vote"] = SoftVote(list(models.values()))
    scores["soft_vote"] = f1_score(y[va], models["soft_vote"].predict_proba(bs[va]).argmax(1), average="macro")
    print(f"soft_vote: val macro-F1 {scores['soft_vote']:.3f}")
    best = max(scores, key=scores.get)
    neutral_mean = bs[tr & (y == EMOTIONS.index("neutral"))].mean(0)
    joblib.dump({"model": models[best], "labels": EMOTIONS, "names": names, "neutral_mean": neutral_mean, "kind": best},
                ARTIFACTS / "emotion.joblib")
    print("saved artifacts/emotion.joblib:", best)


if __name__ == "__main__":
    main()
