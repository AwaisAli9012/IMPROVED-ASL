"""
EMOTION TRAINER (CORRECTED) - single frame, real 68-point features, no leakage tricks
=====================================================================================
What was wrong before and what this does instead:
  * Old: treated 68-point data as MediaPipe's 468 points  -> garbage features.
    Now: features come from emotion_features.py using the real 68-point numbering.
  * Old: "temporal windows" of 5 consecutive rows. Rows are unrelated still photos, so that was
    meaningless, and overlapping windows put the same photo in train AND test.
    Now: one face = one sample.
  * Old: trained on every row, including ~28% synthetic (augmented) rows that carry z-values the
    extractor never produces. Augmented copies of a test face end up in training.
    Now: only rows straight from the extractor (z == 0) are used. Class imbalance is handled with
    sample weights instead of synthetic copies.
  * Old: 466 exact-duplicate rows. Now removed before splitting.

Run:  python train_emotion.py
Needs emotion_keypoints.npy / emotion_labels.npy (or the *_balanced files) in EMOTION_KEYPOINTS_DIR.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    from emotion_config import EMOTIONS, EMOTION_KEYPOINTS_DIR, EMOTION_MODELS_DIR
except ImportError:
    EMOTIONS = {0: 'happy', 1: 'neutral', 2: 'sad', 3: 'angry'}
    EMOTION_KEYPOINTS_DIR = HERE / "emotion_keypoints_extracted"
    EMOTION_MODELS_DIR = HERE / "Emotion_Models"

from emotion_features import (FEATURE_NAMES, N_FEATURES, ORIGIN_IDX, features_from_normalized,
                              ratios5)

KP_DIR = Path(os.getenv("EMOTION_KEYPOINTS_DIR", str(EMOTION_KEYPOINTS_DIR)))
OUT_DIR = Path(os.getenv("EMOTION_MODELS_DIR", str(EMOTION_MODELS_DIR)))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EMOTION TRAINER - SINGLE FRAME, 68-POINT FEATURES, CLEAN ROWS ONLY")
print("=" * 70)

# ------------------------------------------------------------------ load
x_file = next((KP_DIR / n for n in ("emotion_keypoints.npy", "emotion_keypoints_balanced.npy")
               if (KP_DIR / n).exists()), None)
y_file = next((KP_DIR / n for n in ("emotion_labels.npy", "emotion_labels_balanced.npy")
               if (KP_DIR / n).exists()), None)
if x_file is None or y_file is None:
    raise FileNotFoundError(f"No keypoint files in {KP_DIR}")
print(f"Loading {x_file.name} and {y_file.name} from {KP_DIR}")

X_raw = np.load(x_file).astype(np.float32)
y_all = np.load(y_file).astype(np.int64)
assert X_raw.shape[1] >= 1404 and len(X_raw) == len(y_all), "unexpected array shapes"

L = X_raw[:, :1404].reshape(-1, 468, 3)
if np.abs(L[:, 68:, :]).max() != 0:
    print("WARNING: values found beyond landmark 67 - this file is not pure 68-point data")
pts = L[:, :68, :2].copy()

# ------------------------------------------------------------------ keep only extractor-pure rows
has_z = np.abs(L[:, :68, 2]).max(axis=1) > 0
names = {k: v for k, v in EMOTIONS.items()}
print("\nRows per class (all / kept after removing augmented rows with z-values):")
for k in sorted(names):
    m = y_all == k
    print(f"  {names[k]:8s} all={m.sum():6d}   pure={(m & ~has_z).sum():6d}")

keep = ~has_z

# sanity: normalised data must have landmark 30 at the origin and eye distance 1
eye_dist = np.linalg.norm(pts[:, 45] - pts[:, 36], axis=1)
ok_norm = (np.abs(pts[:, ORIGIN_IDX]).max(axis=1) < 1e-4) & (np.abs(eye_dist - 1.0) < 0.01)
ok_range = np.abs(pts).max(axis=(1, 2)) <= 6.0
dropped_bad = int((keep & ~(ok_norm & ok_range)).sum())
keep &= ok_norm & ok_range
print(f"Dropped {dropped_bad} rows with broken normalisation / extreme coordinates")

pts, y = pts[keep], y_all[keep]
X_stored_ratios = X_raw[keep][:, 1404:1409] if X_raw.shape[1] == 1409 else None

# ------------------------------------------------------------------ features
feats = features_from_normalized(pts)
assert feats.shape[1] == N_FEATURES
if X_stored_ratios is not None:
    same = np.allclose(feats[:, 134:139], X_stored_ratios, atol=1e-3)
    print(f"Recomputed ratios match the extractor's stored ratios: {same}")

finite = np.isfinite(feats).all(axis=1)
feats, y = feats[finite], y[finite]

_, uniq_idx = np.unique(np.round(feats, 4), axis=0, return_index=True)
uniq_idx = np.sort(uniq_idx)
print(f"Removed {len(feats) - len(uniq_idx)} exact duplicate rows")
feats, y = feats[uniq_idx], y[uniq_idx]

print("\nFinal training pool:")
for k in sorted(names):
    print(f"  {names[k]:8s} {(y == k).sum():6d}")
print(f"  total    {len(y):6d} samples x {feats.shape[1]} features")

# ------------------------------------------------------------------ split (test set never touched in training)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr_idx, te_idx = next(sss.split(feats, y))
X_trainval, y_trainval = feats[tr_idx], y[tr_idx]
X_test, y_test = feats[te_idx], y[te_idx]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=7)
tr2, va2 = next(sss2.split(X_trainval, y_trainval))
X_tr, y_tr = X_trainval[tr2], y_trainval[tr2]
X_va, y_va = X_trainval[va2], y_trainval[va2]
print(f"\nTrain {len(y_tr):,} | Validation (early stopping) {len(y_va):,} | Test {len(y_test):,}")

w_tr = compute_sample_weight("balanced", y_tr)

# ------------------------------------------------------------------ model
params = dict(
    n_estimators=800, learning_rate=0.05, max_depth=5, min_child_weight=2,
    subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0,
    objective="multi:softprob", tree_method="hist", eval_metric="mlogloss",
    random_state=42, n_jobs=-1,
)
print("\nTraining XGBoost with early stopping on the validation split...")
try:
    model = XGBClassifier(early_stopping_rounds=40, **params)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=50)
except TypeError:   # older xgboost
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)],
              early_stopping_rounds=40, verbose=50)

# ------------------------------------------------------------------ honest evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
macro = f1_score(y_test, y_pred, average="macro")
majority = np.bincount(y_test).max() / len(y_test)
labels = sorted(names)
print("\n" + "=" * 60)
print(f"Test accuracy: {acc * 100:.2f}%   Macro F1: {macro:.4f}   (always-guess-majority = {majority * 100:.1f}%)")
print("=" * 60)
print(classification_report(y_test, y_pred, labels=labels,
                            target_names=[names[k] for k in labels], zero_division=0))
print("Confusion matrix (rows = true, columns = predicted):")
print("          " + " ".join(f"{names[k][:7]:>7s}" for k in labels))
for i, k in enumerate(labels):
    print(f"{names[k]:9s} " + " ".join(f"{v:7d}" for v in confusion_matrix(y_test, y_pred, labels=labels)[i]))
print("\nCAVEAT: the dataset has no person IDs, so photos of the same person can sit in both train and test.")
print("Treat this as an upper bound; live webcam accuracy will be lower.")

# ------------------------------------------------------------------ save
neutral_id = next((k for k, v in names.items() if v == "neutral"), None)
neutral_median = np.median(X_trainval[y_trainval == neutral_id], axis=0) if neutral_id is not None else np.zeros(N_FEATURES)

joblib.dump(model, OUT_DIR / "emotion_model.pkl")
meta = {
    "classes": {str(k): names[k] for k in sorted(names)},
    "n_features": int(N_FEATURES),
    "feature_names": FEATURE_NAMES,
    "neutral_median": [float(v) for v in neutral_median],
    "n_train": int(len(y_tr)), "n_test": int(len(y_test)),
    "test_accuracy": float(acc), "macro_f1": float(macro),
    "trained_utc": datetime.now(timezone.utc).isoformat(),
    "note": "single-frame model, 68-point face_alignment landmarks, clean (non-augmented) rows only",
}
with open(OUT_DIR / "emotion_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"\n✓ Saved emotion_model.pkl and emotion_meta.json to {OUT_DIR}")
