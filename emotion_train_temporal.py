"""
IMPROVED ASL - Native Temporal XGBoost Trainer
==============================================
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from emotion_config import EMOTIONS, EMOTION_KEYPOINTS_DIR, EMOTION_MODELS_DIR

print("=" * 70)
print("NATIVE DATASET TEMPORAL TRAINER")
print("=" * 70)

# Load raw keypoint dataset directly (already 1409 features per frame)
X_raw = np.load(EMOTION_KEYPOINTS_DIR / "emotion_keypoints.npy").astype(np.float32)
y_raw = np.load(EMOTION_KEYPOINTS_DIR / "emotion_labels.npy").astype(np.int64)

WINDOW_SIZE = 5

def create_temporal_windows(X, y, window_size=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        if len(set(y[i : i + window_size])) == 1:
            X_seq.append(X[i : i + window_size].flatten())
            y_seq.append(y[i + window_size - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)

X_temp_win, y_temp_win = create_temporal_windows(X_raw, y_raw, window_size=WINDOW_SIZE)
print(f"Dataset Window Shape: {X_temp_win.shape}") # Expected: (N, 7045)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_temp_win, y_temp_win, test_size=0.20, random_state=42, stratify=y_temp_win
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_full, y_train_full, eval_set=[(X_test, y_test)], verbose=100)

model_path = EMOTION_MODELS_DIR / "emotion_temporal_xgb.pkl"
joblib.dump(xgb_model, model_path)
print(f"\n✓ Saved exact model to: {model_path}")

y_pred = xgb_model.predict(X_test)
print("\n" + classification_report(y_test, y_pred, target_names=[EMOTIONS[i] for i in range(len(EMOTIONS))]))