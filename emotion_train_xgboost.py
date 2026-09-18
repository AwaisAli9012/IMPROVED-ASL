"""
IMPROVED ASL - XGBoost Facial Landmark Emotion Classifier
=========================================================
Computes pairwise landmark distance matrices to solve 
neutral/sad/angry boundary confusion.
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from emotion_config import EMOTIONS, EMOTION_KEYPOINTS_DIR, EMOTION_MODELS_DIR

print("=" * 70)
print("EMOTION LANDMARK XGBOOST CLASSIFIER")
print("=" * 70)

EMOTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
X_path = EMOTION_KEYPOINTS_DIR / "emotion_keypoints.npy"
y_path = EMOTION_KEYPOINTS_DIR / "emotion_labels.npy"

if not X_path.exists() or not y_path.exists():
    raise FileNotFoundError("❌ Extracted keypoint files missing!")

X_raw = np.load(X_path).astype(np.float32)
y_raw = np.load(y_path).astype(np.int64)

# 2. Extract Pairwise Key Landmark Distance Matrix
def extract_pairwise_distances(X_flat):
    N = len(X_flat)
    coords = X_flat[:, :1404].reshape(N, 468, 3)[:, :68, :] # (N, 68, 3)
    
    diffs = coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :] # (N, 68, 68, 3)
    dist_matrices = np.linalg.norm(diffs, axis=-1) # (N, 68, 68)
    
    triu_idx = np.triu_indices(68, k=1)
    pairwise_vecs = dist_matrices[:, triu_idx[0], triu_idx[1]] # (N, 2278 features)
    
    return np.hstack([pairwise_vecs, X_flat[:, 1404:]])

print("Computing pairwise landmark distance matrices...")
X_dist = extract_pairwise_distances(X_raw)
print(f"Engineered Distance Feature Matrix Shape: {X_dist.shape}")

# 3. Stratified Split (80 Train / 10 Val / 10 Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_dist, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

# 4. Train Gradient Boosted Decision Trees
print("\nTraining XGBoost Classifier...")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_full, y_train_full,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# 5. Save Model
model_save_path = EMOTION_MODELS_DIR / "emotion_xgboost_model.pkl"
joblib.dump(xgb_model, model_save_path)
print(f"\n✓ Saved XGBoost model to: {model_save_path}")

# 6. Evaluation
print("\n" + "=" * 70)
print("FINAL HOLDOUT EVALUATION (XGBOOST)")
print("=" * 70)

y_pred = xgb_model.predict(X_test)
test_acc = np.mean(y_pred == y_test)
print(f"🎯 Test Accuracy: {test_acc * 100:.2f}%\n")

target_names = [EMOTIONS[i] for i in range(len(EMOTIONS))]

print("CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))
