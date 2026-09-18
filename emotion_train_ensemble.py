"""
IMPROVED ASL - Procrustes Full-Mesh Ensemble Classifier
======================================================
Aligns full 468-point 3D face mesh using Procrustes analysis
and blends XGBoost + MLP models.
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from emotion_config import EMOTIONS, EMOTION_KEYPOINTS_DIR, EMOTION_MODELS_DIR

print("=" * 70)
print("EMOTION FULL-MESH PROCRUSTES ENSEMBLE TRAINER")
print("=" * 70)

EMOTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
X_path = EMOTION_KEYPOINTS_DIR / "emotion_keypoints.npy"
y_path = EMOTION_KEYPOINTS_DIR / "emotion_labels.npy"

X_raw = np.load(X_path).astype(np.float32)
y_raw = np.load(y_path).astype(np.int64)

# 2. Procrustes Alignment across full 468 mesh
def align_full_mesh(X_flat):
    N = len(X_flat)
    mesh = X_flat[:, :1404].reshape(N, 468, 3)
    
    # Center each face on its mean centroid
    centroids = np.mean(mesh, axis=1, keepdims=True)
    centered_mesh = mesh - centroids
    
    # Scale each face to unit norm
    norms = np.linalg.norm(centered_mesh, axis=(1, 2), keepdims=True)
    norms = np.where(norms == 0, 1e-6, norms)
    aligned_mesh = centered_mesh / norms
    
    # Flatten mesh back to (N, 1404)
    aligned_flat = aligned_mesh.reshape(N, 1404)
    
    # Append any remaining pose/extra features
    return np.hstack([aligned_flat, X_flat[:, 1404:]])

print("Applying Procrustes mesh alignment on full 468 keypoints...")
X_aligned = align_full_mesh(X_raw)
print(f"Full Feature Matrix Shape: {X_aligned.shape}")

# 3. Stratified Split (80 Train / 10 Val / 10 Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_aligned, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 4. Train XGBoost
print("\n[1/2] Training Full-Mesh XGBoost...")
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

# 5. Train Deep MLP
print("\n[2/2] Training Full-Mesh MLP...")
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    early_stopping=True
)
mlp.fit(X_train, y_train)

# 6. Ensemble Predictions via Probability Averaging
print("\nEvaluating Soft-Voting Ensemble on Holdout Test Set...")
prob_xgb = xgb.predict_proba(X_test)
prob_mlp = mlp.predict_proba(X_test)

# Soft Voting (60% XGBoost weight + 40% MLP weight)
ensemble_probs = (0.6 * prob_xgb) + (0.4 * prob_mlp)
y_pred = np.argmax(ensemble_probs, axis=1)

# Save Models
joblib.dump(xgb, EMOTION_MODELS_DIR / "emotion_xgb_mesh.pkl")
joblib.dump(mlp, EMOTION_MODELS_DIR / "emotion_mlp_mesh.pkl")
print(f"\n✓ Models saved to: {EMOTION_MODELS_DIR}")

# 7. Evaluation
print("\n" + "=" * 70)
print(f"🎯 ENSEMBLE TEST ACCURACY: {np.mean(y_pred == y_test) * 100:.2f}%")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=[EMOTIONS[i] for i in range(len(EMOTIONS))]))
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))