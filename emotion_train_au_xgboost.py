"""
IMPROVED ASL - Advanced Action Unit (AU) Feature Engineering
============================================================
Uses normalized facial landmark ratios to resolve Neutral vs Sad vs Angry.
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from emotion_config import EMOTIONS, EMOTION_KEYPOINTS_DIR, EMOTION_MODELS_DIR

print("=" * 70)
print("EMOTION ACTION UNIT (AU) FEATURE EXTRACTOR & TRAINER")
print("=" * 70)

# Load Data
X_path = EMOTION_KEYPOINTS_DIR / "emotion_keypoints.npy"
y_path = EMOTION_KEYPOINTS_DIR / "emotion_labels.npy"

X_raw = np.load(X_path).astype(np.float32)
y_raw = np.load(y_path).astype(np.int64)

def extract_au_features(X_flat):
    N = len(X_flat)
    coords = X_flat[:, :1404].reshape(N, 468, 3)
    
    # Key MediaPipe Facial Indices
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    NOSE_TIP = 1
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    MOUTH_TOP = 0
    MOUTH_BOTTOM = 17
    LEFT_BROW_INNER = 55
    RIGHT_BROW_INNER = 285
    CHIN = 152
    
    # 1. Scale Normalization (Inter-ocular distance)
    eye_dist = np.linalg.norm(coords[:, LEFT_EYE_OUTER] - coords[:, RIGHT_EYE_OUTER], axis=-1, keepdims=True)
    eye_dist = np.where(eye_dist == 0, 1e-6, eye_dist) # Prevent div by zero
    
    # Center landmarks on nose tip
    centered_coords = coords - coords[:, NOSE_TIP:NOSE_TIP+1, :]
    normalized_coords = centered_coords / eye_dist[:, :, np.newaxis]
    
    # 2. Specific Facial Action Unit Ratios
    # AU4: Inner Brow Distance (Angry/Sad)
    inner_brow_dist = np.linalg.norm(coords[:, LEFT_BROW_INNER] - coords[:, RIGHT_BROW_INNER], axis=-1, keepdims=True) / eye_dist
    
    # AU12/AU15: Lip Corner Height Relative to Nose (Sad vs Happy)
    mouth_left_y = normalized_coords[:, MOUTH_LEFT, 1:2]
    mouth_right_y = normalized_coords[:, MOUTH_RIGHT, 1:2]
    mouth_y_avg = (mouth_left_y + mouth_right_y) / 2.0
    
    # Mouth Width & Height
    mouth_width = np.linalg.norm(coords[:, MOUTH_LEFT] - coords[:, MOUTH_RIGHT], axis=-1, keepdims=True) / eye_dist
    mouth_height = np.linalg.norm(coords[:, MOUTH_TOP] - coords[:, MOUTH_BOTTOM], axis=-1, keepdims=True) / eye_dist
    
    # Brow to Eye Vertical Offset
    brow_eye_left = np.abs(normalized_coords[:, LEFT_BROW_INNER, 1:2] - normalized_coords[:, LEFT_EYE_OUTER, 1:2])
    brow_eye_right = np.abs(normalized_coords[:, RIGHT_BROW_INNER, 1:2] - normalized_coords[:, RIGHT_EYE_OUTER, 1:2])
    
    # Flatten Normalized Core Mesh (68 key points)
    core_mesh_norm = normalized_coords[:, :68, :].reshape(N, -1)
    
    # Combine Action Units with Core Normalized Mesh
    au_features = np.hstack([
        core_mesh_norm,
        inner_brow_dist,
        mouth_y_avg,
        mouth_width,
        mouth_height,
        brow_eye_left,
        brow_eye_right,
        X_flat[:, 1404:] # Append pose/extra features
    ])
    
    return au_features

print("Extracting Scale-Invariant Action Unit Features...")
X_au = extract_au_features(X_raw)
print(f"Feature Matrix Shape: {X_au.shape}")

# Stratified Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_au, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

# Train XGBoost with tuned hyper-params
print("\nTraining Action-Unit XGBoost Model...")
xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.75,
    gamma=0.1,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_full, y_train_full,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# Save
model_save_path = EMOTION_MODELS_DIR / "emotion_xgboost_au_model.pkl"
joblib.dump(xgb_model, model_save_path)
print(f"\n✓ Model saved to: {model_save_path}")

# Evaluate
y_pred = xgb_model.predict(X_test)
print("\n" + "=" * 70)
print(f"🎯 NEW TEST ACCURACY: {np.mean(y_pred == y_test) * 100:.2f}%")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=[EMOTIONS[i] for i in range(len(EMOTIONS))]))
print(confusion_matrix(y_test, y_pred))