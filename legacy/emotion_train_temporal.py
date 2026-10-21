"""
IMPROVED ASL - Native Temporal XGBoost Trainer with Fixed Geometric Features
=============================================================================
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from emotion_config import EMOTIONS, EMOTION_KEYPOINTS_DIR, EMOTION_MODELS_DIR

print("=" * 70)
print("NATIVE DATASET TEMPORAL TRAINER (WITH FIXED GEOMETRIC FEATURES)")
print("=" * 70)

# 1. Load Raw Data
X_path = EMOTION_KEYPOINTS_DIR / "emotion_keypoints.npy"
y_path = EMOTION_KEYPOINTS_DIR / "emotion_labels.npy"

if not X_path.exists() or not y_path.exists():
    raise FileNotFoundError("❌ Keypoint files missing. Run landmark extraction first!")

X_raw = np.load(X_path).astype(np.float32)
y_raw = np.load(y_path).astype(np.int64)

# 2. Extract Geometric Relative Features using exact MediaPipe Indices
def extract_advanced_features(X_flat):
    N = len(X_flat)
    coords = X_flat[:, :1404].reshape(N, 468, 3)
    
    # Anchors: Nose tip (Index 1) for relative translation invariance
    nose_tip = coords[:, 1:2, :]
    relative_coords = (coords - nose_tip).reshape(N, -1)
    
    # --- MediaPipe 468 Landmark Mapping ---
    # Lip top (13), Lip bottom (14), Left corner (61), Right corner (291)
    lip_top_bottom = np.linalg.norm(coords[:, 13, :] - coords[:, 14, :], axis=-1, keepdims=True)
    lip_left_right = np.linalg.norm(coords[:, 61, :] - coords[:, 291, :], axis=-1, keepdims=True)
    
    # Eyebrows: Left inner (55), Right inner (285), Left outer (70), Right outer (300)
    brow_inner = np.linalg.norm(coords[:, 55, :] - coords[:, 285, :], axis=-1, keepdims=True)
    
    # Eye-to-brow heights: Left brow (70) to Left eye top (159), Right brow (300) to Right eye top (386)
    left_brow_height = np.linalg.norm(coords[:, 70, :] - coords[:, 159, :], axis=-1, keepdims=True)
    right_brow_height = np.linalg.norm(coords[:, 300, :] - coords[:, 386, :], axis=-1, keepdims=True)
    
    # Smile/Frown curvature: Corner Y relative to center Y
    mouth_center_y = (coords[:, 13, 1] + coords[:, 14, 1]) / 2.0
    left_corner_y = coords[:, 61, 1] - mouth_center_y
    right_corner_y = coords[:, 291, 1] - mouth_center_y
    lip_corners_y = np.stack([left_corner_y, right_corner_y], axis=-1)

    extra_metrics = np.hstack([
        lip_top_bottom, lip_left_right, brow_inner, 
        left_brow_height, right_brow_height, lip_corners_y
    ])
    
    return np.hstack([relative_coords, X_flat[:, 1404:], extra_metrics])

print("Extracting advanced geometric features...")
X_engineered = extract_advanced_features(X_raw)

# 3. Create Temporal Windows
WINDOW_SIZE = 5

def create_temporal_windows(X, y, window_size=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        if len(set(y[i : i + window_size])) == 1:
            X_seq.append(X[i : i + window_size].flatten())
            y_seq.append(y[i + window_size - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)

print(f"Creating temporal windows (Window Size: {WINDOW_SIZE})...")
X_temp_win, y_temp_win = create_temporal_windows(X_engineered, y_raw, window_size=WINDOW_SIZE)
print(f"Dataset Window Shape: {X_temp_win.shape}")

# 4. Data Splits
X_train, X_temp, y_train, y_temp = train_test_split(
    X_temp_win, y_temp_win, test_size=0.20, random_state=42, stratify=y_temp_win
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

# 5. Fit & Save Feature Scaler
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

EMOTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
scaler_path = EMOTION_MODELS_DIR / "emotion_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"✓ Saved feature scaler to: {scaler_path}")

# 6. Regularized XGBoost Training
xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Regularized Temporal XGBoost Model...")
xgb_model.fit(X_train_full_scaled, y_train_full, eval_set=[(X_test_scaled, y_test)], verbose=50)

# 7. Save Model & Evaluate
model_path = EMOTION_MODELS_DIR / "emotion_temporal_xgb.pkl"
joblib.dump(xgb_model, model_path)
print(f"\n✓ Saved temporal model to: {model_path}")

y_pred = xgb_model.predict(X_test_scaled)
print("\n" + classification_report(y_test, y_pred, target_names=[EMOTIONS[i] for i in range(len(EMOTIONS))]))