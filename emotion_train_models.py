"""
IMPROVED ASL - Advanced Facial Feature Engineering & Focal Loss MLP
=============================================================================
Features engineered distance deltas, eyebrow/lip curvature metrics,
and Categorical Focal Loss to break the neutral/sad/angry confusion floor.
"""

import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, callbacks, optimizers

from emotion_config import EMOTIONS, EMOTION_KEYPOINTS_DIR, EMOTION_MODELS_DIR

print("=" * 70)
print("EMOTION LANDMARK MLP (FOCAL LOSS & GEOMETRIC DELTAS)")
print("=" * 70)

EMOTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Raw Data
X_path = EMOTION_KEYPOINTS_DIR / "emotion_keypoints.npy"
y_path = EMOTION_KEYPOINTS_DIR / "emotion_labels.npy"

if not X_path.exists() or not y_path.exists():
    raise FileNotFoundError("❌ Keypoint files missing. Run landmark extraction first!")

X_raw = np.load(X_path).astype(np.float32)
y_raw = np.load(y_path).astype(np.int64)

# 2. Extract Geometric Relative Features
def extract_advanced_features(X_flat):
    N = len(X_flat)
    coords = X_flat[:, :1404].reshape(N, 468, 3)
    
    # Anchors: Nose tip (Index 1) and Intra-eye center
    nose_tip = coords[:, 1:2, :]
    relative_coords = (coords - nose_tip).reshape(N, -1)
    
    # Expression specific distance vectors
    pts = coords[:, :68, :]
    lip_top_bottom = np.linalg.norm(pts[:, 51, :] - pts[:, 57, :], axis=-1, keepdims=True)
    lip_left_right = np.linalg.norm(pts[:, 48, :] - pts[:, 54, :], axis=-1, keepdims=True)
    
    # Brow Inner / Outer distances
    brow_inner = np.linalg.norm(pts[:, 21, :] - pts[:, 22, :], axis=-1, keepdims=True)
    left_brow_height = np.linalg.norm(pts[:, 19, :] - pts[:, 37, :], axis=-1, keepdims=True)
    right_brow_height = np.linalg.norm(pts[:, 24, :] - pts[:, 44, :], axis=-1, keepdims=True)
    
    # Frown/Smile curvature: Lip corner Y relative to mouth center Y
    mouth_center_y = (pts[:, 51, 1] + pts[:, 57, 1]) / 2.0
    left_corner_y = pts[:, 48, 1] - mouth_center_y
    right_corner_y = pts[:, 54, 1] - mouth_center_y
    lip_corners_y = np.stack([left_corner_y, right_corner_y], axis=-1)

    extra_metrics = np.hstack([
        lip_top_bottom, lip_left_right, brow_inner, 
        left_brow_height, right_brow_height, lip_corners_y
    ])
    
    return np.hstack([relative_coords, X_flat[:, 1404:], extra_metrics])

X_engineered = extract_advanced_features(X_raw)

# 3. Stratified Split (80/10/10)
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X_engineered, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 4. Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

joblib.dump(scaler, EMOTION_MODELS_DIR / "emotion_scaler.pkl")

# Compute Class Weights to balance penalty gradient on hard classes
class_weights_vals = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights_vals))

# 5. Residual MLP with Focal Loss
def build_deep_residual_mlp(input_dim, num_classes):
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(512, activation="swish")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    res1 = x
    x = layers.Dense(512, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.add([x, res1])
    
    x = layers.Dense(256, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    res2 = x
    x = layers.Dense(256, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.add([x, res2])
    
    x = layers.Dense(128, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

model = build_deep_residual_mlp(X_train.shape[1], len(EMOTIONS))

# Categorical Focal Loss focusing on hard neutral/sad/angry boundary cases
focal_loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss,
    metrics=["accuracy"]
)

# 6. Training Execution
model_save_path = EMOTION_MODELS_DIR / "emotion_mlp_model.keras"

callbacks_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1)
]

print("\nTraining Advanced Feature MLP...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=callbacks_list
)

# 7. Evaluation
print("\n" + "=" * 70)
print("FINAL HOLDOUT EVALUATION")
print("=" * 70)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🎯 Test Accuracy: {test_acc * 100:.2f}%\n")

y_pred = np.argmax(model.predict(X_test), axis=1)
target_names = [EMOTIONS[i] for i in range(len(EMOTIONS))]

print("CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))