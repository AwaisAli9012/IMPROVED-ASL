import numpy as np
from pathlib import Path
from emotion_config import EMOTION_KEYPOINTS_DIR

# Load training dataset
X_path = EMOTION_KEYPOINTS_DIR / "emotion_keypoints.npy"
y_path = EMOTION_KEYPOINTS_DIR / "emotion_labels.npy"

X_train = np.load(X_path)
y_train = np.load(y_path)

print("=" * 60)
print("DATASET FEATURE DIAGNOSTIC")
print("=" * 60)
print(f"Dataset Shape: {X_train.shape}")
print(f"Sample 0 Feature Range: Min={X_train[0].min():.4f}, Max={X_train[0].max():.4f}, Mean={X_train[0].mean():.4f}")
print(f"Sample 0 First 10 Values:\n{X_train[0][:10]}")
print("=" * 60)