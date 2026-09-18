import numpy as np
from pathlib import Path
from emotion_config import EMOTION_KEYPOINTS_DIR

X_path = EMOTION_KEYPOINTS_DIR / "emotion_keypoints.npy"
X_raw = np.load(X_path)

print(f"Dataset Shape: {X_raw.shape}")
print(f"Sample 0 Non-zero elements: {np.count_nonzero(X_raw[0])}")
print(f"Sample 0 First 10 values:\n{X_raw[0][:10]}")