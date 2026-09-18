import numpy as np
from pathlib import Path

keypoints_path = Path("Emotion_Dataset/emotion_keypoints.npy")
X_raw = np.load(keypoints_path)

print(f"Shape: {X_raw.shape}")
print(f"Min value: {X_raw.min():.4f}")
print(f"Max value: {X_raw.max():.4f}")
print(f"Mean value: {X_raw.mean():.4f}")
print(f"First 10 values of row 0:\n{X_raw[0, :10]}")