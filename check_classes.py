import numpy as np
from emotion_config import EMOTIONS, EMOTION_KEYPOINTS_DIR

y_raw = np.load(EMOTION_KEYPOINTS_DIR / "emotion_labels.npy")
unique, counts = np.unique(y_raw, return_counts=True)

print("Dataset Class Distribution:")
for class_id, count in zip(unique, counts):
    print(f"  {EMOTIONS[class_id]}: {count} samples")
    