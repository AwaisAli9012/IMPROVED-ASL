"""
IMPROVED ASL - Balance Emotion Dataset (Vectorized & 3D Jittered)
=============================================================================
Balanced via vectorized 3D rotation augmentation and safe target scaling.
Updated for 1409-dimensional vectors (1404 coords + 5 engineered ratios).
"""

import numpy as np
from pathlib import Path
from emotion_config import EMOTIONS, EMOTION_KEYPOINTS_DIR

print("=" * 70)
print("EMOTION DATA BALANCING (PRODUCTION-GRADE)")
print("=" * 70)

# Load extracted keypoints
X_path = EMOTION_KEYPOINTS_DIR / "emotion_keypoints.npy"
y_path = EMOTION_KEYPOINTS_DIR / "emotion_labels.npy"

if not X_path.exists() or not y_path.exists():
    raise FileNotFoundError("❌ Extracted keypoints not found. Run extraction first!")

keypoints = np.load(X_path)
labels = np.load(y_path)

print(f"\nBefore balancing:")
print(f"  Total samples: {len(keypoints)}")

class_counts = [np.sum(labels == emotion_id) for emotion_id in range(len(EMOTIONS))]
print("\nOriginal class distribution:")
for emotion_id, count in enumerate(class_counts):
    print(f"  {EMOTIONS[emotion_id]:10}: {count} samples")

target_samples = max(class_counts)
print(f"\n🎯 Target samples per class: {target_samples}")


def augment_landmarks_batch(kp_flat_batch):
    """
    Vectorized geometry augmentation applying subtle 3D rotational jitter.
    Splits the 1409-dim vector into 1404 coordinates + 5 ratios, modifies 3D 
    coordinates, and recalculates key facial ratios dynamically.
    """
    N = len(kp_flat_batch)
    
    # Split coordinates (0:1404) and ratios (1404:1409)
    coords_flat = kp_flat_batch[:, :1404]
    
    # Reshape coordinates to (N, 468, 3)
    kp_3d = coords_flat.reshape(N, 468, 3).copy()

    # Small angular jitter in radians (~ +/- 3 degrees)
    angles = np.random.uniform(-0.05, 0.05, size=(N, 3))

    # Apply 3D rotation matrix per batch element
    for i in range(N):
        ax, ay, az = angles[i]

        # Rotation matrices
        Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])

        R = Rz @ Ry @ Rx
        kp_3d[i] = kp_3d[i] @ R.T

    augmented_coords = kp_3d.reshape(N, 1404)

    # Recalculate expression ratios for augmented 68-point landmarks
    augmented_ratios = np.zeros((N, 5), dtype=np.float32)
    for i in range(N):
        pts = kp_3d[i][:68]
        mouth_width = np.linalg.norm(pts[48] - pts[54])
        mouth_height = np.linalg.norm(pts[51] - pts[57])
        eyebrow_dist = np.linalg.norm(pts[21] - pts[22])
        left_brow_eye = np.linalg.norm(pts[19] - pts[37])
        right_brow_eye = np.linalg.norm(pts[24] - pts[44])

        augmented_ratios[i] = [mouth_width, mouth_height, eyebrow_dist, left_brow_eye, right_brow_eye]

    return np.hstack([augmented_coords, augmented_ratios])


balanced_keypoints = []
balanced_labels = []

np.random.seed(42)

for emotion_id in range(len(EMOTIONS)):
    emotion_name = EMOTIONS[emotion_id]
    mask = labels == emotion_id
    emotion_kp = keypoints[mask]
    count = len(emotion_kp)

    print(f"  {emotion_name:10}: {count:4} → {target_samples}", end="")

    if count == 0:
        print(" ❌ WARNING: 0 samples! Skipping.")
        continue

    if count < target_samples:
        diff = target_samples - count
        
        # Sample base indices randomly for augmentation
        sampled_indices = np.random.choice(count, diff, replace=True)
        base_samples = emotion_kp[sampled_indices]
        
        # Fast vectorized batch augmentation
        augmented_samples = augment_landmarks_batch(base_samples)

        emotion_kp = np.vstack([emotion_kp, augmented_samples])
        print(f" (vectorized 3D augmented +{diff})")
    else:
        indices = np.random.choice(count, target_samples, replace=False)
        emotion_kp = emotion_kp[indices]
        print(f" (exact match / clean select)")

    balanced_keypoints.append(emotion_kp)
    balanced_labels.extend([emotion_id] * len(emotion_kp))

balanced_keypoints = np.vstack(balanced_keypoints).astype(np.float32)
balanced_labels = np.array(balanced_labels, dtype=np.int64)

print(f"\nAfter balancing:")
print(f"  Total balanced samples: {len(balanced_keypoints)}")
print(f"  Balanced Matrix Shape:  {balanced_keypoints.shape}")

# Save output to dedicated target files
np.save(EMOTION_KEYPOINTS_DIR / "emotion_keypoints_balanced.npy", balanced_keypoints)
np.save(EMOTION_KEYPOINTS_DIR / "emotion_labels_balanced.npy", balanced_labels)

print(f"\n✓ Balanced dataset saved as emotion_keypoints_balanced.npy!")