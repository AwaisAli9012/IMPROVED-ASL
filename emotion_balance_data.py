"""
IMPROVED ASL - Balance Emotion Dataset (Vectorized & 3D Jittered)
=============================================================================
Balanced via vectorized 3D rotation augmentation and safe target scaling.
Updated to output directly to emotion_keypoints.npy for seamless training.
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


def standardize_and_augment(kp_flat_batch, do_augment=True):
    """
    Standardizes any input landmark batch to strictly 1409 dimensions 
    (1404 raw coords + 5 engineered geometric ratios), and optionally 
    applies a vectorized 3D rotational jitter augmentation.
    """
    N = len(kp_flat_batch)
    
    # Always slice strictly the first 1404 coordinates to clear any shape drift
    coords_flat = kp_flat_batch[:, :1404]
    kp_3d = coords_flat.reshape(N, 468, 3).copy()

    if do_augment:
        # Small angular jitter in radians (~ +/- 3 degrees)
        angles = np.random.uniform(-0.05, 0.05, size=(N, 3))

        # Apply 3D rotation matrix per batch element
        for i in range(N):
            ax, ay, az = angles[i]
            Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
            Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
            Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            kp_3d[i] = kp_3d[i] @ R.T

    augmented_coords = kp_3d.reshape(N, 1404)

    # Replicate the exact 5 extra metrics used in training
    pts = kp_3d[:, :68, :]
    lip_top_bottom = np.linalg.norm(pts[:, 51, :] - pts[:, 57, :], axis=-1, keepdims=True)
    lip_left_right = np.linalg.norm(pts[:, 48, :] - pts[:, 54, :], axis=-1, keepdims=True)
    brow_inner = np.linalg.norm(pts[:, 21, :] - pts[:, 22, :], axis=-1, keepdims=True)
    left_brow_height = np.linalg.norm(pts[:, 19, :] - pts[:, 37, :], axis=-1, keepdims=True)
    right_brow_height = np.linalg.norm(pts[:, 24, :] - pts[:, 44, :], axis=-1, keepdims=True)
    
    mouth_center_y = (pts[:, 51, 1] + pts[:, 57, 1]) / 2.0
    left_corner_y = pts[:, 48, 1] - mouth_center_y
    right_corner_y = pts[:, 54, 1] - mouth_center_y
    lip_corners_y = np.stack([left_corner_y, right_corner_y], axis=-1)

    extra_metrics = np.hstack([
        lip_top_bottom, lip_left_right, brow_inner, 
        left_brow_height, right_brow_height, lip_corners_y
    ])
    
    # Strictly concatenate to 1404 + 5 = 1409 dimensions
    return np.hstack([augmented_coords, extra_metrics])


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
        sampled_indices = np.random.choice(count, diff, replace=True)
        base_samples = emotion_kp[sampled_indices]
        
        # Generate augmented samples and standardize existing ones to match 1409 shape
        augmented_samples = standardize_and_augment(base_samples, do_augment=True)
        standardized_existing = standardize_and_augment(emotion_kp, do_augment=False)
        
        emotion_kp = np.vstack([standardized_existing, augmented_samples])
        print(f" (vectorized 3D augmented +{diff})")
    else:
        indices = np.random.choice(count, target_samples, replace=False)
        emotion_kp = emotion_kp[indices]
        # Standardize clean select samples to enforce uniform 1409 shape
        emotion_kp = standardize_and_augment(emotion_kp, do_augment=False)
        print(f" (exact match / clean select)")

    balanced_keypoints.append(emotion_kp)
    balanced_labels.extend([emotion_id] * len(emotion_kp))

balanced_keypoints = np.vstack(balanced_keypoints).astype(np.float32)
balanced_labels = np.array(balanced_labels, dtype=np.int64)

print(f"\nAfter balancing:")
print(f"  Total balanced samples: {len(balanced_keypoints)}")
print(f"  Balanced Matrix Shape:  {balanced_keypoints.shape}")

# Save output directly to standard filenames for seamless trainer pick-up
np.save(X_path, balanced_keypoints)
np.save(y_path, balanced_labels)

print(f"\n✓ Balanced dataset successfully overwritten to standard keypoint files!")