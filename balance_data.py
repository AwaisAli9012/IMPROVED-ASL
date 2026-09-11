"""
IMPROVED ASL - Balance Data
============================
Checks class distribution and augments underrepresented classes
Ensures all 65 classes have equal samples (400 each)
Saves balanced keypoints and labels
"""

import numpy as np
from pathlib import Path
from Config import KEYPOINTS_DIR, TRAINING_CONFIG

print("=" * 70)
print("IMPROVED ASL - BALANCE DATA")
print("=" * 70)

# Load data
print(f"\nLoading data from {KEYPOINTS_DIR}...")
keypoints = np.load(KEYPOINTS_DIR / 'keypoints.npy')
labels = np.load(KEYPOINTS_DIR / 'labels.npy')

print(f"✓ Loaded {keypoints.shape[0]} samples")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CHECKING CLASS DISTRIBUTION")
print("=" * 70)

unique_classes = np.unique(labels)
class_counts = {}

for class_id in unique_classes:
    count = np.sum(labels == class_id)
    class_counts[class_id] = count

min_samples = min(class_counts.values())
max_samples = max(class_counts.values())
target_samples = TRAINING_CONFIG['samples_per_class']

print(f"\nTotal classes: {len(unique_classes)}")
print(f"Min samples: {min_samples}")
print(f"Max samples: {max_samples}")
print(f"Target samples: {target_samples}")

# Check imbalance
imbalanced_classes = [cls_id for cls_id, count in class_counts.items() if count < target_samples]
print(f"\nImbalanced classes (< {target_samples}): {len(imbalanced_classes)}")

if imbalanced_classes:
    print("\nClasses needing augmentation:")
    for cls_id in sorted(imbalanced_classes):
        current = class_counts[cls_id]
        needed = target_samples - current
        print(f"  Class {cls_id:2d}: {current:3d} samples → Need {needed} more")

# ═══════════════════════════════════════════════════════════════════════════
# AUGMENTATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def augment_keypoint(keypoint):
    """Apply single augmentation to a keypoint"""
    augmentations = []
    kp = np.array(keypoint)
    
    # 1. Flip (mirror X)
    kp_flip = kp.copy()
    kp_flip[0::3] = 1.0 - kp_flip[0::3]
    augmentations.append(kp_flip)
    
    # 2. Noise
    kp_noise = kp + np.random.normal(0, 0.005, kp.shape)
    kp_noise = np.clip(kp_noise, 0, 1)
    augmentations.append(kp_noise)
    
    # 3. Scale
    scale_factor = np.random.uniform(0.95, 1.05)
    kp_scale = kp * scale_factor
    kp_scale = np.clip(kp_scale, 0, 1)
    augmentations.append(kp_scale)
    
    # 4. Translation
    kp_trans = kp.copy()
    kp_trans[0::3] += np.random.uniform(-0.02, 0.02)  # X
    kp_trans[1::3] += np.random.uniform(-0.02, 0.02)  # Y
    kp_trans = np.clip(kp_trans, 0, 1)
    augmentations.append(kp_trans)
    
    return augmentations

# ═══════════════════════════════════════════════════════════════════════════
# BALANCE CLASSES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BALANCING DATA")
print("=" * 70)

balanced_keypoints = []
balanced_labels = []

for class_id in sorted(unique_classes):
    current_count = class_counts[class_id]
    
    # Get all keypoints for this class
    class_mask = labels == class_id
    class_keypoints = keypoints[class_mask]
    
    # Add original keypoints
    balanced_keypoints.extend(class_keypoints)
    balanced_labels.extend([class_id] * len(class_keypoints))
    
    # If class has fewer than target, augment
    if current_count < target_samples:
        needed = target_samples - current_count
        print(f"  Class {class_id:2d}: Augmenting {needed} samples...")
        
        # Create augmented samples
        augmented_count = 0
        while augmented_count < needed:
            # Pick random sample from this class
            random_idx = np.random.randint(0, len(class_keypoints))
            original_kp = class_keypoints[random_idx]
            
            # Augment it
            augmentations = augment_keypoint(original_kp)
            
            # Add augmented versions until we have enough
            for aug_kp in augmentations:
                if augmented_count < needed:
                    balanced_keypoints.append(aug_kp)
                    balanced_labels.append(class_id)
                    augmented_count += 1
                else:
                    break
    else:
        print(f"  Class {class_id:2d}: Already balanced ({current_count} samples)")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE BALANCED DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SAVING BALANCED DATA")
print("=" * 70)

balanced_keypoints = np.array(balanced_keypoints, dtype=np.float32)
balanced_labels = np.array(balanced_labels, dtype=np.int32)

print(f"\nBalanced data:")
print(f"  - Shape: {balanced_keypoints.shape}")
print(f"  - Labels: {balanced_labels.shape}")
print(f"  - Unique classes: {len(np.unique(balanced_labels))}")

# Verify balance
print(f"\nVerifying balance:")
balanced_counts = {}
for class_id in np.unique(balanced_labels):
    count = np.sum(balanced_labels == class_id)
    balanced_counts[class_id] = count
    if count != target_samples:
        print(f"  ⚠️  Class {class_id}: {count} (expected {target_samples})")
    else:
        print(f"  ✓ Class {class_id}: {count}")

# Save
backup_kp = KEYPOINTS_DIR / 'keypoints_original.npy'
backup_labels = KEYPOINTS_DIR / 'labels_original.npy'

# Backup original
if not backup_kp.exists():
    np.save(backup_kp, keypoints)
    np.save(backup_labels, labels)
    print(f"\n✓ Backed up original: {backup_kp.name}, {backup_labels.name}")

# Save balanced
keypoints_file = KEYPOINTS_DIR / 'keypoints.npy'
labels_file = KEYPOINTS_DIR / 'labels.npy'

np.save(keypoints_file, balanced_keypoints)
np.save(labels_file, balanced_labels)

print(f"✓ Saved balanced: {keypoints_file.name}, {labels_file.name}")

print("\n" + "=" * 70)
print("✅ DATA BALANCING COMPLETE!")
print("=" * 70)
print(f"\nBefore: {keypoints.shape[0]} samples")
print(f"After: {balanced_keypoints.shape[0]} samples")
print(f"Added: {balanced_keypoints.shape[0] - keypoints.shape[0]} augmented samples")
