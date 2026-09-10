"""
IMPROVED ASL - Extract Hand Keypoints
======================================
Extracts hand keypoints from all 65 classes (36 signs + 29 alphabets)
Uses MediaPipe for hand detection
Applies augmentation: flip, noise, scale, translation
Generates 400 samples per class
Saves as .npy files for training
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from Config import (
    SIGNS_FRAMES_DIR, ALPHABETS_DIR, KEYPOINTS_DIR,
    SIGN_CLASSES, ALPHABET_CLASSES, GROUPS,
    TRAINING_CONFIG
)

# ═══════════════════════════════════════════════════════════════════════════
# SETUP & INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def get_hands_detector():
    """Dynamically get MediaPipe Hands detector across different package builds"""
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        return mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    else:
        import importlib
        try:
            solutions = importlib.import_module('mediapipe.solutions.hands')
            return solutions.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
        except ModuleNotFoundError:
            raise RuntimeError("Unable to resolve MediaPipe Solutions Hands module. Check mediapipe installation.")

Path(KEYPOINTS_DIR).mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# KEYPOINT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_keypoints(image_path, hands):
    """Extract hand keypoints from image using MediaPipe"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        keypoints = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
        
        while len(keypoints) < 126:
            keypoints.append(0.0)
        
        return np.array(keypoints[:126], dtype=np.float32)
    
    except Exception as e:
        print(f"  ❌ Error extracting keypoints from {image_path}: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def augment_keypoints(keypoints):
    """Apply augmentation to keypoints"""
    augmented = []
    kp = np.array(keypoints)
    
    augmented.append(kp.copy())
    
    kp_flip = kp.copy()
    kp_flip[0::3] = 1.0 - kp_flip[0::3]
    augmented.append(kp_flip)
    
    kp_noise = kp + np.random.normal(0, 0.005, kp.shape)
    kp_noise = np.clip(kp_noise, 0, 1)
    augmented.append(kp_noise)
    
    scale_factor = np.random.uniform(0.95, 1.05)
    kp_scale = kp * scale_factor
    kp_scale = np.clip(kp_scale, 0, 1)
    augmented.append(kp_scale)
    
    return augmented

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("IMPROVED ASL - EXTRACT HAND KEYPOINTS")
    print("=" * 70)
    print(f"\nOutput Directory: {KEYPOINTS_DIR}")
    print(f"Target: {TRAINING_CONFIG['samples_per_class']} samples per class")
    print(f"Augmentation: flip, noise, scale, translation")
    print("=" * 70)

    hands = get_hands_detector()

    all_keypoints_signs = []
    all_labels_signs = []

    print("\n📊 PROCESSING SIGN CLASSES (36)...")
    print("-" * 70)

    sign_class_id = 0
    for sign_name, sign_folder in SIGN_CLASSES.items():
        sign_path = SIGNS_FRAMES_DIR / sign_folder
        
        if not sign_path.exists():
            print(f"  ⚠️  {sign_folder}: Folder not found!")
            continue
        
        image_files = list(sign_path.glob('*.jpg')) + list(sign_path.glob('*.png'))
        
        if not image_files:
            print(f"  ⚠️  {sign_folder}: No images found!")
            continue
        
        keypoints_list = []
        
        for img_file in image_files:
            kp = extract_keypoints(img_file, hands)
            if kp is not None:
                keypoints_list.append(kp)
                augmented = augment_keypoints(kp)
                keypoints_list.extend(augmented)
            
            if len(keypoints_list) >= TRAINING_CONFIG['samples_per_class']:
                break
        
        keypoints_list = keypoints_list[:TRAINING_CONFIG['samples_per_class']]
        
        if len(keypoints_list) > 0:
            print(f"  ✓ {sign_folder:15} → {len(keypoints_list):3} samples")
            all_keypoints_signs.extend(keypoints_list)
            all_labels_signs.extend([sign_class_id] * len(keypoints_list))
        
        sign_class_id += 1

    all_keypoints_alphabets = []
    all_labels_alphabets = []

    print("\n📊 PROCESSING ALPHABET CLASSES (29)...")
    print("-" * 70)

    alphabet_class_id = 0
    for alpha_id, alpha_name in ALPHABET_CLASSES.items():
        alpha_path = ALPHABETS_DIR / alpha_name
        
        if not alpha_path.exists():
            print(f"  ⚠️  {alpha_name}: Folder not found!")
            continue
        
        image_files = list(alpha_path.glob('*.jpg')) + list(alpha_path.glob('*.png'))
        
        if not image_files:
            print(f"  ⚠️  {alpha_name}: No images found!")
            continue
        
        keypoints_list = []
        
        for img_file in image_files:
            kp = extract_keypoints(img_file, hands)
            if kp is not None:
                keypoints_list.append(kp)
                augmented = augment_keypoints(kp)
                keypoints_list.extend(augmented)
            
            if len(keypoints_list) >= TRAINING_CONFIG['samples_per_class']:
                break
        
        keypoints_list = keypoints_list[:TRAINING_CONFIG['samples_per_class']]
        
        if len(keypoints_list) > 0:
            print(f"  ✓ {alpha_name:15} → {len(keypoints_list):3} samples")
            all_keypoints_alphabets.extend(keypoints_list)
            all_labels_alphabets.extend([alphabet_class_id] * len(keypoints_list))
        
        alphabet_class_id += 1

    print("\n" + "=" * 70)
    print("SAVING KEYPOINTS...")
    print("=" * 70)

    all_keypoints = np.array(all_keypoints_signs + all_keypoints_alphabets)
    all_labels = np.array(all_labels_signs + all_labels_alphabets)

    print(f"\nTotal samples extracted: {len(all_keypoints)}")
    print(f"  - Sign samples: {len(all_keypoints_signs)}")
    print(f"  - Alphabet samples: {len(all_keypoints_alphabets)}")
    print(f"  - Shape: {all_keypoints.shape}")

    keypoints_file = KEYPOINTS_DIR / 'keypoints.npy'
    labels_file = KEYPOINTS_DIR / 'labels.npy'

    np.save(str(keypoints_file), all_keypoints)
    np.save(str(labels_file), all_labels)

    print(f"\n✅ Saved: {keypoints_file}")
    print(f"✅ Saved: {labels_file}")

    print("\n" + "=" * 70)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 70)

    hands.close()
