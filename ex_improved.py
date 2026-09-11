"""
IMPROVED ASL - Extract Hand Keypoints using MediaPipe
=====================================================
Extracts 21 3D hand keypoints (padded to 126 values for 2 hands)
from all 65 classes (36 signs + 29 alphabets).
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from Config import (
    SIGNS_FRAMES_DIR, ALPHABETS_DIR, KEYPOINTS_DIR,
    SIGN_CLASSES, ALPHABET_CLASSES, TRAINING_CONFIG
)

Path(KEYPOINTS_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("IMPROVED ASL - EXTRACT HAND KEYPOINTS (MEDIAPIPE)")
print("=" * 70)

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands

def extract_keypoints_mediapipe(image_path, hands_detector):
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Convert BGR image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(image_rgb)
        
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
        
        # Pad with zeros if fewer than 2 hands are detected (up to 126 float values)
        while len(keypoints) < 126:
            keypoints.append(0.0)
            
        return np.array(keypoints[:126], dtype=np.float32)
    except Exception as e:
        return None

def augment_keypoints(keypoints):
    augmented = []
    kp = np.array(keypoints)
    augmented.append(kp.copy())
    
    # Mirror X coordinates
    kp_flip = kp.copy()
    kp_flip[0::3] = 1.0 - kp_flip[0::3]
    augmented.append(kp_flip)
    
    # Add light Gaussian noise
    kp_noise = kp + np.random.normal(0, 0.005, kp.shape)
    kp_noise = np.clip(kp_noise, 0, 1)
    augmented.append(kp_noise)
    
    # Scale keypoints slightly
    scale_factor = np.random.uniform(0.95, 1.05)
    kp_scale = kp * scale_factor
    kp_scale = np.clip(kp_scale, 0, 1)
    augmented.append(kp_scale)
    
    return augmented

if __name__ == "__main__":
    all_keypoints = []
    all_labels = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        print("\n📊 PROCESSING SIGN CLASSES (36)...")
        print("-" * 70)

        sign_class_id = 0
        for sign_name, sign_folder in SIGN_CLASSES.items():
            sign_path = SIGNS_FRAMES_DIR / sign_folder
            if not sign_path.exists():
                continue
            image_files = list(sign_path.glob('*.jpg')) + list(sign_path.glob('*.png'))
            if not image_files:
                continue
            keypoints_list = []
            for img_file in image_files:
                kp = extract_keypoints_mediapipe(img_file, hands)
                if kp is not None and np.sum(kp) > 0:  # Ensure hand landmark was detected
                    keypoints_list.append(kp)
                    augmented = augment_keypoints(kp)
                    keypoints_list.extend(augmented)
                if len(keypoints_list) >= TRAINING_CONFIG['samples_per_class']:
                    break
            keypoints_list = keypoints_list[:TRAINING_CONFIG['samples_per_class']]
            if len(keypoints_list) > 0:
                print(f"  ✓ {sign_folder:15} → {len(keypoints_list):3} samples (label {sign_class_id})")
                all_keypoints.extend(keypoints_list)
                all_labels.extend([sign_class_id] * len(keypoints_list))
            sign_class_id += 1

        print("\n📊 PROCESSING ALPHABET CLASSES (29)...")
        print("-" * 70)

        alphabet_class_id = 36  # ALPHABETS START AT 36
        for alpha_id, alpha_name in ALPHABET_CLASSES.items():
            alpha_path = ALPHABETS_DIR / alpha_name
            if not alpha_path.exists():
                continue
            image_files = list(alpha_path.glob('*.jpg')) + list(alpha_path.glob('*.png'))
            if not image_files:
                continue
            keypoints_list = []
            for img_file in image_files:
                kp = extract_keypoints_mediapipe(img_file, hands)
                if kp is not None and np.sum(kp) > 0:
                    keypoints_list.append(kp)
                    augmented = augment_keypoints(kp)
                    keypoints_list.extend(augmented)
                if len(keypoints_list) >= TRAINING_CONFIG['samples_per_class']:
                    break
            keypoints_list = keypoints_list[:TRAINING_CONFIG['samples_per_class']]
            if len(keypoints_list) > 0:
                print(f"  ✓ {alpha_name:15} → {len(keypoints_list):3} samples (label {alphabet_class_id})")
                all_keypoints.extend(keypoints_list)
                all_labels.extend([alphabet_class_id] * len(keypoints_list))
            alphabet_class_id += 1

    print("\n" + "=" * 70)
    print("SAVING KEYPOINTS...")
    print("=" * 70)

    all_keypoints = np.array(all_keypoints, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    print(f"\nTotal samples extracted: {len(all_keypoints)}")
    print(f"  - Shape: {all_keypoints.shape}")
    print(f"  - Unique classes: {len(np.unique(all_labels))}")
    print(f"  - Labels range: {np.min(all_labels)}-{np.max(all_labels)}")

    keypoints_file = KEYPOINTS_DIR / 'keypoints.npy'
    labels_file = KEYPOINTS_DIR / 'labels.npy'

    np.save(str(keypoints_file), all_keypoints)
    np.save(str(labels_file), all_labels)

    print(f"\n✅ Saved: {keypoints_file}")
    print(f"✅ Saved: {labels_file}")
    print("\n✅ EXTRACTION COMPLETE!")
