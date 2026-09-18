"""
ALIGNED PUBLIC DATASET EMBEDDING EXTRACTOR
==========================================
Runs MediaPipe FaceAligner on public dataset images before MobileNet extraction.
Ensures identical crop geometry between public data and local webcam frames.
"""

import cv2
import numpy as np
from pathlib import Path
from face_aligner import FaceAligner
from feature_extractor import VisionFeatureExtractor

aligner = FaceAligner()
extractor = VisionFeatureExtractor()

DATASET_DIR = Path("/home/xero1/Documents/IMPROVED ASL/Dataset/emotion_dataset/emotion/train")

# We are strictly focusing on these 4 emotions
TARGET_CLASSES = {
    "angry": "ANGRY",
    "happy": "HAPPY",
    "neutral": "NEUTRAL",
    "sad": "SAD"
}

# Cap at 300 aligned samples per class for fast processing & ideal balance
MAX_SAMPLES_PER_CLASS = 300  

X_public, y_public = [], []

print("[INFO] Processing and aligning public dataset images...")

for folder_name, target_label in TARGET_CLASSES.items():
    folder_path = DATASET_DIR / folder_name
    if not folder_path.exists():
        continue

    image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
    print(f"[INFO] Aligning class '{target_label}'...")

    count = 0
    for img_path in image_files:
        if count >= MAX_SAMPLES_PER_CLASS:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 1. Align face using your custom pipeline logic
        face_crop, _ = aligner.process_frame(img)

        # 2. Extract embedding only if a valid face was aligned
        if face_crop is not None:
            emb = extractor.extract(face_crop)
            if emb is not None:
                X_public.append(emb)
                y_public.append(target_label)
                count += 1

    print(f"  -> Extracted {count} aligned samples for {target_label}.")

# Merge with local webcam data
local_data_path = Path("data/emotion_embeddings.npz")
if local_data_path.exists():
    local_data = np.load(local_data_path)
    X_local, y_local = local_data["X"], local_data["y"]
    
    X_combined = np.vstack([X_public, X_local])
    y_combined = np.concatenate([y_public, y_local])
else:
    X_combined = np.array(X_public)
    y_combined = np.array(y_public)

output_path = Path("data/emotion_embeddings.npz")
output_path.parent.mkdir(exist_ok=True)
np.savez(output_path, X=X_combined, y=y_combined)

print(f"\n[SUCCESS] Saved {len(X_combined)} aligned samples to {output_path}")