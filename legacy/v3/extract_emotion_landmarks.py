import os
import cv2
import numpy as np
import torch
import face_alignment
from pathlib import Path
from emotion_config import EMOTIONS, EMOTION_DATASET_PATH, EMOTION_KEYPOINTS_DIR

print("=" * 70)
print("EXTRACTING FACIAL LANDMARKS (STABLE OPENCV / PYTORCH BACKEND)")
print("=" * 70)

EMOTION_KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Face Alignment model on GPU if available, else CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)

def process_single_image(img_path, emotion_id):
    try:
        image = cv2.imread(str(img_path))
        if image is None or image.size == 0:
            return None, None

        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        preds = fa.get_landmarks(rgb_image)
        if preds is None or len(preds) == 0:
            return None, None

        # Extract 68 2D landmark array (68, 2)
        landmarks = preds[0]

        # Convert to (68, 3) with zeros for z-dimension to align with pipeline structure
        landmarks_3d = np.zeros((68, 3), dtype=np.float32)
        landmarks_3d[:, :2] = landmarks

        # Geometric Translation: Origin at Nose Tip (Landmark index 30)
        base_pt = landmarks_3d[30]
        centered_coords = landmarks_3d - base_pt

        # Geometric Scaling: Outer Eye Distance (Landmarks 36 & 45)
        p_left = centered_coords[36]
        p_right = centered_coords[45]
        scale = np.linalg.norm(p_right - p_left)

        if scale < 1e-5 or np.isnan(scale):
            return None, None

        normalized_coords = centered_coords / scale

        # Feature Engineering: 5 Expression Ratios (68-landmark mapping)
        mouth_width = np.linalg.norm(normalized_coords[48] - normalized_coords[54])
        mouth_height = np.linalg.norm(normalized_coords[51] - normalized_coords[57])
        eyebrow_dist = np.linalg.norm(normalized_coords[21] - normalized_coords[22])
        left_brow_eye = np.linalg.norm(normalized_coords[19] - normalized_coords[37])
        right_brow_eye = np.linalg.norm(normalized_coords[24] - normalized_coords[44])

        expression_ratios = np.array([
            mouth_width, mouth_height, eyebrow_dist, 
            left_brow_eye, right_brow_eye
        ], dtype=np.float32)

        # Pad coordinate vector to 1404 values (468 x 3) + 5 ratios = 1409 total dimensions
        padded_coords = np.zeros(1404, dtype=np.float32)
        flat_coords = normalized_coords.flatten()
        padded_coords[:flat_coords.shape[0]] = flat_coords

        feature_vector = np.hstack([padded_coords, expression_ratios])

        if np.isnan(feature_vector).any() or np.isinf(feature_vector).any():
            return None, None

        return feature_vector, emotion_id
    except Exception:
        return None, None


if __name__ == "__main__":
    name_to_id = {v: k for k, v in EMOTIONS.items()}
    tasks = []

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for emotion_name, emotion_id in name_to_id.items():
        folder_path = EMOTION_DATASET_PATH / emotion_name
        if folder_path.exists():
            images = [p for p in folder_path.glob('**/*') if p.is_file() and p.suffix.lower() in valid_exts]
            print(f"Discovered {len(images)} images for '{emotion_name}'")
            for img_path in images:
                tasks.append((img_path, emotion_id))

    total_processed = len(tasks)
    total_detected = 0

    keypoints_list = []
    labels_list = []

    print("\nStarting landmark extraction loop...")
    for idx, (img_path, emotion_id) in enumerate(tasks):
        feat, label = process_single_image(img_path, emotion_id)
        if feat is not None:
            keypoints_list.append(feat)
            labels_list.append(label)
            total_detected += 1

        if (idx + 1) % 200 == 0 or (idx + 1) == total_processed:
            print(f"Progress: [{idx + 1}/{total_processed}] | Faces detected: {total_detected}")

    X = np.array(keypoints_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)

    print("\n" + "=" * 70)
    print("Extraction Complete!")
    print(f"Total Images Scanned: {total_processed}")
    print(f"Faces Detected & Saved: {total_detected}")
    print(f"Feature Array Shape: {X.shape} (Expected: N x 1409)")
    print("=" * 70)

    np.save(EMOTION_KEYPOINTS_DIR / "emotion_keypoints.npy", X)
    np.save(EMOTION_KEYPOINTS_DIR / "emotion_labels.npy", y)

    print(f"\n✓ Saved normalized keypoints + ratio features to: {EMOTION_KEYPOINTS_DIR}")