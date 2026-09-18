"""
EMOTION DETECTOR - BALANCED NEUTRAL DEADBAND
============================================
Ensures resting face stays NEUTRAL without micro-adjusting eyebrows.
"""

import cv2
import joblib
import time
import numpy as np
import mediapipe as mp
from collections import deque

from emotion_config import EMOTIONS, EMOTION_MODELS_DIR

model_path = EMOTION_MODELS_DIR / "emotion_temporal_xgb.pkl"
emotion_model = joblib.load(model_path)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FACEMESH_TO_68 = [
    162, 234, 93, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 288, 365, 397,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    168, 197, 5, 4, 240, 97, 2, 326, 327,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87
]

def extract_features(landmarks, img_w, img_h):
    coords_2d = np.array([
        [landmarks.landmark[idx].x * img_w, landmarks.landmark[idx].y * img_h] 
        for idx in FACEMESH_TO_68
    ], dtype=np.float32)
    
    base_pt = coords_2d[30]
    centered_coords = coords_2d - base_pt

    scale = np.linalg.norm(centered_coords[45] - centered_coords[36])
    if scale < 1e-5 or np.isnan(scale):
        scale = 1e-6

    normalized_coords = centered_coords / scale

    mouth_w = np.linalg.norm(normalized_coords[48] - normalized_coords[54])
    mouth_h = np.linalg.norm(normalized_coords[51] - normalized_coords[57])
    brow_dist = np.linalg.norm(normalized_coords[21] - normalized_coords[22])
    left_brow_eye = np.linalg.norm(normalized_coords[19] - normalized_coords[37])
    right_brow_eye = np.linalg.norm(normalized_coords[24] - normalized_coords[44])

    ratios = np.array([mouth_w, mouth_h, brow_dist, left_brow_eye, right_brow_eye], dtype=np.float32)

    padded_coords = np.zeros(1404, dtype=np.float32)
    flat_coords = normalized_coords.flatten()
    padded_coords[:flat_coords.shape[0]] = flat_coords

    return np.hstack([padded_coords, ratios]), ratios

cap = cv2.VideoCapture(0)
sequence_buffer = deque(maxlen=5)

# --- Calibration Phase ---
print("[INFO] Relax your face completely for calibration...")
calibration_ratios = []
start_time = time.time()

while cap.isOpened() and (time.time() - start_time) < 3.0:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        _, ratios = extract_features(results.multi_face_landmarks[0], w, h)
        calibration_ratios.append(ratios)
        
    remaining = max(0.0, 3.0 - (time.time() - start_time))
    cv2.putText(frame, f"CALIBRATING (RELAX FACE)... {remaining:.1f}s", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Emotion Detector Live Test", frame)
    cv2.waitKey(1)

if len(calibration_ratios) > 0:
    baseline_ratios = np.mean(calibration_ratios, axis=0)
else:
    baseline_ratios = np.array([0.80, 0.15, 0.35, 0.25, 0.25], dtype=np.float32)

baseline_brow_eye = (baseline_ratios[3] + baseline_ratios[4]) / 2.0
baseline_brow_dist = baseline_ratios[2]

# --- Live Detection Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    current_emotion = "NEUTRAL"
    confidence = 90.0

    if results.multi_face_landmarks:
        features, current_ratios = extract_features(results.multi_face_landmarks[0], w, h)
        sequence_buffer.append(features)

        if len(sequence_buffer) == 5:
            input_seq = np.concatenate(list(sequence_buffer)).reshape(1, -1)
            raw_probs = emotion_model.predict_proba(input_seq)[0]

            raw_pred_idx = np.argmax(raw_probs)
            raw_emotion = EMOTIONS[raw_pred_idx].upper()
            
            # Relative eyebrow metric ratios
            current_brow_eye = (current_ratios[3] + current_ratios[4]) / 2.0
            brow_height_ratio = current_brow_eye / baseline_brow_eye
            brow_pull_ratio = current_ratios[2] / baseline_brow_dist

            # 1. HAPPY (Direct Model Pass-Through)
            if raw_emotion == "HAPPY" and raw_probs[raw_pred_idx] > 0.60:
                current_emotion = "HAPPY"
                confidence = float(raw_probs[raw_pred_idx]) * 100

            # 2. ANGRY (Requires deliberate eyebrow drop/pull)
            elif brow_height_ratio < 0.94 or brow_pull_ratio < 0.90:
                current_emotion = "ANGRY"
                confidence = max(80.0, float(raw_probs[3]) * 100)

            # 3. SAD (Cheat code: eyebrow raise)
            elif brow_height_ratio > 1.08:
                current_emotion = "SAD"
                confidence = min(95.0, float(brow_height_ratio) * 75.0)

            # 4. NEUTRAL (Resting range: 0.94 <= brow_height_ratio <= 1.08)
            else:
                current_emotion = "NEUTRAL"
                confidence = 90.0

            print(f"Brow Ratio: {brow_height_ratio:.3f} | Raw: {raw_emotion} -> Output: {current_emotion}")

    cv2.putText(frame, f"Emotion: {current_emotion}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Emotion Detector Live Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()