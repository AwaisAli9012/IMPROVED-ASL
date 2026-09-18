"""
IMPROVED ASL - Scale-Invariant Live Inference
============================================
"""

import cv2
import pickle
import numpy as np
import mediapipe as mp
from collections import deque

from emotion_config import EMOTIONS, EMOTION_MODELS_DIR

model_path = EMOTION_MODELS_DIR / "emotion_temporal_xgb.pkl"
with open(model_path, "rb") as f:
    emotion_model = pickle.load(f)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_live_scale_invariant(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark[:468]], dtype=np.float32)
    
    scale = np.linalg.norm(coords[33] - coords[263])
    if scale == 0:
        scale = 1e-6
        
    norm_coords = (coords - coords[1]) / scale
    
    mouth_h = np.linalg.norm(norm_coords[13] - norm_coords[14])
    mouth_w = np.linalg.norm(norm_coords[61] - norm_coords[291])
    mouth_ratio = mouth_h / (mouth_w + 1e-6)
    
    l_brow = np.linalg.norm(norm_coords[70] - norm_coords[1])
    r_brow = np.linalg.norm(norm_coords[300] - norm_coords[1])
    
    l_corner_y = norm_coords[61][1]
    r_corner_y = norm_coords[291][1]
    
    return np.hstack([
        norm_coords.flatten(), 
        [mouth_h, mouth_w, mouth_ratio, l_brow, r_brow, l_corner_y, r_corner_y]
    ])

cap = cv2.VideoCapture(0)
sequence_buffer = deque(maxlen=5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_emotion = "Waiting..."
    confidence = 0.0

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        features = extract_live_scale_invariant(face_landmarks)
        sequence_buffer.append(features)

        if len(sequence_buffer) == 5:
            input_seq = np.concatenate(list(sequence_buffer)).reshape(1, -1)
            probs = emotion_model.predict_proba(input_seq)[0]
            
            pred_idx = np.argmax(probs)
            current_emotion = EMOTIONS[pred_idx].upper()
            confidence = float(probs[pred_idx]) * 100

            print(f"H: {probs[0]:.2f} | N: {probs[1]:.2f} | S: {probs[2]:.2f} | A: {probs[3]:.2f} -> {current_emotion}")

    cv2.putText(frame, f"Emotion: {current_emotion}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    if confidence > 0:
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Emotion Detector Live Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()