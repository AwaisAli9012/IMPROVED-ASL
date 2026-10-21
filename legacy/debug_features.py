import numpy as np
import cv2
import mediapipe as mp
from emotion_config import EMOTION_KEYPOINTS_DIR

# 1. Inspect Training Data Format
X_raw = np.load(EMOTION_KEYPOINTS_DIR / "emotion_keypoints.npy")
print(f"Dataset Raw Shape: {X_raw.shape}")
print(f"Dataset Row 0 Min: {X_raw[0].min():.4f} | Max: {X_raw[0].max():.4f} | Mean: {X_raw[0].mean():.4f}")

# 2. Inspect Live Extraction Format
mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        coords = np.array([[l.x, l.y, l.z] for l in lm[:468]]).flatten()
        print(f"Live Raw Shape: {coords.shape}")
        print(f"Live Row 0 Min: {coords.min():.4f} | Max: {coords.max():.4f} | Mean: {coords.mean():.4f}")