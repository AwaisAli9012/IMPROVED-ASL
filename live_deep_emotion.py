"""
LIVE DEEP EMOTION DETECTOR
==========================
Combines MediaPipe alignment, MobileNetV3 embeddings, PyTorch MLP classification,
and temporal window smoothing for live emotion inference.
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from pathlib import Path

from face_aligner import FaceAligner
from feature_extractor import VisionFeatureExtractor

# 1. Define MLP Architecture matching training (RobustEmotionClassifier layout with BatchNorm)
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Corrected mapping sequence: Index 0 -> 'angry', Index 1 -> 'happy', Index 2 -> 'neutral', Index 3 -> 'sad'
CLASSES = np.array(['angry', 'happy', 'neutral', 'sad'])
np.save("models/classes.npy", CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionClassifier(input_dim=1024, num_classes=len(CLASSES)).to(device)
state_dict = torch.load("models/emotion_mlp.pth", map_location=device)

if "state_dict" in state_dict:
    model.load_state_dict(state_dict["state_dict"])
else:
    model.load_state_dict(state_dict)

model.eval()

# 3. Initialize pipeline components
aligner = FaceAligner()
extractor = VisionFeatureExtractor(device=device)

# Temporal smoothing buffer (expanded to 15 frames for stability)
prob_buffer = deque(maxlen=15)

cap = cv2.VideoCapture(0)
print(f"[INFO] Launching Live Deep Emotion Detector on {device}. Active classes: {CLASSES}. Press 'q' to exit.")

current_emotion = "SEARCHING..."
confidence = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    start_t = time.time()

    # Align & Crop Face
    face_crop, bbox = aligner.process_frame(frame)

    if face_crop is not None and face_crop.size > 0:
        # Convert OpenCV BGR format to RGB for feature extraction
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Extract 1024D embedding
        embedding = extractor.extract(face_crop_rgb)

        if embedding is not None:
            embedding = embedding.flatten()

            # L2 Normalize feature embedding vector before passing to MLP
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            tensor_emb = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor_emb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            prob_buffer.append(probs)

            # Apply temporal averaging over last 15 frames
            avg_probs = np.mean(prob_buffer, axis=0)
            pred_idx = np.argmax(avg_probs)
            raw_confidence = float(avg_probs[pred_idx]) * 100.0

            # Confidence hysteresis thresholding to prevent rapid switching
            if raw_confidence > 40.0:
                current_emotion = str(CLASSES[pred_idx]).upper()
                confidence = raw_confidence

            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    latency = (time.time() - start_t) * 1000.0

    # Overlay UI
    cv2.putText(frame, f"Emotion: {current_emotion}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Latency: {latency:.1f}ms", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Production Deep Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()