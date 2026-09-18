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

# 1. Define MLP Architecture matching training
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 2. Load classes and model weights
classes = np.load("models/classes.npy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionClassifier(input_dim=1024, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("models/emotion_mlp.pth", map_location=device))
model.eval()

# 3. Initialize pipeline components
aligner = FaceAligner()
extractor = VisionFeatureExtractor(device=device)

# Temporal smoothing buffer (preserves existing 5-frame queue)
prob_buffer = deque(maxlen=5)

cap = cv2.VideoCapture(0)
print("[INFO] Launching Live Deep Emotion Detector. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    start_t = time.time()

    # Align & Crop Face
    face_crop, bbox = aligner.process_frame(frame)

    current_emotion = "NEUTRAL"
    confidence = 0.0

    if face_crop is not None:
        # Extract 1024D embedding
        embedding = extractor.extract(face_crop)

        if embedding is not None:
            # Prepare tensor for MLP
            tensor_emb = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor_emb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            prob_buffer.append(probs)

            # Apply temporal averaging over last 5 frames
            avg_probs = np.mean(prob_buffer, axis=0)
            pred_idx = np.argmax(avg_probs)

            current_emotion = str(classes[pred_idx])
            confidence = float(avg_probs[pred_idx]) * 100.0

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