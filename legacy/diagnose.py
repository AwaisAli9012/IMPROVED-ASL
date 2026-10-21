"""
LIVE EMOTION DIAGNOSTIC SCRIPT (1024-dim Vision Embeddings + CLAHE Contrast Enhancement)
"""
import cv2
import torch
import torch.nn as nn
import numpy as np

from face_aligner import FaceAligner
from feature_extractor import VisionFeatureExtractor

# 1. Matching architecture for 1024-dim input
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier(input_dim=1024, num_classes=4).to(device)

# Load retrained weights
checkpoint = torch.load("models/emotion_mlp.pth", map_location=device)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

aligner = FaceAligner()
extractor = VisionFeatureExtractor()

# Correct mapping matching np.unique(data['y']) order: ['ANGRY', 'HAPPY', 'NEUTRAL', 'SAD']
EMOTIONS = {0: 'ANGRY', 1: 'HAPPY', 2: 'NEUTRAL', 3: 'SAD'}

cap = cv2.VideoCapture(0)
print("\n--- LIVE DIAGNOSTIC MODE: Testing 1024-dim Retrained Vision MLP ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    face_crop, bbox = aligner.process_frame(frame)

    if face_crop is not None and face_crop.size > 0:
        # Apply CLAHE Contrast Enhancement to handle dark shadows / underexposure
        yuv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        face_crop_enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # Extract feature embedding from contrast-enhanced crop
        emb = extractor.extract(face_crop_enhanced)

        if emb is not None:
            tensor_emb = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor_emb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            idx = np.argmax(probs)
            pred_label = EMOTIONS.get(idx, f"Class {idx}")

            print(f"ANGRY: {probs[0]:.2f} | HAPPY: {probs[1]:.2f} | NEUTRAL: {probs[2]:.2f} | SAD: {probs[3]:.2f} -> PREDICTION: {pred_label}")

            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, pred_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Diagnostic Window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()