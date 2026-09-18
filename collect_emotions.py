"""
EMOTION DATA COLLECTOR
======================
Captures 1024D embeddings for target emotions into a dataset file.
Press 'h' for HAPPY, 's' for SAD, 'a' for ANGRY, 'n' for NEUTRAL.
"""

import cv2
import numpy as np
from pathlib import Path
from face_aligner import FaceAligner
from feature_extractor import VisionFeatureExtractor

aligner = FaceAligner()
extractor = VisionFeatureExtractor()
cap = cv2.VideoCapture(0)

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

X_data, y_data = [], []
counts = {"NEUTRAL": 0, "HAPPY": 0, "SAD": 0, "ANGRY": 0}

print("[INFO] Press 'n' (Neutral), 'h' (Happy), 's' (Sad), 'a' (Angry) to collect frames. Press 'q' to save & exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    face_crop, bbox = aligner.process_frame(frame)

    if face_crop is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display counts
    cv2.putText(frame, f"N: {counts['NEUTRAL']} | H: {counts['HAPPY']} | S: {counts['SAD']} | A: {counts['ANGRY']}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    label_map = {ord('n'): "NEUTRAL", ord('h'): "HAPPY", ord('s'): "SAD", ord('a'): "ANGRY"}
    if key in label_map and face_crop is not None:
        label = label_map[key]
        emb = extractor.extract(face_crop)
        if emb is not None:
            X_data.append(emb)
            y_data.append(label)
            counts[label] += 1

cap.release()
cv2.destroyAllWindows()

if len(X_data) > 0:
    np.savez(output_dir / "emotion_embeddings.npz", X=np.array(X_data), y=np.array(y_data))
    print(f"[SUCCESS] Saved {len(X_data)} samples to data/emotion_embeddings.npz")