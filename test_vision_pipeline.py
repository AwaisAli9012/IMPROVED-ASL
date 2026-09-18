import cv2
import time
from face_aligner import FaceAligner
from feature_extractor import VisionFeatureExtractor

aligner = FaceAligner()
extractor = VisionFeatureExtractor()
cap = cv2.VideoCapture(0)

print("[INFO] Testing vision feature extraction pipeline...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    start_t = time.time()
    
    # 1. Align & Crop Face Patch
    face_crop, bbox = aligner.process_frame(frame)

    if face_crop is not None:
        # 2. Extract Deep Embedding Vector
        embedding = extractor.extract(face_crop)
        proc_time = (time.time() - start_t) * 1000

        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Embedding Size: {embedding.shape[0]}D", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {proc_time:.1f}ms", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Vision Pipeline Live Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()