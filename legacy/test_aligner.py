import cv2
from face_aligner import FaceAligner

aligner = FaceAligner()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    face_crop, bbox = aligner.process_frame(frame)

    if face_crop is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Normalized Face Patch (224x224)", face_crop)

    cv2.imshow("Main Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()