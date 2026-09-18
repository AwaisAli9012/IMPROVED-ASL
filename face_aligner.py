"""
FACE ALIGNER MODULE
===================
Detects faces via MediaPipe, aligns roll orientation (eye level), 
and returns a standardized 224x224 cropped face patch.
"""

import cv2
import numpy as np
import mediapipe as mp

class FaceAligner:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.LEFT_EYE_CORNER = 33
        self.RIGHT_EYE_CORNER = 263

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = np.array([landmarks[self.LEFT_EYE_CORNER].x * w, landmarks[self.LEFT_EYE_CORNER].y * h])
        right_eye = np.array([landmarks[self.RIGHT_EYE_CORNER].x * w, landmarks[self.RIGHT_EYE_CORNER].y * h])

        d_y = right_eye[1] - left_eye[1]
        d_x = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(d_y, d_x))

        eye_center = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))

        rot_mat = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        rotated_frame = cv2.warpAffine(frame, rot_mat, (w, h), flags=cv2.INTER_CUBIC)

        all_coords = np.array([[l.x * w, l.y * h] for l in landmarks])
        ones = np.ones((all_coords.shape[0], 1))
        transformed_coords = np.hstack([all_coords, ones]) @ rot_mat.T

        x_min = max(0, int(np.min(transformed_coords[:, 0])))
        y_min = max(0, int(np.min(transformed_coords[:, 1])))
        x_max = min(w, int(np.max(transformed_coords[:, 0])))
        y_max = min(h, int(np.max(transformed_coords[:, 1])))

        box_w = x_max - x_min
        box_h = y_max - y_min
        pad_x = int(box_w * 0.15)
        pad_y = int(box_h * 0.15)

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        face_crop = rotated_frame[y_min:y_max, x_min:x_max]

        if face_crop.size == 0:
            return None, None

        resized_crop = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_AREA)

        return resized_crop, (x_min, y_min, x_max - x_min, y_max - y_min)