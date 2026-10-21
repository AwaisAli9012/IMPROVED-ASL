"""
CALIBRATED EMOTION DETECTOR - EFFORTLESS ANGRY SENSITIVITY
===========================================================
1. Run script and keep a neutral expression.
2. Press 'c' to capture your baseline face structure.
3. Express subtle movements to trigger emotions.
"""

import cv2
import numpy as np
import mediapipe as mp

print("=" * 70)
print("INITIALIZING EFFORTLESS CALIBRATED EMOTION DETECTOR")
print("=" * 70)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Baseline Neutral Reference Data
neutral_baseline = {
    'smile': None,
    'brow_squeeze': None,
    'brow_eye': None,
    'calibrated': False
}

# Smoothing buffer
smoothed = {
    'smile': 0.0,
    'brow_squeeze': 0.0,
    'brow_eye': 0.0,
    'mar': 0.0
}
ALPHA = 0.20  # Smoother transitions to prevent rapid toggling

def extract_raw_features(landmarks, w, h):
    coords = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)

    face_width = np.linalg.norm(coords[234, :2] - coords[454, :2]) + 1e-6
    face_height = np.linalg.norm(coords[10, :2] - coords[150, :2]) + 1e-6

    # Pitch ratio for head tilt compensation
    forehead_to_nose = coords[1, 1] - coords[10, 1]
    nose_to_chin = coords[152, 1] - coords[1, 1] + 1e-6
    pitch_ratio = forehead_to_nose / nose_to_chin

    # 1. Lip Curvature (Smile / Frown)
    lip_center_y = (coords[13, 1] + coords[14, 1]) / 2.0
    left_corner_lift = lip_center_y - coords[61, 1]
    right_corner_lift = lip_center_y - coords[291, 1]
    raw_smile = ((left_corner_lift + right_corner_lift) / 2.0) / face_width

    if pitch_ratio > 0.65:
        raw_smile += (pitch_ratio - 0.65) * 0.045

    # 2. Inner Eyebrow Squeeze Distance (Landmarks 55 & 285)
    inner_brow_dist = np.linalg.norm(coords[55, :2] - coords[285, :2])
    raw_brow_squeeze = inner_brow_dist / face_width

    # 3. Eyebrow Height relative to Eyes
    left_brow_dist = np.linalg.norm(coords[70, :2] - coords[159, :2])
    right_brow_dist = np.linalg.norm(coords[300, :2] - coords[386, :2])
    raw_brow_eye = ((left_brow_dist + right_brow_dist) / 2.0) / face_height

    # 4. Mouth Opening Ratio
    lip_height = np.linalg.norm(coords[13, :2] - coords[14, :2])
    lip_width = np.linalg.norm(coords[61, :2] - coords[291, :2]) + 1e-6
    raw_mar = lip_height / lip_width

    return raw_smile, raw_brow_squeeze, raw_brow_eye, raw_mar

def classify_emotion(landmarks, w, h):
    global smoothed, neutral_baseline

    raw_smile, raw_brow_squeeze, raw_brow_eye, raw_mar = extract_raw_features(landmarks, w, h)

    # Apply Smoothing
    smoothed['smile'] = ALPHA * raw_smile + (1 - ALPHA) * smoothed['smile']
    smoothed['brow_squeeze'] = ALPHA * raw_brow_squeeze + (1 - ALPHA) * smoothed['brow_squeeze']
    smoothed['brow_eye'] = ALPHA * raw_brow_eye + (1 - ALPHA) * smoothed['brow_eye']
    smoothed['mar'] = ALPHA * raw_mar + (1 - ALPHA) * smoothed['mar']

    if not neutral_baseline['calibrated']:
        return "PRESS 'C' TO CALIBRATE NEUTRAL FACE", 0.0

    # Calculate differences relative to neutral calibration baseline
    delta_smile = smoothed['smile'] - neutral_baseline['smile']
    delta_brow_eye = smoothed['brow_eye'] - neutral_baseline['brow_eye']
    delta_squeeze = smoothed['brow_squeeze'] - neutral_baseline['brow_squeeze']

    # --- Distinct Priority Cascade ---

    # 1. HAPPY: Smile lifted above neutral baseline
    if delta_smile > 0.010:
        conf = min(0.99, 0.75 + delta_smile * 20)
        return "HAPPY", conf

    # 2. SAD: Frown corners pulled down below neutral baseline (independent of eyebrows)
    elif delta_smile < -0.012:
        conf = min(0.99, 0.75 + abs(delta_smile) * 20)
        return "SAD", conf

    # 3. ANGRY: Eyebrows strictly lower than neutral (very soft threshold, zero facial strain)
    elif delta_brow_eye < -0.003 or delta_squeeze < -0.008:
        drop = max(abs(delta_brow_eye), abs(delta_squeeze))
        conf = min(0.99, 0.70 + drop * 35)
        return "ANGRY", conf

    # 4. NEUTRAL: Default resting state
    else:
        return "NEUTRAL", 0.85

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam.")

print("\n--- INFERENCE ENGINE RUNNING ---")
print("Look at the camera with a neutral face and press 'c' to calibrate.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    current_prediction_text = "Searching face..."
    conf_val = 0.0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            emotion, conf_val = classify_emotion(face_landmarks.landmark, w, h)
            
            if neutral_baseline['calibrated']:
                current_prediction_text = f"Emotion: {emotion} ({conf_val * 100:.1f}%)"
            else:
                current_prediction_text = emotion

            # Calibration trigger
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                raw_s, raw_sq, raw_be, _ = extract_raw_features(face_landmarks.landmark, w, h)
                neutral_baseline['smile'] = raw_s
                neutral_baseline['brow_squeeze'] = raw_sq
                neutral_baseline['brow_eye'] = raw_be
                neutral_baseline['calibrated'] = True
                print("✅ Personal Neutral Face Calibrated Successfully!")

            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

    # Status Overlay
    color = (0, 255, 0) if neutral_baseline['calibrated'] else (0, 165, 255)
    cv2.putText(frame, current_prediction_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    
    if not neutral_baseline['calibrated']:
        cv2.putText(frame, "Hold resting face & press 'c'", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Calibrated Live Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()