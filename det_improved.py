"""
IMPROVED ASL - Real-Time Detection (Fullscreen + Side Panel)
=============================================================
"""

import cv2
import numpy as np
import pickle
import mediapipe as mp
from pathlib import Path
from Config import GROUPS, MODELS_DIR

print("=" * 70)
print("IMPROVED ASL - REAL-TIME DETECTION")
print("=" * 70)

# Load ensemble models
models = {}
for group_id in GROUPS:
    try:
        with open(MODELS_DIR / f"{group_id}_rf.pkl", 'rb') as f:
            models[group_id] = {
                'rf': pickle.load(f),
                'xgb': pickle.load(open(MODELS_DIR / f"{group_id}_xgb.pkl", 'rb')),
                'meta': pickle.load(open(MODELS_DIR / f"{group_id}_meta.pkl", 'rb'))
            }
    except Exception:
        pass

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    while len(keypoints) < 126:
        keypoints.append(0.0)
    return np.array(keypoints[:126], dtype=np.float32)

def predict_ensemble(keypoints, group_id):
    if group_id not in models or keypoints is None:
        return None, None
    try:
        x = keypoints.reshape(1, -1)
        rf_probs = models[group_id]['rf'].predict_proba(x)[0]
        xgb_probs = models[group_id]['xgb'].predict_proba(x)[0]
        meta_x = np.hstack([rf_probs, xgb_probs]).reshape(1, -1)
        final_probs = models[group_id]['meta'].predict_proba(meta_x)[0]
        class_idx = int(np.argmax(final_probs))
        confidence = float(np.max(final_probs))
        return class_idx, confidence
    except Exception:
        return None, None

def get_class_name(group_id, class_idx):
    group_classes = GROUPS[group_id]['classes']
    if class_idx < len(group_classes):
        return group_classes[class_idx]
    return "Unknown"

CURRENT_GROUP = 'ALPHA1'
cap = cv2.VideoCapture(0)

cv2.namedWindow("IMPROVED ASL - Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("IMPROVED ASL - Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Process MediaPipe detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        hand_detected = bool(results.multi_hand_landmarks)
        if hand_detected:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            keypoints = extract_keypoints(results)
            class_idx, conf = predict_ensemble(keypoints, CURRENT_GROUP)
            class_name = get_class_name(CURRENT_GROUP, class_idx) if (class_idx is not None and conf > 0.5) else None
        else:
            class_name, conf = None, 0.0

        # Create canvas with side panel overlay
        sidebar_w = 320
        canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        
        # Sidebar background
        canvas[:, w:] = (30, 30, 30)
        cv2.line(canvas, (w, 0), (w, h), (70, 70, 70), 2)

        # Draw main camera HUD
        cv2.putText(canvas, f"GROUP: {CURRENT_GROUP}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        if not hand_detected:
            cv2.putText(canvas, "NO HAND DETECTED", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        elif class_name:
            cv2.putText(canvas, f"DETECTED: {class_name} ({conf:.1%})", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        # Draw Side Panel Controls & Signs
        cv2.putText(canvas, "AVAILABLE SIGNS", (w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(canvas, (w + 20, 65), (w + sidebar_w - 20, 65), (100, 100, 100), 1)

        classes = GROUPS[CURRENT_GROUP]['classes']
        for i, sign_label in enumerate(classes):
            y_pos = 110 + (i * 45)
            is_active = (class_name == sign_label)
            color = (0, 255, 0) if is_active else (200, 200, 200)
            prefix = "► " if is_active else "  "
            cv2.putText(canvas, f"{prefix}{sign_label}", (w + 30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2 if is_active else 1)

        # Draw Hotkey Instructions at bottom
        cv2.putText(canvas, "HOTKEYS:", (w + 20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 1)
        cv2.putText(canvas, "1-6: Alphabets (ALPHA1-6)", (w + 20, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(canvas, "a-g: Signs (SIGN1-7)", (w + 20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(canvas, "q: Quit", (w + 20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("IMPROVED ASL - Detection", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in '123456':
            CURRENT_GROUP = f'ALPHA{key - ord("0")}'
        elif chr(key) in 'abcdefg':
            CURRENT_GROUP = f'SIGN{ord(chr(key)) - ord("a") + 1}'

cap.release()
cv2.destroyAllWindows()
