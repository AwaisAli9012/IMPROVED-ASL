"""
IMPROVED ASL - Multimodal Wizard Engine
=============================================================================================
Integrates:
1. Ensemble Gesture Recognition (RF + XGB + Meta)
2. Live Permission & Calibrated MediaPipe Emotion Detector
=============================================================================================
"""

import os
import gc
import pickle
import threading
import time
from queue import Queue
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS

from google import genai
from dotenv import load_dotenv
from Config import GROUPS, MODELS_DIR, APP_CONFIG

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app)

print("=" * 70)
print("IMPROVED ASL - MULTIMODAL WIZARD ENGINE")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1. LOAD GESTURE ENSEMBLE MODELS
# -----------------------------------------------------------------------------
models = {}
for group_id in GROUPS:
    try:
        with open(MODELS_DIR / f"{group_id}_rf.pkl", 'rb') as f:
            models[group_id] = {
                'rf': pickle.load(f),
                'xgb': pickle.load(open(MODELS_DIR / f"{group_id}_xgb.pkl", 'rb')),
                'meta': pickle.load(open(MODELS_DIR / f"{group_id}_meta.pkl", 'rb'))
            }
    except Exception as e:
        print(f"   ❌ Gesture Model {group_id}: {e}")

print(f"✓ Loaded {len(models)} gesture model groups")

# -----------------------------------------------------------------------------
# 2. INITIALIZE MEDIAPIPE ENGINES
# -----------------------------------------------------------------------------
try:
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ MediaPipe Face Mesh initialized")
except Exception as e:
    print(f"❌ MediaPipe Face Mesh initialization failed: {e}")
    face_mesh = None

try:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    print("✓ MediaPipe Hands initialized")
except Exception as e:
    print(f"❌ MediaPipe Hands initialization failed: {e}")
    hands = None

# -----------------------------------------------------------------------------
# 3. GLOBAL STATE & CALIBRATION BUFFERS
# -----------------------------------------------------------------------------
session_state = {
    "signs": [],             
    "cursor_index": 0,       
    "last_valid_sign": None, 
    "last_confidence": 0.0,
    "camera_detected_emotion": "MANUAL / DISABLED", 
    "emotion_confidence": 0.0,
    "locked_emotion": "NEUTRAL",          
    "active_tab": "SIGN",
    "permission_granted": False,
    "is_calibrated": False
}
state_lock = threading.Lock()

current_group = 'SIGN1'
capture_queue = Queue(maxsize=1)
render_queue = Queue(maxsize=1)
executor = ThreadPoolExecutor(max_workers=2)

neutral_baseline = {
    'smile': None,
    'brow_squeeze': None,
    'brow_eye': None,
    'calibrated': False
}

smoothed_features = {
    'smile': 0.0,
    'brow_squeeze': 0.0,
    'brow_eye': 0.0,
    'mar': 0.0
}
ALPHA = 0.20

trigger_calibration_flag = False

# -----------------------------------------------------------------------------
# 4. CALIBRATED EMOTION EXTRACTION & CLASSIFICATION
# -----------------------------------------------------------------------------
def extract_raw_features(landmarks, w, h):
    coords = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)

    face_width = np.linalg.norm(coords[234, :2] - coords[454, :2]) + 1e-6
    face_height = np.linalg.norm(coords[10, :2] - coords[150, :2]) + 1e-6

    forehead_to_nose = coords[1, 1] - coords[10, 1]
    nose_to_chin = coords[152, 1] - coords[1, 1] + 1e-6
    pitch_ratio = forehead_to_nose / nose_to_chin

    lip_center_y = (coords[13, 1] + coords[14, 1]) / 2.0
    left_corner_lift = lip_center_y - coords[61, 1]
    right_corner_lift = lip_center_y - coords[291, 1]
    raw_smile = ((left_corner_lift + right_corner_lift) / 2.0) / face_width

    if pitch_ratio > 0.65:
        raw_smile += (pitch_ratio - 0.65) * 0.045

    inner_brow_dist = np.linalg.norm(coords[55, :2] - coords[285, :2])
    raw_brow_squeeze = inner_brow_dist / face_width

    left_brow_dist = np.linalg.norm(coords[70, :2] - coords[159, :2])
    right_brow_dist = np.linalg.norm(coords[300, :2] - coords[386, :2])
    raw_brow_eye = ((left_brow_dist + right_brow_dist) / 2.0) / face_height

    lip_height = np.linalg.norm(coords[13, :2] - coords[14, :2])
    lip_width = np.linalg.norm(coords[61, :2] - coords[291, :2]) + 1e-6
    raw_mar = lip_height / lip_width

    return raw_smile, raw_brow_squeeze, raw_brow_eye, raw_mar

def classify_emotion(landmarks, w, h):
    global smoothed_features, neutral_baseline, trigger_calibration_flag

    raw_s, raw_sq, raw_be, raw_m = extract_raw_features(landmarks, w, h)

    # Perform calibration ONLY when explicitly triggered by button
    if trigger_calibration_flag:
        neutral_baseline['smile'] = raw_s
        neutral_baseline['brow_squeeze'] = raw_sq
        neutral_baseline['brow_eye'] = raw_be
        neutral_baseline['calibrated'] = True
        trigger_calibration_flag = False
        with state_lock:
            session_state["is_calibrated"] = True
        print("✅ Neutral Calibration Successful!")

    if not neutral_baseline['calibrated']:
        return "NEUTRAL CALIBRATION NEEDED", 0.0

    smoothed_features['smile'] = ALPHA * raw_s + (1 - ALPHA) * smoothed_features['smile']
    smoothed_features['brow_squeeze'] = ALPHA * raw_sq + (1 - ALPHA) * smoothed_features['brow_squeeze']
    smoothed_features['brow_eye'] = ALPHA * raw_be + (1 - ALPHA) * smoothed_features['brow_eye']
    smoothed_features['mar'] = ALPHA * raw_m + (1 - ALPHA) * smoothed_features['mar']

    delta_smile = smoothed_features['smile'] - neutral_baseline['smile']
    delta_brow_eye = smoothed_features['brow_eye'] - neutral_baseline['brow_eye']
    delta_squeeze = smoothed_features['brow_squeeze'] - neutral_baseline['brow_squeeze']

    if delta_smile > 0.010:
        conf = min(0.99, 0.75 + delta_smile * 20)
        return "HAPPY", conf * 100.0

    elif delta_smile < -0.012:
        conf = min(0.99, 0.75 + abs(delta_smile) * 20)
        return "SAD", conf * 100.0

    elif delta_brow_eye < -0.003 or delta_squeeze < -0.008:
        drop = max(abs(delta_brow_eye), abs(delta_squeeze))
        conf = min(0.99, 0.70 + drop * 35)
        return "ANGRY", conf * 100.0

    else:
        return "NEUTRAL", 85.0

# -----------------------------------------------------------------------------
# 5. THREADING & PROCESSING PIPELINE
# -----------------------------------------------------------------------------
def camera_capture_thread():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✓ Camera thread started")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            capture_queue.get_nowait()
        except Exception:
            pass
        capture_queue.put(frame)

def extract_hand_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
    while len(keypoints) < 126:
        keypoints.append(0.0)
    return np.array(keypoints[:126], dtype=np.float32)

def predict_gesture(keypoints, group_id):
    if group_id not in models or keypoints is None:
        return None, None
    try:
        x = keypoints.reshape(1, -1)
        rf_probs = models[group_id]['rf'].predict_proba(x)[0]
        xgb_probs = models[group_id]['xgb'].predict_proba(x)[0]
        meta_x = np.hstack([rf_probs, xgb_probs]).reshape(1, -1)
        final_probs = models[group_id]['meta'].predict_proba(meta_x)[0]
        return int(np.argmax(final_probs)), float(np.max(final_probs))
    except Exception:
        return None, None

def processing_thread():
    global session_state
    prediction_window = deque(maxlen=3)
    frame_count = 0

    print("✓ Processing thread started")
    
    while True:
        try:
            frame = capture_queue.get(timeout=1)
        except Exception:
            continue
        
        frame_count += 1
        frame_flipped = cv2.flip(frame, 1)
        h, w, _ = frame_flipped.shape
        
        with state_lock:
            group_id = current_group
            perm_granted = session_state["permission_granted"]

        rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # --- A. EMOTION DETECTION (ONLY WHEN PERMISSION IS GRANTED) ---
        current_emotion = "PERMISSION REQUIRED"
        confidence = 0.0

        if face_mesh:
            face_results = face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    if perm_granted:
                        current_emotion, confidence = classify_emotion(face_landmarks.landmark, w, h)
                    
                    # Draw Face Mesh contour overlay
                    mp_drawing.draw_landmarks(
                        image=frame_flipped,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 0) if perm_granted else (128, 128, 128), 
                            thickness=1, circle_radius=1
                        )
                    )

        # --- B. HAND GESTURE DETECTION ---
        hand_results = hands.process(rgb_frame) if hands else None
        hands_detected = bool(hand_results and hand_results.multi_hand_landmarks)
        
        if hands_detected:
            keypoints = extract_hand_keypoints(hand_results)
            class_idx, conf_val = predict_gesture(keypoints, group_id)
            if conf_val and conf_val > 0.60:
                prediction_window.append((class_idx, conf_val))
            else:
                prediction_window.append((None, 0.0))
        else:
            prediction_window.clear()

        stable_class_name = None
        stable_confidence = 0.0

        if len(prediction_window) > 0:
            valid_votes = [p[0] for p in prediction_window if p[0] is not None]
            if valid_votes:
                most_common = max(set(valid_votes), key=valid_votes.count)
                if sum(1 for v in valid_votes if v == most_common) >= 2:
                    group_classes = GROUPS[group_id]['classes']
                    stable_class_name = group_classes[most_common] if most_common < len(group_classes) else "Unknown"
                    stable_confidence = np.mean([p[1] for p in prediction_window if p[0] == most_common])

        # --- C. UPDATE SESSION STATE ---
        with state_lock:
            if stable_class_name is not None and stable_confidence > 0.60:
                session_state["last_valid_sign"] = stable_class_name
                session_state["last_confidence"] = round(stable_confidence * 100, 1)
            
            if perm_granted:
                session_state["camera_detected_emotion"] = current_emotion
                session_state["emotion_confidence"] = round(confidence, 1)
            else:
                session_state["camera_detected_emotion"] = "PERMISSION REQUIRED"
                session_state["emotion_confidence"] = 0.0

        if hands_detected and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_flipped, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        try:
            render_queue.get_nowait()
        except Exception:
            pass
        render_queue.put(frame_flipped)

        if frame_count % 50 == 0:
            gc.collect()

def generate_frames():
    while True:
        try:
            frame = render_queue.get(timeout=1)
        except Exception:
            continue
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n' + buffer.tobytes() + b'\r\n')

# -----------------------------------------------------------------------------
# 6. REST API ENDPOINTS
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html', groups=GROUPS)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state', methods=['GET'])
def get_state():
    with state_lock:
        return jsonify(session_state)

@app.route('/api/grant_emotion_permission', methods=['POST'])
def grant_emotion_permission():
    with state_lock:
        session_state["permission_granted"] = True
    return jsonify({'status': 'success', 'message': 'Permission granted. Please calibrate neutral face.'})

@app.route('/api/calibrate_neutral', methods=['POST'])
def calibrate_neutral():
    global trigger_calibration_flag
    with state_lock:
        if not session_state["permission_granted"]:
            return jsonify({'status': 'error', 'message': 'Permission not granted yet'}), 400
    
    trigger_calibration_flag = True
    return jsonify({'status': 'success', 'message': 'Neutral calibration triggered'})

@app.route('/api/groups_detail', methods=['GET'])
def get_groups_detail():
    detailed_groups = {}
    for g_id, g_data in GROUPS.items():
        classes = g_data.get('classes', [])
        detailed_groups[g_id] = {
            'classes': classes,
            'display_name': f"{g_id} — [{', '.join(classes)}]"
        }
    return jsonify(detailed_groups)

@app.route('/api/group', methods=['GET', 'POST'])
def handle_group():
    global current_group
    if request.method == 'POST':
        data = request.get_json(force=True, silent=True) or {}
        group_id = data.get('group_id')
        if group_id in GROUPS:
            with state_lock:
                current_group = group_id
            return jsonify({'status': 'success', 'group': group_id, 'classes': GROUPS[group_id]['classes']})
        return jsonify({'status': 'error', 'message': 'Invalid Group ID'}), 400
    
    with state_lock:
        return jsonify({'current_group': current_group, 'classes': GROUPS[current_group]['classes']})

@app.route('/api/confirm_sign', methods=['POST'])
def confirm_sign():
    with state_lock:
        sign = session_state.get("last_valid_sign")
        if sign:
            idx = session_state["cursor_index"]
            session_state["signs"].insert(idx, sign)
            session_state["cursor_index"] += 1
            return jsonify({'status': 'success', 'state': session_state})
        return jsonify({'status': 'error', 'message': 'No sign latched to confirm'}), 400

@app.route('/api/navigate', methods=['POST'])
def navigate_cursor():
    data = request.get_json(force=True, silent=True) or {}
    direction = data.get('direction')
    
    with state_lock:
        signs_len = len(session_state["signs"])
        if direction == 'left' and session_state["cursor_index"] > 0:
            session_state["cursor_index"] -= 1
        elif direction == 'right' and session_state["cursor_index"] < signs_len:
            session_state["cursor_index"] += 1
        return jsonify({'status': 'success', 'cursor_index': session_state["cursor_index"]})

@app.route('/api/delete_sign', methods=['POST'])
def delete_sign():
    with state_lock:
        idx = session_state["cursor_index"]
        signs = session_state["signs"]
        
        if signs and idx > 0:
            signs.pop(idx - 1)
            session_state["cursor_index"] -= 1
        elif signs and idx == 0 and len(signs) > 0:
            signs.pop(0)
            
        return jsonify({'status': 'success', 'state': session_state})

@app.route('/api/clear', methods=['POST'])
def clear_all():
    with state_lock:
        session_state["signs"].clear()
        session_state["cursor_index"] = 0
        session_state["last_valid_sign"] = None
        session_state["locked_emotion"] = "NEUTRAL"
        return jsonify({'status': 'success', 'state': session_state})

@app.route('/api/confirm_emotion', methods=['POST'])
def confirm_emotion():
    data = request.get_json(force=True, silent=True) or {}
    emotion = data.get('emotion')
    
    with state_lock:
        if emotion:
            session_state["locked_emotion"] = emotion.upper()
        else:
            session_state["locked_emotion"] = session_state["camera_detected_emotion"]
            
        return jsonify({
            'status': 'success', 
            'locked_emotion': session_state["locked_emotion"],
            'camera_detected': session_state["camera_detected_emotion"]
        })

@app.route('/api/tab', methods=['POST'])
def switch_tab():
    data = request.get_json(force=True, silent=True) or {}
    tab = data.get('tab', 'SIGN')
    with state_lock:
        session_state["active_tab"] = tab
        return jsonify({'status': 'success', 'active_tab': session_state["active_tab"]})

@app.route('/api/generate', methods=['POST', 'GET'])
def generate_sentence():
    if request.method == 'GET':
        return jsonify({'error': 'Please use POST to generate sentences'}), 405

    data = request.get_json(force=True, silent=True) or {}
    
    with state_lock:
        signs = session_state.get('signs', []) or data.get('signs', [])
        emotion = session_state.get('locked_emotion', 'NEUTRAL')
    
    if not signs:
        return jsonify({'error': 'No signs in buffer'}), 400
    
    cleaned_signs = [s if s != "—" else " " for s in signs]
    signs_text = "".join(cleaned_signs) if any(len(s) == 1 for s in cleaned_signs) else ' '.join(cleaned_signs)
    
    prompt = (
        f"You are a multimodal ASL interpreter system.\n"
        f"1. Input Gestures (Signed Concepts): {signs_text.strip()}\n"
        f"2. Speaker's Facial Emotion Affect: {emotion.upper()}\n\n"
        f"Task: Translate these signed concepts into a natural, complete English sentence.\n"
        f"CRITICAL REQUIREMENT: The sentence MUST strongly express and be modified by the speaker's '{emotion.upper()}' emotion. "
        f"For example, if HAPPY, express enthusiasm; if ANGRY, express frustration/intensity; if SAD, express distress or regret.\n"
        f"Return ONLY the final translated sentence with no extra commentary."
    )
    
    if not GEMINI_API_KEY:
        return jsonify({'error': 'No Gemini API key configured'}), 500

    def _call_gemini():
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return response.text

    try:
        future = executor.submit(_call_gemini)
        sentence = future.result(timeout=15)
        
        if sentence:
            return jsonify({'sentence': sentence.strip(), 'signs': signs, 'emotion': emotion})
        else:
            return jsonify({'error': 'Empty response from Gemini API'}), 400
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Native SDK Generation Error: {error_msg}")
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    t1 = threading.Thread(target=camera_capture_thread, daemon=True)
    t2 = threading.Thread(target=processing_thread, daemon=True)
    t1.start()
    t2.start()
    
    print("=" * 70)
    print("✓ Flask server running on http://127.0.0.1:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)