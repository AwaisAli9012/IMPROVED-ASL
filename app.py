"""
IMPROVED ASL - Production Web Backend
================================================
- Clean MJPEG Stream
- Thread-safe predictions & persistent JSON State API
- Fully coordinated endpoints for UI population & searching
"""

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import threading
from queue import Queue
from collections import deque
import mediapipe as mp
import gc

from Config import GROUPS, MODELS_DIR

app = Flask(__name__)
CORS(app)

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

# Initialize MediaPipe
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except Exception as e:
    hands = None

capture_queue = Queue(maxsize=2)
render_queue = Queue(maxsize=1)

current_group = 'SIGN1'
latest_detection_state = {
    'hands_detected': False,
    'class_name': None,
    'confidence': 0.0,
    'group_id': 'SIGN1'
}
state_lock = threading.Lock()

def camera_capture_thread():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            capture_queue.get_nowait()
        except Exception:
            pass
        capture_queue.put(frame)

def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
        for hand_landmarks in sorted_hands[:2]:
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
        return int(np.argmax(final_probs)), float(np.max(final_probs))
    except Exception:
        return None, None

def processing_thread():
    global latest_detection_state
    prediction_window = deque(maxlen=5)
    frame_count = 0
    
    while True:
        try:
            frame = capture_queue.get(timeout=1)
        except Exception:
            continue
        
        frame_count += 1
        frame_flipped = cv2.flip(frame, 1)
        
        with state_lock:
            group_id = current_group

        rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = hands.process(rgb_frame) if hands else None
        hands_detected = bool(results and results.multi_hand_landmarks)
        
        class_idx, confidence = None, 0.0
        if hands_detected:
            keypoints = extract_keypoints(results)
            class_idx, confidence = predict_ensemble(keypoints, group_id)
            if confidence > 0.60:
                prediction_window.append((class_idx, confidence))
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
                classes = GROUPS[group_id]['classes']
                stable_class_name = classes[most_common] if most_common < len(classes) else "Unknown"
                stable_confidence = max([p[1] for p in prediction_window if p[0] == most_common])

        # Draw skeletal landmarks for rendering
        if hands_detected and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_flipped, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        with state_lock:
            latest_detection_state = {
                'hands_detected': hands_detected,
                'class_name': stable_class_name,
                'confidence': round(stable_confidence * 100, 1),
                'group_id': group_id
            }

        try:
            render_queue.get_nowait()
        except Exception:
            pass
        render_queue.put(frame_flipped)

        if frame_count % 100 == 0:
            gc.collect()

def generate_frames():
    while True:
        try:
            frame = render_queue.get(timeout=1)
        except Exception:
            continue
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ═══════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html', groups=GROUPS)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/groups')
def get_groups():
    """Returns all available sign groups for the frontend dropdown."""
    return jsonify(GROUPS)

@app.route('/api/state')
def get_state():
    """Returns the latest detection state (for live JS polling)."""
    with state_lock:
        return jsonify(latest_detection_state)

@app.route('/api/group', methods=['GET', 'POST'])
def handle_group():
    """Gets or updates the active detection group."""
    global current_group
    if request.method == 'POST':
        data = request.json or {}
        group_id = data.get('group_id')
        if group_id in GROUPS:
            with state_lock:
                current_group = group_id
            return jsonify({'status': 'success', 'group': group_id})
        return jsonify({'status': 'error', 'message': 'Invalid Group ID'}), 400
    
    with state_lock:
        return jsonify({'current_group': current_group})

@app.route('/api/search', methods=['POST'])
def search_class():
    """Searches for a sign class and sets the group automatically if found."""
    global current_group
    data = request.json or {}
    query = data.get('query', '').strip().lower()
    
    if not query:
        return jsonify({'found': False}), 400

    for group_id, group_info in GROUPS.items():
        classes = group_info.get('classes', [])
        for class_name in classes:
            if class_name.lower() == query:
                with state_lock:
                    current_group = group_id
                return jsonify({'found': True, 'group': group_id, 'class': class_name})
                
    return jsonify({'found': False})

if __name__ == '__main__':
    threading.Thread(target=camera_capture_thread, daemon=True).start()
    threading.Thread(target=processing_thread, daemon=True).start()
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)