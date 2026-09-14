"""
IMPROVED ASL - Optimized Low-Lag Version
=============================================================================================
"""

import os
import gc
import pickle
import threading
from queue import Queue
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp

from google import genai

from Config import GROUPS, MODELS_DIR, APP_CONFIG

GEMINI_API_KEY = "AQ.Ab8RN6I5ugDUkIzQm8JpoIRZUVYWmY5vEZq1Q0S6eFBFLMBGqg"

app = Flask(__name__)
CORS(app)

print("=" * 70)
print("IMPROVED ASL - OPTIMIZED LOW-LAG ENGINE")
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
    except Exception as e:
        print(f"   ❌ {group_id}: {e}")

print(f"✓ Loaded {len(models)} model groups")

# Initialize MediaPipe (Optimized for speed)
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Track up to 2 hands
        model_complexity=0,  # Lowest complexity for zero latency
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ MediaPipe initialized (Optimized mode)")
except Exception as e:
    print(f"❌ MediaPipe error: {e}")
    hands = None

sign_buffer = []  
buffer_lock = threading.Lock()

current_group = 'SIGN1'
latest_detection_state = {
    'hands_detected': False,
    'class_name': None,
    'confidence': 0.0,
    'group_id': 'SIGN1'
}
state_lock = threading.Lock()

capture_queue = Queue(maxsize=1)
render_queue = Queue(maxsize=1)
executor = ThreadPoolExecutor(max_workers=2)

def camera_capture_thread():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✓ Camera thread started (640x480)")
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
        return int(np.argmax(final_probs)), float(np.max(final_probs))
    except Exception:
        return None, None

def processing_thread():
    global latest_detection_state
    prediction_window = deque(maxlen=3)
    last_added_sign = None
    frames_since_last_add = 0
    frame_count = 0

    print("✓ Processing thread started")
    
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
                if sum(1 for v in valid_votes if v == most_common) >= 2:
                    classes = GROUPS[group_id]['classes']
                    stable_class_name = classes[most_common] if most_common < len(classes) else "Unknown"
                    stable_confidence = np.mean([p[1] for p in prediction_window if p[0] == most_common])

        # Debounced Auto-Add with Smart Word & Space Handling
        if stable_class_name is not None and stable_confidence > 0.65:
            frames_since_last_add += 1
            if stable_class_name != last_added_sign and frames_since_last_add >= 15:
                with buffer_lock:
                    class_lower = stable_class_name.lower()
                    
                    if class_lower == 'space':
                        # If there are items in the buffer, insert a clear visual separator or space marker
                        if sign_buffer and sign_buffer[-1] != "—":
                            sign_buffer.append("—")  # Adds a clean visual word divider in your UI
                    elif class_lower == 'del':
                        if sign_buffer:
                            sign_buffer.pop()
                    else:
                        if not sign_buffer or sign_buffer[-1] != stable_class_name:
                            sign_buffer.append(stable_class_name)
                            
                last_added_sign = stable_class_name
                frames_since_last_add = 0
        else:
            if frames_since_last_add > 5:
                last_added_sign = None
            frames_since_last_add = 0

        # Draw lightweight landmarks
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
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', groups=GROUPS)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/groups')
def get_groups():
    return jsonify(GROUPS)

@app.route('/api/group', methods=['GET', 'POST'])
def handle_group():
    global current_group
    if request.method == 'POST':
        data = request.get_json(force=True, silent=True) or {}
        group_id = data.get('group_id')
        if group_id in GROUPS:
            with state_lock:
                current_group = group_id
            return jsonify({'status': 'success', 'group': group_id})
        return jsonify({'status': 'error', 'message': 'Invalid Group ID'}), 400
    
    with state_lock:
        return jsonify({'current_group': current_group})

@app.route('/api/buffer', methods=['GET', 'POST', 'DELETE'])
def manage_buffer():
    if request.method == 'GET':
        with buffer_lock:
            return jsonify({'signs': sign_buffer.copy()})
            
    elif request.method == 'POST':
        data = request.get_json(force=True, silent=True) or {}
        action = data.get('action')
        
        with buffer_lock:
            if action == 'clear':
                sign_buffer.clear()
            elif action == 'backspace' and sign_buffer:
                sign_buffer.pop()
            elif action == 'add':
                sign = data.get('sign')
                if sign:
                    sign_buffer.append(sign)
            return jsonify({'status': 'success', 'signs': sign_buffer.copy()})
            
    elif request.method == 'DELETE':
        with buffer_lock:
            sign_buffer.clear()
        return jsonify({'status': 'deleted', 'signs': []})

@app.route('/api/generate', methods=['POST', 'GET'])
def generate_sentence():
    """Generate sentence using the native google-genai SDK"""
    if request.method == 'GET':
        return jsonify({'error': 'Please use POST to generate sentences'}), 405

    data = request.get_json(force=True, silent=True) or {}
    signs = data.get('signs', [])
    
    if not signs:
        return jsonify({'error': 'No signs in buffer'}), 400
    
    # Clean up the buffer text for Gemini, replacing visual dividers with proper spaces
    cleaned_signs = [s if s != "—" else " " for s in signs]
    signs_text = "".join(cleaned_signs) if any(len(s) == 1 for s in cleaned_signs) else ' '.join(cleaned_signs)
    
    prompt = f"Convert these ASL signs into a natural, grammatically correct English sentence: {signs_text.strip()}. Return only the sentence, nothing else."
    
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
            return jsonify({'sentence': sentence.strip()})
        else:
            return jsonify({'error': 'Empty response from Gemini API'}), 400
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Native SDK Generation Error: {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/health')
def health():
    with state_lock:
        state = latest_detection_state.copy()
    with buffer_lock:
        buf_size = len(sign_buffer)
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'current_group': state['group_id'],
        'hands_detected': state['hands_detected'],
        'active_prediction': state['class_name'],
        'confidence': state['confidence'],
        'signs_in_buffer': buf_size
    })

if __name__ == '__main__':
    t1 = threading.Thread(target=camera_capture_thread, daemon=True)
    t2 = threading.Thread(target=processing_thread, daemon=True)
    t1.start()
    t2.start()
    
    print("=" * 70)
    print("✓ Flask server running on http://127.0.0.1:5000 (Optimized)")
    print("=" * 70 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)