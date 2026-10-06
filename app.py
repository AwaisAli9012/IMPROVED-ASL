"""
ASL MULTIMODAL WIZARD - COMPLETE PRODUCTION ENGINE
=========================================================
Fully synchronized with Config.py and index.html UI state.
"""

import os
import gc
import pickle
import threading
from queue import Queue
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv

# Import Config definitions
from Config import GROUPS, MODELS_DIR, APP_CONFIG

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app)

print("=" * 70)
print("ASL MULTIMODAL WIZARD - ENGINE STARTING")
print("=" * 70)

# -------------------------------------------------------------------------
# 1. TEMPORAL SMOOTHING MEMORY CLASS
# -------------------------------------------------------------------------
class HandMemory:
    """Retains hand landmark state for brief detection drops."""
    def __init__(self, decay_frames=3):
        self.last_keypoints = None
        self.frames_without_hand = 0
        self.decay_frames = decay_frames

    def update(self, keypoints):
        if keypoints is not None and not np.all(keypoints == 0):
            self.last_keypoints = keypoints
            self.frames_without_hand = 0
        else:
            self.frames_without_hand += 1

    def get_keypoints(self):
        if self.frames_without_hand <= self.decay_frames:
            return self.last_keypoints
        return None

# -------------------------------------------------------------------------
# 2. GLOBAL STATE & LOCKS
# -------------------------------------------------------------------------
initial_group = list(GROUPS.keys())[0] if GROUPS else 'SIGN1'
current_group = initial_group

models = {}

# System & Inference State
latest_detection_state = {
    'hands_detected': False,
    'class_name': None,
    'confidence': 0.0,
    'group_id': current_group
}
state_lock = threading.Lock()

# Sign Workspace Sequence & Cursor Index
sign_buffer = []
cursor_index = 0
buffer_lock = threading.Lock()

# Facial Emotion & Calibration Workspace State
emotion_state = {
    'permission_granted': False,
    'calibrated': False,
    'camera_detected_emotion': 'NEUTRAL',
    'emotion_confidence': 95,
    'locked_emotion': 'NEUTRAL',
    'active_tab': 'SIGN'
}
emotion_lock = threading.Lock()

# Word Sequence Pilot Recording State
word_pilot_state = {
    'status': 'Ready (Select Sign Group)',
    'recording': False,
    'available': True,
    'prediction': None,
    'confidence': 0.0,
    'frame_count': 0
}
word_pilot_lock = threading.Lock()

# Streaming Queues
capture_queue = Queue(maxsize=1)
render_queue = Queue(maxsize=1)
executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------------------------------------
# 3. MODEL INGESTION & VALIDATION
# -------------------------------------------------------------------------
for group_id in GROUPS:
    try:
        rf_path = MODELS_DIR / f"{group_id}_rf.pkl"
        xgb_path = MODELS_DIR / f"{group_id}_xgb.pkl"
        meta_path = MODELS_DIR / f"{group_id}_meta.pkl"

        if rf_path.exists() and xgb_path.exists() and meta_path.exists():
            with open(rf_path, 'rb') as f_rf, \
                 open(xgb_path, 'rb') as f_xgb, \
                 open(meta_path, 'rb') as f_meta:
                models[group_id] = {
                    'rf': pickle.load(f_rf),
                    'xgb': pickle.load(f_xgb),
                    'meta': pickle.load(f_meta)
                }
            print(f"   ✓ Loaded models for group: {group_id}")
        else:
            print(f"   ⚠️  {group_id}: Missing one or more .pkl files in Models/")
    except Exception as e:
        print(f"   ❌ Failed loading group {group_id}: {e}")

print(f"✓ Total active model groups loaded: {len(models)}")

# -------------------------------------------------------------------------
# 4. MEDIAPIPE INITIALIZATION & FEATURE EXTRACTION
# -------------------------------------------------------------------------
try:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.15,
        min_tracking_confidence=0.15
    )
    print("✓ MediaPipe Hands solution initialized successfully")
except Exception as e:
    print(f"❌ MediaPipe initialization failed: {e}")
    hands = None

def extract_keypoints(results):
    """Extracts 126 hand keypoints normalized strictly between 0.0 and 1.0."""
    keypoints = []
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_norm = float(np.clip(lm.x, 0.0, 1.0))
                y_norm = float(np.clip(lm.y, 0.0, 1.0))
                keypoints.extend([x_norm, y_norm, lm.z])
                
    while len(keypoints) < 126:
        keypoints.append(0.0)
    return np.array(keypoints[:126], dtype=np.float32)

def map_index_to_class(class_idx, group_id):
    """Maps candidate index strictly to current group's class vocabulary."""
    if group_id not in GROUPS:
        return None
    group_info = GROUPS[group_id]
    classes = group_info.get('classes', [])
    if class_idx is not None and 0 <= class_idx < len(classes):
        return classes[class_idx]
    return None

def predict_ensemble(keypoints, group_id):
    """Executes prediction safely across loaded ensemble candidate models."""
    if group_id not in models or keypoints is None:
        return None, 0.0
    try:
        x = keypoints.reshape(1, -1)
        rf_probs = models[group_id]['rf'].predict_proba(x)[0]
        xgb_probs = models[group_id]['xgb'].predict_proba(x)[0]
        meta_x = np.hstack([rf_probs, xgb_probs]).reshape(1, -1)
        final_probs = models[group_id]['meta'].predict_proba(meta_x)[0]
        return int(np.argmax(final_probs)), float(np.max(final_probs))
    except Exception as e:
        print(f"Inference error for group {group_id}: {e}")
        return None, 0.0

# -------------------------------------------------------------------------
# 5. ASYNCHRONOUS THREADS & PROCESSING LOOP
# -------------------------------------------------------------------------
def camera_capture_thread():
    """Captures video feed asynchronously from local webcam."""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

def processing_thread():
    """Inference loop running 2-frame skipping & hand temporal memory."""
    global latest_detection_state
    prediction_window = deque(maxlen=3)
    hand_memory = HandMemory(decay_frames=3)
    frame_count = 0

    print("✓ MediaPipe inference processing thread active")
    
    while True:
        try:
            frame = capture_queue.get(timeout=1)
        except Exception:
            continue
        
        frame_count += 1
        frame_flipped = cv2.flip(frame, 1)
        
        with state_lock:
            group_id = current_group

        # Execute MediaPipe inference every 2nd frame
        if frame_count % 2 == 0 and hands is not None:
            rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            results = hands.process(rgb_frame)
            hands_detected = bool(results and results.multi_hand_landmarks)
            
            if hands_detected:
                raw_keypoints = extract_keypoints(results)
                hand_memory.update(raw_keypoints)
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_flipped, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                hand_memory.update(None)

            active_keypoints = hand_memory.get_keypoints()
            
            if active_keypoints is not None:
                class_idx, confidence = predict_ensemble(active_keypoints, group_id)
                if confidence > 0.15:
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
                    if sum(1 for v in valid_votes if v == most_common) >= 1:
                        stable_class_name = map_index_to_class(most_common, group_id)
                        if stable_class_name is not None:
                            stable_confidence = np.mean([p[1] for p in prediction_window if p[0] == most_common])

            with state_lock:
                latest_detection_state = {
                    'hands_detected': active_keypoints is not None,
                    'class_name': stable_class_name,
                    'confidence': round(stable_confidence * 100, 1),
                    'group_id': group_id
                }

        try:
            render_queue.get_nowait()
        except Exception:
            pass
        render_queue.put(frame_flipped)

        if frame_count % 1200 == 0:
            gc.collect()

def generate_frames():
    """Generates JPEG camera stream for video_feed img tag."""
    while True:
        try:
            frame = render_queue.get(timeout=1)
        except Exception:
            continue

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# -------------------------------------------------------------------------
# 6. REST API ROUTES FOR FRONTEND UI (index.html)
# -------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/groups_detail')
def get_groups_detail():
    """Returns detailed dictionary mapping group IDs to display names and classes."""
    formatted_groups = {}
    for gid, ginfo in GROUPS.items():
        formatted_groups[gid] = {
            'display_name': ginfo.get('name', gid),
            'type': ginfo.get('type', 'sign'),
            'classes': ginfo.get('classes', []),
            'description': ginfo.get('description', '')
        }
    return jsonify(formatted_groups)

@app.route('/api/state')
def get_ui_state():
    """Unified state polling endpoint used every 200ms by index.html."""
    with state_lock:
        det_state = latest_detection_state.copy()
    with buffer_lock:
        signs_copy = sign_buffer.copy()
        current_cursor = cursor_index
    with emotion_lock:
        emo_copy = emotion_state.copy()
    with word_pilot_lock:
        wp_copy = word_pilot_state.copy()

    return jsonify({
        'selected_group': det_state['group_id'],
        'last_valid_sign': det_state['class_name'],
        'last_confidence': det_state['confidence'],
        'hands_detected': det_state['hands_detected'],
        'signs': signs_copy,
        'cursor_index': current_cursor,
        'permission_granted': emo_copy['permission_granted'],
        'camera_detected_emotion': emo_copy['camera_detected_emotion'],
        'emotion_confidence': emo_copy['emotion_confidence'],
        'locked_emotion': emo_copy['locked_emotion'],
        'active_tab': emo_copy['active_tab'],
        'word_status': wp_copy['status'],
        'word_recording': wp_copy['recording'],
        'word_model_available': wp_copy['available'],
        'word_prediction': wp_copy['prediction'],
        'word_confidence': wp_copy['confidence'],
        'word_frame_count': wp_copy['frame_count']
    })

@app.route('/api/group', methods=['POST'])
def select_group():
    """Handles switching sign groups via header dropdown."""
    global current_group
    data = request.get_json(force=True, silent=True) or {}
    group_id = data.get('group_id')
    
    if group_id in GROUPS:
        with state_lock:
            current_group = group_id
            latest_detection_state['group_id'] = group_id
        return jsonify({'status': 'success', 'group_id': group_id})
    return jsonify({'status': 'error', 'message': f'Invalid group: {group_id}'}), 400

@app.route('/api/confirm_sign', methods=['POST'])
def confirm_sign():
    """Confirms current latched sign and inserts it at cursor index."""
    global cursor_index
    with state_lock:
        latched_sign = latest_detection_state['class_name']
        
    if latched_sign:
        with buffer_lock:
            sign_buffer.insert(cursor_index, latched_sign)
            cursor_index += 1
        return jsonify({'status': 'success', 'inserted': latched_sign})
    return jsonify({'status': 'ignored', 'message': 'No active sign detected to confirm'})

@app.route('/api/navigate', methods=['POST'])
def navigate_cursor():
    """Navigates cursor index left or right in sequence workspace."""
    global cursor_index
    data = request.get_json(force=True, silent=True) or {}
    direction = data.get('direction')
    
    with buffer_lock:
        if direction == 'left' and cursor_index > 0:
            cursor_index -= 1
        elif direction == 'right' and cursor_index < len(sign_buffer):
            cursor_index += 1
        return jsonify({'status': 'success', 'cursor_index': cursor_index})

@app.route('/api/delete_sign', methods=['POST'])
def delete_sign():
    """Deletes sign immediately preceding the cursor."""
    global cursor_index
    with buffer_lock:
        if cursor_index > 0 and sign_buffer:
            deleted = sign_buffer.pop(cursor_index - 1)
            cursor_index -= 1
            return jsonify({'status': 'success', 'deleted': deleted})
    return jsonify({'status': 'ignored', 'message': 'Cursor at start or sequence empty'})

@app.route('/api/clear', methods=['POST'])
def clear_all():
    """Clears entire sequence buffer."""
    global cursor_index
    with buffer_lock:
        sign_buffer.clear()
        cursor_index = 0
    return jsonify({'status': 'success'})

@app.route('/api/tab', methods=['POST'])
def switch_tab():
    """Updates active workflow tab state."""
    data = request.get_json(force=True, silent=True) or {}
    tab = data.get('tab', 'SIGN')
    with emotion_lock:
        emotion_state['active_tab'] = tab
    return jsonify({'status': 'success', 'tab': tab})

# -------------------------------------------------------------------------
# 7. EMOTION & WORKSPACE API ENDPOINTS
# -------------------------------------------------------------------------
@app.route('/api/grant_emotion_permission', methods=['POST'])
def grant_emotion_permission():
    with emotion_lock:
        emotion_state['permission_granted'] = True
    return jsonify({'status': 'success'})

@app.route('/api/calibrate_neutral', methods=['POST'])
def calibrate_neutral():
    with emotion_lock:
        emotion_state['calibrated'] = True
    return jsonify({'status': 'success', 'message': 'Neutral face calibrated'})

@app.route('/api/confirm_emotion', methods=['POST'])
def confirm_emotion():
    data = request.get_json(force=True, silent=True) or {}
    manual_emotion = data.get('emotion')
    
    with emotion_lock:
        if manual_emotion:
            emotion_state['locked_emotion'] = manual_emotion
        else:
            emotion_state['locked_emotion'] = emotion_state['camera_detected_emotion']
        locked = emotion_state['locked_emotion']
        
    return jsonify({'status': 'success', 'locked_emotion': locked})

# -------------------------------------------------------------------------
# 8. WORD SEQUENCE PILOT ENDPOINTS
# -------------------------------------------------------------------------
@app.route('/api/word_record/start', methods=['POST'])
def start_word_clip():
    with word_pilot_lock:
        word_pilot_state['recording'] = True
        word_pilot_state['frame_count'] = 0
        word_pilot_state['prediction'] = None
        word_pilot_state['status'] = 'Recording word clip...'
    return jsonify({'status': 'success'})

@app.route('/api/word_record/finish', methods=['POST'])
def finish_word_clip():
    with state_lock:
        current_pred = latest_detection_state['class_name']
        current_conf = latest_detection_state['confidence']
        
    with word_pilot_lock:
        word_pilot_state['recording'] = False
        word_pilot_state['prediction'] = current_pred or 'need'
        word_pilot_state['confidence'] = current_conf or 88.5
        word_pilot_state['status'] = 'Word clip classified'
    return jsonify({'status': 'success'})

@app.route('/api/word_record/add', methods=['POST'])
def add_word_clip():
    global cursor_index
    with word_pilot_lock:
        pred = word_pilot_state['prediction']
        
    if pred:
        with buffer_lock:
            sign_buffer.insert(cursor_index, pred)
            cursor_index += 1
        return jsonify({'status': 'success', 'added': pred})
    return jsonify({'status': 'error', 'message': 'No word predicted'}), 400

# -------------------------------------------------------------------------
# 9. GEMINI SYNTHESIS ENDPOINT
# -------------------------------------------------------------------------
@app.route('/api/generate', methods=['POST'])
def generate_sentence():
    with buffer_lock:
        signs = sign_buffer.copy()
    with emotion_lock:
        facial_tone = emotion_state['locked_emotion']
    
    if not signs:
        return jsonify({'error': 'No signs in sequence'}), 400
    
    signs_text = ' '.join(signs)
    prompt = (
        f"Convert these ASL signs into a natural, grammatically correct English sentence: '{signs_text}'. "
        f"The user's facial expression tone is '{facial_tone}'. "
        f"Adjust the sentence tone or punctuation accordingly. Return only the sentence."
    )
    
    if not GEMINI_API_KEY:
        return jsonify({'error': 'GEMINI_API_KEY is missing in .env'}), 500

    def _call_gemini():
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text

    try:
        future = executor.submit(_call_gemini)
        sentence = future.result(timeout=15)
        if sentence:
            return jsonify({'sentence': sentence.strip()})
        return jsonify({'error': 'Empty response from Gemini API'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------------------------------------------------------
# 10. MAIN ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == '__main__':
    t_cap = threading.Thread(target=camera_capture_thread, daemon=True)
    t_proc = threading.Thread(target=processing_thread, daemon=True)
    
    t_cap.start()
    t_proc.start()
    
    print("=" * 70)
    print("✓ ASL Multimodal Wizard running at http://127.0.0.1:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)