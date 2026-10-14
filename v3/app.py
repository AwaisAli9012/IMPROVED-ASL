"""
ASL MULTIMODAL WIZARD - HIGH-PERFORMANCE ENGINE
========================================================================
Synchronized with Config.py, extract_multimodal_3.py (126-dim pure geometry),
and candidate models saved from train.py.
Spatial features, centroids, and presence flags stripped to eliminate data leakage.
"""

import os
import gc
import math
import joblib
import json
import time
import threading
import sys
import warnings
from queue import Queue
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))

try:
    import Config
except ImportError:
    import config as Config



def _build_groups(raw):
    """Effective groups = the real alphabet/word groups from Config, plus ONE auto-scan mode per section.
    Auto-scan is not a group of its own: it predicts from ALL groups of its section (letters or words).
    Any combined 'ALL' group, or other mixed group, is dropped so words and letters never compete."""
    groups = {gid: dict(g) for gid, g in raw.items()
              if g.get('type') in ('alphabet', 'sign') and gid != 'ALL'}
    for section, gid, name in (('alphabet', 'ALPHA_ALL', 'Alphabet Auto-Scan'),
                               ('sign', 'SIGN_ALL', 'Word Auto-Scan')):
        classes = []
        for g in groups.values():
            if g['type'] == section:
                for c in g.get('classes', []):
                    if c not in classes:
                        classes.append(c)
        groups[gid] = {'name': name, 'type': section + '_auto', 'classes': classes,
                       'description': f"Predicts from all {'letter' if section == 'alphabet' else 'word'} groups"}
    return groups


_RAW_GROUPS = getattr(Config, 'GROUPS', {})
RAW_GROUP_IDS = list(_RAW_GROUPS.keys())
GROUPS = _build_groups(_RAW_GROUPS)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")   # your current choice; override with GEMINI_MODEL=...
# FLIP_FIRST=1 flips the frame BEFORE MediaPipe (selfie view); default 0 matches the unflipped training photos
FLIP_BEFORE_MEDIAPIPE = os.getenv("FLIP_FIRST", "0") == "1"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)

@app.after_request
def no_cache(resp):
    # always serve the freshest page and state, never a cached copy
    resp.headers['Cache-Control'] = 'no-store'
    return resp

print("=" * 70)
print("ASL MULTIMODAL WIZARD - ENGINE STARTING (126-DIM PURE GEOMETRY)")
print("=" * 70)
print(f"Config file in use: {getattr(Config, '__file__', '?')}")
print(f"Group IDs in that file: {RAW_GROUP_IDS}")

class HandMemory:
    def __init__(self, decay_frames=15):
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

current_group = 'ALPHA_ALL'      # starts in Alphabet Auto-Scan; press W for Word Auto-Scan

candidate_model = {}
scaler = None
idx_to_class = {}
class_to_idx = {}

latest_detection_state = {
    'hands_detected': False,
    'class_name': None,
    'confidence': 0.0,
    'group_id': current_group,
    'timer_progress': 0.0
}
state_lock = threading.Lock()

sign_buffer = []
cursor_index = 0
buffer_lock = threading.Lock()

# Auto-confirm: a sign held steadily for AUTO_CONFIRM_HOLD_DURATION seconds is added automatically.
# 'fired' stops the same held sign from being added again until it changes or the hand leaves.
AUTO_CONFIRM_HOLD_DURATION = 1.5
auto_state = {'sign': None, 'start': None, 'fired': False}
timer_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Facial tone: on-demand emotion detection (face_alignment 68 landmarks + trained classifier)
# ---------------------------------------------------------------------------
emotion_state = {
    'permission_granted': False,
    'calibrated': False,
    'camera_detected_emotion': 'neutral',
    'emotion_confidence': 0,
    'locked_emotion': 'neutral',
    'active_tab': 'SIGN',
    'emotion_status': 'idle',        # idle | loading | capturing | done | no_face | error
    'emotion_message': '',
    'emotion_probs': {},
    'emotion_signals': {},           # measured face changes, for the on-screen readout
    'emotion_kind': '',              # 'calibrate' or 'detect' while a job runs
    'emotion_progress': '',          # e.g. '2/3' frames analysed
    'emotion_personalised': False,   # True when the last detection used the person's own neutral
}
emotion_lock = threading.Lock()
emotion_job = {'busy': False}
face_scan = {'active': False, 'start': 0.0, 'end': 0.0, 'ok': None}   # drives the on-screen face scan
neutral_ref = None                   # this signer's own neutral-face features (set by Calibrate Neutral)
latest_raw_frame = None              # newest unflipped camera frame, grabbed for emotion analysis
frame_lock = threading.Lock()
EMOTION_FRAMES = int(os.getenv("EMOTION_FRAMES", "3"))                    # frames averaged per detection
USE_CALIBRATION = os.getenv("EMOTION_USE_CALIBRATION", "1") == "1"        # shift the user's neutral onto the dataset's

emotion_engine = None
try:
    from emotion_engine import EmotionEngine, locate_model_dir
    _cands = [os.getenv("EMOTION_MODELS_DIR"), BASE_DIR / "Emotion_Models", BASE_DIR.parent / "Emotion_Models"]
    try:
        import emotion_config
        _cands.insert(1, emotion_config.EMOTION_MODELS_DIR)
    except Exception:
        pass
    _edir = locate_model_dir(_cands)
    emotion_engine = EmotionEngine(_edir)
    if _edir:
        print(f"✓ Emotion model found in {_edir} (face model loads on first use)")
    else:
        print("⚠️ Emotion model not found (run train_emotion.py). Facial tone will be manual only.")
except Exception as e:
    print(f"⚠️ Emotion engine unavailable ({e}). Facial tone will be manual only.")

capture_queue = Queue(maxsize=1)
render_queue = Queue(maxsize=1)

candidate_dir = BASE_DIR / "Models_candidate" / "models"
scaler_path = candidate_dir / "candidate_scaler.pkl"
mapping_path = candidate_dir / "candidate_label_mapping.json"
rf_path = candidate_dir / "candidate_rf.pkl"
et_path = candidate_dir / "candidate_et.pkl"
xgb_path = candidate_dir / "candidate_xgb.pkl"
meta_path = candidate_dir / "candidate_meta.pkl"

if scaler_path.exists():
    try:
        scaler = joblib.load(scaler_path)
        print("✓ Scaler loaded successfully.")
    except Exception as e:
        print(f"Failed scaler loading: {e}")

if mapping_path.exists():
    try:
        with open(mapping_path, "r") as f:
            mapping_data = json.load(f)
            idx_to_class = {int(k): v for k, v in mapping_data.get("idx_to_class", {}).items()}
            class_to_idx = {v: int(k) for k, v in idx_to_class.items()}
        print(f"✓ Loaded label mapping for {len(idx_to_class)} classes.")
    except Exception as e:
        print(f"Failed mapping loading: {e}")

try:
    if rf_path.exists() and et_path.exists() and xgb_path.exists() and meta_path.exists():
        candidate_model = {
            'rf': joblib.load(rf_path),
            'et': joblib.load(et_path),
            'xgb': joblib.load(xgb_path),
            'meta': joblib.load(meta_path)
        }
        print("✓ All 4 candidate ensemble models loaded successfully.")
    else:
        print("⚠️ Candidate model files missing from Models_candidate/models/")
except Exception as e:
    print(f"Model load error: {e}")

try:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.30,
        min_tracking_confidence=0.30
    )
except Exception as e:
    print(f"MediaPipe initialization error: {e}")
    hands = None

def extract_frame_features(results, img_w, img_h):
    if not results or not results.multi_hand_landmarks:
        return np.zeros(126, dtype=np.float32)

    left_hand_coords = np.zeros((21, 3), dtype=np.float32)
    right_hand_coords = np.zeros((21, 3), dtype=np.float32)
    left_present, right_present = False, False
    scale_factors = np.array([img_w, img_h, img_w], dtype=np.float32)

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness_label = "Right"
        if results.multi_handedness and idx < len(results.multi_handedness):
            handedness_label = results.multi_handedness[idx].classification[0].label

        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
        ) * scale_factors

        if handedness_label == "Left":
            left_hand_coords = coords
            left_present = True
        else:
            right_hand_coords = coords
            right_present = True

    def process_geometry(raw_coords, is_present):
        if not is_present:
            return np.zeros(63, dtype=np.float32)
        rel = raw_coords - raw_coords[0]
        span = float(np.linalg.norm(rel[9]))
        if span < 1e-6:
            span = 1e-6
        return (rel / span).flatten()

    left_norm = process_geometry(left_hand_coords, left_present)
    right_norm = process_geometry(right_hand_coords, right_present)
    return np.concatenate([left_norm, right_norm]).astype(np.float32)

def map_index_to_class(class_idx):
    if class_idx is not None and class_idx in idx_to_class:
        return idx_to_class[class_idx]
    return None

def predict_ensemble(keypoints, group_id='ALPHA1'):
    if not candidate_model or keypoints is None:
        return None, 0.0

    try:
        x = keypoints.reshape(1, -1).astype(np.float32)
        if scaler is not None:
            x = scaler.transform(x)

        rf_probs = candidate_model['rf'].predict_proba(x)
        et_probs = candidate_model['et'].predict_proba(x)
        xgb_probs = candidate_model['xgb'].predict_proba(x)

        meta_x = np.hstack([rf_probs, et_probs, xgb_probs])
        final_probs = candidate_model['meta'].predict_proba(meta_x)[0]

        _top = int(np.argmax(final_probs))
        _now = time.time()
        if _now - getattr(predict_ensemble, '_last_print', 0) > 0.5:
            predict_ensemble._last_print = _now
            print(f"RAW top: {idx_to_class.get(_top)} | {final_probs[_top]:.2f}")

        n_allowed = len(final_probs)   # how many classes compete in this group
        if group_id not in GROUPS:
            return None, 0.0           # unknown group: never guess across letters AND words
        allowed_indices = [class_to_idx[c] for c in GROUPS[group_id].get('classes', []) if c in class_to_idx]
        if not allowed_indices:
            return None, 0.0           # none of this group's classes exist in the trained model
        n_allowed = len(allowed_indices)
        masked_probs = np.zeros_like(final_probs)
        masked_probs[allowed_indices] = final_probs[allowed_indices]
        prob_sum = np.sum(masked_probs)
        if prob_sum > 0:
            masked_probs /= prob_sum
        final_probs = masked_probs

        best_idx = int(np.argmax(final_probs))
        best_conf = float(final_probs[best_idx])

        min_conf = 0.05
        if best_conf < min_conf:
            _now2 = time.time()
            if _now2 - getattr(predict_ensemble, '_last_rej', 0) > 0.5:
                predict_ensemble._last_rej = _now2
                print(f"REJECT {group_id}: {idx_to_class.get(best_idx)} {best_conf:.2f} < {min_conf:.2f} (n={n_allowed})")
            return None, 0.0

        return best_idx, best_conf
    except Exception as e:
        print(f"Inference error: {e}")
        return None, 0.0

def camera_capture_thread():
    global latest_raw_frame
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ CRITICAL ERROR: Could not open video device /dev/video0")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Not Available (/dev/video0)", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        while True:
            try:
                capture_queue.get_nowait()
            except Exception:
                pass
            capture_queue.put(error_frame)
            time.sleep(0.5)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    consecutive_failures = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_failures += 1
            if consecutive_failures % 50 == 0:
                print(f"⚠️ Warning: {consecutive_failures} consecutive frame read failures on /dev/video0")
            time.sleep(0.01)
            continue
            
        consecutive_failures = 0
        with frame_lock:
            latest_raw_frame = frame
        try:
            capture_queue.get_nowait()
        except Exception:
            pass
        capture_queue.put(frame)

infer_slot = {'frame': None, 'id': 0}
infer_lock = threading.Lock()
hand_view = {'results': None}
perf = {'frames': 0, 'infer': 0, 'mp_ms': 0.0, 'pred_ms': 0.0, 't0': time.time()}
PERF_LOG = os.getenv("ASL_PERF", "0") == "1"
INFER_FPS = float(os.getenv("ASL_INFER_FPS", "12"))

FACE_PARTS = [(range(0, 17), False), (range(17, 22), False), (range(22, 27), False), (range(27, 31), False),
              (range(31, 36), False), (range(36, 42), True), (range(42, 48), True),
              (range(48, 60), True), (range(60, 68), True)]
SCAN_PERIOD = 1.8


def draw_face_scan(frame):
    now = time.time()
    active = face_scan['active']
    ended = face_scan['end']
    if not active and not (ended and face_scan['ok'] and now - ended < 1.5):
        return
    h, w = frame.shape[:2]
    ov = emotion_engine.last_overlay if emotion_engine is not None else None
    fresh = ov is not None and ov['t'] >= face_scan['start'] - 0.05 and (now - ov['t']) < 6.0
    done = not active
    phase = ((now - face_scan['start']) % SCAN_PERIOD) / SCAN_PERIOD
    base = (80, 255, 120) if done else (255, 230, 0)
    dim = tuple(int(c * 0.6) for c in base)

    if not fresh:
        if done:
            return
        y = int(h * phase)
        band = frame.copy()
        cv2.rectangle(band, (0, max(0, y - 12)), (w, min(h, y + 12)), base, -1)
        cv2.addWeighted(band, 0.16, frame, 0.84, 0, frame)
        cv2.line(frame, (0, y), (w, y), base, 1, cv2.LINE_AA)
        return

    pts = np.asarray(ov['pts'], dtype=np.float32).copy()
    pts[:, 0] = (w - 1) - pts[:, 0] * (w / float(ov['w']))
    pts[:, 1] = pts[:, 1] * (h / float(ov['h']))
    brow_y = float(pts[17:27, 1].min())
    chin_y = float(pts[8, 1])
    face_h = max(chin_y - brow_y, 20.0)
    top_y = brow_y - 0.45 * face_h
    bottom_y = chin_y + 0.06 * face_h
    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    pad = 0.12 * face_h
    y_line = top_y + (bottom_y - top_y) * phase

    if not done:
        band = frame.copy()
        bh = 0.07 * face_h
        cv2.rectangle(band, (int(x0 - pad), int(y_line - bh)), (int(x1 + pad), int(y_line + bh)), base, -1)
        cv2.addWeighted(band, 0.20, frame, 0.80, 0, frame)

    ipts = pts.astype(np.int32)
    for idx, closed in FACE_PARTS:
        cv2.polylines(frame, [ipts[list(idx)].reshape(-1, 1, 2)], closed, dim, 1, cv2.LINE_AA)

    for i in range(68):
        glow = 0.0 if done else math.exp(-((pts[i, 1] - y_line) / (0.09 * face_h)) ** 2)
        col = tuple(int(dim[k] * (1.0 - glow) + 255 * glow) for k in range(3))
        cv2.circle(frame, (int(ipts[i, 0]), int(ipts[i, 1])), 2 + int(round(2 * glow)), col, -1, cv2.LINE_AA)

    if not done:
        cv2.line(frame, (int(x0 - pad), int(y_line)), (int(x1 + pad), int(y_line)), base, 2, cv2.LINE_AA)

    bx0, bx1, by0, by1 = int(x0 - pad), int(x1 + pad), int(top_y), int(bottom_y)
    arm = int(0.14 * face_h)
    for (cx, cy, dx, dy) in ((bx0, by0, 1, 1), (bx1, by0, -1, 1), (bx0, by1, 1, -1), (bx1, by1, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * arm, cy), base, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * arm), base, 2, cv2.LINE_AA)


def processing_thread():
    frame_count = 0
    while True:
        try:
            frame = capture_queue.get(timeout=1)
        except Exception:
            continue

        frame_count += 1
        perf['frames'] += 1
        with infer_lock:
            infer_slot['frame'] = frame
            infer_slot['id'] += 1

        frame_display = cv2.flip(frame, 1) if FLIP_BEFORE_MEDIAPIPE else frame.copy()
        try:
            res = hand_view['results']
            if res is not None and res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if not FLIP_BEFORE_MEDIAPIPE:
                frame_display = cv2.flip(frame_display, 1)

            draw_face_scan(frame_display)
        except Exception as e:
            print(f"Frame drawing error: {e}")
        finally:
            try:
                render_queue.get_nowait()
            except Exception:
                pass
            render_queue.put(frame_display)

        if frame_count % 1200 == 0:
            gc.collect()


def inference_thread():
    global latest_detection_state, cursor_index
    prediction_window = deque(maxlen=3)
    hand_memory = HandMemory(decay_frames=15)
    last_id = 0
    last_run = 0.0
    min_dt = 1.0 / max(INFER_FPS, 1.0)

    while True:
        with emotion_lock:
            paused = emotion_state['emotion_status'] in ('loading', 'capturing')
        if paused:
            hand_view['results'] = None
            prediction_window.clear()
            hand_memory = HandMemory(decay_frames=15)
            with timer_lock:
                auto_state.update(sign=None, start=None, fired=False)
            with state_lock:
                latest_detection_state = {'hands_detected': False, 'class_name': None, 'confidence': 0.0,
                                          'group_id': current_group, 'timer_progress': 0.0}
            time.sleep(0.1)
            continue

        wait = min_dt - (time.time() - last_run)
        if wait > 0:
            time.sleep(wait)
        with infer_lock:
            frame = infer_slot['frame']
            fid = infer_slot['id']
        if frame is None or fid == last_id:
            time.sleep(0.005)
            continue
        last_id = fid
        last_run = time.time()

        img_h, img_w = frame.shape[:2]
        with state_lock:
            group_id = current_group

        try:
            if hands is not None:
                t0 = time.time()
                input_frame = cv2.flip(frame, 1) if FLIP_BEFORE_MEDIAPIPE else frame
                rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False

                results = hands.process(rgb_frame)
                hand_view['results'] = results
                t1 = time.time()
                hands_detected = bool(results and results.multi_hand_landmarks)

                if hands_detected:
                    raw_keypoints = extract_frame_features(results, img_w, img_h)
                    hand_memory.update(raw_keypoints)
                else:
                    hand_memory.update(None)

                active_keypoints = hand_memory.get_keypoints()

                if active_keypoints is not None:
                    class_idx, confidence = predict_ensemble(active_keypoints, group_id=group_id)
                    if class_idx is not None:
                        prediction_window.append((class_idx, confidence))
                    else:
                        prediction_window.append((None, 0.0))
                else:
                    prediction_window.clear()
                t2 = time.time()

                stable_class_name = None
                stable_confidence = 0.0

                if len(prediction_window) > 0:
                    valid_votes = [p[0] for p in prediction_window if p[0] is not None]
                    if valid_votes:
                        most_common = max(set(valid_votes), key=valid_votes.count)
                        stable_class_name = map_index_to_class(most_common)
                        if stable_class_name is not None:
                            stable_confidence = np.mean([p[1] for p in prediction_window if p[0] == most_common])

                now = time.time()
                timer_progress = 0.0
                with timer_lock:
                    if stable_class_name:
                        if auto_state['sign'] != stable_class_name:
                            auto_state.update(sign=stable_class_name, start=now, fired=False)
                        elif not auto_state['fired']:
                            elapsed = now - auto_state['start']
                            timer_progress = min(1.0, elapsed / AUTO_CONFIRM_HOLD_DURATION)
                            if elapsed >= AUTO_CONFIRM_HOLD_DURATION:
                                with buffer_lock:
                                    sign_buffer.insert(cursor_index, stable_class_name)
                                    cursor_index += 1
                                auto_state['fired'] = True
                                timer_progress = 0.0
                    else:
                        auto_state.update(sign=None, start=None, fired=False)

                with state_lock:
                    latest_detection_state = {
                        'hands_detected': active_keypoints is not None,
                        'class_name': stable_class_name,
                        'confidence': round(stable_confidence * 100, 1),
                        'group_id': group_id,
                        'timer_progress': round(timer_progress * 100, 1)
                    }

                perf['infer'] += 1
                perf['mp_ms'] += (t1 - t0) * 1000
                perf['pred_ms'] += (t2 - t1) * 1000
        except Exception as e:
            print(f"Frame processing error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------------------------------------------------------------
# UPDATED: Client-side browser webcam prediction endpoint with landmarks & auto-add timer support
# ---------------------------------------------------------------------------
@app.route('/api/predict_frame', methods=['POST'])
def predict_client_frame():
    global cursor_index
    file = request.files.get('frame')
    if not file:
        return jsonify({'error': 'No frame provided'}), 400

    np_img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    with frame_lock:
        global latest_raw_frame
        latest_raw_frame = frame

    img_h, img_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with state_lock:
        group_id = current_group

    landmarks_data = []
    try:
        if hands is not None:
            results = hands.process(rgb_frame)
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_pts = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
                    landmarks_data.append(hand_pts)

                raw_keypoints = extract_frame_features(results, img_w, img_h)
                class_idx, confidence = predict_ensemble(raw_keypoints, group_id=group_id)
                sign_name = map_index_to_class(class_idx)
                
                if sign_name:
                    with state_lock:
                        latest_detection_state.update({
                            'hands_detected': True,
                            'class_name': sign_name,
                            'confidence': round(confidence * 100, 1),
                            'group_id': group_id
                        })
                    
                    now = time.time()
                    timer_progress = 0.0
                    with timer_lock:
                        if auto_state['sign'] != sign_name:
                            auto_state.update(sign=sign_name, start=now, fired=False)
                        elif not auto_state['fired']:
                            elapsed = now - auto_state['start']
                            timer_progress = min(1.0, elapsed / AUTO_CONFIRM_HOLD_DURATION)
                            with state_lock:
                                latest_detection_state['timer_progress'] = round(timer_progress * 100, 1)
                            if elapsed >= AUTO_CONFIRM_HOLD_DURATION:
                                with buffer_lock:
                                    sign_buffer.insert(cursor_index, sign_name)
                                    cursor_index += 1
                                auto_state['fired'] = True
                                with state_lock:
                                    latest_detection_state['timer_progress'] = 0.0

                    return jsonify({
                        'hands_detected': True,
                        'class_name': sign_name,
                        'confidence': round(confidence * 100, 1),
                        'landmarks': landmarks_data
                    })
    except Exception as e:
        print(f"Client frame inference error: {e}")

    with state_lock:
        latest_detection_state.update({
            'hands_detected': False,
            'class_name': None,
            'confidence': 0.0,
            'group_id': group_id,
            'timer_progress': 0.0
        })
    with timer_lock:
        auto_state.update(sign=None, start=None, fired=False)

    return jsonify({'hands_detected': False, 'class_name': None, 'confidence': 0.0, 'landmarks': []})

def generate_frames():
    while True:
        try:
            frame = render_queue.get(timeout=1)
        except Exception:
            continue

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/api/groups_detail')
def get_groups_detail():
    formatted_groups = {}
    for gid, ginfo in GROUPS.items():
        classes = ginfo.get('classes', [])
        if class_to_idx:
            classes = [c for c in classes if c in class_to_idx]
        gtype = ginfo.get('type', 'sign')
        formatted_groups[gid] = {
            'display_name': ginfo.get('name', gid),
            'type': gtype,
            'section': 'alphabet' if gtype.startswith('alphabet') else 'word',
            'is_auto': gtype.endswith('_auto'),
            'classes': classes,
            'description': ginfo.get('description', '')
        }
    return jsonify(formatted_groups)

@app.route('/api/state')
def get_ui_state():
    with state_lock:
        det_state = latest_detection_state.copy()
    with buffer_lock:
        signs_copy = sign_buffer.copy()
        current_cursor = cursor_index

    with emotion_lock:
        emo = emotion_state.copy()

    return jsonify({
        'selected_group': det_state['group_id'],
        'last_valid_sign': det_state['class_name'],
        'last_confidence': det_state['confidence'],
        'hands_detected': det_state['hands_detected'],
        'timer_progress': det_state.get('timer_progress', 0.0),
        'signs': signs_copy,
        'permission_granted': emo['permission_granted'],
        'calibrated': emo['calibrated'],
        'camera_detected_emotion': emo['camera_detected_emotion'],
        'emotion_confidence': emo['emotion_confidence'],
        'locked_emotion': emo['locked_emotion'],
        'active_tab': emo['active_tab'],
        'emotion_status': emo['emotion_status'],
        'emotion_message': emo['emotion_message'],
        'emotion_probs': emo['emotion_probs'],
        'emotion_signals': emo['emotion_signals'],
        'emotion_progress': emo['emotion_progress'],
        'emotion_personalised': emo['emotion_personalised'],
        'emotion_available': bool(emotion_engine is not None and emotion_engine.available),
        'cursor_index': current_cursor
    })

@app.route('/api/group', methods=['POST'])
def select_group():
    global current_group
    data = request.get_json(force=True, silent=True) or {}
    group_id = data.get('group_id')
    groups_dict = GROUPS
    
    if group_id in groups_dict:
        print(f"-> group switched to {group_id}")
        with state_lock:
            current_group = group_id
            latest_detection_state['group_id'] = group_id
        return jsonify({'status': 'success', 'group_id': group_id})
    return jsonify({'status': 'error', 'message': f'Invalid group: {group_id}'}), 400

@app.route('/api/confirm_sign', methods=['POST'])
def confirm_sign():
    global cursor_index
    with state_lock:
        latched_sign = latest_detection_state['class_name']
    if latched_sign:
        with buffer_lock:
            sign_buffer.insert(cursor_index, latched_sign)
            cursor_index += 1
        with timer_lock:
            auto_state.update(sign=latched_sign, start=time.time(), fired=True)
        return jsonify({'status': 'success', 'inserted': latched_sign})
    return jsonify({'status': 'ignored', 'message': 'No active sign detected'})

@app.route('/api/add_space', methods=['POST'])
def add_space():
    global cursor_index
    with buffer_lock:
        sign_buffer.insert(cursor_index, "[SPACE]")
        cursor_index += 1
    return jsonify({'status': 'success', 'cursor_index': cursor_index})

@app.route('/api/navigate', methods=['POST'])
def navigate_cursor():
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
    global cursor_index
    with buffer_lock:
        if cursor_index > 0 and sign_buffer:
            deleted = sign_buffer.pop(cursor_index - 1)
            cursor_index -= 1
            return jsonify({'status': 'success', 'deleted': deleted})
    return jsonify({'status': 'ignored', 'message': 'Cursor at start or sequence empty'})

@app.route('/api/clear', methods=['POST'])
def clear_all():
    global cursor_index
    with buffer_lock:
        sign_buffer.clear()
        cursor_index = 0
    return jsonify({'status': 'success'})

@app.route('/api/tab', methods=['POST'])
def switch_tab():
    data = request.get_json(force=True, silent=True) or {}
    tab = data.get('tab', 'SIGN')
    with emotion_lock:
        emotion_state['active_tab'] = tab
    return jsonify({'status': 'success', 'tab': tab})

def _set_emotion(**kw):
    with emotion_lock:
        emotion_state.update(kw)

def _latest_frame_copy():
    with frame_lock:
        return None if latest_raw_frame is None else latest_raw_frame.copy()

def _emotion_worker(kind):
    global neutral_ref
    label = 'Calibrating' if kind == 'calibrate' else 'Detecting tone'
    face_found = False
    face_scan.update(active=True, start=time.time(), end=0.0, ok=None)
    try:
        if not emotion_engine.loaded:
            _set_emotion(emotion_status='loading', emotion_kind=kind, emotion_progress='',
                         emotion_message='Loading the face model...')
            emotion_engine.load()

        n = EMOTION_FRAMES
        feats = []
        for i in range(n):
            _set_emotion(emotion_status='capturing', emotion_kind=kind, emotion_progress=f'{i + 1}/{n}',
                         emotion_message=f'{label} - hold still ({i + 1}/{n})...')
            frame = _latest_frame_copy()
            if frame is None:
                raise RuntimeError('no camera frames available')
            f = emotion_engine.frame_features(frame)
            if f is not None:
                feats.append(f)
        face_found = bool(feats)

        if kind == 'calibrate':
            ref, hint = emotion_engine.calibrate_features(feats)
            if ref is None:
                _set_emotion(emotion_status='no_face',
                             emotion_message='No face found. Face the camera in good light and try again.')
            else:
                with emotion_lock:
                    neutral_ref = ref
                    emotion_state.update(
                        calibrated=True, emotion_status='done', emotion_progress='',
                        emotion_message=hint or 'Neutral face saved.')
        else:
            with emotion_lock:
                ref = neutral_ref if USE_CALIBRATION else None
            res = emotion_engine.analyze_features(feats, ref)
            if not res['ok']:
                _set_emotion(emotion_status='no_face',
                             emotion_message='No face found. Face the camera in good light and try again.')
            else:
                conf = round(res['confidence'] * 100, 1)
                _set_emotion(
                    camera_detected_emotion=res['emotion'],
                    emotion_confidence=conf,
                    emotion_probs={k: round(v * 100, 1) for k, v in res['probs'].items()},
                    emotion_signals={k: round(v, 3) for k, v in res['signals'].items()},
                    emotion_personalised=res['personalised'],
                    locked_emotion=res['emotion'],
                    emotion_status='done', emotion_progress='',
                    emotion_message=f"Locked: {res['emotion']} ({conf}%)")
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        _set_emotion(emotion_status='error', emotion_message=f'Emotion analysis failed: {e}')
    finally:
        face_scan.update(active=False, end=time.time(), ok=face_found)
        with emotion_lock:
            emotion_job['busy'] = False

def _start_emotion_job(kind):
    if emotion_engine is None or not emotion_engine.available:
        return jsonify({'status': 'error', 'message': 'Emotion model not found.'}), 503
    with emotion_lock:
        if not emotion_state['permission_granted']:
            return jsonify({'status': 'error', 'message': 'Enable facial tone first.'}), 403
        if emotion_job['busy']:
            return jsonify({'status': 'busy', 'message': 'Still working...'}), 409
        emotion_job['busy'] = True
    threading.Thread(target=_emotion_worker, args=(kind,), daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/api/grant_emotion_permission', methods=['POST'])
def grant_emotion_permission():
    with emotion_lock:
        emotion_state['permission_granted'] = True
    return jsonify({'status': 'success'})

@app.route('/api/calibrate_neutral', methods=['POST'])
def calibrate_neutral():
    return _start_emotion_job('calibrate')

@app.route('/api/reset_calibration', methods=['POST'])
def reset_calibration():
    global neutral_ref
    with emotion_lock:
        neutral_ref = None
        emotion_state.update(calibrated=False, emotion_message='Calibration cleared.')
    return jsonify({'status': 'success'})

@app.route('/api/confirm_emotion', methods=['POST'])
def confirm_emotion():
    data = request.get_json(force=True, silent=True) or {}
    manual = (data.get('emotion') or '').strip().lower()
    if manual:
        _set_emotion(locked_emotion=manual, emotion_status='done', emotion_message=f'Locked manually: {manual}.')
        return jsonify({'status': 'success', 'locked_emotion': manual})
    return _start_emotion_job('detect')

TONE_STYLE = {
    'neutral': "Use plain, matter-of-fact wording and normal punctuation.",
    'happy': "Make it warm, cheerful and enthusiastic.",
    'sad': "Make it subdued and sorrowful.",
    'angry': "Make it firm, frustrated and forceful.",
}

GEMINI_TIMEOUT_S = float(os.getenv("GEMINI_TIMEOUT", "10"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
GEMINI_RETRIES = int(os.getenv("GEMINI_RETRIES", "2"))
GEMINI_RETRY_WAIT_S = float(os.getenv("GEMINI_RETRY_WAIT", "1.0"))
RETRYABLE_CODES = {429, 500, 502, 503, 504}
_gemini_client = None
_gemini_lock = threading.Lock()

ASL_RULES = (
    "The input is a list of ASL glosses, NOT English. '[SPACE]' markers represent word boundaries. "
    "Turn them into ONE correct, natural English sentence that uses every sign."
)
ASL_EXAMPLES = "Examples:\nI go school -> I go to school."

def _http_options(types):
    ms = int((GEMINI_TIMEOUT_S + 2) * 1000)
    try:
        import httpx
        return types.HttpOptions(timeout=ms, client_args={
            'limits': httpx.Limits(max_keepalive_connections=2, keepalive_expiry=120)})
    except Exception:
        return types.HttpOptions(timeout=ms)

def _attempt(model, prompt):
    global _gemini_client
    from google.genai import types
    with _gemini_lock:
        if _gemini_client is None:
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY, http_options=_http_options(types))
        client = _gemini_client
    response = client.models.generate_content(model=model, contents=prompt, config=types.GenerateContentConfig(max_output_tokens=512))
    return (response.text or '').strip()

@app.route('/api/generate', methods=['POST'])
def generate_sentence():
    with buffer_lock:
        signs = sign_buffer.copy()
    if not signs:
        return jsonify({'error': 'No signs in sequence to synthesize.'}), 400

    raw_signs = ' '.join(signs)
    if not GEMINI_API_KEY:
        return jsonify({'sentence': " ".join(signs).capitalize(), 'fallback': True})

    with emotion_lock:
        tone = emotion_state['locked_emotion']
    style = TONE_STYLE.get(tone, TONE_STYLE['neutral'])
    
    prompt = f"{ASL_RULES}\n{ASL_EXAMPLES}\n\nSigns: {raw_signs}\nTone: {tone}. {style}\nReply with only the sentence."

    try:
        sentence = _attempt(GEMINI_MODEL, prompt)
        return jsonify({'sentence': sentence})
    except Exception as e:
        return jsonify({'sentence': " ".join(signs).capitalize(), 'fallback': True})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)