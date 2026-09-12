"""
IMPROVED ASL - Flask Web App with MediaPipe Hand Detection
===========================================================
- Proper MediaPipe 21-joint hand landmarks (126 dims total)
- Threading: Capture → Processing → Render (zero lag)
- Temporal smoothing (5-frame rolling window)
- Flask routes for group switching & search
- Real-time streaming video feed
"""

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
from pathlib import Path
import threading
from queue import Queue
from collections import deque
import mediapipe as mp

from Config import GROUPS, MODELS_DIR, APP_CONFIG

# ═══════════════════════════════════════════════════════════════════════════
# FLASK SETUP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

print("=" * 70)
print("IMPROVED ASL - FLASK WEB APP WITH MEDIAPIPE")
print("=" * 70)

# Load models
print("\nLoading 39 models...")
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
        print(f"  ❌ {group_id}")

print(f"✓ Loaded {len(models)} groups")

# ═══════════════════════════════════════════════════════════════════════════
# MEDIAPIPE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    print("✓ MediaPipe Hands initialized")
except Exception as e:
    print(f"❌ MediaPipe initialization failed: {e}")
    hands = None

# ═══════════════════════════════════════════════════════════════════════════
# THREADING ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

capture_queue = Queue(maxsize=2)
render_queue = Queue(maxsize=1)

current_group = 'SIGN1'
state_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════
# THREAD 1: CAMERA CAPTURE
# ═══════════════════════════════════════════════════════════════════════════

def camera_capture_thread():
    """Capture frames continuously"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✓ Camera thread started")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        try:
            capture_queue.get_nowait()
        except:
            pass
        
        capture_queue.put(frame)
    
    cap.release()

# ═══════════════════════════════════════════════════════════════════════════
# THREAD 2: PROCESSING (MediaPipe + Prediction)
# ═══════════════════════════════════════════════════════════════════════════

def processing_thread():
    """Process hands with MediaPipe & predict"""
    
    # Temporal smoothing
    prediction_window = deque(maxlen=5)
    
    print("✓ Processing thread started")
    
    while True:
        try:
            frame = capture_queue.get(timeout=1)
        except:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe hand detection
        keypoints = None
        hands_detected = False
        
        try:
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hands_detected = True
                keypoints = []
                
                # Process up to 2 hands
                for hand_landmarks in results.multi_hand_landmarks[:2]:
                    # 21 joints × 3 coords = 63 per hand
                    for landmark in hand_landmarks.landmark:
                        keypoints.extend([landmark.x, landmark.y, landmark.z])
                
                # Pad to 126 (2 hands × 63)
                while len(keypoints) < 126:
                    keypoints.append(0.0)
                
                keypoints = np.array(keypoints[:126], dtype=np.float32)
        except Exception as e:
            print(f"MediaPipe error: {e}")
        
        # Get current group
        with state_lock:
            group_id = current_group
        
        # Predict
        class_idx = None
        confidence = 0
        
        if hands_detected and keypoints is not None:
            class_idx, confidence = predict_ensemble(keypoints, group_id)
        
        # Temporal smoothing: 5-frame majority voting
        prediction_window.append((class_idx, confidence))
        
        stable_class_idx = None
        stable_confidence = 0
        
        if len(prediction_window) == 5:
            class_votes = [p[0] for p in prediction_window if p[0] is not None]
            if class_votes:
                most_common = max(set(class_votes), key=class_votes.count)
                if sum(1 for v in class_votes if v == most_common) >= 3:  # 3/5 votes
                    stable_class_idx = most_common
                    stable_confidence = np.mean([p[1] for p in prediction_window if p[0] == stable_class_idx])
        
        class_name = None
        if stable_class_idx is not None and stable_confidence > 0.65:
            class_name = get_class_name(group_id, stable_class_idx)
        
        # Put result (no .copy() to avoid memory leak)
        try:
            render_queue.get_nowait()
        except:
            pass
        
        render_queue.put({
            'frame': frame,
            'hands_detected': hands_detected,
            'class_name': class_name,
            'confidence': stable_confidence
        })

# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_ensemble(keypoints, group_id):
    """Predict using ensemble"""
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
    except:
        return None, None

def get_class_name(group_id, class_idx):
    """Get class name from index"""
    classes = GROUPS[group_id]['classes']
    return classes[class_idx] if class_idx < len(classes) else "Unknown"

def search_class(query):
    """Search for class by name"""
    query_lower = query.lower()
    for group_id, group_info in GROUPS.items():
        for class_name in group_info['classes']:
            if class_name.lower() == query_lower:
                return group_id
    return None

# ═══════════════════════════════════════════════════════════════════════════
# VIDEO STREAMING
# ═══════════════════════════════════════════════════════════════════════════

def generate_frames():
    """Generate video frames with UI overlay"""
    while True:
        try:
            result = render_queue.get(timeout=1)
        except:
            continue
        
        frame = result['frame']
        class_name = result['class_name']
        confidence = result['confidence']
        hands_detected = result['hands_detected']
        
        with state_lock:
            group = current_group
        
        # Draw UI
        frame = draw_ui(frame, group, class_name, confidence, hands_detected)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + frame_bytes + b'\r\n')

def draw_ui(frame, group_id, class_name, confidence, hands_detected):
    """Draw UI on frame"""
    h, w = frame.shape[:2]
    
    # Left panel
    cv2.rectangle(frame, (0, 0), (280, h), (20, 20, 40), -1)
    cv2.putText(frame, f"GROUP: {group_id}", (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
    
    y = 90
    cv2.putText(frame, "AVAILABLE:", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    y += 25
    
    for cls in GROUPS[group_id]['classes']:
        cv2.putText(frame, f"• {cls}", (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        y += 20
    
    # Center prediction
    if hands_detected:
        if class_name:
            cv2.putText(frame, class_name, (w//2 - 120, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 0), 3)
            
            bar_w = int(confidence * 300)
            cv2.rectangle(frame, (w//2 - 150, h//2 + 50), (w//2 + 150, h//2 + 70),
                         (100, 100, 100), 2)
            cv2.rectangle(frame, (w//2 - 150, h//2 + 50), (w//2 - 150 + bar_w, h//2 + 70),
                         (0, 255, 0), -1)
            cv2.putText(frame, f"{confidence:.0%}", (w//2 - 25, h//2 + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Analyzing...", (w//2 - 100, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 165, 0), 2)
    else:
        cv2.putText(frame, "SHOW YOUR HANDS", (w//2 - 150, h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 255), 2)
    
    # Bottom info
    cv2.putText(frame, "Real-time ASL Detection", (w - 350, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
    
    return frame

# ═══════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', groups=GROUPS)

@app.route('/video_feed')
def video_feed():
    """Stream video"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/group', methods=['GET', 'POST'])
def manage_group():
    """Get/set group"""
    global current_group
    
    if request.method == 'POST':
        data = request.json
        group_id = data.get('group_id')
        if group_id in GROUPS:
            with state_lock:
                current_group = group_id
            return jsonify({'status': 'success', 'group': group_id})
        return jsonify({'status': 'error'}), 400
    
    with state_lock:
        group = current_group
    
    return jsonify({'current_group': group, 'groups': list(GROUPS.keys())})

@app.route('/api/search', methods=['POST'])
def search():
    """Search for class"""
    data = request.json
    query = data.get('query', '').strip()
    
    found_group = search_class(query)
    if found_group:
        with state_lock:
            current_group = found_group
        return jsonify({'found': True, 'group': found_group, 'class': query})
    return jsonify({'found': False})

@app.route('/api/groups')
def get_groups():
    """Get all groups"""
    groups_info = {}
    for gid, ginfo in GROUPS.items():
        groups_info[gid] = {
            'name': ginfo['name'],
            'classes': ginfo['classes'],
            'type': ginfo['type']
        }
    return jsonify(groups_info)

@app.route('/api/health')
def health():
    """Health check"""
    with state_lock:
        group = current_group
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'current_group': group,
        'mediapipe': hands is not None
    })

# ═══════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\nStarting threads...")
    
    t1 = threading.Thread(target=camera_capture_thread, daemon=True)
    t2 = threading.Thread(target=processing_thread, daemon=True)
    t1.start()
    t2.start()
    
    print("=" * 70)
    print("✓ Flask server starting on http://127.0.0.1:5000")
    print("=" * 70 + "\n")
    
    try:
        app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n✓ Shutting down...")
