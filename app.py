"""
IMPROVED ASL - Flask Web App
=============================
Real-time ASL detection web interface
- Live camera feed
- Group selection
- Search functionality
- Ensemble predictions (RF + XGBoost + Meta)
"""

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
from pathlib import Path
import threading
import json

from Config import GROUPS, MODELS_DIR, APP_CONFIG

# ═══════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

print("=" * 70)
print("IMPROVED ASL - FLASK WEB APP")
print("=" * 70)

# Load models
print("\nLoading all 39 models...")
models = {}
for group_id in GROUPS:
    try:
        with open(MODELS_DIR / f"{group_id}_rf.pkl", 'rb') as f:
            models[group_id] = {
                'rf': pickle.load(f),
                'xgb': pickle.load(open(MODELS_DIR / f"{group_id}_xgb.pkl", 'rb')),
                'meta': pickle.load(open(MODELS_DIR / f"{group_id}_meta.pkl", 'rb'))
            }
        print(f"  ✓ {group_id}")
    except Exception as e:
        print(f"  ❌ {group_id}: {e}")

print(f"✓ Loaded {len(models)} groups")

# Global variables
current_group = 'ALPHA1'
current_frame = None
frame_lock = threading.Lock()
camera = None

# ═══════════════════════════════════════════════════════════════════════════
# HAND DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_hand_region(frame):
    """Extract hand keypoints using skin detection"""
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, False
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        h, w = frame.shape[:2]
        
        if not ((h * w) * 0.01 < area < (h * w) * 0.7):
            return None, False
        
        keypoints = []
        points = largest.reshape(-1, 2)
        
        for i in range(min(63, len(points))):
            keypoints.extend([
                float(points[i][0]) / w,
                float(points[i][1]) / h,
                0.5
            ])
        
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"]) / w
            cy = int(M["m01"] / M["m00"]) / h
            for _ in range(min(10, 63 - len(points))):
                keypoints.extend([cx, cy, 0.5])
        
        while len(keypoints) < 126:
            keypoints.append(0.0)
        
        kp = np.array(keypoints[:126], dtype=np.float32)
        return kp if np.sum(np.abs(kp)) > 0.5 else None, True
    except:
        return None, False

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
    """Get class name"""
    classes = GROUPS[group_id]['classes']
    return classes[class_idx] if class_idx < len(classes) else "Unknown"

def search_class(query):
    """Search for class"""
    query_lower = query.lower()
    for group_id, group_info in GROUPS.items():
        for class_name in group_info['classes']:
            if class_name.lower() == query_lower:
                return group_id
    return None

# ═══════════════════════════════════════════════════════════════════════════
# CAMERA THREAD
# ═══════════════════════════════════════════════════════════════════════════

def camera_thread_func():
    """Capture frames from camera"""
    global current_frame, current_group, camera
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, APP_CONFIG['frame_width'])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, APP_CONFIG['frame_height'])
    camera.set(cv2.CAP_PROP_FPS, APP_CONFIG['fps_target'])
    
    print("✓ Camera started")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Extract hand and predict
        keypoints, hand_detected = extract_hand_region(frame)
        class_name = None
        confidence = 0
        
        if hand_detected and keypoints is not None:
            class_idx, conf = predict_ensemble(keypoints, current_group)
            if class_idx is not None and conf > 0.6:
                class_name = get_class_name(current_group, class_idx)
                confidence = conf
        
        # Draw UI
        cv2.putText(frame, f"GROUP: {current_group}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Classes: {', '.join(GROUPS[current_group]['classes'])}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        if not hand_detected:
            cv2.putText(frame, "NO HAND DETECTED", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif class_name:
            cv2.putText(frame, f"DETECTED: {class_name} ({confidence:.1%})", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            bar_w = int(confidence * 300)
            cv2.rectangle(frame, (20, 160), (20 + bar_w, 180), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 160), (320, 180), (200, 200, 200), 2)
        else:
            cv2.putText(frame, "Analyzing...", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        
        # Encode and store
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        with frame_lock:
            current_frame = buffer.tobytes()

# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', groups=GROUPS)

@app.route('/video_feed')
def video_feed():
    """Video streaming"""
    def generate():
        while True:
            with frame_lock:
                if current_frame is None:
                    continue
                frame = current_frame
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/group', methods=['GET', 'POST'])
def manage_group():
    """Get/set current group"""
    global current_group
    
    if request.method == 'POST':
        data = request.json
        group_id = data.get('group_id')
        if group_id in GROUPS:
            current_group = group_id
            return jsonify({'status': 'success', 'group': group_id})
        return jsonify({'status': 'error', 'message': 'Invalid group'}), 400
    
    return jsonify({'current_group': current_group, 'groups': list(GROUPS.keys())})

@app.route('/api/search', methods=['POST'])
def search():
    """Search for class"""
    data = request.json
    query = data.get('query', '').strip()
    
    found_group = search_class(query)
    if found_group:
        return jsonify({'found': True, 'group': found_group, 'class': query})
    return jsonify({'found': False})

@app.route('/api/groups')
def get_groups():
    """Get all groups info"""
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
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'current_group': current_group
    })

# ═══════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\nStarting camera thread...")
    camera_t = threading.Thread(target=camera_thread_func, daemon=True)
    camera_t.start()
    
    print("=" * 70)
    print("✓ Flask server starting on http://127.0.0.1:5000")
    print("=" * 70)
    
    try:
        app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n✓ Shutting down...")
        if camera:
            camera.release()
