"""Per-frame MediaPipe landmarks from a video file, with the same .task models and hand/pose/lip rules as the browser."""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

from signbridge.config import LIP_IDX, POSE_IDX, WEB
from signbridge.landmarks import assemble_frame, assign_hands

MODELS = WEB / "static" / "models"
VIDEO = vision.RunningMode.VIDEO
_landmarkers, _clock = None, 0  # one set per process: creating and closing them per clip leaks ~50 MB each


def make_landmarkers():
    def base(name):
        return mpt.BaseOptions(model_asset_path=str(MODELS / name))
    return {
        "hands": vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base("hand_landmarker.task"), running_mode=VIDEO, num_hands=2)),
        "pose": vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base("pose_landmarker_lite.task"), running_mode=VIDEO, num_poses=1)),
        "face": vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base("face_landmarker.task"), running_mode=VIDEO, num_faces=1)),
    }


def _xy(landmarks, idx):
    return np.array([[landmarks[i].x, landmarks[i].y] for i in idx], dtype=np.float32)


def video_landmarks(path):
    """-> {"seq": (T, N_POINTS, 2), "hands": [(lh, rh) per frame], "aspect": width / height} or None if unreadable.

    The face runs on every 2nd frame and the last lips are reused, as in the browser.
    Video mode needs ever-increasing timestamps and carries tracking state, so each clip starts later on the
    shared clock and two blank frames clear the tracking left over from the previous clip.
    """
    global _landmarkers, _clock
    _landmarkers = _landmarkers or make_landmarkers()
    lm, base = _landmarkers, _clock + 1000
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 30.0
    aspect = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / max(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 1)
    seq, hands_per_frame, lips, i = [], [], None, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if i == 0:
            blank = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros_like(rgb))
            for t in (base - 66, base - 33):
                for m in lm.values():
                    m.detect_for_video(blank, t)
        ts = base + int(i * 1000 / fps)
        hands = lm["hands"].detect_for_video(img, ts).hand_landmarks
        pose = lm["pose"].detect_for_video(img, ts).pose_landmarks
        pose = _xy(pose[0], POSE_IDX) if pose else None
        if i % 2 == 0:
            face = lm["face"].detect_for_video(img, ts).face_landmarks
            lips = _xy(face[0], LIP_IDX) if face else None
        hands = [np.array([[p.x, p.y, p.z] for p in h], dtype=np.float32) for h in hands]
        seq.append(assemble_frame(hands, pose, lips))
        hands_per_frame.append(assign_hands(hands, pose))
        i += 1
    cap.release()
    _clock = base + int(i * 1000 / fps)
    if not seq:
        return None
    return {"seq": np.stack(seq), "hands": hands_per_frame, "aspect": float(aspect)}
