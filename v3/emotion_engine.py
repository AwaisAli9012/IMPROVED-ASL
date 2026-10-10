"""
EMOTION ENGINE (live inference for app.py)
==========================================
Webcam frame -> face_alignment 68 landmarks -> emotion_features (same code as training) -> XGBoost.

Designed for ON-DEMAND use (Calibrate / Detect & Lock buttons), not continuous tracking:
face_alignment on a CPU takes roughly a second per frame. Nothing heavy is imported until load().
"""

import json
import threading
from pathlib import Path

import numpy as np

from emotion_features import features_from_raw, N_FEATURES

MODEL_FILE = "emotion_model.pkl"
META_FILE = "emotion_meta.json"


def locate_model_dir(candidates):
    """First candidate folder that contains the trained model, else None."""
    for c in candidates:
        if c and (Path(c) / MODEL_FILE).exists():
            return Path(c)
    return None


class EmotionEngine:
    def __init__(self, model_dir, max_width=480):
        self.model_dir = Path(model_dir) if model_dir else None
        self.max_width = max_width
        self.model = None
        self.meta = None
        self.fa = None
        self.classes = []
        self.neutral_median = None
        self.loaded = False
        self.error = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def available(self):
        return self.model_dir is not None and (self.model_dir / MODEL_FILE).exists()

    # ------------------------------------------------------------------ loading
    def load(self):
        """Load the classifier and the face landmark model (idempotent, thread-safe)."""
        with self._load_lock:
            if self.loaded:
                return
            if not self.available:
                raise FileNotFoundError(
                    f"Emotion model not found. Run train_emotion.py and make sure {MODEL_FILE} and "
                    f"{META_FILE} are in the Emotion_Models folder.")
            import joblib
            self.model = joblib.load(self.model_dir / MODEL_FILE)
            with open(self.model_dir / META_FILE) as f:
                self.meta = json.load(f)
            if int(self.meta.get("n_features", -1)) != N_FEATURES:
                raise RuntimeError(
                    f"Model expects {self.meta.get('n_features')} features but emotion_features.py "
                    f"produces {N_FEATURES}. Retrain with the current emotion_features.py.")
            self.classes = [self.meta["classes"][str(i)] for i in range(len(self.meta["classes"]))]
            self.neutral_median = np.asarray(self.meta["neutral_median"], dtype=np.float32)

            import torch
            import face_alignment
            try:
                torch.set_num_threads(2)   # keep the video threads responsive
            except Exception:
                pass
            lm_type = getattr(face_alignment.LandmarksType, "TWO_D", None) or \
                getattr(face_alignment.LandmarksType, "_2D")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.fa = face_alignment.FaceAlignment(lm_type, flip_input=False, device=device)
            self.loaded = True
            self.error = None

    # ------------------------------------------------------------------ per frame
    def get_landmarks(self, bgr):
        """BGR frame -> (68,2) landmarks of the largest face, or None."""
        import cv2
        h, w = bgr.shape[:2]
        if w > self.max_width:
            s = self.max_width / float(w)
            bgr = cv2.resize(bgr, (self.max_width, int(h * s)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._infer_lock:
            preds = self.fa.get_landmarks(rgb)
        if not preds:
            return None
        best = max(preds, key=lambda p: (p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min()))
        return np.asarray(best, dtype=np.float32)

    def frame_features(self, bgr):
        lm = self.get_landmarks(bgr)
        return None if lm is None else features_from_raw(lm)

    # ------------------------------------------------------------------ model side
    def probs_from_features(self, feats, neutral_ref=None):
        """Class probabilities for one feature vector. With a neutral reference the person's own
        neutral face is shifted onto the dataset's typical neutral face first."""
        x = np.asarray(feats, dtype=np.float32)
        if neutral_ref is not None:
            x = x - np.asarray(neutral_ref, dtype=np.float32) + self.neutral_median
        return self.model.predict_proba(x.reshape(1, -1))[0]

    def analyze_features(self, feat_list, neutral_ref=None):
        """Average the probabilities over several frames' features."""
        if not feat_list:
            return {"ok": False, "reason": "no_face"}
        probs = np.mean([self.probs_from_features(f, neutral_ref) for f in feat_list], axis=0)
        k = int(np.argmax(probs))
        return {
            "ok": True,
            "emotion": self.classes[k],
            "confidence": float(probs[k]),
            "probs": {c: float(p) for c, p in zip(self.classes, probs)},
            "faces_used": len(feat_list),
        }

    def analyze(self, frames, neutral_ref=None):
        feats = [f for f in (self.frame_features(fr) for fr in frames) if f is not None]
        res = self.analyze_features(feats, neutral_ref)
        res["frames"] = len(frames)
        return res

    def calibrate(self, frames):
        """The person's neutral reference: mean feature vector over several frames, or None."""
        feats = [f for f in (self.frame_features(fr) for fr in frames) if f is not None]
        return np.mean(feats, axis=0).astype(np.float32) if feats else None
