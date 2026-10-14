"""
EMOTION ENGINE (live inference for app.py)
==========================================
Webcam frame -> face_alignment 68 landmarks -> emotion_features (same code as training) -> classifier.

On-demand use (Calibrate Neutral / Detect Tone buttons), not continuous tracking: face_alignment on a CPU
takes roughly a second per frame. Nothing heavy is imported until load().

Personal baseline ("Option B"):
  * Calibrate Neutral stores the person's own neutral-face feature vector.
  * At detection time the person's CHANGE from their own neutral is added onto the dataset's typical neutral
    face before classifying, so a naturally downturned mouth or heavy brow does not read as sad/angry.
  * An optional interpretable "cue" model can be blended in (weight stored in emotion_meta.json by
    train_emotion.py, or overridden with EMOTION_CUE_WEIGHT). It is 0 unless it helped on validation data.
  * `signals()` returns the person's measured changes (smile, brows, mouth...) for the on-screen readout.
"""

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from emotion_features import features_from_raw, N_FEATURES, FEATURE_NAMES

MODEL_FILE = "emotion_model.pkl"
CUE_FILE = "emotion_cue_model.pkl"
META_FILE = "emotion_meta.json"

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def locate_model_dir(candidates):
    """First candidate folder that contains the trained model, else None."""
    for c in candidates:
        if c and (Path(c) / MODEL_FILE).exists():
            return Path(c)
    return None


def _avg(feats, *names):
    return float(np.mean([feats[_IDX[n]] for n in names]))


class EmotionEngine:
    def __init__(self, model_dir, max_width=480):
        self.model_dir = Path(model_dir) if model_dir else None
        self.max_width = max_width
        self.model = None
        self.cue_model = None
        self.cue_weight = 0.0
        self.meta = None
        self.fa = None
        self.classes = []
        self.neutral_median = None
        self.expr_start = 134
        self.loaded = False
        self.error = None
        self.last_overlay = None   # newest landmarks in ORIGINAL frame pixels, for the on-screen face scan
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def available(self):
        return self.model_dir is not None and (self.model_dir / MODEL_FILE).exists()

    # ------------------------------------------------------------------ loading
    def load_classifier(self):
        """Load only the trained classifier + metadata (no face model). Idempotent."""
        if self.model is not None:
            return
        if not self.available:
            raise FileNotFoundError(
                f"Emotion model not found. Run train_emotion.py and make sure {MODEL_FILE} and "
                f"{META_FILE} are in the Emotion_Models folder.")
        import joblib
        model = joblib.load(self.model_dir / MODEL_FILE)
        with open(self.model_dir / META_FILE) as f:
            meta = json.load(f)
        if int(meta.get("n_features", -1)) != N_FEATURES:
            raise RuntimeError(
                f"Model expects {meta.get('n_features')} features but emotion_features.py "
                f"produces {N_FEATURES}. Retrain with the current emotion_features.py.")
        self.meta = meta
        self.classes = [meta["classes"][str(i)] for i in range(len(meta["classes"]))]
        self.neutral_median = np.asarray(meta["neutral_median"], dtype=np.float32)
        self.expr_start = int(meta.get("expr_start", 134))

        w = float(meta.get("cue_weight", 0.0))
        env_w = os.getenv("EMOTION_CUE_WEIGHT")
        if env_w not in (None, ""):
            w = float(env_w)
        self.cue_weight = max(0.0, min(1.0, w))
        cue_path = self.model_dir / CUE_FILE
        if self.cue_weight > 0 and cue_path.exists():
            self.cue_model = joblib.load(cue_path)
        else:
            self.cue_weight = 0.0
        self.model = model

    def load(self):
        """Load the classifier and the face landmark model (idempotent, thread-safe)."""
        with self._load_lock:
            if self.loaded:
                return
            self.load_classifier()
            import torch
            import face_alignment
            try:
                torch.set_num_threads(int(os.getenv("EMOTION_TORCH_THREADS", "3")))   # sign detection is paused while this runs
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
        scale = 1.0
        if w > self.max_width:
            scale = self.max_width / float(w)
            bgr = cv2.resize(bgr, (self.max_width, int(h * scale)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._infer_lock:
            preds = self.fa.get_landmarks(rgb)
        if not preds:
            return None
        best = max(preds, key=lambda p: (p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min()))
        lm = np.asarray(best, dtype=np.float32) / scale      # back to original frame pixels
        self.last_overlay = {"pts": lm, "w": w, "h": h, "t": time.time()}
        return lm

    def frame_features(self, bgr):
        lm = self.get_landmarks(bgr)
        return None if lm is None else features_from_raw(lm)

    # ------------------------------------------------------------------ model side
    def _shifted(self, feats, neutral_ref):
        x = np.asarray(feats, dtype=np.float32)
        if neutral_ref is not None:
            x = x - np.asarray(neutral_ref, dtype=np.float32) + self.neutral_median
        return x

    def probs_from_features(self, feats, neutral_ref=None):
        """Class probabilities for one feature vector (personal-baseline shifted when calibrated)."""
        x = self._shifted(feats, neutral_ref).reshape(1, -1)
        p = self.model.predict_proba(x)[0]
        if self.cue_model is not None and self.cue_weight > 0:
            pc = self.cue_model.predict_proba(x[:, self.expr_start:])[0]
            p = (1.0 - self.cue_weight) * p + self.cue_weight * pc
        return p

    def signals(self, feats, neutral_ref=None):
        """The person's measured facial changes, in eye-distance units, for the on-screen readout.
        Relative to their own neutral when calibrated, otherwise to a typical neutral face.
        Positive = more of that movement."""
        ref = np.asarray(neutral_ref, dtype=np.float32) if neutral_ref is not None else self.neutral_median
        d = np.asarray(feats, dtype=np.float32) - ref
        return {
            "smile (mouth corners up)": float(d[_IDX["corner_lift"]]),
            "mouth wider": float(d[_IDX["mouth_width"]]),
            "cheeks up (eyes narrower)": float(-_avg(d, "eye_open_r", "eye_open_l")),
            "brows lowered": float(-_avg(d, "brow_raise_r", "brow_raise_l")),
            "inner brows down": float(_avg(d, "inner_drop_r", "inner_drop_l")),
            "brows drawn together": float(-d[_IDX["inner_brow_dist"]]),
            "mouth open": float(d[_IDX["mouth_open_inner"]]),
        }

    def analyze_features(self, feat_list, neutral_ref=None):
        """Average probabilities and signals over several frames."""
        if not feat_list:
            return {"ok": False, "reason": "no_face"}
        probs = np.mean([self.probs_from_features(f, neutral_ref) for f in feat_list], axis=0)
        sig_list = [self.signals(f, neutral_ref) for f in feat_list]
        sig = {k: float(np.mean([s[k] for s in sig_list])) for k in sig_list[0]}
        k = int(np.argmax(probs))
        return {
            "ok": True,
            "emotion": self.classes[k],
            "confidence": float(probs[k]),
            "probs": {c: float(p) for c, p in zip(self.classes, probs)},
            "signals": sig,
            "personalised": neutral_ref is not None,
            "faces_used": len(feat_list),
        }

    def analyze(self, frames, neutral_ref=None):
        feats = [f for f in (self.frame_features(fr) for fr in frames) if f is not None]
        res = self.analyze_features(feats, neutral_ref)
        res["frames"] = len(frames)
        return res

    # ------------------------------------------------------------------ calibration
    def calibrate_features(self, feat_list):
        """Neutral reference from feature vectors -> (ref, hint). ref is None if there are none.
        hint is a warning string when the model is fairly sure the face was NOT neutral."""
        if not feat_list:
            return None, None
        ref = np.mean(feat_list, axis=0).astype(np.float32)
        hint = None
        try:
            p = np.mean([self.probs_from_features(f, None) for f in feat_list], axis=0)
            k = int(np.argmax(p))
            if self.classes[k] != "neutral" and p[k] >= 0.65:
                hint = (f"That face looked {self.classes[k]} rather than neutral. "
                        f"If so, relax your face and calibrate again.")
        except Exception:
            pass
        return ref, hint

    def calibrate(self, frames):
        feats = [f for f in (self.frame_features(fr) for fr in frames) if f is not None]
        return self.calibrate_features(feats)