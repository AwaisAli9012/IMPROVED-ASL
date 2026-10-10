"""
EMOTION FEATURES (68-point face_alignment landmarks)
====================================================
ONE place that defines how a face becomes a feature vector. Training (train_emotion.py) and the
live app (emotion_engine.py) both import this file, so they can never compute different features.

Landmarks use the standard iBUG 68-point numbering produced by `face_alignment`:
    jaw 0-16 | right brow 17-21 | left brow 22-26 | nose 27-35 | right eye 36-41 | left eye 42-47
    outer lip 48-59 | inner lip 60-67

Normalisation is identical to extract_emotion_landmarks.py:
    centre on landmark 30 (nose tip), divide by the outer-eye distance (landmarks 36 and 45).

Feature vector (150 values):
    134  normalised x,y of the 67 points other than landmark 30 (which is always 0,0)
      5  expression ratios (same five, same order as the extractor)
     11  extra expression features (eye openness, brow raise / inner-brow drop, smile lift, mouth opening ...)
"""

import numpy as np

N_POINTS = 68
ORIGIN_IDX = 30


def _d(n, a, b):
    """Euclidean distance between landmarks a and b. Works for (68,2) or (N,68,2)."""
    return np.linalg.norm(n[..., a, :] - n[..., b, :], axis=-1)


def normalize_landmarks(raw):
    """Raw (68,2) pixel landmarks -> normalised (68,2), or None if the face is degenerate."""
    pts = np.asarray(raw, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] != N_POINTS or pts.shape[1] < 2:
        return None
    pts = pts[:, :2]
    centered = pts - pts[ORIGIN_IDX]
    scale = float(np.linalg.norm(centered[45] - centered[36]))
    if not np.isfinite(scale) or scale < 1e-5:
        return None
    out = centered / scale
    return out if np.all(np.isfinite(out)) else None


def ratios5(n):
    """The five ratios from the original extractor (same order)."""
    return np.stack([
        _d(n, 48, 54),   # mouth width
        _d(n, 51, 57),   # mouth height
        _d(n, 21, 22),   # distance between inner brows
        _d(n, 19, 37),   # right brow to right eye
        _d(n, 24, 44),   # left brow to left eye
    ], axis=-1)


def extras11(n):
    """Extra expression features. y grows downward in image coordinates."""
    eye_w_r = _d(n, 36, 39) + 1e-6
    eye_w_l = _d(n, 42, 45) + 1e-6
    eye_open_r = (_d(n, 37, 41) + _d(n, 38, 40)) / 2.0 / eye_w_r
    eye_open_l = (_d(n, 43, 47) + _d(n, 44, 46)) / 2.0 / eye_w_l

    brow_raise_r = n[..., 37, 1] - n[..., 19, 1]      # bigger = brow further above the eye
    brow_raise_l = n[..., 44, 1] - n[..., 24, 1]
    inner_drop_r = n[..., 21, 1] - n[..., 17, 1]      # positive = inner end lower than outer end (frown)
    inner_drop_l = n[..., 22, 1] - n[..., 26, 1]

    lip_mid_y = (n[..., 51, 1] + n[..., 57, 1]) / 2.0
    corner_y = (n[..., 48, 1] + n[..., 54, 1]) / 2.0
    corner_lift = lip_mid_y - corner_y                # positive = corners above lip centre (smile)

    mouth_open = _d(n, 62, 66)
    chin_drop = _d(n, 8, 33)
    corner_to_nose_r = _d(n, 48, 33)
    corner_to_nose_l = _d(n, 54, 33)

    return np.stack([
        eye_open_r, eye_open_l, brow_raise_r, brow_raise_l, inner_drop_r, inner_drop_l,
        corner_lift, mouth_open, chin_drop, corner_to_nose_r, corner_to_nose_l,
    ], axis=-1)


def features_from_normalized(n):
    """Normalised landmarks (68,2) or (N,68,2) -> features (150,) or (N,150)."""
    n = np.asarray(n, dtype=np.float32)
    coords = np.delete(n, ORIGIN_IDX, axis=-2)
    coords = coords.reshape(*n.shape[:-2], -1)
    return np.concatenate([coords, ratios5(n), extras11(n)], axis=-1).astype(np.float32)


def features_from_raw(raw):
    """Raw (68,2) pixel landmarks -> features (150,), or None if the face is unusable."""
    n = normalize_landmarks(raw)
    if n is None:
        return None
    return features_from_normalized(n)


FEATURE_NAMES = (
    [f"p{i}_{ax}" for i in range(N_POINTS) if i != ORIGIN_IDX for ax in "xy"]
    + ["mouth_width", "mouth_height", "inner_brow_dist", "r_brow_eye", "l_brow_eye"]
    + ["eye_open_r", "eye_open_l", "brow_raise_r", "brow_raise_l", "inner_drop_r", "inner_drop_l",
       "corner_lift", "mouth_open_inner", "chin_drop", "corner_nose_r", "corner_nose_l"]
)
N_FEATURES = len(FEATURE_NAMES)   # 150
