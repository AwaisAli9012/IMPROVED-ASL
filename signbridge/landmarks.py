"""Landmark normalization and features. Shared by training and live inference: never reimplement elsewhere.

A sequence is a float32 array (T, N_POINTS, 2); NaN marks a missing point.
"""
import warnings

import numpy as np

from .config import HAND_N, LH, LIP_IDX, LIPS, N_POINTS, POSE, POSE_IDX, RH, T_MODEL, T_RAW

_BLOCKS = (("lh", LH, HAND_N), ("rh", RH, HAND_N), ("pose", POSE, len(POSE_IDX)), ("lips", LIPS, len(LIP_IDX)))
_PAIRS = ((43, 44), (45, 46), (47, 48))  # L/R pose pairs: shoulders, elbows, wrists


def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a)


def _present(seq, block):
    """Per-frame bool: the hand block has any point."""
    return np.isfinite(seq[:, block, 0]).any(axis=1)


def frames_to_array(frames):
    """Client frames [{"lh","rh","pose","lips"}] -> (T, N_POINTS, 2). Trust boundary: validates shapes."""
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty list")
    out = np.full((len(frames), N_POINTS, 2), np.nan, dtype=np.float32)
    for t, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frame {t} is not an object")
        for key, block, n in _BLOCKS:
            pts = frame.get(key)
            if pts is None:
                continue
            try:
                arr = np.asarray(pts, dtype=np.float32)
            except (TypeError, ValueError):
                raise ValueError(f"frame {t} {key}: not numeric") from None
            if arr.shape != (n, 2) or not np.isfinite(arr).all():
                raise ValueError(f"frame {t} {key}: expected {n} finite [x, y] points")
            out[t, block] = arr
    return out


def canonicalize(seq):
    """Make the dominant hand (more frames with a hand; tie -> right) the right hand."""
    # shortcut: lip points are mirrored in x but not paired left/right, so left-handed clips have
    # slightly different lip features, build a face L/R index map if left-handed accuracy is low.
    if _present(seq, LH).sum() <= _present(seq, RH).sum():
        return seq
    m = seq.copy()
    m[..., 0] = 1 - m[..., 0]
    out = m.copy()
    out[:, LH], out[:, RH] = m[:, RH], m[:, LH]
    for a, b in _PAIRS:
        out[:, a], out[:, b] = m[:, b], m[:, a]
    return out


def trim(seq):
    """Drop leading and trailing frames with no hand. Unchanged if no frame has a hand."""
    idx = np.flatnonzero(_present(seq, LH) | _present(seq, RH))
    return seq[idx[0]:idx[-1] + 1] if len(idx) else seq


def resample(seq, n):
    return seq[np.linspace(0, len(seq) - 1, n).round().astype(int)]


def prepare_sequence(seq):
    """Raw (T, N_POINTS, 2) -> (T_RAW, N_POINTS, 2)."""
    return resample(trim(canonicalize(seq)), T_RAW)


def augment_sequence(seq, rng):
    """Training-time augmentation of a (T_RAW, N_POINTS, 2) sequence. Returns a new array."""
    seq = seq.copy()
    if rng.random() < 0.5:
        c = _nanmean(seq.reshape(-1, 2), axis=0)
        if np.isfinite(c).all():
            th = np.deg2rad(rng.uniform(-15, 15))
            rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
            p = (seq - c) * rng.uniform(0.8, 1.2) @ rot.T
            p[..., 1] *= rng.uniform(0.75, 1.33)  # unknown video aspect ratios
            seq = (p + c + rng.uniform(-0.08, 0.08, size=2)).astype(np.float32)
    if rng.random() < 0.5:
        n = len(seq)
        length = int(rng.uniform(0.8, 1.0) * n)
        start = rng.integers(0, n - length + 1)
        seq = resample(seq[start:start + length], n)
    if rng.random() < 0.1:
        seq[:, LH] = np.nan
    if rng.random() < 0.5:
        seq = seq + rng.normal(0, 0.008, seq.shape).astype(np.float32)
    return seq


def sequence_features(seq):
    """(T_RAW, N_POINTS, 2) -> (T_MODEL, 400): position, velocity, dominant-hand local shape, presence."""
    s = resample(seq, T_MODEL)
    presence = np.stack([_present(s, LH), _present(s, RH)], axis=1).astype(np.float32)
    shoulders = s[:, 43:45]
    center = _nanmean(shoulders.reshape(-1, 2), axis=0)
    scale = _nanmean(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=-1))
    if np.isnan(center).any():
        center = _nanmean(s.reshape(-1, 2), axis=0)
    if np.isnan(scale) or scale < 1e-3:
        scale = 2 * _nanstd(s) + 1e-6
    if np.isnan(center).any():
        center = np.zeros(2, dtype=np.float32)
    if np.isnan(scale):
        scale = 1.0
    pos = (s - center) / scale
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    h = pos[:, RH]
    span = np.linalg.norm(h[:, 9] - h[:, 0], axis=-1)[:, None, None]
    local = (h - h[:, 0:1]) / (span + 1e-6)
    t = len(s)
    feats = np.concatenate([pos.reshape(t, -1), vel.reshape(t, -1), local.reshape(t, -1), presence], axis=1)
    return np.nan_to_num(feats, nan=0.0).astype(np.float32)


_IU = np.triu_indices(HAND_N, 1)


def hand_features(hand, aspect):
    """(21, 3) image-normalized hand + width/height -> (273,): 63 local coords + 210 pairwise distances."""
    p = np.asarray(hand, dtype=np.float32).copy()
    p[:, 0] *= aspect
    p -= p[0]
    p /= np.linalg.norm(p[9, :2]) + 1e-6
    dist = np.linalg.norm(p[:, None, :2] - p[None, :, :2], axis=-1)[_IU]
    return np.concatenate([p.ravel(), dist]).astype(np.float32)


def mirror_hand(hand):
    """Training augmentation for static data: x -> 1 - x."""
    h = np.asarray(hand, dtype=np.float32).copy()
    h[:, 0] = 1 - h[:, 0]
    return h


def emotion_features(bs, baseline, neutral_mean):
    """Shift the user's neutral face onto the training neutral. baseline may be None."""
    bs = np.asarray(bs, dtype=np.float32)
    if baseline is None:
        return bs
    shift = np.asarray(baseline, dtype=np.float32) - np.asarray(neutral_mean, dtype=np.float32)
    return np.clip(bs - shift, 0.0, 1.0).astype(np.float32)
