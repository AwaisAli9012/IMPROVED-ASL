import numpy as np
import pytest

from signbridge.config import LH, LIPS, N_POINTS, POSE, RH, T_MODEL, T_RAW, WORD_FEATURES
from signbridge.landmarks import (assemble_frame, assign_hands, augment_sequence, canonicalize, emotion_features, frames_to_array,
                                  hand_features, mirror_hand, prepare_sequence, sequence_features, trim)

_PAIRS = ((43, 44), (45, 46), (47, 48))


def _clip(seed=0, t=T_RAW):
    """Right-dominant clip: every point present except the left hand."""
    seq = np.random.default_rng(seed).uniform(0.2, 0.8, (t, N_POINTS, 2)).astype(np.float32)
    seq[:, LH] = np.nan
    return seq


def _mirror(seq):
    m = seq.copy()
    m[..., 0] = 1 - m[..., 0]
    out = m.copy()
    out[:, LH], out[:, RH] = m[:, RH], m[:, LH]
    for a, b in _PAIRS:
        out[:, a], out[:, b] = m[:, b], m[:, a]
    return out


def test_shapes_and_dtype():
    f = sequence_features(_clip())
    assert f.shape == (T_MODEL, WORD_FEATURES) and f.dtype == np.float32
    assert not np.isnan(f).any()
    assert prepare_sequence(_clip(t=40)).shape == (T_RAW, N_POINTS, 2)
    assert hand_features(np.random.rand(21, 3), 1.5).shape == (273,)


def test_translation_and_scale_invariance():
    seq = _clip()
    base = sequence_features(seq)
    np.testing.assert_allclose(sequence_features(seq + np.float32([0.3, -0.2])), base, atol=1e-4)
    np.testing.assert_allclose(sequence_features(seq * 1.7), base, atol=1e-4)


def test_left_dominant_matches_mirrored_twin():
    right = _clip(1)
    left = _mirror(right)
    assert np.isfinite(left[:, LH]).all() and np.isnan(left[:, RH]).all()
    np.testing.assert_allclose(canonicalize(left), right, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(sequence_features(prepare_sequence(left)),
                               sequence_features(prepare_sequence(right)), atol=1e-5)


def test_hand_features_invariance():
    hand = np.random.default_rng(2).uniform(0.2, 0.8, (21, 3)).astype(np.float32)
    base = hand_features(hand, 1.0)
    moved = hand.copy()
    moved[:, :2] += 0.1
    np.testing.assert_allclose(hand_features(moved, 1.0), base, atol=1e-4)
    np.testing.assert_allclose(hand_features(hand * 2, 1.0), base, atol=1e-4)


def test_trim_drops_only_empty_ends():
    seq = _clip(t=10)
    seq[:3] = np.nan
    seq[8:] = np.nan
    assert trim(seq).shape[0] == 5
    empty = np.full((6, N_POINTS, 2), np.nan, dtype=np.float32)
    assert trim(empty).shape[0] == 6


def test_emotion_features_shift():
    neutral, own = np.full(52, 0.1, np.float32), np.full(52, 0.3, np.float32)
    np.testing.assert_allclose(emotion_features(own, own, neutral), neutral, atol=1e-6)
    np.testing.assert_array_equal(emotion_features(own, None, neutral), own)
    assert emotion_features(np.ones(52), own, neutral).max() <= 1.0


def test_mirror_hand_involution():
    hand = np.random.rand(21, 3).astype(np.float32)
    np.testing.assert_allclose(mirror_hand(mirror_hand(hand)), hand, atol=1e-6)


def test_augment_keeps_shape_and_input():
    seq = _clip()
    before = seq.copy()
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = augment_sequence(seq, rng)
        assert out.shape == seq.shape and out.dtype == np.float32
    np.testing.assert_array_equal(seq, before)


def test_frames_to_array():
    hand = [[0.5, 0.5]] * 21
    arr = frames_to_array([{"rh": hand, "lh": None, "pose": None, "lips": None}, {}])
    assert arr.shape == (2, N_POINTS, 2)
    assert np.isfinite(arr[0, RH]).all() and np.isnan(arr[0, LH]).all() and np.isnan(arr[1]).all()
    with pytest.raises(ValueError):
        frames_to_array([{"rh": [[0.5, 0.5]] * 20}])
    with pytest.raises(ValueError):
        frames_to_array([])


def _pose():
    p = np.zeros((7, 2), np.float32)
    p[1], p[2] = (0.6, 0.4), (0.4, 0.4)   # left shoulder is on the image right in an un-mirrored video
    p[5], p[6] = (0.7, 0.6), (0.3, 0.6)   # left / right wrist
    return p


def _hand(x, y):
    h = np.zeros((21, 3), np.float32)
    h[:, 0], h[:, 1] = x, y
    return h


def test_assign_hands_by_nearest_wrist():
    left, right = _hand(0.72, 0.6), _hand(0.28, 0.6)
    lh, rh = assign_hands([right, left], _pose())
    assert lh is left and rh is right
    lh, rh = assign_hands([right], _pose())
    assert lh is None and rh is right


def test_assign_hands_rest_rule_and_no_pose():
    resting = _hand(0.3, 1.5)  # far below the shoulders
    assert assign_hands([resting], _pose()) == (None, None)
    lh, rh = assign_hands([_hand(0.3, 0.5), _hand(0.7, 0.5)], None)
    assert lh[0, 0] == pytest.approx(0.7) and rh[0, 0] == pytest.approx(0.3)


def test_assemble_frame_layout():
    frame = assemble_frame([_hand(0.3, 0.6)], _pose(), None)
    assert frame.shape == (N_POINTS, 2)
    assert np.isfinite(frame[RH]).all() and np.isnan(frame[LH]).all() and np.isnan(frame[LIPS]).all()
    assert np.isfinite(frame[POSE]).all()
