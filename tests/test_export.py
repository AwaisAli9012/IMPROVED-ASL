import numpy as np

from scripts.export_references import hand_frame, word_frames
from signbridge.config import LH, N_POINTS, RH


def test_word_frames_are_shoulder_normalised_and_json_ready():
    seq = np.random.default_rng(0).uniform(0.2, 0.8, (64, N_POINTS, 2)).astype(np.float32)
    seq[:, LH] = np.nan
    seq[:, 43], seq[:, 44] = (0.4, 0.5), (0.6, 0.5)  # steady shoulders, as in real signing
    frames = word_frames(seq)
    assert len(frames) == 64 and frames[0]["lh"] is None and len(frames[0]["rh"]) == 21 and len(frames[0]["lips"]) == 40
    shoulders = np.array([[f["pose"][1], f["pose"][2]] for f in frames])
    assert abs(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=-1).mean() - 1) < 0.05  # one shoulder width


def test_word_frames_need_shoulders():
    seq = np.random.default_rng(0).uniform(0.2, 0.8, (64, N_POINTS, 2)).astype(np.float32)
    seq[:, 43:45] = np.nan
    assert word_frames(seq) is None


def test_hand_frame_is_upright_and_unit_sized():
    hand = np.random.default_rng(1).uniform(0.2, 0.8, (21, 3)).astype(np.float32)
    p = np.array(hand_frame(hand, 1.5)[0]["rh"])
    assert np.allclose(p[0], 0) and np.allclose(p[9], [0, -1], atol=1e-2)  # wrist at origin, middle knuckle straight up
