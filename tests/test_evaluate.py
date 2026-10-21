import numpy as np

from scripts.evaluate import accept_threshold, ece


def test_ece_is_zero_when_confidence_matches_accuracy():
    conf = np.array([0.5] * 10)
    correct = np.array([True] * 5 + [False] * 5)
    assert ece(conf, correct) < 1e-9
    assert ece(np.array([0.9] * 10), correct) > 0.3  # overconfident


def test_accept_threshold_picks_smallest_precise_cutoff():
    conf = np.concatenate([np.linspace(0.3, 0.55, 30), np.linspace(0.6, 0.99, 70)])
    correct = conf >= 0.6  # everything below 0.6 is wrong, everything above is right
    t = accept_threshold(conf, correct)
    assert correct[conf >= t].mean() >= 0.95
    assert correct[conf >= t - 0.01].mean() < 0.95  # one step lower would no longer meet the bar


def test_accept_threshold_none_when_too_few_predictions():
    assert accept_threshold(np.array([0.5, 0.6]), np.array([True, True])) is None
