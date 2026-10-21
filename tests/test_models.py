import pytest
import torch

from signbridge.config import ARTIFACTS, T_MODEL, WORD_FEATURES
from signbridge.models import WordNet
from signbridge.vocab import display, word_labels


def test_wordnet_output_shape():
    model = WordNet(n_classes=7).eval()
    with torch.inference_mode():
        assert model(torch.randn(3, T_MODEL, WORD_FEATURES)).shape == (3, 7)


@pytest.mark.skipif(not (ARTIFACTS / "vocab_words.json").exists(), reason="vocab not built yet")
def test_every_word_has_a_display_name():
    labels = word_labels()
    assert len(labels) == len(set(labels)) == 100
    assert all(display(w).strip() for w in labels)
