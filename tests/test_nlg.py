import pytest

from signbridge import nlg
from signbridge.config import ARTIFACTS

pytestmark = pytest.mark.skipif(not (ARTIFACTS / "vocab_words.json").exists(), reason="vocab not built yet")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_gloss_to_english_offline():
    assert nlg.gloss_to_english(["me", "go", "school"]) == {"english": "I go school.", "source": "offline"}
    assert nlg.gloss_to_english(["you", "fine"], question="yesno")["english"] == "You fine?"
    assert nlg.gloss_to_english(["my", "name", "fs:ali"])["english"] == "My name Ali."
    with pytest.raises(ValueError):
        nlg.gloss_to_english([])


def test_english_to_gloss_offline():
    out = nlg.english_to_gloss("Thank you, Ali! I am happy to help.")
    assert out["source"] == "offline"
    assert out["tokens"] == [{"type": "sign", "label": "thankyou"}, {"type": "spell", "text": "ALI"},
                             {"type": "sign", "label": "me"}, {"type": "sign", "label": "happy"},
                             {"type": "sign", "label": "help"}]


def test_unknown_words_and_digits_are_spelled():
    tokens = nlg.english_to_gloss("room 42")["tokens"]
    assert tokens[-1] == {"type": "spell", "text": "42"}


def test_clean_rejects_model_inventions():
    vocab = {"hello"}
    tokens = nlg._clean([{"type": "sign", "label": "Hello"}, {"type": "sign", "label": "banana!"},
                         {"type": "spell", "text": "a-b 1"}, {"type": "wat"}, {"type": "spell", "text": "--"}], vocab)
    assert tokens == [{"type": "sign", "label": "hello"}, {"type": "spell", "text": "BANANA"}, {"type": "spell", "text": "AB1"}]
