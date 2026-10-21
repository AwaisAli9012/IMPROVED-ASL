"""Word labels and display names. The label list is written by scripts/build_words_dataset.py."""
import json
from functools import cache

from .config import ARTIFACTS

DISPLAY = {"thankyou": "thank you"}  # labels are lowercase ASL Citizen glosses; only joined words need a display name


def display(label):
    return DISPLAY.get(label, label)


@cache
def word_labels():
    return json.loads((ARTIFACTS / "vocab_words.json").read_text())["labels"]
