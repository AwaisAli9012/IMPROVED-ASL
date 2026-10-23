"""Request bodies. Fields are loose on purpose: services.py validates them and keeps the original error messages."""
from typing import Any

from pydantic import BaseModel


class WordRequest(BaseModel):
    frames: Any = None


class StaticRequest(BaseModel):
    hand: Any = None
    aspect: Any = None
    mode: Any = None


class EmotionRequest(BaseModel):
    bs: Any = None
    baseline: Any = None


class SentenceRequest(BaseModel):
    glosses: Any = None
    tone: Any = "neutral"
    question: Any = "none"


class GlossRequest(BaseModel):
    text: Any = None


class ProfileRequest(BaseModel):
    name: Any = None


class AttemptRequest(BaseModel):
    profile_id: Any = None
    item: Any = None
    correct: Any = None


class UnitPassRequest(BaseModel):
    profile_id: Any = None
    unit_id: Any = None
    score: Any = None
