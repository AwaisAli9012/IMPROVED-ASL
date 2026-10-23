"""Sentences from glosses and glosses from sentences."""
from fastapi import APIRouter

from .. import services
from ..schemas import GlossRequest, SentenceRequest

router = APIRouter()


@router.post("/sentence")
def make_sentence(req: SentenceRequest):
    return services.make_sentence(req.glosses, req.tone, req.question)


@router.post("/gloss")
def make_gloss(req: GlossRequest):
    return services.make_gloss(req.text)
