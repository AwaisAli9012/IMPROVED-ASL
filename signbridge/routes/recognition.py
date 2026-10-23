"""Sign and face recognition."""
from fastapi import APIRouter

from .. import services
from ..schemas import EmotionRequest, StaticRequest, WordRequest

router = APIRouter()


@router.post("/predict/word")
def predict_word(req: WordRequest):
    return services.predict_word(req.frames)


@router.post("/predict/static")
def predict_static(req: StaticRequest):
    return services.predict_static(req.hand, req.aspect, req.mode)


@router.post("/emotion")
def predict_emotion(req: EmotionRequest):
    return services.predict_emotion(req.bs, req.baseline)
