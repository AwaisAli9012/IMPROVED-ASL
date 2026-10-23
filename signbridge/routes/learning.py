"""Profiles, the course and progress."""
from fastapi import APIRouter

from .. import learn
from ..schemas import AttemptRequest, ProfileRequest, UnitPassRequest

router = APIRouter()


@router.get("/profiles")
def get_profiles():
    return learn.list_profiles()


@router.post("/profiles")
def post_profile(req: ProfileRequest):
    return learn.create_profile(req.name)


@router.delete("/profiles/{profile_id}")
def remove_profile(profile_id: int):
    learn.delete_profile(profile_id)
    return {"deleted": True}


@router.get("/progress/{profile_id}")
def get_progress_report(profile_id: int):
    return learn.progress_report(profile_id)


@router.get("/learn/{profile_id}")
def get_progress(profile_id: int):
    return learn.curriculum_with_progress(profile_id)


@router.get("/learn/{profile_id}/review")
def get_review(profile_id: int):
    return learn.due_items(profile_id)


@router.post("/learn/attempt")
def post_attempt(req: AttemptRequest):
    if not isinstance(req.correct, bool):
        raise ValueError("correct must be true or false")
    return learn.record_attempt(req.profile_id, req.item, req.correct)


@router.post("/learn/unit_pass")
def post_unit_pass(req: UnitPassRequest):
    if not isinstance(req.score, (int, float)) or isinstance(req.score, bool):
        raise ValueError("score must be a number")
    return learn.record_unit_pass(req.profile_id, req.unit_id, req.score)
