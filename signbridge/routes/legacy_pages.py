"""Temporary: the Jinja pages, until each one exists in the SvelteKit app (removed in Phase 8)."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .. import config, learn

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(config.WEB / "templates"))
templates.env.globals["url_for"] = lambda name, filename: f"/static/{filename}"
LANDING = config.WEB / "scrollcraft" / "builds" / "signbridge"
LANDING_FILES = {"scrollcraft.js", "scrollcraft.css", "landing.css", "landing.js"}  # a whitelist: BRIEF.md stays private
PAGES = {"app": "converse", "learn": "learn", "dictionary": "dictionary", "metrics": "metrics", "certificate": "certificate",
         "progress": "progress", "settings": "settings", "about": "about", "phrasebook": "phrasebook", "captions": "captions"}


@router.get("/")
def landing():
    if not (LANDING / "index.html").exists():
        return RedirectResponse("/app")
    return FileResponse(LANDING / "index.html")


@router.get("/learn/{unit_id}")
def lesson_page(request: Request, unit_id: str):
    if unit_id not in learn.UNITS and unit_id != "review":  # "review" is the spaced-repetition session
        raise HTTPException(404, "unknown unit")
    return templates.TemplateResponse(request, "lesson.html", {"unit_id": unit_id})


@router.get("/{name}")
def page(request: Request, name: str):
    if name in LANDING_FILES:
        return FileResponse(LANDING / name)
    if name not in PAGES:
        raise HTTPException(404)
    return templates.TemplateResponse(request, f"{PAGES[name]}.html")
