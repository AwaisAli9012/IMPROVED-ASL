"""Configuration, health, reports, curriculum and phrases."""
import json
from pathlib import Path

from fastapi import APIRouter

from .. import config, learn, services

router = APIRouter()


@router.get("/config")
def get_config():
    return services.app_config()


@router.get("/health")
def get_health():
    return services.health()


@router.get("/reports")
def get_reports():
    return {f.stem: json.loads(f.read_text()) for f in sorted(config.REPORTS.glob("*.json"))}


@router.get("/curriculum")
def get_curriculum():
    return learn.CURRICULUM


@router.get("/phrases")
def get_phrases():
    return json.loads((Path(__file__).parent.parent / "phrases.json").read_text())
