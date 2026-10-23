import pytest
from fastapi.testclient import TestClient

from signbridge import config
from signbridge.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return TestClient(app)
