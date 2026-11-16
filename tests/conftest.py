import pytest


@pytest.fixture(autouse=True)
def _offline_drafter_by_default(monkeypatch):
    """Tests never call a real LLM, even if ARC_DRAFTER=groq is exported in the shell or set in .env."""
    monkeypatch.setenv("ARC_DRAFTER", "extractive")
