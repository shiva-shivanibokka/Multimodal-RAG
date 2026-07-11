# backend/tests/test_providers.py
from app.generate.providers import route_provider
def test_route_openai_compatible():
    assert route_provider("groq")["kind"] == "openai_compat"
    assert route_provider("gemini")["kind"] == "openai_compat"
    assert route_provider("openai")["kind"] == "openai_compat"
def test_route_anthropic():
    assert route_provider("anthropic")["kind"] == "anthropic"
def test_unknown_provider_raises():
    import pytest
    with pytest.raises(ValueError):
        route_provider("cohere")
