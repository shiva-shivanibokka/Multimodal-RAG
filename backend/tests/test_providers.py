# backend/tests/test_providers.py
import httpx
import pytest

from app.generate.providers import ProviderError, generate, route_provider
def test_route_openai_compatible():
    assert route_provider("groq")["kind"] == "openai_compat"
    assert route_provider("gemini")["kind"] == "openai_compat"
    assert route_provider("openai")["kind"] == "openai_compat"
def test_route_anthropic():
    assert route_provider("anthropic")["kind"] == "anthropic"
def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        route_provider("cohere")


_SECRET_KEY = "sk-this-should-never-appear-in-an-error"


def test_generate_401_maps_to_bad_key_provider_error(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return httpx.Response(401, json={"error": "invalid api key"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(ProviderError) as exc_info:
        generate("openai", "gpt-4o", _SECRET_KEY, [{"role": "user", "content": "hi"}])

    err = exc_info.value
    assert 400 <= err.status_code < 500
    assert _SECRET_KEY not in err.detail
    assert _SECRET_KEY not in str(err)


def test_generate_5xx_maps_to_upstream_provider_error(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return httpx.Response(500, json={"error": "boom"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(ProviderError) as exc_info:
        generate("anthropic", "claude-x", _SECRET_KEY, [{"role": "user", "content": "hi"}])

    assert exc_info.value.status_code == 502
    assert _SECRET_KEY not in exc_info.value.detail
