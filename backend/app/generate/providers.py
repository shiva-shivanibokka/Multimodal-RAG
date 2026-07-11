# backend/app/generate/providers.py
import httpx
_OPENAI_COMPAT = {
    "openai": "https://api.openai.com/v1",
    "groq":   "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


class ProviderError(Exception):
    """Raised when the LLM provider call fails. ``status_code`` is the HTTP
    status the API route should return (never the api_key, never leaked)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def route_provider(provider: str) -> dict:
    if provider in _OPENAI_COMPAT:
        return {"kind": "openai_compat", "base_url": _OPENAI_COMPAT[provider]}
    if provider == "anthropic":
        return {"kind": "anthropic", "base_url": "https://api.anthropic.com/v1"}
    raise ValueError(f"unsupported provider: {provider}")


def _post(url, headers, payload, timeout):
    """POST to the provider and map failures to ProviderError. Never include
    the request headers (which carry the api_key) in any error message."""
    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        return r
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403):
            raise ProviderError(
                400, "provider rejected the request (check your API key / model name)"
            ) from None
        if 400 <= status < 500:
            raise ProviderError(
                400, "provider rejected the request (check your API key / model name)"
            ) from None
        raise ProviderError(502, "upstream provider error") from None
    except httpx.HTTPError:
        # timeouts, connection errors, etc. -- no response to inspect
        raise ProviderError(502, "upstream provider error") from None


def generate(provider, model, api_key, messages, images=None, timeout=120) -> str:
    cfg = route_provider(provider)
    if cfg["kind"] == "openai_compat":
        payload = {"model": model, "messages": _attach_images_openai(messages, images)}
        r = _post(f"{cfg['base_url']}/chat/completions",
                  headers={"Authorization": f"Bearer {api_key}"},
                  payload=payload, timeout=timeout)
        return r.json()["choices"][0]["message"]["content"]
    # anthropic
    payload = {"model": model, "max_tokens": 1024,
               "messages": _attach_images_anthropic(messages, images)}
    r = _post(f"{cfg['base_url']}/messages",
              headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
              payload=payload, timeout=timeout)
    return "".join(b.get("text", "") for b in r.json()["content"])

def _attach_images_openai(messages, images):
    if not images: return messages
    msgs = [dict(m) for m in messages]
    last = msgs[-1]
    parts = [{"type": "text", "text": last["content"]}]
    parts += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images]
    last["content"] = parts
    return msgs

def _attach_images_anthropic(messages, images):
    if not images: return messages
    msgs = [dict(m) for m in messages]
    last = msgs[-1]
    parts = [{"type": "text", "text": last["content"]}]
    parts += [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b}} for b in images]
    last["content"] = parts
    return msgs
