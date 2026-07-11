# backend/app/generate/providers.py
import httpx
_OPENAI_COMPAT = {
    "openai": "https://api.openai.com/v1",
    "groq":   "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}
def route_provider(provider: str) -> dict:
    if provider in _OPENAI_COMPAT:
        return {"kind": "openai_compat", "base_url": _OPENAI_COMPAT[provider]}
    if provider == "anthropic":
        return {"kind": "anthropic", "base_url": "https://api.anthropic.com/v1"}
    raise ValueError(f"unsupported provider: {provider}")

def generate(provider, model, api_key, messages, images=None, timeout=120) -> str:
    cfg = route_provider(provider)
    if cfg["kind"] == "openai_compat":
        payload = {"model": model, "messages": _attach_images_openai(messages, images)}
        r = httpx.post(f"{cfg['base_url']}/chat/completions",
                       headers={"Authorization": f"Bearer {api_key}"},
                       json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    # anthropic
    payload = {"model": model, "max_tokens": 1024,
               "messages": _attach_images_anthropic(messages, images)}
    r = httpx.post(f"{cfg['base_url']}/messages",
                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                   json=payload, timeout=timeout)
    r.raise_for_status()
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
