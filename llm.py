"""Thin wrapper around a local Ollama model.

Uses only `requests` so there are no heavy SDK dependencies. Talks to the
Ollama REST API directly. Provides plain text generation and JSON-constrained
generation (used by the retriever).
"""

import json

import requests

import config


class LLMError(RuntimeError):
    pass


def _post(path: str, payload: dict) -> dict:
    url = f"{config.OLLAMA_HOST}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise LLMError(
            f"Could not reach Ollama at {config.OLLAMA_HOST}. "
            "Is it running? Try `ollama serve` and `ollama pull "
            f"{config.MODEL}`."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise LLMError(f"Ollama returned an error: {resp.text}") from exc
    return resp.json()


def generate(prompt: str, system: str = "", temperature: float = 0.2) -> str:
    """Return a plain-text completion for `prompt`."""
    payload = {
        "model": config.MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": temperature},
    }
    return _post("/api/generate", payload)["response"].strip()


def generate_json(prompt: str, system: str = "", temperature: float = 0.0) -> dict:
    """Return parsed JSON. Ollama's `format=json` forces valid JSON output."""
    payload = {
        "model": config.MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
    }
    raw = _post("/api/generate", payload)["response"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model did not return valid JSON: {raw!r}") from exc


def health_check() -> bool:
    """True if the Ollama server is reachable and the model is available."""
    try:
        tags = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10).json()
    except requests.exceptions.RequestException:
        return False
    names = {m["name"].split(":")[0] for m in tags.get("models", [])}
    return config.MODEL.split(":")[0] in names
