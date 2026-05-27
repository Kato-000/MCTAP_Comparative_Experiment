"""
src/llm_client.py — Unified LLM API client
Supports: LM Studio (OpenAI-compatible), Anthropic, HuggingFace
"""
from __future__ import annotations
import os
import time
from typing import Optional, List

import openai
from loguru import logger

from src.models import Model, AnthropicModel, HFModel


def _make_openai_client(model: Model) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=model.api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url=model.api_base,
    )


def chat_completion(
    model_cfg,
    messages: List[dict],
    max_tokens: int = 1024,
) -> str:
    """
    Send a chat completion request and return the assistant's text.
    Handles retries with exponential backoff.
    """
    for attempt in range(5):
        try:
            if isinstance(model_cfg, Model):
                client = _make_openai_client(model_cfg)
                # Strip the "openai/" prefix that LM Studio doesn't need
                model_name = model_cfg.model_name
                if model_name.startswith("openai/"):
                    model_name = model_name[len("openai/"):]
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=model_cfg.temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""

            elif isinstance(model_cfg, AnthropicModel):
                import anthropic
                client = anthropic.Anthropic(
                    api_key=model_cfg.api_key or os.getenv("ANTHROPIC_API_KEY", "")
                )
                # Separate system message if present
                system = ""
                filtered = []
                for m in messages:
                    if m["role"] == "system":
                        system = m["content"]
                    else:
                        filtered.append(m)
                kwargs = dict(
                    model=model_cfg.model_name,
                    max_tokens=max_tokens,
                    messages=filtered,
                    temperature=model_cfg.temperature,
                )
                if system:
                    kwargs["system"] = system
                resp = client.messages.create(**kwargs)
                return resp.content[0].text

            elif isinstance(model_cfg, HFModel):
                client = openai.OpenAI(
                    api_key=model_cfg.api_key or os.getenv("HF_TOKEN", ""),
                    base_url=model_cfg.api_base or "https://api-inference.huggingface.co/v1",
                )
                resp = client.chat.completions.create(
                    model=model_cfg.model_name,
                    messages=messages,
                    temperature=model_cfg.temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""

        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"LLM call failed (attempt {attempt+1}/5): {e}. Retrying in {wait}s.")
            time.sleep(wait)

    logger.error("All LLM call attempts failed.")
    return "[ERROR: LLM call failed]"


def get_embedding(model_cfg: Model, text: str) -> List[float]:
    """Get text embedding vector."""
    try:
        client = _make_openai_client(model_cfg)
        model_name = model_cfg.model_name
        if model_name.startswith("openai/"):
            model_name = model_name[len("openai/"):]
        resp = client.embeddings.create(model=model_name, input=[text])
        return resp.data[0].embedding
    except Exception as e:
        logger.warning(f"Embedding call failed: {e}")
        return []
