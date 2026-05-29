"""
src/llm_client.py — Unified LLM API client
Supports: LM Studio (OpenAI-compatible), Anthropic, HuggingFace

モデルロード失敗（LM Studio が一時的にアンロードした場合）に対して
長めの待機時間でリトライする。
"""
from __future__ import annotations
import os
import time
from typing import Optional, List

import openai
from loguru import logger

from src.models import Model, AnthropicModel, HFModel

# モデルロード失敗と判定するキーワード
_MODEL_LOAD_ERROR_KEYWORDS = [
    "failed to load model",
    "error loading model",
    "model is unloaded",
    "model not found",
]

_MAX_RETRIES         = 10   # 最大リトライ回数
_MODEL_LOAD_WAIT_SEC = 60   # モデルロードエラー時の待機時間（秒）
_BASE_WAIT_SEC       = 2    # 通常エラー時のベース待機時間


def _is_model_load_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(kw in msg for kw in _MODEL_LOAD_ERROR_KEYWORDS)


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
    モデルロードエラー時は長めに待機してリトライする。
    """
    for attempt in range(_MAX_RETRIES):
        try:
            if isinstance(model_cfg, Model):
                client = _make_openai_client(model_cfg)
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
            if _is_model_load_error(e):
                # モデルロードエラー: LM Studio がロード中のため長めに待機
                logger.warning(
                    f"[LLM] Model load error (attempt {attempt+1}/{_MAX_RETRIES}): {e}. "
                    f"Waiting {_MODEL_LOAD_WAIT_SEC}s for model to load..."
                )
                time.sleep(_MODEL_LOAD_WAIT_SEC)
            else:
                # 通常エラー: 指数バックオフ
                wait = _BASE_WAIT_SEC ** attempt
                logger.warning(
                    f"[LLM] Call failed (attempt {attempt+1}/{_MAX_RETRIES}): {e}. "
                    f"Retrying in {wait}s."
                )
                time.sleep(wait)

    logger.error("[LLM] All attempts failed. Returning error string.")
    return "[ERROR: LLM call failed]"


def get_embedding(model_cfg: Model, text: str) -> List[float]:
    """Get text embedding vector."""
    for attempt in range(5):
        try:
            client = _make_openai_client(model_cfg)
            model_name = model_cfg.model_name
            if model_name.startswith("openai/"):
                model_name = model_name[len("openai/"):]
            resp = client.embeddings.create(model=model_name, input=[text])
            return resp.data[0].embedding
        except Exception as e:
            if _is_model_load_error(e):
                logger.warning(f"[Embedding] Model load error. Waiting {_MODEL_LOAD_WAIT_SEC}s...")
                time.sleep(_MODEL_LOAD_WAIT_SEC)
            else:
                logger.warning(f"[Embedding] Call failed: {e}")
                return []
    return []
