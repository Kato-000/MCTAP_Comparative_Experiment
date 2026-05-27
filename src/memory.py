"""
src/memory.py — RAG Memory Store for MCTAP_NC
Stores successful attack prompts as embeddings per VulnCategory.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
from loguru import logger

from src.models import Model
from src.llm_client import get_embedding


class CategoryMemory:
    """Per-category vector store."""

    def __init__(self, category: str):
        self.category = category
        self.embeddings: torch.Tensor = torch.zeros(0, 768)  # (N, D)
        self.records: List[dict] = []

    def add(self, text: str, embedding: List[float], meta: dict) -> None:
        if not embedding:
            return
        vec = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        # Pad/truncate to match stored dim
        d = self.embeddings.shape[1] if self.embeddings.shape[0] > 0 else len(embedding)
        if vec.shape[1] != d:
            if vec.shape[1] > d:
                vec = vec[:, :d]
            else:
                vec = torch.nn.functional.pad(vec, (0, d - vec.shape[1]))
        if self.embeddings.shape[0] == 0:
            self.embeddings = vec
        else:
            self.embeddings = torch.cat([self.embeddings, vec], dim=0)
        self.records.append({"text": text, **meta})

    def query(self, embedding: List[float], top_k: int = 3) -> List[dict]:
        if not embedding or self.embeddings.shape[0] == 0:
            return []
        vec = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        d = self.embeddings.shape[1]
        if vec.shape[1] != d:
            if vec.shape[1] > d:
                vec = vec[:, :d]
            else:
                vec = torch.nn.functional.pad(vec, (0, d - vec.shape[1]))
        # Cosine similarity
        norm_q = vec / (vec.norm(dim=1, keepdim=True) + 1e-8)
        norm_db = self.embeddings / (self.embeddings.norm(dim=1, keepdim=True) + 1e-8)
        sims = (norm_db @ norm_q.T).squeeze(-1)
        top_k = min(top_k, len(self.records))
        indices = sims.topk(top_k).indices.tolist()
        return [self.records[i] for i in indices]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.embeddings, path / f"{self.category}_embeddings.pt")
        with open(path / f"{self.category}_records.json", "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, category: str, path: Path) -> "CategoryMemory":
        mem = cls(category)
        emb_path = path / f"{category}_embeddings.pt"
        rec_path = path / f"{category}_records.json"
        if emb_path.exists():
            mem.embeddings = torch.load(emb_path, weights_only=True)
        if rec_path.exists():
            with open(rec_path, "r", encoding="utf-8") as f:
                mem.records = json.load(f)
        return mem


class MemoryStore:
    """Top-level store: one CategoryMemory per VulnCategory."""

    def __init__(self, memory_dir: str, embedding_model: Optional[Model] = None):
        self.memory_dir = Path(memory_dir)
        self.embedding_model = embedding_model
        self._stores: Dict[str, CategoryMemory] = {}

    def _get_store(self, category: str) -> CategoryMemory:
        if category not in self._stores:
            self._stores[category] = CategoryMemory(category)
        return self._stores[category]

    def remember(self, goal: str, prompt: str, category: str, score: int) -> None:
        if self.embedding_model is None:
            return
        emb = get_embedding(self.embedding_model, goal)
        self._get_store(category).add(prompt, emb, {"goal": goal, "score": score})
        self._get_store(category).save(self.memory_dir)
        logger.debug(f"[Memory] Stored success for category={category}")

    def recall(self, goal: str, category: str, top_k: int = 3) -> List[dict]:
        if self.embedding_model is None:
            return []
        emb = get_embedding(self.embedding_model, goal)
        return self._get_store(category).query(emb, top_k=top_k)

    def stats(self) -> Dict[str, int]:
        return {k: len(v.records) for k, v in self._stores.items()}

    @classmethod
    def load(cls, memory_dir: str, embedding_model: Optional[Model] = None) -> "MemoryStore":
        """既存のベクトルストアをディスクからロードする。"""
        store = cls(memory_dir, embedding_model)
        path = Path(memory_dir)
        if path.exists():
            # "general" キーのストアを事前ロード（カテゴリなし版で使用）
            store._stores["general"] = CategoryMemory.load("general", path)
        return store
