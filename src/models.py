"""
src/models.py — Data classes and configuration
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class AttackMethod(str, Enum):
    TAP_BASELINE = "TAP_Baseline"  # Baseline TAP（論文オリジナル）
    PAIR = "PAIR"                  # PAIR（論文オリジナル）
    MCTAP_NC = "MCTAP_NC"          # 提案手法 ablation: Memory あり・Category なし


@dataclass
class Model:
    api_base: str
    api_key: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class AnthropicModel:
    model_name: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class HFModel:
    model_name: str
    api_key: str
    provider: str = "hf-inference"
    api_base: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class LocalModel:
    model_path: str
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass
class AttackConfig:
    # Models
    attacker_remote: Optional[Model] = None
    target_remote: Optional[Model] = None
    target_claude: Optional[AnthropicModel] = None
    target_hf: Optional[HFModel] = None
    target_local: Optional[LocalModel] = None
    embedding_model: Optional[Model] = None

    # TAP tree parameters
    branching_factor: int = 3
    root_width: int = 5
    depth: int = 10
    stop_score: int = 4
    width: int = 10

    # Memory directory (MCTAP_NC のみ使用; None で無効)
    memory_dir: Optional[str] = "data/memory"

    # Output paths
    output_name: str = "results"
    output_base: str = "data/jailbreaks"
    trace_base: str = "data/traces"
    benchmark_path: str = "Dataset/Original_Prompt/goals.jsonl"


@dataclass
class AttackResult:
    goal: str
    method: AttackMethod
    success: bool
    score: int
    num_queries: int
    duration: float
    final_prompt: str
    final_response: str
    attempts: List[dict] = field(default_factory=list)
