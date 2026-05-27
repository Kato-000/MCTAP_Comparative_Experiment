"""
main.py — Comparison Experiment Entry Point

Runs three methods on the same set of goals:
  1. Baseline TAP          → TAP_Baseline        (メモリなし・カテゴリなし / 論文オリジナル)
  2. PAIR                  → PAIR                (論文オリジナル)
  3. 提案手法 カテゴリなし版 → MCTAP_NC (メモリあり・カテゴリなし)

Results are saved per method to data/jailbreaks/<method>/<output_name>.jsonl
Summary is saved to data/summary.json

Usage:
  python main.py

Configuration:
  Edit the `config` block below.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.models import AttackConfig, AttackMethod, AttackResult, Model, AnthropicModel, HFModel
from src.memory import MemoryStore
from src.tap import run_tap
from src.pair import run_pair
from src.utils import save_result_jsonl, load_goals, save_summary

logger.add(sink="logs/experiment.log", level=20, rotation="10 MB")

# =============================================================================
# ★ CONFIGURATION — Edit this section ★
# =============================================================================

# ── Model examples (uncomment one set) ──────────────────────────────────────

# [A] LM Studio (local)
ATTACKER_MODEL = Model(
    api_base="http://localhost:1234/v1",
    api_key="lm-studio",
    model_name="openai/mistralai/mistral-nemo-instruct-2407",
    temperature=0.7,
)
TARGET_MODEL = Model(
    api_base="http://localhost:1234/v1",
    api_key="lm-studio",
    model_name="openai/llama-3.2-1b-instruct",
    temperature=0.0,
)
EMBEDDING_MODEL = Model(
    api_base="http://localhost:1234/v1",
    api_key="lm-studio",
    model_name="text-embedding-nomic-embed-text-v1.5",
)

# [B] Anthropic Claude as target (uncomment to use)
# TARGET_MODEL = AnthropicModel(
#     model_name="claude-haiku-4-5-20251001",
#     api_key=os.getenv("ANTHROPIC_API_KEY", ""),
#     temperature=0.0,
# )

# [C] HuggingFace Inference API (uncomment to use)
# TARGET_MODEL = HFModel(
#     model_name="meta-llama/Llama-3.3-70B-Instruct",
#     api_key=os.getenv("HF_TOKEN", ""),
#     provider="hf-inference",
#     temperature=0.0,
# )

# ── Experiment parameters ────────────────────────────────────────────────────

config = AttackConfig(
    # Models
    attacker_remote=ATTACKER_MODEL,
    target_remote=TARGET_MODEL,
    embedding_model=EMBEDDING_MODEL,

    # TAP tree parameters
    branching_factor=3,
    root_width=5,
    depth=10,
    stop_score=4,
    width=10,

    # Memory (MCTAP_NC のみ使用)
    memory_dir="data/memory",

    # Output
    output_name="results",
    output_base="data/jailbreaks",
    trace_base="data/traces",

    # Dataset
    benchmark_path="Dataset/Original_Prompt/goals.jsonl",
)

# Which methods to run
METHODS_TO_RUN: List[AttackMethod] = [
    AttackMethod.TAP_BASELINE,          # Baseline TAP: メモリなし・カテゴリなし（論文オリジナル）
    AttackMethod.PAIR,                  # PAIR（論文オリジナル）
    AttackMethod.MCTAP_NC,  # 提案手法 カテゴリなし版: メモリあり・カテゴリなし
]

# PAIR-specific: max iterations per goal
PAIR_MAX_ITERATIONS = 20

# =============================================================================
# Execution
# =============================================================================

def main() -> None:
    # ── Setup directories ──
    for method in METHODS_TO_RUN:
        Path(config.output_base, method.value).mkdir(parents=True, exist_ok=True)
    Path(config.trace_base).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    # ── Load goals ──
    goals = load_goals(config.benchmark_path)
    logger.info(f"[+] Loaded {len(goals)} goals from {config.benchmark_path}")

    # ── Initialize Memory (MCTAP_NC のみ使用) ──
    memory: Optional[MemoryStore] = None
    if AttackMethod.MCTAP_NC in METHODS_TO_RUN and config.memory_dir:
        memory = MemoryStore.load(config.memory_dir, embedding_model=config.embedding_model)
        logger.info(f"[+] MemoryStore loaded: {memory.stats()}")

    all_results: List[AttackResult] = []

    # ── Main loop ──
    for goal_idx, goal in enumerate(goals, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[Goal {goal_idx}/{len(goals)}] {goal[:80]}")

        for method in METHODS_TO_RUN:
            logger.info(f"  ── Running: {method.value} ──")

            try:
                if method == AttackMethod.PAIR:
                    result = run_pair(goal, config, max_iterations=PAIR_MAX_ITERATIONS)
                else:
                    result = run_tap(goal, config, method=method, memory=memory)

                # Save
                out_path = Path(config.output_base) / method.value / f"{config.output_name}.jsonl"
                save_result_jsonl(result, out_path)
                all_results.append(result)

                status = "✓ SUCCESS" if result.success else "✗ FAIL"
                logger.info(
                    f"  {status} | score={result.score}/4 | "
                    f"queries={result.num_queries} | time={result.duration}s"
                )

            except Exception as e:
                logger.error(f"  ERROR running {method.value} on goal '{goal[:50]}': {e}")
                import traceback
                logger.debug(traceback.format_exc())

    # ── Save summary ──
    summary_path = Path("data/summary.json")
    save_summary(all_results, summary_path)

    # ── Print summary table ──
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Method':<25} {'ASR':>6} {'Avg Queries':>12} {'Avg Time(s)':>12}")
    print("-"*70)

    with open(summary_path, "r") as f:
        summary = json.load(f)

    for method_name, stats in summary.items():
        print(
            f"{method_name:<25} "
            f"{stats['asr']:>6.1%} "
            f"{stats['avg_queries']:>12.1f} "
            f"{stats['avg_duration_sec']:>12.1f}"
        )
    print("="*70)
    print(f"\nDetailed results: {config.output_base}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
