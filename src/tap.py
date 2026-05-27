"""
src/tap.py — TAP (Tree of Attacks with Pruning) core

Two variants:
  - TAP_Baseline        : メモリなし（論文オリジナル）
  - MCTAP_NC : メモリあり（提案手法カテゴリなし版）

Attacker システムプロンプトは PAIR と完全に共通（src/prompts.py）。
手法間の差異はアルゴリズム（木探索）とメモリの有無のみ。
"""
from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List

from loguru import logger

from src.llm_client import chat_completion
from src.judge import judge_response
from src.models import AttackConfig, AttackResult, AttackMethod
from src.memory import MemoryStore
from src.prompts import ATTACKER_SYSTEM, ATTACKER_INIT_MSG, ATTACKER_FEEDBACK_MSG


# ────────────────────────────────────────────────
# Tree node
# ────────────────────────────────────────────────

@dataclass
class TreeNode:
    prompt: str
    score: int = 0
    depth: int = 0
    children: List["TreeNode"] = field(default_factory=list)
    response: str = ""


# ────────────────────────────────────────────────
# TAP runner
# ────────────────────────────────────────────────

def run_tap(
    goal: str,
    config: AttackConfig,
    method: AttackMethod,
    memory: Optional[MemoryStore] = None,
) -> AttackResult:
    """
    TAP_Baseline        : use_memory=False
    MCTAP_NC : use_memory=True
    """
    t_start = time.time()
    attacker = config.attacker_remote
    target   = config.target_remote or config.target_claude or config.target_hf
    judge    = attacker  # attacker を judge として再利用

    use_memory = (method == AttackMethod.MCTAP_NC) and (memory is not None)

    # ── メモリ検索（提案手法のみ）──
    memory_prefix = ""
    if use_memory:
        recalls = memory.recall(goal, "general")
        if recalls:
            examples = "\n".join(f"- {r['text'][:200]}" for r in recalls[:3])
            memory_prefix = (
                f"Similar successful prompts from past experience:\n{examples}\n"
                f"Use these as inspiration.\n\n"
            )

    # ── ルートノード生成 ──
    roots: List[TreeNode] = []
    query_count = 0

    for _ in range(config.root_width):
        init_user = memory_prefix + ATTACKER_INIT_MSG.format(goal=goal)
        messages = [
            {"role": "system", "content": ATTACKER_SYSTEM},
            {"role": "user",   "content": init_user},
        ]
        attack_prompt = _parse_prompt(chat_completion(attacker, messages, max_tokens=512))
        query_count += 1

        response = chat_completion(
            target,
            [{"role": "user", "content": attack_prompt}],
            max_tokens=512,
        )
        query_count += 1

        score, _ = judge_response(judge, goal, response)
        query_count += 1

        node = TreeNode(prompt=attack_prompt, score=score, depth=0, response=response)
        roots.append(node)
        logger.debug(f"[TAP/{method.value}] root score={score}")

        if score >= config.stop_score:
            _maybe_remember(use_memory, memory, goal, attack_prompt, score)
            return _make_result(goal, method, node, query_count, t_start, True)

    # ── 木探索 ──
    best_node = max(roots, key=lambda n: n.score)
    frontier  = sorted(roots, key=lambda n: n.score, reverse=True)[:config.width]

    for depth in range(1, config.depth + 1):
        next_frontier: List[TreeNode] = []

        for parent in frontier:
            feedback_user = ATTACKER_FEEDBACK_MSG.format(
                response=parent.response[:600],
                goal=goal,
                score=parent.score,
            )
            messages = [
                {"role": "system", "content": ATTACKER_SYSTEM},
                {"role": "user",   "content": ATTACKER_INIT_MSG.format(goal=goal)},
                {"role": "assistant", "content": json.dumps({"improvement": "initial attempt", "prompt": parent.prompt})},
                {"role": "user",   "content": feedback_user},
            ]

            for _ in range(config.branching_factor):
                attack_prompt = _parse_prompt(chat_completion(attacker, messages, max_tokens=512))
                query_count += 1

                response = chat_completion(
                    target,
                    [{"role": "user", "content": attack_prompt}],
                    max_tokens=512,
                )
                query_count += 1

                score, _ = judge_response(judge, goal, response)
                query_count += 1

                child = TreeNode(prompt=attack_prompt, score=score, depth=depth, response=response)
                parent.children.append(child)
                next_frontier.append(child)

                if score > best_node.score:
                    best_node = child
                    logger.info(f"[TAP/{method.value}] depth={depth} new best score={score}")

                if score >= config.stop_score:
                    _maybe_remember(use_memory, memory, goal, attack_prompt, score)
                    return _make_result(goal, method, child, query_count, t_start, True)

        # Prune: top-width ノードを残す
        frontier = sorted(next_frontier, key=lambda n: n.score, reverse=True)[:config.width]
        if not frontier:
            break

    _maybe_remember(use_memory, memory, goal, best_node.prompt, best_node.score)
    return _make_result(goal, method, best_node, query_count, t_start,
                        best_node.score >= config.stop_score)


# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────

def _parse_prompt(raw: str) -> str:
    """JSON の prompt フィールドを抽出。失敗時は raw をそのまま返す。"""
    try:
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return data.get("prompt", raw).strip()
    except Exception:
        pass
    return raw.strip()


def _maybe_remember(
    use_memory: bool,
    memory: Optional[MemoryStore],
    goal: str,
    prompt: str,
    score: int,
) -> None:
    if use_memory and memory and score >= 3:
        memory.remember(goal, prompt, "general", score)


def _make_result(
    goal: str,
    method: AttackMethod,
    node: TreeNode,
    query_count: int,
    t_start: float,
    success: bool,
) -> AttackResult:
    return AttackResult(
        goal=goal,
        method=method,
        success=success,
        score=node.score,
        num_queries=query_count,
        duration=round(time.time() - t_start, 2),
        final_prompt=node.prompt,
        final_response=node.response,
    )
