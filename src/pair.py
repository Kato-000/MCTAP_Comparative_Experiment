"""
src/pair.py — PAIR (Prompt Automatic Iterative Refinement) attack

Reference: Chao et al., "Jailbreaking Black Box Large Language Models in Twenty Queries"
           NeurIPS Workshop R0-FoMo, 2023.

Algorithm:
  For up to max_iterations:
    1. Attacker が {improvement, prompt} JSON を生成（会話履歴を積み上げ）
    2. prompt をターゲットに送信 → 応答取得
    3. Judge がスコアリング（1–4）
    4. score >= stop_score → 成功終了
    5. LANGUAGE MODEL OUTPUT / OBJECTIVE / SCORE を Attacker にフィードバック → 次ターンへ

Attacker システムプロンプトは TAP と完全に共通（src/prompts.py）。
手法間の差異はアルゴリズム（反復ループ）のみ。
"""
from __future__ import annotations
import json
import re
import time

from loguru import logger

from src.llm_client import chat_completion
from src.judge import judge_response
from src.models import AttackConfig, AttackResult, AttackMethod
from src.prompts import ATTACKER_SYSTEM, ATTACKER_INIT_MSG, ATTACKER_FEEDBACK_MSG


def run_pair(
    goal: str,
    config: AttackConfig,
    max_iterations: int = 20,
) -> AttackResult:
    """PAIR: 1本の会話履歴を積み上げながら最大 max_iterations 回反復改善する。"""
    t_start = time.time()
    attacker = config.attacker_remote
    target   = config.target_remote or config.target_claude or config.target_hf
    judge    = attacker

    query_count = 0
    best_score    = 0
    best_prompt   = ""
    best_response = ""

    # Attacker との会話履歴（システムプロンプト以外）
    attacker_history = []

    # ── 初回メッセージ ──
    attacker_history.append({
        "role": "user",
        "content": ATTACKER_INIT_MSG.format(goal=goal),
    })

    for iteration in range(1, max_iterations + 1):

        # ── Step 1: Attacker がプロンプトを生成 ──
        messages = [{"role": "system", "content": ATTACKER_SYSTEM}] + attacker_history
        attacker_output = chat_completion(attacker, messages, max_tokens=512)
        query_count += 1

        attack_prompt = _parse_prompt(attacker_output)

        # ── Step 2: ターゲットにプロンプトを送信 ──
        response = chat_completion(
            target,
            [{"role": "user", "content": attack_prompt}],
            max_tokens=512,
        )
        query_count += 1

        # ── Step 3: Judge がスコアリング ──
        score, _ = judge_response(judge, goal, response)
        query_count += 1

        logger.info(f"[PAIR] iter={iteration}/{max_iterations} score={score}")

        if score > best_score:
            best_score    = score
            best_prompt   = attack_prompt
            best_response = response

        # ── Step 4: 成功判定 ──
        if score >= config.stop_score:
            logger.success(f"[PAIR] SUCCESS at iter={iteration} score={score}")
            return AttackResult(
                goal=goal,
                method=AttackMethod.PAIR,
                success=True,
                score=score,
                num_queries=query_count,
                duration=round(time.time() - t_start, 2),
                final_prompt=attack_prompt,
                final_response=response,
            )

        # ── Step 5: Attacker 会話履歴にフィードバックを追記 ──
        attacker_history.append({"role": "assistant", "content": attacker_output})
        attacker_history.append({
            "role": "user",
            "content": ATTACKER_FEEDBACK_MSG.format(
                response=response[:600],
                goal=goal,
                score=score,
            ),
        })

    return AttackResult(
        goal=goal,
        method=AttackMethod.PAIR,
        success=best_score >= config.stop_score,
        score=best_score,
        num_queries=query_count,
        duration=round(time.time() - t_start, 2),
        final_prompt=best_prompt,
        final_response=best_response,
    )


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
