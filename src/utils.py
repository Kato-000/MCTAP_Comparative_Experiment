"""
src/utils.py — File I/O utilities

MCTAP本家 src/utils.py の log_jsonl / save_trace に準拠したtrace保存を追加。
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Optional

from src.models import AttackResult


# =============================================================================
# Result saving
# =============================================================================

def save_result_jsonl(result: AttackResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "goal":           result.goal,
        "method":         result.method.value,
        "success":        result.success,
        "score":          result.score,
        "num_queries":    result.num_queries,
        "duration":       result.duration,
        "final_prompt":   result.final_prompt,
        "final_response": result.final_response,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_goals(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    goals = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    goals.append(obj.get("harmful") or obj.get("goal") or str(obj))
                else:
                    goals.append(str(obj))
            except json.JSONDecodeError:
                goals.append(line)
    return goals


def save_summary(results: List[AttackResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from collections import defaultdict
    stats = defaultdict(lambda: {"total": 0, "success": 0, "total_queries": 0, "total_duration": 0.0})
    for r in results:
        s = stats[r.method.value]
        s["total"] += 1
        s["success"] += int(r.success)
        s["total_queries"] += r.num_queries
        s["total_duration"] += r.duration

    summary = {}
    for method, s in stats.items():
        n = s["total"]
        summary[method] = {
            "total":            n,
            "success":          s["success"],
            "asr":              round(s["success"] / n, 4) if n > 0 else 0.0,
            "avg_queries":      round(s["total_queries"] / n, 1) if n > 0 else 0.0,
            "avg_duration_sec": round(s["total_duration"] / n, 1) if n > 0 else 0.0,
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# =============================================================================
# Trace utilities（MCTAP本家 src/utils.py の log_jsonl / save_trace に準拠）
# =============================================================================

def log_jsonl(path: Path, record: dict) -> None:
    """1レコードを JSONL ファイルに逐次追記する。本家 log_jsonl に準拠。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_trace(
    trace_file: Path,
    goal: str,
    method: str,
    all_attempts: List[dict],
    best_score_history: List[dict],
    success: bool,
    duration: float,
) -> None:
    """
    1ゴール分の全試行を JSON ファイルに保存する。本家 save_trace に準拠。

    保存構造:
    {
      "goal": str,
      "method": str,
      "success": bool,
      "duration": float,
      "best_score_history": [{"iteration": int, "score": int, "prompt": str}, ...],
      "all_attempts": [
        {
          "iteration":   int,        # 深さ / イテレーション番号（0=初回クエリ）
          "node_id":     str,        # ノード識別子（例: "1-2"）
          "child_id":    str,        # 子ノード識別子（例: "1-2-3"）
          "prompt":      str,        # 攻撃プロンプト
          "response":    str,        # ターゲットの応答
          "score":       int,        # Judgeスコア
          "improvement": str,        # Attackerの改善方針
          "on_topic":    bool,       # on-topicチェック結果
        },
        ...
      ]
    }
    """
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace = {
        "goal":               goal,
        "method":             method,
        "success":            success,
        "duration":           duration,
        "best_score_history": best_score_history,
        "all_attempts":       all_attempts,
    }
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)


def make_goal_slug(goal: str) -> str:
    """ゴール文字列をファイル名に使えるスラグに変換する。本家準拠。"""
    return goal[:60].replace("/", "_").replace(" ", "_").replace('"', "").replace("'", "")
