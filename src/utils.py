"""
src/utils.py — File I/O utilities
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List

from src.models import AttackResult


def save_result_jsonl(result: AttackResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "goal": result.goal,
        "method": result.method.value,
        "success": result.success,
        "score": result.score,
        "num_queries": result.num_queries,
        "duration": result.duration,
        "final_prompt": result.final_prompt,
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
                # Support both {"harmful": "..."} and {"goal": "..."} and plain string
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
            "total": n,
            "success": s["success"],
            "asr": round(s["success"] / n, 4) if n > 0 else 0.0,
            "avg_queries": round(s["total_queries"] / n, 1) if n > 0 else 0.0,
            "avg_duration_sec": round(s["total_duration"] / n, 1) if n > 0 else 0.0,
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
