"""
evaluate.py — Post-hoc evaluation of attack results

Computes:
  - Attack Success Rate (ASR) per method
  - Average number of queries
  - Average duration
  - Score distribution
  - Per-goal comparison across methods

Usage:
  python evaluate.py --input data/jailbreaks --output data/eval_report.json

Options:
  --input   Directory containing method subdirectories with *.jsonl files
  --output  Output JSON report path
  --csv     Also export a CSV summary
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict


def load_all_results(input_dir: str) -> Dict[str, List[dict]]:
    """Load all JSONL results, grouped by method."""
    base = Path(input_dir)
    results: Dict[str, List[dict]] = defaultdict(list)

    for method_dir in sorted(base.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for jsonl_file in sorted(method_dir.glob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results[method].append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    return results


def compute_stats(records: List[dict]) -> dict:
    if not records:
        return {}
    n = len(records)
    successes = sum(1 for r in records if r.get("success"))
    scores = [r.get("score", 0) for r in records]
    queries = [r.get("num_queries", 0) for r in records]
    durations = [r.get("duration", 0.0) for r in records]

    score_dist = defaultdict(int)
    for s in scores:
        score_dist[str(s)] += 1

    return {
        "total": n,
        "success": successes,
        "asr": round(successes / n, 4),
        "avg_score": round(sum(scores) / n, 3),
        "avg_queries": round(sum(queries) / n, 1),
        "avg_duration_sec": round(sum(durations) / n, 1),
        "score_distribution": dict(score_dist),
    }


def build_per_goal_table(all_results: Dict[str, List[dict]]) -> List[dict]:
    """Build per-goal cross-method comparison."""
    goal_data: Dict[str, Dict[str, dict]] = defaultdict(dict)

    for method, records in all_results.items():
        for r in records:
            goal = r.get("goal", "")
            goal_data[goal][method] = {
                "success": r.get("success"),
                "score": r.get("score"),
                "num_queries": r.get("num_queries"),
            }

    table = []
    for goal, methods in sorted(goal_data.items()):
        row = {"goal": goal}
        row.update({f"{m}_success": v["success"] for m, v in methods.items()})
        row.update({f"{m}_score": v["score"] for m, v in methods.items()})
        row.update({f"{m}_queries": v["num_queries"] for m, v in methods.items()})
        table.append(row)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adversarial attack results")
    parser.add_argument("--input", default="data/jailbreaks", help="Input directory")
    parser.add_argument("--output", default="data/eval_report.json", help="Output JSON report")
    parser.add_argument("--csv", action="store_true", help="Also export CSV summary")
    args = parser.parse_args()

    print(f"Loading results from: {args.input}")
    all_results = load_all_results(args.input)

    if not all_results:
        print("No results found. Run main.py first.")
        return

    # Compute stats
    report = {
        "methods": {},
        "per_goal": build_per_goal_table(all_results),
    }

    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    print(f"{'Method':<25} {'N':>5} {'ASR':>6} {'Avg Score':>10} {'Avg Q':>8} {'Avg T(s)':>10}")
    print("-"*70)

    for method, records in sorted(all_results.items()):
        stats = compute_stats(records)
        report["methods"][method] = stats
        print(
            f"{method:<25} "
            f"{stats['total']:>5} "
            f"{stats['asr']:>6.1%} "
            f"{stats['avg_score']:>10.3f} "
            f"{stats['avg_queries']:>8.1f} "
            f"{stats['avg_duration_sec']:>10.1f}"
        )

    print("="*70)

    # Score distribution
    print("\nScore Distribution (1=refusal … 4=success):")
    for method, stats in report["methods"].items():
        dist = stats.get("score_distribution", {})
        dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        print(f"  {method:<25} {dist_str}")

    # Save JSON report
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {out}")

    # Optional CSV
    if args.csv:
        csv_path = out.with_suffix(".csv")
        method_names = list(report["methods"].keys())
        fieldnames = ["goal"] + [
            f"{m}_{k}" for m in method_names for k in ["success", "score", "queries"]
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in report["per_goal"]:
                writer.writerow(row)
        print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
