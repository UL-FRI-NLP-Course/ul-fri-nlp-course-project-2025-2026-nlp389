"""
Aggregates LLM judge scores across all 4 systems and writes a Markdown report.

Usage:
    python aggregate.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", default="results")
parser.add_argument("--out", default="results/final_report.md")
args = parser.parse_args()

results_dir = Path(args.results_dir)
files = sorted(results_dir.glob("judged_*.json"))
if not files:
    raise SystemExit(f"No judged_*.json files in {results_dir}/")

SYSTEM_LABELS = {
    "base":           "A: Base (no RAG, no FT)",
    "finetuned":      "B: Fine-tuned (no RAG)",
    "base_rag":       "C: Base + RAG",
    "finetuned_rag":  "D: Fine-tuned + RAG",
    "llama70b":       "E: LLaMA-3.3-70B (reference, no RAG)",
}


def avg(xs):
    return mean(xs) if xs else 0.0


# ── Load all judgements ──────────────────────────────────────────────────
systems = {}
for f in files:
    name = f.stem.replace("judged_", "")
    systems[name] = json.loads(f.read_text(encoding="utf-8"))

# ── Compute overall averages ─────────────────────────────────────────────
lines = ["# LoL RAG Evaluation Report\n"]
lines.append(f"Systems evaluated: {len(systems)}")
lines.append(f"Questions per system: {len(next(iter(systems.values())))}\n")

lines.append("## Overall Scores\n")
lines.append("| System | Correctness | Relevance | Combined |")
lines.append("|--------|:-----------:|:---------:|:--------:|")
for name, label in SYSTEM_LABELS.items():
    if name not in systems:
        continue
    rows = systems[name]
    c = avg([r["correctness"] for r in rows])
    r = avg([r["relevance"] for r in rows])
    lines.append(f"| {label} | {c:.2f} | {r:.2f} | {(c + r) / 2:.2f} |")

# ── Per-category breakdown ───────────────────────────────────────────────
lines.append("\n## Scores by Category\n")
all_categories = set()
for rows in systems.values():
    for r in rows:
        all_categories.add(r["category"])

for cat in sorted(all_categories):
    lines.append(f"\n### {cat}\n")
    lines.append("| System | Correctness | Relevance |")
    lines.append("|--------|:-----------:|:---------:|")
    for name, label in SYSTEM_LABELS.items():
        if name not in systems:
            continue
        rows = [r for r in systems[name] if r["category"] == cat]
        if not rows:
            continue
        c = avg([r["correctness"] for r in rows])
        rel = avg([r["relevance"] for r in rows])
        lines.append(f"| {label} | {c:.2f} | {rel:.2f} |")

# ── Per-question table (compact) ─────────────────────────────────────────
lines.append("\n## Per-Question Correctness\n")
header = ["#", "Category"] + [SYSTEM_LABELS.get(n, n) for n in SYSTEM_LABELS if n in systems]
lines.append("| " + " | ".join(header) + " |")
lines.append("|" + "|".join([":-:"] * len(header)) + "|")

# Build a lookup id -> {system: correctness}
by_id = defaultdict(dict)
categories = {}
for name, rows in systems.items():
    for r in rows:
        by_id[r["id"]][name] = r["correctness"]
        categories[r["id"]] = r["category"]

for qid in sorted(by_id):
    row = [str(qid), categories.get(qid, "?")]
    for name in SYSTEM_LABELS:
        if name not in systems:
            continue
        row.append(str(by_id[qid].get(name, "-")))
    lines.append("| " + " | ".join(row) + " |")

# ── Save ─────────────────────────────────────────────────────────────────
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"\n[done] Report saved to {out_path}")
