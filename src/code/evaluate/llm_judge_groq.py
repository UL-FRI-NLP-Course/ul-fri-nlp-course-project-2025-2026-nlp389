"""
LLM-as-a-Judge using Groq (free, fast, LLaMA-3.3-70B).

Scores each answer on Correctness (0-5) and Relevance (0-5)
against a ground-truth reference.

Setup:
    1. Get free API key at https://console.groq.com/keys
    2. pip install groq
    3. Set key: see GROQ_API_KEY below

Usage:
    python llm_judge_groq.py
    python llm_judge_groq.py --systems base finetuned_rag
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from groq import Groq
except ImportError:
    sys.exit("Run: pip install groq")

# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--answers-dir", default="answers")
parser.add_argument("--results-dir", default="results")
parser.add_argument("--ground-truth", default="ground_truth.json")
parser.add_argument("--model", default="llama-3.3-70b-versatile",
                    help="Groq model. Alternatives: llama-3.1-70b-versatile, llama-3.1-8b-instant")
parser.add_argument("--systems", nargs="*", default=None)
parser.add_argument("--sleep", type=float, default=2.0,
                    help="seconds between calls (Groq free = 30 req/min)")
parser.add_argument("--max-retries", type=int, default=5)
args = parser.parse_args()

# Paste your Groq key here OR set GROQ_API_KEY env var
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "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"

if GROQ_API_KEY == "PASTE_YOUR_GROQ_KEY_HERE":
    sys.exit("ERROR: paste your Groq API key in llm_judge_groq.py or set GROQ_API_KEY env var.\n"
             "Get one free at https://console.groq.com/keys")

client = Groq(api_key=GROQ_API_KEY)

# ── Judge prompt ─────────────────────────────────────────────────────────
JUDGE_PROMPT = """You are an expert evaluator for League of Legends question-answering systems.

Score the candidate answer on TWO axes, each 0-5:

CORRECTNESS (0-5): Does the answer match the ground-truth facts?
  5 = fully correct, all key facts present
  4 = mostly correct, minor omissions
  3 = partially correct, some right and some wrong
  2 = mostly wrong, only minor overlap with truth
  1 = almost entirely wrong
  0 = completely wrong, contradicts the truth, or refuses to answer when truth is known

RELEVANCE (0-5): Does the answer actually address the question that was asked?
  5 = directly answers the question
  4 = answers the question with extra info
  3 = partially addresses, goes off-topic somewhat
  2 = mostly off-topic
  1 = barely related
  0 = completely off-topic

Output JSON ONLY in this exact format:
{{"correctness": <int 0-5>, "relevance": <int 0-5>, "reason": "<one short sentence>"}}

QUESTION: {question}

GROUND TRUTH: {ground_truth}

CANDIDATE ANSWER: {answer}

JSON output:"""


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"correctness": 0, "relevance": 0, "reason": f"could not parse: {text[:120]}"}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        return {"correctness": 0, "relevance": 0, "reason": f"json error: {e}"}


def judge_one(question: str, ground_truth: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, answer=answer)
    last_err = ""
    for attempt in range(args.max_retries):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content
            return parse_json_response(text)
        except Exception as e:
            last_err = str(e)
            msg = last_err.lower()
            if "429" in msg or "rate" in msg or "quota" in msg:
                m = re.search(r"try again in (\d+\.?\d*)s", last_err)
                wait = float(m.group(1)) + 1 if m else 20
                print(f" [rate-limited {wait:.0f}s]", end="", flush=True)
                time.sleep(wait)
                continue
            break
    return {"correctness": 0, "relevance": 0, "reason": f"api error: {last_err[:200]}"}


# ── Load ground truth ────────────────────────────────────────────────────
gt_path = Path(args.ground_truth)
if not gt_path.is_file():
    sys.exit(f"Ground truth not found: {gt_path}")
ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
gt_by_q = {item["question"]: item for item in ground_truth}
print(f"Loaded {len(ground_truth)} ground-truth entries")

# ── Discover answer files ───────────────────────────────────────────────
answers_dir = Path(args.answers_dir)
results_dir = Path(args.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

answer_files = sorted(answers_dir.glob("*.json"))
if args.systems:
    answer_files = [p for p in answer_files if p.stem in args.systems]
if not answer_files:
    sys.exit(f"No answer files in {answers_dir}/")

print(f"Judge   : {args.model}")
print(f"Sleep   : {args.sleep}s between calls")
print(f"Systems : {[p.stem for p in answer_files]}\n")

# ── Judge each system ────────────────────────────────────────────────────
for ans_path in answer_files:
    sys_name = ans_path.stem
    print(f"\n=== Judging {sys_name} ===")
    answers = json.loads(ans_path.read_text(encoding="utf-8"))

    judged = []
    for i, item in enumerate(answers, 1):
        q = item["question"]
        a = item["answer"]
        gt = gt_by_q.get(q)
        if not gt:
            print(f"  [{i}/{len(answers)}] WARN: no GT for '{q[:60]}'")
            continue
        print(f"  [{i}/{len(answers)}] {q[:60]}...", end=" ", flush=True)
        score = judge_one(q, gt["ground_truth"], a)
        print(f"C={score.get('correctness')} R={score.get('relevance')}")
        judged.append({
            "id": gt["id"],
            "category": gt["category"],
            "question": q,
            "ground_truth": gt["ground_truth"],
            "answer": a,
            "correctness": score.get("correctness", 0),
            "relevance": score.get("relevance", 0),
            "reason": score.get("reason", ""),
        })
        time.sleep(args.sleep)

    out_path = results_dir / f"judged_{sys_name}.json"
    out_path.write_text(json.dumps(judged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved {len(judged)} judgements to {out_path}")

print("\n[done] All systems judged. Run aggregate.py for the report.")
