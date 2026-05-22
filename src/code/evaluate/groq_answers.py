"""
Sends all 30 evaluation questions to Groq LLaMA-3.3-70B and saves answers.
Used as a strong reference model baseline (no RAG, no fine-tuning, but 70B params).

Usage:
    python groq_answers.py --api-key YOUR_GROQ_KEY
    # or set GROQ_API_KEY env variable
"""

import argparse
import json
import os
import time
from pathlib import Path

from groq import Groq

QUESTIONS_FILE = Path("full_questions.txt")
OUT_FILE = Path("answers/llama70b.json")
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are a knowledgeable League of Legends expert. "
    "Answer the following question about champion builds, patch notes, or ranked statistics. "
    "Be concise and factual. If you don't know the exact current patch data, "
    "provide the best general answer you can based on your knowledge."
)

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", default=os.environ.get("GROQ_API_KEY", ""))
parser.add_argument("--sleep", type=float, default=2.5,
                    help="Seconds between requests (Groq free tier: ~30 rpm)")
args = parser.parse_args()

if not args.api_key:
    raise SystemExit("No API key. Use --api-key or set GROQ_API_KEY env variable.")

client = Groq(api_key=args.api_key)

questions = [
    line.strip()
    for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]
print(f"Loaded {len(questions)} questions. Model: {MODEL}")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

answers = []
for i, question in enumerate(questions, 1):
    print(f"[{i}/{len(questions)}] {question[:70]}...")
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            answer = response.choices[0].message.content.strip()
            break
        except Exception as e:
            wait = (attempt + 1) * 15
            print(f"  Error (attempt {attempt+1}): {e} — retrying in {wait}s")
            time.sleep(wait)
    else:
        answer = "ERROR: failed after 5 attempts"

    answers.append({
        "question": question,
        "answer": answer,
        "sources": []
    })
    print(f"  -> {answer[:80]}...")
    time.sleep(args.sleep)

OUT_FILE.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[done] Saved {len(answers)} answers to {OUT_FILE}")
