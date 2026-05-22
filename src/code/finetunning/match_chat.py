"""
Interactive match analyst — tests the fine-tuned model on match data WITHOUT RAG.

Loads a random match from match_data.json, shows you the stats, then lets you
ask questions about it exactly the way the model was trained.

Usage:
    python match_chat.py
    python match_chat.py --base-only          # compare base vs fine-tuned
    python match_chat.py --champion Tristana  # filter to a specific champion
    python match_chat.py --index 0            # pick a specific match by index
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from huggingface_hub import login

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_DATA       = os.path.join(_HERE, "..", "..", "data") if os.path.isdir(os.path.join(_HERE, "..", "..", "data")) else os.path.join(_HERE, "..", "data")
MATCH_DATA  = os.path.join(_HERE, "match_data.json")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", os.path.join(_HERE, "lol-llama-finetuned"))

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--base-only",  action="store_true", help="skip LoRA adaptesr (compare base vs finetuned)")
parser.add_argument("--adapter",    default=ADAPTER_PATH)
parser.add_argument("--champion",   default=None, help="filter matches to this champion")
parser.add_argument("--index",      type=int, default=None, help="pick match by index instead of random")
parser.add_argument("--max-tokens", type=int, default=300)
args = parser.parse_args()

HF_TOKEN = os.getenv("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")
if HF_TOKEN:
    login(token=HF_TOKEN)

# ── Load match data ───────────────────────────────────────────────────────────
with open(MATCH_DATA, encoding="utf-8") as f:
    matches = json.load(f)

if args.champion:
    matches = [m for m in matches if m["champion"].lower() == args.champion.lower()]
    if not matches:
        print(f"No matches found for champion '{args.champion}'. Available: "
              f"{sorted(set(m['champion'] for m in json.load(open(MATCH_DATA))))}")
        sys.exit(1)

if args.index is not None:
    match = matches[args.index]
else:
    match = random.choice(matches)

def build_context(m):
    items = ", ".join(m.get("items", [])) or "unknown"
    spells = " and ".join(m.get("summoner_spells", [])) or "unknown"
    allies = ", ".join(m.get("ally_team", [])) or "unknown"
    enemies = ", ".join(m.get("enemy_team", [])) or "unknown"
    return (
        f"Champion: {m['champion']} | Position: {m['position'].capitalize()} | Patch: {m['patch']}\n"
        f"Result: {'win' if m['win'] else 'loss'} | Duration: {m['game_duration_min']} min\n"
        f"KDA: {m['kills']}/{m['deaths']}/{m['assists']} | CS: {m['cs']} | Gold: {m['gold']} | Level: {m['champ_level']}\n"
        f"Items: {items}\n"
        f"Keystone: {m.get('keystone_rune','?')} | Summoner spells: {spells}\n"
        f"Allies: {allies}\n"
        f"Enemies: {enemies}\n"
        f"Damage dealt: {m.get('damage_dealt','?')} | Vision score: {m.get('vision_score','?')} | "
        f"Kill participation: {int(m.get('kill_participation',0)*100)}% | Turret kills: {m.get('turret_kills','?')}"
    )

context = build_context(match)

# ── Load model ────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"[init] device={DEVICE}", flush=True)

if DEVICE == "cuda":
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
else:
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="cpu", torch_dtype=torch.float32, trust_remote_code=True
    )

if args.base_only or not os.path.isdir(args.adapter):
    if not args.base_only:
        print(f"[warn] adapter not found at '{args.adapter}' — using base model", flush=True)
    model = base
    label = "BASE model"
else:
    print(f"[init] loading LoRA adapter from {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    label = "FINE-TUNED model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer,
    return_full_text=False, max_new_tokens=args.max_tokens,
    do_sample=True, temperature=0.7, repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
)

# ── REPL ──────────────────────────────────────────────────────────────────────
BANNER = f"""
{'='*60}
  LoL Match Analyst  [{label}]
  Champion : {match['champion']} | Patch {match['patch']} | {match['position'].capitalize()}
  Result   : {'WIN' if match['win'] else 'LOSS'} in {match['game_duration_min']} min
  KDA      : {match['kills']}/{match['deaths']}/{match['assists']}
  Items    : {', '.join(match.get('items', []))}

  Ask anything about this match. Type 'exit' to quit.
  Type '/context' to see the full match data.
{'='*60}"""
print(BANNER, flush=True)

SUGGESTED = [
    f"What items and runes did {match['champion']} use in this Diamond EUW game on patch {match['patch']}?",
    f"How did {match['champion']} perform in this {match['position'].capitalize()} game on patch {match['patch']}?",
    f"Analyse the outcome of this {match['champion']} game — did they win or lose and why?",
    f"Describe the team matchup in this Diamond ranked game featuring {match['champion']}.",
]
print("Suggested questions:")
for i, q in enumerate(SUGGESTED, 1):
    print(f"  {i}. {q}")
print()

while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not q:
        continue
    if q.lower() in ("exit", "quit", "/exit"):
        break
    if q.lower() == "/context":
        print(f"\n{context}\n")
        continue
    if q.isdigit() and 1 <= int(q) <= len(SUGGESTED):
        q = SUGGESTED[int(q) - 1]
        print(f"  → {q}")

    messages = [
        {"role": "system", "content": (
            "You are a League of Legends analyst specializing in Diamond EUW ranked games. "
            "Give clear, natural, insightful answers of 2-4 sentences using the exact numbers from the match data. "
            "Vary your language and don't repeat the question back."
        )},
        {"role": "user", "content": f"Match data:\n{context}\n\nQuestion: {q}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = pipe(prompt)[0]["generated_text"].strip()
    print(f"\nBot: {out}\n")

print("bye.")
