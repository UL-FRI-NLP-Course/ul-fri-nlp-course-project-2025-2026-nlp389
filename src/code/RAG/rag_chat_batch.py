"""
Non-interactive batch wrapper around rag_chat logic.

Reads questions from a plain text file (one question per line, # = comment)
and writes answers to a JSON file. Suitable for running under SLURM where
there is no interactive terminal.

Usage:
    python rag_chat_batch.py --questions questions.txt --out chat_batch_results.json
    python rag_chat_batch.py --base-only      # ablation: base model only
    python rag_chat_batch.py --no-history     # ablation: no conversation history
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from huggingface_hub import login

# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--questions",  default="questions.txt",           help="text file with one question per line")
parser.add_argument("--out",        default="diamond_builds_chat_batch_results.json", help="output JSON file")
parser.add_argument("--base-only",  action="store_true", help="skip LoRA adapter (base model only)")
parser.add_argument("--no-history", action="store_true", help="answer each question independently")
parser.add_argument("--no-rag",     action="store_true", help="skip FAISS retrieval (pure LLM, no context)")
parser.add_argument("--adapter",    default=os.environ.get("ADAPTER_PATH", "./lol-llama-finetuned"))
parser.add_argument("--faiss",      default="patch_notes/faiss_index")
parser.add_argument("--k",          type=int, default=7)
parser.add_argument("--max-tokens", type=int, default=512)
args = parser.parse_args()

BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HF_TOKEN = os.getenv("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")
if HF_TOKEN:
    login(token=HF_TOKEN)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] device={DEVICE}  adapter={'NO' if args.base_only else args.adapter}", flush=True)

# ── Model ────────────────────────────────────────────────────────────────
if DEVICE == "cuda":
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="cpu", torch_dtype=torch.float32, trust_remote_code=True,
    )

if args.base_only or not os.path.isdir(args.adapter):
    model  = base_model
    tok_src = BASE_MODEL
else:
    print(f"[init] attaching LoRA adapter from {args.adapter}", flush=True)
    model   = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    tok_src = args.adapter if os.path.isfile(os.path.join(args.adapter, "tokenizer_config.json")) else BASE_MODEL

tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
hf_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=args.max_tokens,
    do_sample=False,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=[tokenizer.eos_token_id, eot_id],
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ── Vector store ─────────────────────────────────────────────────────────
print(f"[init] loading embeddings", flush=True)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

if args.no_rag:
    print("[init] --no-rag enabled: skipping FAISS retrieval", flush=True)
    retriever = None
else:
    if not os.path.isdir(args.faiss):
        sys.exit(f"[err] FAISS index not found at '{args.faiss}'. Run rag_finetuned.py first.")
    vectorstore = FAISS.load_local(args.faiss, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": args.k, "fetch_k": max(20, args.k * 4), "lambda_mult": 0.6},
    )
    print(f"[init] index: {vectorstore.index.ntotal} vectors", flush=True)

# ── Prompt helpers ────────────────────────────────────────────────────────
SYSTEM_RULES = (
    "You are a League of Legends expert assistant. "
    "Answer using only the context provided below.\n\n"
    "The context may include:\n"
    "- Patch notes (balance changes, new features)\n"
    "- Champion descriptions (abilities, lore)\n"
    "- Item descriptions (stats, passives)\n"
    "- Diamond-tier build data (runes, items, skill order, win rates per champion/role/patch)\n\n"
    "Rules:\n"
    "- Be clear, concise, and accurate.\n"
    "- Cite the patch version for every change you mention.\n"
    "- For build data, mention the patch and win rate when available.\n"
    "- A win rate above 52% is strong, 50-52% is average, below 49% is weak.\n"
    "- If several patches are relevant, summarize how things evolved over time.\n"
    "- Rewrite numeric changes as natural sentences "
    "(e.g. 'the cooldown dropped from 10s to 6s at max rank'); "
    "do not output raw formats like 'X => Y', 'X -> Y', or '6/5/4/3'.\n"
    "- If the context does not contain the answer, reply exactly: "
    "I don't have that information."
)

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('patch_version','?')} | {d.metadata.get('section','?')} | "
        f"{d.metadata.get('subject','?')} | {d.metadata.get('change_type','?')}]\n{d.page_content}"
        for d in docs
    )

def clean_stats(text):
    text = re.sub(r'(\S+)\s*⇒\s*(\S+)', r'from \1 to \2', text)
    text = re.sub(r'(\S+)\s*→\s*(\S+)', r'from \1 to \2', text)
    return text

def render_prompt(question, context, history_text=""):
    extras = (
        f"\n\nConversation so far:\n{history_text}"
        "Use the conversation only to understand follow-up references; "
        "ground every factual claim in the patch notes above."
        if history_text else ""
    )
    system = f"{SYSTEM_RULES}\n\nPatch notes context:\n{context}{extras}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def rewrite_question(question, history):
    if not history:
        return question
    hist = ""
    for m in history:
        if isinstance(m, HumanMessage): hist += f"User: {m.content}\n"
        elif isinstance(m, AIMessage):  hist += f"Assistant: {m.content[:300]}\n"
    messages = [
        {"role": "system", "content":
            "Rewrite the follow-up question as a standalone question. Output ONLY the rewritten question."},
        {"role": "user", "content": f"Conversation:\n{hist}\nFollow-up: {question}\n\nStandalone question:"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = (RunnableLambda(lambda _: prompt) | llm | StrOutputParser()).invoke({})
    return out.strip().split("\n")[0].strip() or question

def answer_one(question, history):
    standalone = question if args.no_history else rewrite_question(question, history)
    if standalone != question:
        print(f"  [rewritten]: {standalone}", flush=True)
    if args.no_rag:
        docs = []
    else:
        docs = retriever.invoke(standalone)
    hist_text = ""
    if not args.no_history:
        for m in history:
            if isinstance(m, HumanMessage): hist_text += f"User: {m.content}\n"
            elif isinstance(m, AIMessage):  hist_text += f"Assistant: {m.content[:300]}\n"
    context = format_docs(docs) if docs else "(no context provided; answer from your general knowledge)"
    prompt  = render_prompt(question, context, hist_text)
    raw     = (RunnableLambda(lambda _: prompt) | llm | StrOutputParser()).invoke({})
    sources = [
        f"{d.metadata.get('patch_version','?')} | {d.metadata.get('subject','?')} | {d.metadata.get('change_type','?')}"
        for d in docs
    ]
    return clean_stats(raw).strip(), sources

# ── Load questions ────────────────────────────────────────────────────────
if not os.path.isfile(args.questions):
    sys.exit(f"[err] questions file '{args.questions}' not found.")

questions = []
with open(args.questions, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            questions.append(line)

print(f"[run] {len(questions)} questions loaded from {args.questions}", flush=True)

# ── Answer loop ───────────────────────────────────────────────────────────
history = []
results = []

for i, q in enumerate(questions, 1):
    print(f"\n[{i}/{len(questions)}] {q}", flush=True)
    print("-" * 60, flush=True)
    answer, sources = answer_one(q, history)
    print(answer, flush=True)
    results.append({"question": q, "answer": answer, "sources": sources})
    if not args.no_history:
        history.append(HumanMessage(content=q))
        history.append(AIMessage(content=answer))

# ── Save ──────────────────────────────────────────────────────────────────
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n[done] {len(results)} answers saved to {args.out}")
