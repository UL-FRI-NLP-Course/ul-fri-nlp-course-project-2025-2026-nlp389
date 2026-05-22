"""
Interactive chat CLI for the fine-tuned + RAG system.

Loads everything ONCE (base model + LoRA adapter + saved FAISS index),
then enters a REPL that answers questions from stdin until you type 'exit'.

Prerequisites (run these once, in any order):
    1. python model_finetune.py       -> creates ./lol-llama-finetuned/
    2. python rag_finetuned.py        -> creates patch_notes/faiss_index/
                                         (or just run the indexing portion)

Usage:
    python rag_chat.py
    python rag_chat.py --no-history     # single-turn mode
    python rag_chat.py --base-only      # skip the LoRA adapter (for ablation)

Commands inside the REPL:
    /reset    clear conversation history
    /sources  show retrieved chunks for the last question
    /save     dump the current session to chat_session.json
    /exit     quit (also: 'exit', 'quit', Ctrl-D)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
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

import re


# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--base-only",  action="store_true", help="don't load the LoRA adapter")
parser.add_argument("--no-history", action="store_true", help="single-turn mode (no follow-up rewriting)")
parser.add_argument("--adapter",    default=os.environ.get("ADAPTER_PATH", "./lol-llama-finetuned"))
parser.add_argument("--faiss",      default="patch_notes/faiss_index")
parser.add_argument("--k",          type=int, default=5)
parser.add_argument("--max-tokens", type=int, default=512)
args = parser.parse_args()

BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

HF_TOKEN = os.getenv("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")
if HF_TOKEN:
    login(token=HF_TOKEN)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] device={DEVICE}  adapter={'NO' if args.base_only else args.adapter}", flush=True)

# ── Model + adapter ──────────────────────────────────────────────────────
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
    if not args.base_only:
        print(f"[warn] adapter dir '{args.adapter}' not found — falling back to base model", flush=True)
    model = base_model
    tok_src = BASE_MODEL
else:
    print(f"[init] attaching LoRA adapter from {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(base_model, args.adapter)
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
print(f"[init] loading embeddings ({EMBED_MODEL})", flush=True)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

if not os.path.isdir(args.faiss):
    sys.exit(f"[err] FAISS index not found at '{args.faiss}'. "
             f"Run rag_finetuned.py first to build it.")

print(f"[init] loading FAISS index from {args.faiss}", flush=True)
vectorstore = FAISS.load_local(args.faiss, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": args.k, "fetch_k": max(20, args.k * 4), "lambda_mult": 0.6},
)
print(f"[init] index ready: {vectorstore.index.ntotal} vectors", flush=True)

# ── Prompt + chains ──────────────────────────────────────────────────────
SYSTEM_RULES = (
    "You are a League of Legends expert assistant. "
    "Answer using only the patch notes provided in the context.\n\n"
    "Rules:\n"
    "- Be clear, concise, and accurate.\n"
    "- Cite the patch version for every change you mention.\n"
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

def clean_stats(text: str) -> str:
    text = re.sub(r'(\S+)\s*⇒\s*(\S+)', r'from \1 to \2', text)
    text = re.sub(r'(\S+)\s*→\s*(\S+)', r'from \1 to \2', text)
    return text

def render_prompt(question: str, context: str, history_text: str = "") -> str:
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

def rewrite_question(question: str, history) -> str:
    if not history:
        return question
    hist = ""
    for m in history:
        if isinstance(m, HumanMessage): hist += f"User: {m.content}\n"
        elif isinstance(m, AIMessage):  hist += f"Assistant: {m.content[:300]}\n"
    messages = [
        {"role": "system", "content":
            "Rewrite the follow-up question as a standalone question using the conversation history. "
            "Output ONLY the rewritten question."},
        {"role": "user", "content": f"Conversation:\n{hist}\nFollow-up: {question}\n\nStandalone question:"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = (RunnableLambda(lambda _: prompt) | llm | StrOutputParser()).invoke({})
    return out.strip().split("\n")[0].strip() or question


# ── REPL ─────────────────────────────────────────────────────────────────
chat_history = []
session = []
last_docs = []

BANNER = (
    "\n" + "=" * 60 +
    "\nLoL Patch-Notes RAG Chat"
    f"\n  model: Llama-3-8B-Instruct"
    f" {'(BASE only)' if (args.base_only or not os.path.isdir(args.adapter)) else '+ LoRA adapter'}"
    f"\n  index: {vectorstore.index.ntotal} chunks   k={args.k}"
    f"\n  history: {'OFF' if args.no_history else 'ON'}"
    "\n  commands: /reset  /sources  /save  /exit"
    "\n" + "=" * 60
)
print(BANNER, flush=True)

def answer_one(question: str) -> str:
    global last_docs
    standalone = question if args.no_history else rewrite_question(question, chat_history)
    if standalone != question:
        print(f"  [rewritten]: {standalone}", flush=True)
    last_docs = retriever.invoke(standalone)
    prompt = render_prompt(
        question,
        format_docs(last_docs),
        history_text="".join(
            (f"User: {m.content}\n" if isinstance(m, HumanMessage) else f"Assistant: {m.content[:300]}\n")
            for m in chat_history
        ) if not args.no_history else "",
    )
    raw = (RunnableLambda(lambda _: prompt) | llm | StrOutputParser()).invoke({})
    return clean_stats(raw).strip()

while True:
    try:
        q = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not q:
        continue
    cmd = q.lower()
    if cmd in ("/exit", "exit", "quit", "/quit"):
        break
    if cmd == "/reset":
        chat_history.clear()
        print("[history cleared]")
        continue
    if cmd == "/sources":
        if not last_docs:
            print("[no retrieval yet]")
        else:
            for i, d in enumerate(last_docs, 1):
                print(f"[{i}] {d.metadata.get('patch_version','?')} | {d.metadata.get('subject','?')} "
                      f"| {d.metadata.get('change_type','?')}")
                print(f"    {d.page_content[:160]}")
        continue
    if cmd == "/save":
        with open("chat_session.json", "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        print("[saved chat_session.json]")
        continue

    a = answer_one(q)
    print(f"\nBot: {a}")
    if not args.no_history:
        chat_history.append(HumanMessage(content=q))
        chat_history.append(AIMessage(content=a))
    session.append({"user": q, "bot": a})

# auto-save on exit if anything happened
if session:
    with open("chat_session.json", "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    print(f"[saved chat_session.json — {len(session)} turns]")
print("bye.")
