"""
RAG on top of the LoRA-fine-tuned Llama-3-8B-Instruct.

How it works:
  1. Load the base model `meta-llama/Meta-Llama-3-8B-Instruct` in 4-bit.
  2. Attach the LoRA adapter saved by `model_finetune.py` (./lol-llama-finetuned).
  3. Build (or reuse a saved) FAISS index over the patch-notes chunks.
  4. Run plain / RAG / conversational-RAG flows and save the answers.

Note: there is no "merged final model" to save. The deployable artifact is:
    - the LoRA adapter folder  (./lol-llama-finetuned)
    - the FAISS index          (patch_notes/faiss_index/)
Together they ARE the fine-tuned + RAG system.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import re

import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from huggingface_hub import login

# ── Config ──────────────────────────────────────────────────────────────────
BASE_MODEL    = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH  = os.environ.get("ADAPTER_PATH", "./lol-llama-finetuned")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH   = "patch_notes/rag_chunks.jsonl"
CHAMP_CHUNKS   = os.environ.get("CHAMP_CHUNKS",   "champion_chunks.jsonl")
ITEM_CHUNKS    = os.environ.get("ITEM_CHUNKS",    "item_chunks.jsonl")
DIAMOND_CHUNKS = os.environ.get("DIAMOND_CHUNKS", "diamond_builds.jsonl")
FAISS_PATH    = "patch_notes/faiss_index"
RESULTS_PATH  = "builds_champ_item_patch_notes_rag_finetuned_results.json"

HF_TOKEN = os.getenv("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")
if HF_TOKEN:
    login(token=HF_TOKEN)

# ── Device banner ───────────────────────────────────────────────────────────
print("=" * 60)
print(f"PyTorch        : {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version   : {torch.version.cuda}")
    print(f"GPU count      : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}          : {props.name}  ({props.total_memory / 1e9:.1f} GB)")
    DEVICE = "cuda"
else:
    print("WARNING: No CUDA GPU — running on CPU (will be very slow)")
    DEVICE = "cpu"
print(f"Using device   : {DEVICE.upper()}")
print(f"Adapter path   : {ADAPTER_PATH}  (exists={os.path.isdir(ADAPTER_PATH)})")
print("=" * 60)
sys.stdout.flush()

# ── Load base + LoRA adapter ────────────────────────────────────────────────
if DEVICE == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("Loading base model in 4-bit NF4 on CUDA...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    print("Loading base model in float32 on CPU (no quantization)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

if os.path.isdir(ADAPTER_PATH):
    print(f"Attaching LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("Adapter attached. (Using fine-tuned model.)")
else:
    print(f"WARNING: adapter dir '{ADAPTER_PATH}' not found — running BASE model only.")
    model = base_model

# Prefer tokenizer saved with the adapter (chat template / special tokens stay in sync);
# fall back to base tokenizer if not present.
tok_src = ADAPTER_PATH if os.path.isfile(os.path.join(ADAPTER_PATH, "tokenizer_config.json")) else BASE_MODEL
tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
tokenizer.pad_token   = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"Tokenizer loaded from: {tok_src}")

if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved  = torch.cuda.memory_reserved(0) / 1e9
    print(f"VRAM after model load: {allocated:.1f} GB allocated / {reserved:.1f} GB reserved")
sys.stdout.flush()

# ── Generation pipeline ─────────────────────────────────────────────────────
eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

hf_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=2048,
    do_sample=False,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=[tokenizer.eos_token_id, eot_token_id],
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
print("Pipeline ready.")
sys.stdout.flush()


def clean_stats(text: str) -> str:
    text = re.sub(r'(\S+)\s*⇒\s*(\S+)', r'from \1 to \2', text)
    text = re.sub(r'(\S+)\s*→\s*(\S+)', r'from \1 to \2', text)
    return text


# ── Plain (no-RAG) chain — useful for ablation ──────────────────────────────
def make_plain_prompt(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

plain_chain = RunnableLambda(make_plain_prompt) | llm | StrOutputParser()

print("\n=== FINETUNED, WITHOUT RAG ===")
print(plain_chain.invoke("Is Anivia viable in the current patch? What changed for her recently?"))
sys.stdout.flush()

# ── Embeddings + FAISS (reuse if already built) ─────────────────────────────
print(f"\nLoading embedding model on {DEVICE.upper()}...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding model loaded.")

if os.path.isdir(FAISS_PATH):
    print(f"Loading existing FAISS index from {FAISS_PATH}...")
    vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print(f"Loaded index: {vectorstore.index.ntotal} vectors, dim {vectorstore.index.d}")
else:
    chunks = []

    print(f"Loading patch notes chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks.append(Document(
                page_content=c["text"],
                metadata={
                    "subject":       c.get("subject", ""),
                    "section":       c.get("section", "patch_notes"),
                    "patch_version": c.get("patch_version", ""),
                    "patch_date":    c.get("patch_date", ""),
                    "change_type":   c.get("change_type", ""),
                    "source":        c.get("patch_url", ""),
                },
            ))
    print(f"  {len(chunks)} patch note chunks loaded.")

    for extra_path, source_type in [(CHAMP_CHUNKS, "champion"), (ITEM_CHUNKS, "item"), (DIAMOND_CHUNKS, "diamond_build")]:
        if os.path.isfile(extra_path):
            before = len(chunks)
            with open(extra_path, "r", encoding="utf-8") as f:
                for line in f:
                    c = json.loads(line)
                    meta = c.get("metadata", {})
                    # diamond_build chunks store champion/role/patch in metadata
                    subj = meta.get("champion", c.get("subject", ""))
                    role = meta.get("role", "")
                    patch = meta.get("patch", "")
                    if role:
                        subj = f"{subj} ({role})"
                    chunks.append(Document(
                        page_content=c["text"],
                        metadata={
                            "subject":       subj,
                            "section":       source_type,
                            "patch_version": patch,
                            "patch_date":    "",
                            "change_type":   source_type,
                            "source":        source_type,
                        },
                    ))
            print(f"  {len(chunks) - before} {source_type} chunks loaded from {extra_path}.")
        else:
            print(f"  [skip] {extra_path} not found — run get_item_champ_desc.py or convert_diamond_to_rag.py first.")

    print(f"Indexing {len(chunks)} total chunks into FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(FAISS_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_PATH)
    print(f"Index saved to {FAISS_PATH}/")
sys.stdout.flush()

# ── RAG chain (single-turn) ─────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6},
)

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{d.metadata['patch_version']} | {d.metadata['section']} | {d.metadata['subject']} | {d.metadata['change_type']}]\n{d.page_content}"
        for d in docs
    )

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


def make_rag_prompt(inputs: dict) -> str:
    system = f"{SYSTEM_RULES}\n\nPatch notes context:\n{inputs['context']}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": inputs["question"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RunnableLambda(make_rag_prompt)
    | llm
    | StrOutputParser()
)

results = {}

def ask(key: str, question: str) -> str:
    print(f"\nQuestion [{key}]: {question}")
    print("-" * 60)
    answer = clean_stats(rag_chain.invoke(question))
    print(answer)
    print("=" * 60)
    results[key] = {"question": question, "answer": answer}
    sys.stdout.flush()
    return answer

ask("kassadin_viability",
    "Is Kassadin still strong in mid lane? What nerfs did he receive?")
ask("gwen_nerfs",
    "What nerfs did Gwen receive across the 26.x patches and why?")
ask("item_changes",
    "What changes were made to Hubris and other items in the 26.x patches?")
ask("seraphine_bugfix",
    "Are there any bug fixes related to Seraphine?")
ask("anivia_viability",
    "Is Anivia viable right now? What changes did she receive?")

print("\n=== FINETUNED, WITHOUT RAG ===")
q = "What changed for Varus in League of Legends patch 26.2?"
results["no_rag"] = {"question": q, "answer": plain_chain.invoke(q)}
print(results["no_rag"]["answer"])

print("\n=== FINETUNED, WITH RAG ===")
results["with_rag"] = {"question": q, "answer": rag_chain.invoke(q)}
print(results["with_rag"]["answer"])
sys.stdout.flush()

# ── Conversational RAG ──────────────────────────────────────────────────────
def rewrite_question(question: str, history: list) -> str:
    if not history:
        return question
    history_text = ""
    for msg in history:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content[:300]}\n"
    messages = [
        {"role": "system", "content": (
            "Rewrite the follow-up question as a standalone question using the conversation history. "
            "Output ONLY the rewritten question. No explanation, no prefix, nothing else."
        )},
        {"role": "user", "content": (
            f"Conversation:\n{history_text}\nFollow-up: {question}\n\nStandalone question:"
        )},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rewritten = (RunnableLambda(lambda _: prompt) | llm | StrOutputParser()).invoke({})
    rewritten = rewritten.strip().split("\n")[0].strip()
    return rewritten if rewritten else question


def build_conv_inputs(inputs: dict) -> dict:
    standalone_q = rewrite_question(inputs["question"], inputs.get("chat_history", []))
    print(f"  [rewritten query]: {standalone_q}")
    return {
        "context":      format_docs(retriever.invoke(standalone_q)),
        "question":     inputs["question"],
        "chat_history": inputs.get("chat_history", []),
    }


def make_conv_rag_prompt(inputs: dict) -> str:
    history_text = ""
    for msg in inputs.get("chat_history", []):
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content[:300]}\n"
    extras = (
        f"\n\nConversation so far:\n{history_text}"
        "Use the conversation only to understand follow-up references; "
        "ground every factual claim in the patch notes above."
        if history_text else ""
    )
    system = (
        f"{SYSTEM_RULES}\n\n"
        f"Patch notes context:\n{inputs['context']}"
        f"{extras}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": inputs["question"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


conv_rag_chain = (
    RunnableLambda(build_conv_inputs)
    | RunnableLambda(make_conv_rag_prompt)
    | llm
    | StrOutputParser()
)

chat_history = []
conv_results = []

def chat(question: str) -> str:
    print(f"\nYou: {question}")
    answer = clean_stats(conv_rag_chain.invoke({"question": question, "chat_history": chat_history}))
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    conv_results.append({"user": question, "bot": answer})
    print(f"Bot: {answer}")
    sys.stdout.flush()
    return answer

print("\n=== CONVERSATIONAL RAG (FINETUNED) ===")
chat("What happened to Lillia across the recent patches?")
chat("So was she nerfed too hard? What did Riot say about it?")
chat("What item or build changes in those patches might have affected her performance?")

results["conversation"] = conv_results

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nAll results saved to {RESULTS_PATH}")
print("DONE!")
sys.stdout.flush()
