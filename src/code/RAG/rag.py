import warnings
warnings.filterwarnings("ignore")

import sys
import torch
import json
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from huggingface_hub import login
import os

HF_TOKEN = os.getenv("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")
login(token=HF_TOKEN)

print("=" * 60)
print(f"PyTorch        : {torch.__version__}")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version   : {torch.version.cuda}")
    print(f"GPU count      : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}          : {props.name}")
        print(f"VRAM {i}         : {props.total_memory / 1e9:.1f} GB")
    DEVICE = "cuda"
else:
    print("WARNING: No CUDA GPU — running on CPU (will be very slow)")
    DEVICE = "cpu"
print(f"Using device   : {DEVICE.upper()}")
print("=" * 60)
sys.stdout.flush()

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

if DEVICE == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("Loading model in 4-bit NF4 quantization on CUDA...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    print("Loading model in float32 on CPU (no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Model loaded.")

if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved  = torch.cuda.memory_reserved(0) / 1e9
    print(f"VRAM after model load: {allocated:.1f} GB allocated / {reserved:.1f} GB reserved")
sys.stdout.flush()

# Build generation pipeline 
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
import re

def clean_stats(text: str) -> str:
    # "80 / 150 / 220 / 290 / 360" → "80 / 150 / ... / 360" stays but ⇒ arrows get replaced
    # Replace "X ⇒ Y" with "from X to Y"
    text = re.sub(r'(\S+)\s*⇒\s*(\S+)', r'from \1 to \2', text)
    # Replace "X → Y" with "from X to Y"  
    text = re.sub(r'(\S+)\s*→\s*(\S+)', r'from \1 to \2', text)
    return text



# Plain chain (no RAG) 
def make_plain_prompt(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

plain_chain = RunnableLambda(make_plain_prompt) | llm | StrOutputParser()

# Example: Anivia is NOT in the patch data — the plain LLM will hallucinate,
# hopefully RAG will correctly say it has no information. Intentional contrast.
print("\n=== WITHOUT RAG — Llama-3 from memory only ===")
print(plain_chain.invoke("Is Anivia viable in the current patch? What changed for her recently?"))
sys.stdout.flush()

print("\nLoading patch notes chunks...")
chunks = []
with open("patch_notes/rag_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        chunks.append(Document(
            page_content=c["text"],
            metadata={
                "subject":       c["subject"],
                "section":       c["section"],
                "patch_version": c["patch_version"],
                "patch_date":    c["patch_date"],
                "change_type":   c["change_type"],
                "source":        c["patch_url"],
            }
        ))

print(f"Loaded {len(chunks)} chunks")
print(f"Example: {chunks[0].page_content[:200]}")
print(f"Metadata: {chunks[0].metadata}")
sys.stdout.flush()

# Embedding model 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
print(f"\nLoading embedding model on {DEVICE.upper()}...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding model loaded.")

# Cosine similarity sanity check — using champions confirmed in the data
test_sentences = [
    "Kassadin health growth reduced — nerf",
    "Kassadin E slow decreased — nerf",
    "Garen movement speed buff — buff",
]
vecs = np.array(embeddings.embed_documents(test_sentences))
sim  = vecs @ vecs.T
print("\nCosine similarity check:")
for i, s in enumerate(test_sentences):
    row = "  ".join(f"{sim[i,j]:.3f}" for j in range(len(test_sentences)))
    print(f"  {s[:45]:<45} {row}")
sys.stdout.flush()

# FAISS vector store
print(f"\nIndexing {len(chunks)} chunks into FAISS...")
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"Done. Index: {vectorstore.index.ntotal} vectors, dim {vectorstore.index.d}")
vectorstore.save_local("patch_notes/faiss_index")
print("Index saved to patch_notes/faiss_index/")
sys.stdout.flush()

#Retrieval demo
sim_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5})

query = "What changed for Kassadin in recent patches?"
print("\n=== Similarity Search ===")
for i, doc in enumerate(sim_retriever.invoke(query)):
    print(f"[{i+1}] {doc.metadata['patch_version']} | {doc.metadata['subject']} | {doc.metadata['change_type']}")
    print(f"     {doc.page_content[:150]}\n")

print("=== MMR Search ===")
for i, doc in enumerate(mmr_retriever.invoke(query)):
    print(f"[{i+1}] {doc.metadata['patch_version']} | {doc.metadata['subject']} | {doc.metadata['change_type']}")
    print(f"     {doc.page_content[:150]}\n")
sys.stdout.flush()

# RAG chain (single-turn)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6},
)

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{d.metadata['patch_version']} | {d.metadata['section']} | {d.metadata['subject']} | {d.metadata['change_type']}]\n{d.page_content}"
        for d in docs
    )

def make_rag_prompt(inputs: dict) -> str:
    system = (
        "You are a League of Legends expert assistant. "
        "Answer questions about champion performance, patch changes, item builds, rune choices, and meta viability "
        "using ONLY the patch notes context provided below.\n"
        "If the answer is not in the context, say: \"I don't have that information in the current patch data.\"\n"
        "Always mention which patch version the information comes from.\n"
        "If multiple patches are relevant, summarise how things changed over time.\n"
        "You MUST explain all numerical changes as plain English sentences. "
        "NEVER write stat values in the form 'X ⇒ Y', 'X → Y', or slash-separated lists like '6/5/4/3'. "
        "ALWAYS write full sentences like 'the cooldown was reduced from 10 seconds to 6 seconds at max rank'.\n\n"
        f"CONTEXT:\n{inputs['context']}"
    )
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
    answer = clean_stats(rag_chain.invoke(question))  # ← wrap here
    print(answer)
    print("=" * 60)
    results[key] = {"question": question, "answer": answer}
    sys.stdout.flush()
    return answer

# RAG questions (all confirmed in the data)
ask("kassadin_viability",
    "Is Kassadin still strong in mid lane? What nerfs did he receive?")

ask("gwen_nerfs",
    "What nerfs did Gwen receive across the 26.x patches and why?")

ask("item_changes",
    "What changes were made to Hubris and other items in the 26.x patches?")

ask("seraphine_bugfix",
    "Are there any bug fixes related to Seraphine?")

# Anivia intentionally NOT in data 
ask("anivia_viability",
    "Is Anivia viable right now? What changes did she receive?")

# ── Without vs With RAG comparison ──────────────────────────
print("\n=== WITHOUT RAG ===")
q = "What changed for Varus in League of Legends patch 26.2?"
results["no_rag"] = {"question": q, "answer": plain_chain.invoke(q)}
print(results["no_rag"]["answer"])
sys.stdout.flush()

print("\n=== WITH RAG ===")
results["with_rag"] = {"question": q, "answer": rag_chain.invoke(q)}
print(results["with_rag"]["answer"])
sys.stdout.flush()

# Conversational RAG (manual history-aware retrieval)

def rewrite_question(question: str, history: list) -> str:
    """Rewrite a follow-up question into a standalone question using chat history."""
    if not history:
        return question
    history_text = ""
    for msg in history:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            # Truncate long answers to keep the prompt short
            history_text += f"Assistant: {msg.content[:300]}\n"
    messages = [
        {"role": "system", "content": (
            "Rewrite the follow-up question as a standalone question using the conversation history. "
            "Output ONLY the rewritten question. No explanation, no prefix, nothing else."
        )},
        {"role": "user", "content": (
            f"Conversation:\n{history_text}\n"
            f"Follow-up: {question}\n\n"
            "Standalone question:"
        )},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rewritten = (RunnableLambda(lambda _: prompt) | llm | StrOutputParser()).invoke({})
    # Take only the first line — prevents the model from continuing into an answer
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
    context_block = (
        f"CONTEXT:\n{inputs['context']}\n\nCONVERSATION SO FAR:\n{history_text}"
        if history_text else
        f"CONTEXT:\n{inputs['context']}"
    )
    system = (
        "You are a League of Legends expert assistant. "
        "Answer using ONLY the patch notes context below. "
        "Always cite the patch version. "
        "You MUST explain all numerical changes as plain English sentences. "
        "NEVER write stat values in the form 'X ⇒ Y', 'X → Y', or slash-separated lists like '6/5/4/3'. "
        "ALWAYS write full sentences like 'the cooldown was reduced from 10 seconds to 6 seconds at max rank'. "
        "If the answer is not in the context, say: "
        "\"I don't have that information in the current patch data.\"\n\n"
        f"{context_block}"
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

print("\n=== CONVERSATIONAL RAG ===")
chat("What happened to Lillia across the recent patches?")
chat("So was she nerfed too hard? What did Riot say about it?")
chat("What item or build changes in those patches might have affected her performance?")

results["conversation"] = conv_results

with open("rag_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nAll results saved to rag_results.json")
print("DONE!")
sys.stdout.flush()