import warnings
warnings.filterwarnings("ignore")
import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from huggingface_hub import login

HF_TOKEN = os.getenv("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")
login(token=HF_TOKEN)
print("HuggingFace login OK.", flush=True)

MIN_GAME_DURATION = 10.0
MIN_CS = 15

# Load model — same as rag2.py 
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

if DEVICE == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="cpu", torch_dtype=torch.float32, trust_remote_code=True
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

hf_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    batch_size=4,
)
print("Model loaded.", flush=True)  

# Build context string
def build_context(m):
    return (
        f"Champion: {m['champion']} | Position: {m['position'].capitalize()} | Patch: {m['patch']}\n"
        f"Result: {'win' if m['win'] else 'loss'} | Duration: {m['game_duration_min']} min\n"
        f"KDA: {m['kills']}/{m['deaths']}/{m['assists']} | CS: {m['cs']} | Gold: {m['gold']} | Level: {m['champ_level']}\n"
        f"Items: {', '.join(m['items'])}\n"
        f"Keystone: {m['keystone_rune']} | Summoner spells: {' and '.join(m['summoner_spells'])}\n"
        f"Allies: {', '.join(m['ally_team'])}\n"
        f"Enemies: {', '.join(m['enemy_team'])}\n"
        f"Damage dealt: {m['damage_dealt']} | Vision score: {m['vision_score']} | "
        f"Kill participation: {int(m['kill_participation']*100)}% | Turret kills: {m['turret_kills']}"
    )

def generate_answer(question, context):
    messages = [
        {"role": "system", "content": (
            "You are a League of Legends analyst specializing in Diamond EUW ranked games. "
            "Give clear, natural, insightful answers of 2-4 sentences using the exact numbers from the match data. "
            "Vary your language and don't repeat the question back."
        )},
        {"role": "user", "content": f"Match data:\n{context}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = hf_pipeline(prompt)
    return result[0]["generated_text"].strip()

# ── Questions per match 
def get_questions(m):
    champ = m["champion"]
    patch = m["patch"]
    pos   = m["position"].capitalize()
    return [
        f"What items and runes did {champ} use in this Diamond EUW game on patch {patch}?",
        f"How did {champ} perform in this {pos} game on patch {patch}?",
        f"Analyse the outcome of this {champ} game — did they win or lose and why?",
        f"Describe the team matchup in this Diamond ranked game featuring {champ}.",
    ]

def main(input_json, output_json):
    with open(input_json, "r", encoding="utf-8") as f:
        matches = json.load(f)
    print(f"Loaded {len(matches)} matches.", flush=True)

    existing = []
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Resuming — {len(existing)} examples already done.", flush=True)
    except FileNotFoundError:
        pass

    done_matches      = len(existing) // 4
    matches_to_process = matches[done_matches:]
    all_examples      = list(existing)
    skipped           = 0

    for i, m in enumerate(matches_to_process):
        if m.get("game_duration_min", 0) < MIN_GAME_DURATION:
            skipped += 1
            continue
        if m.get("cs", 0) < MIN_CS and m.get("position") != "UTILITY":
            skipped += 1
            continue

        context   = build_context(m)
        questions = get_questions(m)

        for question in questions:
            answer = generate_answer(question, context)
            all_examples.append({
                "text": f"### Human: {question}\n\n### Context: {context}\n\n### Assistant: {answer}"
            })

        # Save checkpoint every 25 matches
        if (i + 1) % 25 == 0:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(all_examples, f, indent=2, ensure_ascii=False)
            print(f"[checkpoint] {len(all_examples)} examples | match {i+1}/{len(matches_to_process)}", flush=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(all_examples)} examples saved to {output_json}. Skipped {skipped} bad games.", flush=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
