import warnings
warnings.filterwarnings("ignore")
import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
import os

HF_TOKEN = os.getenv("HF_API_KEY")
login(token=HF_TOKEN)

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Model loaded.", flush=True)

print("Loading dataset...", flush=True)
with open("finetune_data_llm_half.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
print(f"Dataset size: {len(dataset)} examples", flush=True)
print("Sample:", dataset["text"][0][:300], flush=True)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

sft_config = SFTConfig(
    output_dir="./lol-llama-finetuned",
    dataset_text_field="text",
    max_length=512,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    fp16=False,
    bf16=True,
    logging_steps=25,
    save_steps=200,
    save_total_limit=2,
    dataloader_num_workers=4,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_config,
)

print("Trainable parameters:", flush=True)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} ({100*trainable/total:.4f}%)", flush=True)

print("Starting training...", flush=True)
trainer.train()

trainer.model.save_pretrained("./lol-llama-finetuned")
tokenizer.save_pretrained("./lol-llama-finetuned")
print("Model saved to ./lol-llama-finetuned", flush=True)