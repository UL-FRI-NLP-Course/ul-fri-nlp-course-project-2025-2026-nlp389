# Natural Language Processing Course: League of Legends AI Assistant

A domain-specific AI assistant for **League of Legends** that leverages **Retrieval-Augmented Generation (RAG)** and **Fine-Tuning** to provide accurate, context-aware, and up-to-date insights about gameplay, champion performance, patch changes, and strategic recommendations.


---

## Running the Model on the HPC

The trained LoRA adapter and the FAISS index are already prepared on the HPC at:

```
/d/hpc/projects/onj_fri/nlp389/FIN
```

This folder contains everything `rag_chat.py` needs: the script itself, the `lol-llama-finetuned/` adapter, and the `patch_notes/faiss_index/` directory. 

### 1. Connect and navigate to the project folder

```bash
ssh <your-user>@hpc.fri.uni-lj.si
cd /d/hpc/projects/onj_fri/nlp389/FIN
```

### 2. Launch the chatbot with `srun`

Two modes are available:

**With history (default).**
The model remembers previous questions, so you can ask follow-ups like *"and what about Jungle?"* after asking about Top.

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=02:00:00 \
     --pty python rag_chat.py
```

**Without history (`--no-history`).**
Each question is answered independently. Use this for clean, isolated answers unaffected by previously asked questions.

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=40G --time=02:00:00 \
     --pty python rag_chat.py --no-history
```

Inside the REPL you can also use:
- `/sources` — show the retrieved chunks for the last question
- `/reset` — clear the conversation history
- `/save` — dump the session to `chat_session.json`
- `/exit` — quit

### 3. Example questions to try

The model is best at questions that touch the indexed knowledge base. Good examples (one per category) are:

- **Build data:** *What is the best first item for Jinx ADC?*
- **Leaderboard:** *Which top lane champions have the highest win rates this patch?*
- **Patch notes:** *Which keystone from the Sorcery tree replaced Phase Rush?*
- **Champion description:** *Why is Malphite considered a good pick against attack damage heavy champions?*

The full list of 30 evaluation questions is in `src/code/evaluate/full_questions.txt`.

---

## Running Locally

If you want to run the project on your own machine (you will need a CUDA GPU for reasonable performance):

### 1. Clone the repository

```bash
git clone https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2025-2026-nlp389.git
cd ul-fri-nlp-course-project-2025-2026-nlp389
```

### 2. Install dependencies

We recommend a fresh Python 3.10+ environment (conda or venv):

```bash
pip install "numpy<2.0"
pip install torch transformers bitsandbytes accelerate peft
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-huggingface
pip install sentence-transformers faiss-cpu tiktoken huggingface_hub trl
```

### 3. Set your HuggingFace token

The base model (`meta-llama/Meta-Llama-3-8B-Instruct`) is gated, so you need access on HuggingFace and a token in your environment:

```bash
export HF_API_KEY=hf_xxxxxxxxxxxxxxxxxx
```


### 4. Obtain the fine-tuned model

You have two options:

**Option A: Fine-tune yourself**
Run the fine-tuning script (requires GPU, ~4–6 hours on a single A100):
```bash
cd src/code/fine-tune
python model_finetune.py
```

**Option B: Copy from HPC**
If you have access to the project HPC directory, copy the existing adapter:
```bash
cp -r /d/hpc/projects/onj_fri/nlp389/FIN/lol-llama-finetuned ./lol-llama-finetuned
```

### 5. Build the FAISS index (first run only)

From `src/code/RAG/`:

```bash
cd src/code/RAG
python rag_finetuned.py
```

This indexes patch notes, champion/item descriptions, and Diamond-tier builds into `patch_notes/faiss_index/`.

### 6. Run the chatbot

```bash
python rag_chat.py             # multi-turn with history
python rag_chat.py --no-history   # single-turn, independent answers
```
This indexes patch notes, champion/item descriptions, and Diamond-tier builds into `patch_notes/faiss_index/`.

### 5. Run the chatbot

```bash
python rag_chat.py             # multi-turn with history
python rag_chat.py --no-history   # single-turn, independent answers
```




---

## Project Structure

```
.
├── LICENSE                                     # Project license
├── README.md                                  
│
├── src/                                        # Main source code and data
│   │
│   ├── code/                                   # Implementation scripts
│   │   │
│   │   ├── create_data/                        # Data collection & preprocessing
│   │   │   ├── getMatchData.py                 # Riot API match history fetcher
│   │   │   ├── getPlayerPuuids.py              # Resolve player names to PUUIDs
│   │   │   ├── tagsAndNamesDiamond.py          # Scrape Diamond player tags from op.gg
│   │   │   ├── scrape_patches.ipynb            # Patch-notes scraper
│   │   │   ├── get_build_data.py               # Pull u.gg per-champion build/matchup JSON
│   │   │   ├── convert_diamond_to_rag.py       # Convert u.gg builds into RAG chunks
│   │   │   ├── get_item_champ_desc.py          # Fetch champion/item data from Data Dragon
│   │   │   └── get_item_champ_desc.ipynb       # Notebook variant of the above
│   │   │
│   │   ├── finetunning/                        # Model fine-tuning
│   │   │   ├── generate_finetune_data.py       # Build instruction-response pairs from matches
│   │   │   ├── model_finetune.py               # QLoRA fine-tuning script
│   │   │   └── match_chat.py                   # Standalone match-analyst demo (no RAG)
│   │   │
│   │   ├── RAG/                                # Retrieval-augmented generation
│   │   │   ├── rag.py                          # Original single-source RAG (patch notes only)
│   │   │   ├── rag_finetuned.py                # Builds FAISS index over all data sources
│   │   │   ├── rag_chat.py                     # Interactive RAG chatbot (REPL)
│   │   │   ├── rag_chat_batch.py               # Batch QA wrapper for evaluation
│   │   │   └── run_rag_finetuned.sh            # SLURM launcher for index build
│   │   │
│   │   └── evaluate/                           # Evaluation pipeline
│   │       ├── full_questions.txt              # 30 evaluation questions
│   │       ├── ground_truth.json               # Reference answers + categories
│   │       ├── run_all_systems.sh              # Generate answers for all 5 systems    
│   │       ├── groq_answers.py                 # Groq Llama-3.3-70B answer generator
│   │       ├── llm_judge_groq.py               # Groq Llama-3.3-70B judge
│   │       ├── aggregate.py                    # Aggregate judged scores into report
│   │       ├── answers/                        # Per-model generated answers
│   │       │   ├── base.json
│   │       │   ├── base_rag.json
│   │       │   ├── finetuned.json
│   │       │   ├── finetuned_rag.json
│   │       │   └── llama70b.json
│   │       └── results/                        # Judged scores + final report
│   │           ├── judged_*.json
│   │           └── final_report.md
│   │
│   ├── data/                                   # Collected data
│   │   ├── diamond_leaderboards_all_23K.csv    # Diamond+ player rankings (~23K)
│   │   ├── diamond_puuids_100.csv              # Sample of 100 player PUUIDs
│   │   ├── match_data.json                     # Per-match Diamond ranked data
│   │   ├── diamond_16_6_named.json             # u.gg overview/matchups, patch 16.6 -> (26.6)
│   │   ├── diamond_16_7_named.json             # u.gg overview/matchups, patch 16.7 -> (26.7)
│   │   ├── diamond_16_8_named.json             # u.gg overview/matchups, patch 16.8 -> (26.8)
│   │   ├── diamond_16_9_named.json             # u.gg overview/matchups, patch 16.9 -> (26.9)
│   │   ├── diamond_builds.jsonl                # u.gg builds reformatted as RAG chunks
│   │   ├── champion_chunks.jsonl               # Champion descriptions (RAG chunks)
│   │   ├── item_chunks.jsonl                   # Item descriptions (RAG chunks)
│   │   ├── game_knowledge.jsonl                # General LoL knowledge chunks
│   │   ├── finetune_data_llm_8000.json         # Full SFT dataset (~8K pairs)
│   │   ├── finetune_data_llm_half.json         # Reduced SFT dataset (~4K pairs)
│   │   │
│   │   └── patch_notes/                        # Processed patch data
│   │       ├── all_patches.json
│   │       ├── rag_chunks.jsonl
│   │       └── rag_chunks_pretty.json
│   │
│   └── results/                                # Model training outputs & artifacts (but the meaningful ones are in the evaluate/results folder)
│       ├── finetune_test_results.json
│       └── rag_results.json
│
└── report/                                     # Project report
  
```

---

## Folder Details

### `/src/code/create_data/`
Scripts for collecting and preparing raw data: Riot Games API match history, op.gg player scraping, official patch-note scraping, u.gg build/matchup statistics, and Data Dragon champion/item descriptions.

### `/src/code/finetunning/`
Scripts for generating the synthetic instruction-response dataset from real Diamond matches and QLoRA fine-tuning of Llama-3-8B-Instruct on it.

### `/src/code/RAG/`
Retrieval-augmented generation system: FAISS index construction over patch notes, champion/item descriptions, and Diamond-tier build data, plus interactive and batch chat interfaces.

### `/src/code/evaluate/`
End-to-end evaluation pipeline: 30 curated questions with ground truth, answer generation for five system configurations, automated LLM-as-judge scoring (Groq Llama-3.3-70B), and aggregation into a final report.

### `/src/data/`
All raw and processed datasets — leaderboards, PUUIDs, match histories, u.gg snapshots per patch, fine-tuning data, and the RAG chunk files for each knowledge source.

### `/src/data/patch_notes/`
Processed patch notes segmented and formatted for the RAG retrieval system.

### `/src/results/`
Model evaluation outputs and the final fine-tuned LoRA adapter (with intermediate training checkpoints).

### `/report/`
LaTeX project documentation including the final report, bibliography, custom style file, and figures.
