# Natural Language Processing Course: League of Legends AI Assistant

A domain-specific AI assistant for **League of Legends** that leverages **Retrieval-Augmented Generation (RAG)** and **Fine-Tuning** to provide accurate, context-aware, and up-to-date insights about gameplay, champion performance, patch changes, and strategic recommendations.

---

## Project Structure

```
│
├── src/                            # Main source code and data
│   │
│   ├── code/                      # Implementation scripts
│   │   ├── create_data/           # Data collection
│   │   │   ├── getMatchData.py
│   │   │   ├── getPlayerPuuids(1).py
│   │   │   ├── scrape_patches.ipynb
│   │   │   └── tagsAndNamesDiamond.py
│   │   │
│   │   ├── finetunning/           # Model fine-tuning
│   │   │   ├── generate_finetune_data.py
│   │   │   └── model_finetune.py
│   │   │
│   │   └── RAG/                    # Retrieval-augmented generation
│   │       └── rag.py
│   │
│   ├── data/                       # Datasets
│   │   ├── diamond_leaderboards_all_23K.csv    # Player rankings (~23K)
│   │   ├── diamond_puuids_100.csv              # Sample player PUUIDs
│   │   ├── match_data.json                     # Match history data
│   │   ├── finetune_data_llm_8000.json         # Full training dataset (8K)
│   │   ├── finetune_data_llm_half.json         # Reduced training dataset (4K)
│   │   │
│   │   └── patch_notes/            # Processed patch data
│   │       ├── all_patches.json
│   │       ├── rag_chunks.jsonl
│   │       └── rag_chunks_pretty.json
│   │
│   └── results/                     # Model outputs
│       ├── finetune_test_results.json
│       └──rag_results.json 
│
└── report/                          # Project report
   
```

---

## Folder Details

### `/src/code/create_data/`
Scripts for collecting and preparing raw data from League of Legends API and patch sources.

### `/src/code/finetunning/`
Scripts for generating fine-tuning datasets and adapting the base LLM model to League of Legends domain knowledge.

### `/src/code/RAG/`
Retrieval-augmented generation system that retrieves relevant patch notes and game data to enhance model responses.

### `/src/data/`
Raw and processed datasets including player statistics, match history, and training data for fine-tuning.

### `/src/data/patch_notes/`
Processed patch notes segmented and formatted for RAG retrieval system.

### `/src/results/`
Model evaluation results and the final fine-tuned model with LoRA adapters and checkpoints.

### `/report/`
LaTeX project documentation including the final project report, bibliography, and code examples.
