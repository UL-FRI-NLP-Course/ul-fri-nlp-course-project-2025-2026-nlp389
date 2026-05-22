# LoL RAG Evaluation Report

Systems evaluated: 5
Questions per system: 30

## Overall Scores

| System | Correctness | Relevance | Combined |
|--------|:-----------:|:---------:|:--------:|
| A: Base (no RAG, no FT) | 1.30 | 4.33 | 2.82 |
| B: Fine-tuned (no RAG) | 1.60 | 4.10 | 2.85 |
| C: Base + RAG | 2.60 | 4.60 | 3.60 |
| D: Fine-tuned + RAG | 2.87 | 4.63 | 3.75 |
| E: LLaMA-3.3-70B (reference, no RAG) | 2.60 | 4.83 | 3.72 |

## Scores by Category


### build_data

| System | Correctness | Relevance |
|--------|:-----------:|:---------:|
| A: Base (no RAG, no FT) | 1.18 | 4.29 |
| B: Fine-tuned (no RAG) | 1.94 | 4.06 |
| C: Base + RAG | 2.76 | 4.47 |
| D: Fine-tuned + RAG | 2.76 | 4.59 |
| E: LLaMA-3.3-70B (reference, no RAG) | 2.29 | 4.82 |

### champion_desc

| System | Correctness | Relevance |
|--------|:-----------:|:---------:|
| A: Base (no RAG, no FT) | 1.80 | 4.40 |
| B: Fine-tuned (no RAG) | 1.40 | 3.00 |
| C: Base + RAG | 3.40 | 4.60 |
| D: Fine-tuned + RAG | 2.80 | 4.40 |
| E: LLaMA-3.3-70B (reference, no RAG) | 4.40 | 5.00 |

### leaderboard

| System | Correctness | Relevance |
|--------|:-----------:|:---------:|
| A: Base (no RAG, no FT) | 1.60 | 4.80 |
| B: Fine-tuned (no RAG) | 1.20 | 4.80 |
| C: Base + RAG | 2.00 | 5.00 |
| D: Fine-tuned + RAG | 4.20 | 5.00 |
| E: LLaMA-3.3-70B (reference, no RAG) | 2.20 | 4.80 |

### patch_notes

| System | Correctness | Relevance |
|--------|:-----------:|:---------:|
| A: Base (no RAG, no FT) | 0.67 | 3.67 |
| B: Fine-tuned (no RAG) | 0.67 | 5.00 |
| C: Base + RAG | 1.33 | 4.67 |
| D: Fine-tuned + RAG | 1.33 | 4.67 |
| E: LLaMA-3.3-70B (reference, no RAG) | 2.00 | 4.67 |

## Per-Question Correctness

| # | Category | A: Base (no RAG, no FT) | B: Fine-tuned (no RAG) | C: Base + RAG | D: Fine-tuned + RAG | E: LLaMA-3.3-70B (reference, no RAG) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | patch_notes | 0 | 0 | 0 | 0 | 2 |
| 2 | patch_notes | 1 | 2 | 0 | 0 | 2 |
| 3 | leaderboard | 2 | 2 | 4 | 5 | 2 |
| 4 | build_data | 0 | 2 | 2 | 5 | 2 |
| 5 | build_data | 0 | 0 | 1 | 1 | 1 |
| 6 | build_data | 0 | 0 | 5 | 2 | 2 |
| 7 | build_data | 1 | 0 | 2 | 2 | 1 |
| 8 | patch_notes | 1 | 0 | 4 | 4 | 2 |
| 9 | build_data | 0 | 0 | 5 | 5 | 1 |
| 10 | build_data | 4 | 2 | 0 | 2 | 4 |
| 11 | champion_desc | 2 | 2 | 2 | 2 | 4 |
| 12 | build_data | 0 | 2 | 0 | 0 | 2 |
| 13 | leaderboard | 2 | 1 | 0 | 5 | 2 |
| 14 | champion_desc | 2 | 0 | 4 | 4 | 4 |
| 15 | leaderboard | 0 | 1 | 2 | 5 | 1 |
| 16 | build_data | 4 | 2 | 2 | 2 | 2 |
| 17 | build_data | 1 | 1 | 1 | 2 | 1 |
| 18 | build_data | 2 | 5 | 5 | 5 | 1 |
| 19 | build_data | 1 | 2 | 5 | 5 | 4 |
| 20 | build_data | 2 | 5 | 2 | 2 | 4 |
| 21 | champion_desc | 2 | 1 | 5 | 2 | 5 |
| 22 | build_data | 2 | 2 | 4 | 2 | 2 |
| 23 | build_data | 0 | 4 | 4 | 4 | 2 |
| 24 | champion_desc | 1 | 2 | 4 | 4 | 4 |
| 25 | build_data | 2 | 2 | 4 | 4 | 4 |
| 26 | build_data | 1 | 4 | 0 | 0 | 4 |
| 27 | build_data | 0 | 0 | 5 | 4 | 2 |
| 28 | champion_desc | 2 | 2 | 2 | 2 | 5 |
| 29 | leaderboard | 4 | 2 | 2 | 4 | 4 |
| 30 | leaderboard | 0 | 0 | 2 | 2 | 2 |