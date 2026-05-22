# LoL RAG — Automatic Metrics Report (ROUGE-L & BERTScore)

Metrics computed against `ground_truth.json` reference answers.

> **Note:** ROUGE-L measures lexical overlap; BERTScore measures semantic similarity.
> Neither captures factual correctness for numerical/specific facts as well as the LLM judge.

## Overall Scores

| System | ROUGE-L | BERTScore F1 |
|--------|:-------:|:------------:|
| A: Base (no RAG, no FT) | 0.121 | 0.850 |
| B: Fine-tuned (no RAG) | 0.138 | 0.862 |
| C: Base + RAG | 0.201 | 0.858 |
| D: Fine-tuned + RAG | 0.185 | 0.865 |

## Scores by Category


### build_data

| System | ROUGE-L | BERTScore F1 |
|--------|:-------:|:------------:|
| A: Base (no RAG, no FT) | 0.115 | 0.847 |
| B: Fine-tuned (no RAG) | 0.143 | 0.864 |
| C: Base + RAG | 0.191 | 0.853 |
| D: Fine-tuned + RAG | 0.168 | 0.862 |

### champion_desc

| System | ROUGE-L | BERTScore F1 |
|--------|:-------:|:------------:|
| A: Base (no RAG, no FT) | 0.121 | 0.856 |
| B: Fine-tuned (no RAG) | 0.113 | 0.856 |
| C: Base + RAG | 0.163 | 0.864 |
| D: Fine-tuned + RAG | 0.135 | 0.864 |

### leaderboard

| System | ROUGE-L | BERTScore F1 |
|--------|:-------:|:------------:|
| A: Base (no RAG, no FT) | 0.107 | 0.839 |
| B: Fine-tuned (no RAG) | 0.142 | 0.853 |
| C: Base + RAG | 0.266 | 0.849 |
| D: Fine-tuned + RAG | 0.321 | 0.868 |

### patch_notes

| System | ROUGE-L | BERTScore F1 |
|--------|:-------:|:------------:|
| A: Base (no RAG, no FT) | 0.182 | 0.871 |
| B: Fine-tuned (no RAG) | 0.151 | 0.875 |
| C: Base + RAG | 0.211 | 0.893 |
| D: Fine-tuned + RAG | 0.137 | 0.879 |

## Per-Question ROUGE-L

| # | Category | A: Base (no RAG, no FT) | B: Fine-tuned (no RAG) | C: Base + RAG | D: Fine-tuned + RAG |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | patch_notes | 0.121 | 0.164 | 0.116 | 0.188 |
| 2 | patch_notes | 0.169 | 0.137 | 0.311 | 0.104 |
| 3 | leaderboard | 0.106 | 0.105 | 0.345 | 0.283 |
| 4 | build_data | 0.137 | 0.202 | 0.259 | 0.210 |
| 5 | build_data | 0.176 | 0.163 | 0.156 | 0.090 |
| 6 | build_data | 0.104 | 0.063 | 0.169 | 0.241 |
| 7 | build_data | 0.110 | 0.118 | 0.146 | 0.111 |
| 8 | patch_notes | 0.256 | 0.152 | 0.205 | 0.118 |
| 9 | build_data | 0.182 | 0.169 | 0.485 | 0.296 |
| 10 | build_data | 0.186 | 0.219 | 0.126 | 0.133 |
| 11 | champion_desc | 0.173 | 0.142 | 0.205 | 0.101 |
| 12 | build_data | 0.000 | 0.100 | 0.196 | 0.214 |
| 13 | leaderboard | 0.124 | 0.132 | 0.253 | 0.411 |
| 14 | champion_desc | 0.117 | 0.075 | 0.258 | 0.097 |
| 15 | leaderboard | 0.096 | 0.165 | 0.309 | 0.383 |
| 16 | build_data | 0.137 | 0.187 | 0.110 | 0.077 |
| 17 | build_data | 0.103 | 0.083 | 0.237 | 0.169 |
| 18 | build_data | 0.093 | 0.137 | 0.190 | 0.158 |
| 19 | build_data | 0.093 | 0.111 | 0.176 | 0.160 |
| 20 | build_data | 0.126 | 0.176 | 0.075 | 0.100 |
| 21 | champion_desc | 0.100 | 0.145 | 0.092 | 0.122 |
| 22 | build_data | 0.158 | 0.149 | 0.146 | 0.182 |
| 23 | build_data | 0.085 | 0.082 | 0.171 | 0.151 |
| 24 | champion_desc | 0.083 | 0.099 | 0.136 | 0.206 |
| 25 | build_data | 0.106 | 0.220 | 0.192 | 0.277 |
| 26 | build_data | 0.096 | 0.153 | 0.176 | 0.162 |
| 27 | build_data | 0.057 | 0.091 | 0.236 | 0.131 |
| 28 | champion_desc | 0.135 | 0.102 | 0.124 | 0.149 |
| 29 | leaderboard | 0.128 | 0.198 | 0.256 | 0.345 |
| 30 | leaderboard | 0.080 | 0.110 | 0.165 | 0.180 |