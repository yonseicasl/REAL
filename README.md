# REAL: REtrieval-reAsoning and Logic-constructed Attention Behaviors for Long-Context KV Cache Compression

## Table of Contents

- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Download Model Weights](#download-model-weights)
- [Download Datasets](#download-datasets)
- [Run Multiple-Behavior Head Detection](#run-multiple-behavior-head-detection)
- [Requirements Summary](#requirements-summary)

---

## Repository Structure

```
REAL/
├── Important_Head/
│   ├── retrieval_head_detection.py      # Main retrieval head detection script
│   ├── retrieval_head_detection_r2.py   # Extended version with additional needle types
│   ├── haystack_for_detect/             # Needle-in-a-haystack data (v1)
│   └── haystack_for_detect_r2/          # Needle-in-a-haystack data (v2)
├── csrc/                                # CUDA extension source
├── requirements.txt                     # Python dependencies
└── *.sh                                 # Shell scripts for running experiments
```

> **Note:** The `data/` directory (LongBench datasets) and model weights are not included in this repository due to file size limits. Follow the instructions below to download them.

---

## Environment Setup

### 1. Clone this repository

```bash
git clone https://github.com/yonseicasl/REAL.git
cd REAL
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the CUDA extension

```bash
cd csrc
make
cd ..
```

---

## Download Model Weights

Download **Mistral-7B-Instruct-v0.2** from HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
    --local-dir ./Mistral-7B-Instruct-v0.2
```

Or using Python:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                  local_dir="./Mistral-7B-Instruct-v0.2")
```

---

## Download Datasets

### LongBench

```bash
mkdir -p data/LongBench
python3 - <<'EOF'
from datasets import load_dataset
import json, os

tasks = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
    "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
    "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
    "lcc", "repobench-p"
]
for task in tasks:
    ds = load_dataset("THUDM/LongBench", task, split="test")
    with open(f"data/LongBench/{task}.jsonl", "w") as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Downloaded: {task}")
EOF
```

### LongBench v2

```bash
mkdir -p data/longbenchv2
python3 - <<'EOF'
from datasets import load_dataset
import json

ds = load_dataset("THUDM/LongBench-v2", split="train")
# split by category
from collections import defaultdict
buckets = defaultdict(list)
for item in ds:
    buckets[item["domain"]].append(item)
for domain, items in buckets.items():
    fname = domain.lower().replace(" ", "_") + ".jsonl"
    with open(f"data/longbenchv2/{fname}", "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Downloaded: {fname}")
EOF
```

---

## Run Multiple-Behavior Head Detection

```bash
python Important_Head/retrieval_head_detection.py \
    --model_path ./Mistral-7B-Instruct-v0.2 \
    --s_len 0 \
    --e_len 128000
```

```bash
python Important_Head/retrieval_head_detection_r2.py \
    --model_path ./Mistral-7B-Instruct-v0.2 \
    --s_len 0 \
    --e_len 128000
```

Or use the provided shell scripts:

```bash
bash head_start_mistral.sh
bash head_start_llama.sh
```

---

## Requirements Summary

| Item | Source | Size |
|------|--------|------|
| Mistral-7B-Instruct-v0.2 | [HuggingFace](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | ~14 GB |
| LongBench | [HuggingFace](https://huggingface.co/datasets/THUDM/LongBench) | ~500 MB |
| LongBench v2 | [HuggingFace](https://huggingface.co/datasets/THUDM/LongBench-v2) | ~200 MB |
