# REAL: REtrieval-reAsoning and Logic-constructed Attention Behaviors for Long-Context KV Cache Compression

The growing sequence length of large language models poses significant challenges for key-value caches. Existing state-of-the-art cache eviction methods primarily analyze the inference behavior of attention heads in successful retrieval-reasoning cases, often overlooking diverse behaviors in failure cases, such as bias and distraction. This oversight limits the potential to leverage heterogeneous head behaviors for improved eviction performance. Inspired by the confusion matrix, we introduce an Attention Behavior Matrix to comprehensively analyze attention head behaviors in both success and failure scenarios. By maximizing the signal-to-noise ratio — strengthening valid reasoning pathways in success cases while inhibiting noise from bias and distraction in failure cases — we propose REtrieval-reAsoning and Logic-constructed (REAL). REAL is the first KV cache eviction method that leverages multi-behavior analysis.

## Table of Contents

- [Updated Timeline](#updated-timeline)
- [Note](#note)
- [Environment Setup](#environment-setup)
- [Download Model Weights and Datasets](#download-model-weights-and-datasets)
- [Workflow](#workflow)

---
## Updated Timeline

😊 [2026.4.15] Source Code Commitment. 

---

## Note

⚠️ The `data/` directory (LongBench datasets) and model weights are not included in this repository due to file size limits. Follow the instructions below to download them.

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

## Download Model Weights and Datasets

### 1. Download Model Weights

| Item | Source | Size |
|------|--------|------|
| Meta-Llama-3-8B-Instruct | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)  | ~16 GB
| Mistral-7B-Instruct-v0.2 | [HuggingFace](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | ~14 GB |

#### Meta-Llama-3-8B-Instruct

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir ./Meta-Llama-3-8B-Instruct
```

Or using Python:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                  local_dir="./Meta-Llama-3-8B-Instruct")
```


#### Mistral-7B-Instruct-v0.2

```bash
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

### 2. Download Datasets

| Item | Source | Size |
|------|--------|------|
| LongBench | [HuggingFace](https://huggingface.co/datasets/THUDM/LongBench) | ~500 MB |
| LongBench v2 | [HuggingFace](https://huggingface.co/datasets/THUDM/LongBench-v2) | ~200 MB |

#### LongBench

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

#### LongBench v2

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

## Workflow

### 1. Compute INFsc

```bash
python structure_head_InfScore.py --model meta-llama/Meta-Llama-3-8B-Instruct --max_len 2048 --depths 15
```

### 2. Allocate KV budget
```bash
 ./head_base.sh 0 ReasonKV 0.1 flash_attention_2 meta-llama/Meta-Llama-3-8B-Instruct reason 1.351 0.98
```

```bash
 ./InfKV_head_base.sh 0 InfKV 128 flash_attention_2 meta-llama/Meta-Llama-3-8B-Instruct 1.5 1
```
### 3. Inference
```bash
./head_base.sh 0 ReasonKV 0.05 flash_attention_2 meta-llama/Meta-Llama-3-8B-Instruct reason 1 1
```
### 4. Evaluate
```bash
python InfKV_eval.py \
    --results_dir results_long_bench_reason_base0.1_beta1.351_temp0.98 \
    --model Meta-Llama-3-8B-Instruct \
    --capacity_ratio 0.1
```

```bash
python eval.py \
    --results_dir HeadKV/results/results_long_bench_reason_base128_beta1.351_temp0.98 \
    --model meta-llama-3-8b-instruct \
    --capacity 128
```





