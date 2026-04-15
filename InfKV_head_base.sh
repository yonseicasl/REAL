#!/bin/bash

# 指定使用 GPU 0 (或者你需要的 GPU ID)
export CUDA_VISIBLE_DEVICES=0,1
# 参数定义
method=$1
max_capacity_prompts=$2
attn_implementation=$3
model_path=$4
beta=$5
temp=$6
head_scores_path=$7 


# 内存环境变量 (可以保留)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 结果目录
save_dir="./results/${method}/results_long_bench_base${max_capacity_prompts}_beta${beta}_temp${temp}"

# 使用 python3 直接运行
echo "Starting with python3 on single GPU..."
python3 InfKV_run_longbench.py \
    --method "${method}" \
    --model_path "${model_path}" \
    --max_capacity_prompts "${max_capacity_prompts}" \
    --attn_implementation "${attn_implementation}" \
    --beta "${beta}" \
    --temp "${temp}" \
    --save_dir "${save_dir}" \
    --use_cache True \
    --head_scores_path "${head_scores_path}" \
    --eval_batch_size 1

echo "Python script finished."