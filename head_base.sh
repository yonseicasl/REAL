export CUDA_VISIBLE_DEVICES=0,1  # 双GPU模式
# export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

method=$2 # Support AdativeKV, ReasonKV
max_capacity_prompts=$3 # 128,2048 in paper
# max_capacity_prompts_ratio=$3
attn_implementation=$4 # Support "flash_attention_2"
model_path=$5
head_choice=$6
beta=$7
temp=$8

save_dir="./results/results_long_bench_${head_choice}_base${max_capacity_prompts}_beta${beta}_temp${temp}" # path to result save_dir

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --max_capacity_prompts_ratio -1 \
    --head_choice ${head_choice} \
    --beta ${beta} \
    --temp ${temp} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True \
    --max_num_examples 20
