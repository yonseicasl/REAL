#!/bin/bash

mkdir ./longbench_logss/
mkdir ./needle_resultss/

# devices=(0 1 2 3 4 5 6 7)
devices=(0)
head_choices=('reason') # copy, reason
betas=(2.5)
max_capacity_prompts=128
counter=0

for((i=0;i<1;i++));do 
    for((j=0;j<1;j++));do
        device=${devices[counter]}
        head_choice=${head_choices[i]}
        beta=${betas[j]}
        temp=1
        alpha=1

        bash needle_base.sh \
            $device \
            ReasonKV \
            ${max_capacity_prompts} \
            flash_attention_2 \
            meta-llama/Meta-Llama-3-8B-Instruct \
            $head_choice \
            $beta \
            $temp \
            1000 \
            8001 \
            1000 \
            LLaMA3 \
            Meta-Llama-3-8B-Instruct > ./longbench_logs/llama3_ReasonKV_${head_choice}_base${max_capacity_prompts}_beta${beta}_temp${temp}.txt 2>&1 &
        ((counter+=1))
    done
done