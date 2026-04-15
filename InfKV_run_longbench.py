import os
import json
import random
import argparse
import traceback
import types
import sys

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# --- START OF EDIT: Import hijack functions --- 
from headkv.InfKV_llama_hijack import (
    adaptive_LlamaModel_forward,
    inf_llama_flash_attn2_forward,
    # Import others if needed based on args.method or attn_implementation
    # reason_llama_flash_attn2_forward,
    # adaptive_llama_flash_attn2_forward,
    prepare_inputs_for_generation_llama # Assuming this is also needed
)
# --- END OF EDIT ---

# --- Remove subclass import, revert to AutoModel ---
# from headkv.InfKV_llama_hijack import InfKVLlamaForCausalLM, prepare_inputs_for_generation_llama 

datasets = [
    "narrativeqa"
    # "qasper",
    # "multifieldqa_en",
    # "multifieldqa_zh",
    # "hotpotqa",
    # "2wikimqa",
    # "musique",
    # "dureader",
    # "gov_report",
    # "qmsum",
    # "multi_news",
    # "vcsum",
    # "trec",
    # "triviaqa",
    # "samsum",
    # "lsht",
    # "passage_count",
    # "passage_retrieval_en",
    # "passage_retrieval_zh",
    # "lcc",
    # "repobench-p",
    # "comprehension_and_reasoning",
    # "multiple_information_retrieval",
    # "timeline_reorder",
    # "computation"
]




dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
    'comprehension_and_reasoning': 64,
    'multiple_information_retrieval': 64,
    'timeline_reorder': 32,
    'computation': 32,
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    'comprehension_and_reasoning': 'Please answer the question based on the long texts below. \n{context}\nQuestion: {input}\nAnswer:',
    'computation': 'Please answer the question based on the long texts below. \n{context}\nQuestion: {input}\nAnswer:',
    'multiple_information_retrieval': 'Please answer the question based on the long texts below. \n{context}\nQuestion: {input}\nAnswer:',
    'timeline_reorder': 'Please answer the question based on the long texts below. \n{context}\nQuestion: {input}\nAnswer:',

    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}


model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 31500,
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt

def main(args, model, tokenizer):

    print("Loading data...")

    test_data = []
    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []

    input_max_len = 0
    model_path = args.model_path.lower() 

    model_max_len = args.model_max_len 
    output_max_len = dataset2maxlen.get(args.dataset, 128) 

    try:
        with open(args.data_file) as fp:
            for line in fp: 
                example = json.loads(line)
                length = example["length"]
                if length > input_max_len: input_max_len = length
                template = model2prompt[args.dataset]
                prompt = template.format(**example)
                if "llama-2" in model_path or "llama2" in model_path: # Check model type correctly
                    prompt = build_chat(prompt)
                example["prompt"] = prompt
                test_data.append(example)
    except FileNotFoundError:
        print(f"[ERROR] Data file not found: {args.data_file}")
        return # Exit main if data file not found

    print(f"Max Length for dataset {args.dataset} is {input_max_len}")


    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]

    current_prompts = [ex["prompt"] for ex in test_data]
    current_inputs = [ex["input"] for ex in test_data]
    current_contexts = [ex["context"] for ex in test_data]
    current_answerss = [ex["answers"] for ex in test_data]
    current_lengths = [ex["length"] for ex in test_data]
    current_datasets = [ex["dataset"] for ex in test_data]
    current_languages = [ex["language"] for ex in test_data]
    current_all_classess = [ex["all_classes"] for ex in test_data]
    current_ids_info = [ex["_id"] for ex in test_data]

    print(f"Finished preparing data for {args.dataset}.")

    model_name = args.model_path.split("/")[-1] # Use args.model_path


    dataset_save_dir = os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset)
    os.makedirs(dataset_save_dir, exist_ok=True)
    output_file_path = os.path.join(dataset_save_dir, f"{args.method}.json")

    print(f"Results will be saved to: {output_file_path}")

    print("model type:", type(model))
    if hasattr(model, "model"):
        print("model.model type:", type(model.model))
        print("model.model.prepare_inputs_for_generation:", model.model.prepare_inputs_for_generation)
    print("model.prepare_inputs_for_generation:", model.prepare_inputs_for_generation)

    # --- START OF EDIT: Apply Monkey Patching --- 
    print("\nApplying monkey patches for InfKV...")
    try:
        # Patch LlamaModel.forward (assuming model has 'model' attribute)
        if hasattr(model, 'model') and hasattr(model.model, 'forward'):
            model.model.forward = types.MethodType(adaptive_LlamaModel_forward, model.model)
            print("  Patched model.model.forward with adaptive_LlamaModel_forward.")
        else:
            print("  [WARN] Could not patch model.model.forward.")

        # Patch attention layer forward based on implementation argument
        target_attn_forward_func = None
        if args.attn_implementation == "flash_attention_2":
            # Assuming InfKV logic is mainly within inf_llama_flash_attn2_forward
            target_attn_forward_func = inf_llama_flash_attn2_forward 
        # Add elif for other attn_implementations if needed
        # elif args.attn_implementation == "sdpa":
        #     target_attn_forward_func = inf_llama_sdpa_forward # Example
        # elif args.attn_implementation == "eager":
        #     target_attn_forward_func = inf_llama_eager_forward # Example
        else:
            print(f"  [WARN] Unknown or unsupported attn_implementation '{args.attn_implementation}'. No attention patch applied.")

        if target_attn_forward_func and hasattr(model, 'model') and hasattr(model.model, 'layers'):
            patched_layers = 0
            for i, layer in enumerate(model.model.layers):
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'forward'):
                    layer.self_attn.forward = types.MethodType(target_attn_forward_func, layer.self_attn)
                    patched_layers += 1
                else:
                    print(f"  [WARN] Layer {i} does not have self_attn.forward to patch.")
            print(f"  Patched {patched_layers} attention layers with {target_attn_forward_func.__name__}.")
        elif target_attn_forward_func:
            print("  [WARN] Could not patch attention layers (model structure unexpected).")

        # Patch prepare_inputs_for_generation
        if hasattr(model, 'prepare_inputs_for_generation'):
            model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation_llama, model)
            print("  Patched model.prepare_inputs_for_generation.")
        else:
            print("  [WARN] Could not patch model.prepare_inputs_for_generation.")

        print("Monkey patching finished.")

    except Exception as e:
        print(f"[ERROR] Failed during monkey patching: {e}")
        # Decide if you want to exit or continue with the original model
        # return 
    
    # Optional: Verify patches by printing methods after patching
    if hasattr(model, 'model'): print("model.model.forward after patch:", model.model.forward)
    if hasattr(model, 'model') and len(model.model.layers) > 0: print("Layer 0 self_attn.forward after patch:", model.model.layers[0].self_attn.forward)
    print("model.prepare_inputs_for_generation after patch:", model.prepare_inputs_for_generation)
    # --- END OF EDIT: Apply Monkey Patching --- 

    # --- Set config attributes BEFORE the loop --- 
    if args.method.lower() == 'infkv':
        if hasattr(args, 'beta') and args.beta is not None:
            setattr(model.config, 'infkv_beta', args.beta) # Use setattr for safety
            print(f"[INFO] Set model.config.infkv_beta = {model.config.infkv_beta}")
        if hasattr(args, 'temp') and args.temp is not None:
            setattr(model.config, 'infkv_temp', args.temp)
            print(f"[INFO] Set model.config.infkv_temp = {model.config.infkv_temp}")
        # Ensure base_capacity and head_scores_path are also set if needed by init_infkv
        # (The code added previously already does this before model loading, which is fine)

    # Define the tqdm bar format here (this was accepted by user)
    # tqdm_bar_format = "{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    # The line above is now commented out or removed as we'll use tqdm defaults for the bar.

    # Calculate num_batches once before the loops
    num_total_samples = len(current_prompts)
    num_batches = (num_total_samples + args.eval_batch_size - 1) // args.eval_batch_size

    # Create the tqdm progress bar object manually, total is number of individual samples
    with tqdm(total=num_total_samples,
              desc=f"Evaluating {args.dataset}",
              # bar_format=tqdm_bar_format, # Removed to use tqdm default
              # ncols=100, # Removed to use tqdm default
              # dynamic_ncols=False, # Removed to use tqdm default
              mininterval=0.1 # Kept for responsiveness
             ) as pbar:
        with open(output_file_path, "w") as fout:
            # Iterate through the data in batches using range
            for i in range(0, num_total_samples, args.eval_batch_size):
                current_batch_number = i // args.eval_batch_size + 1
                # Print a separator for clarity between batches/samples (optional)
                # --- Start of Edit: Comment out the interfering print statement ---
                # print(f"\n--- Starting Batch {current_batch_number}/{num_batches} (Processing items {i+1} to {min(i+args.eval_batch_size, num_total_samples)}) ---")
                # --- End of Edit ---

                batch_prompts = current_prompts[i:i+args.eval_batch_size]
                batch_answerss = current_answerss[i:i+args.eval_batch_size]
                batch_lengths = current_lengths[i:i+args.eval_batch_size]
                batch_datasets_info = current_datasets[i:i+args.eval_batch_size]
                batch_languages_info = current_languages[i:i+args.eval_batch_size]
                batch_all_classess_info = current_all_classess[i:i+args.eval_batch_size]
                batch_ids_info = current_ids_info[i:i+args.eval_batch_size]

                try:
                    tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
                    batch_input_ids = tokenized_prompts.input_ids
                    attention_mask = tokenized_prompts.attention_mask

                    current_seq_len = batch_input_ids.shape[1]
                    print(f"[DEBUG] Batch {current_batch_number}: Original input length: {current_seq_len}, Current args.max_capacity_prompts: {args.max_capacity_prompts}")

                    if current_seq_len > model_max_len:
                        print(f"[DEBUG] Input too long ({current_seq_len} > {model_max_len}). Truncating... Current args.max_capacity_prompts: {args.max_capacity_prompts}")
                        prefix_len = model_max_len // 2
                        suffix_len = model_max_len - prefix_len
                        batch_input_ids = torch.cat([batch_input_ids[:, :prefix_len], batch_input_ids[:, -suffix_len:]], dim=1)
                        attention_mask = torch.ones_like(batch_input_ids)
                        tokenized_prompts = {'input_ids': batch_input_ids, 'attention_mask': attention_mask}
                        current_seq_len = batch_input_ids.shape[1]
                        print(f"[DEBUG] Truncated input length: {current_seq_len}, Current args.max_capacity_prompts: {args.max_capacity_prompts}")

                    context_length = batch_input_ids.shape[-1] # This context_length is for the whole batch (due to padding)

                    print(f"[DEBUG] Calling model.generate for batch {current_batch_number}...")
                    generate_kwargs = {
                        "max_new_tokens": output_max_len,
                        "num_beams": 1,
                        "do_sample": False,
                        "temperature": 1.0,
                        "eos_token_id": [tokenizer.eos_token_id],
                        "pad_token_id": tokenizer.pad_token_id
                    }
                    output = model.generate(
                        **tokenized_prompts,
                        **generate_kwargs
                    )
                    print(f"[DEBUG] model.generate finished for batch {current_batch_number}.")

                    batch_generations = []
                    for idx_in_batch in range(output.shape[0]):
                        # Determine the actual prompt length for *this specific item* in the batch
                        # The prompt part in batch_input_ids might be shorter than context_length due to padding
                        item_prompt_length = tokenized_prompts.input_ids[idx_in_batch].ne(tokenizer.pad_token_id).sum().item()
                        
                        # Slice the generation part for this item
                        item_generation_ids = output[idx_in_batch][item_prompt_length:]
                        decoded_text = tokenizer.decode(item_generation_ids, skip_special_tokens=True)
                        batch_generations.append(decoded_text)

                    for j in range(len(batch_prompts)):
                        example = {}
                        original_index = i + j
                        if original_index >= num_total_samples:
                            print(f"[ERROR] Invalid original_index {original_index} (num_total_samples: {num_total_samples}). Skipping.")
                            continue
                            
                        example["prompt"] = current_prompts[original_index]
                        example["input"] = current_inputs[original_index]
                        example["context"] = current_contexts[original_index]
                        example["answers"] = current_answerss[original_index]
                        example["pred"] = batch_generations[j] if j < len(batch_generations) else "Decoding Error"
                        example["length"] = current_lengths[original_index]
                        example["dataset"] = current_datasets[original_index]
                        example["language"] = current_languages[original_index]
                        example["all_classes"] = current_all_classess[original_index]
                        example["_id"] = current_ids_info[original_index]

                        fout.write(json.dumps(example) + "\n")
                        pbar.update(1) # Update progress bar for each successfully processed item

                except Exception as e:
                    print(f"[ERROR] Failed processing batch starting at index {i}: {e}")
                    traceback.print_exc()
                    error_message = f"Error processing batch starting at index {i}: {str(e)}"
                    first_item_id = current_ids_info[i] if i < num_total_samples else f"unknown_id_at_batch_start_{i}"
                    error_example = {"_id": f"BATCH_ERROR_STARTING_AT_{first_item_id}", "error": error_message}
                    fout.write(json.dumps(error_example) + "\n")
                    
                    num_items_in_batch = len(batch_prompts) # Actual number of items intended for this batch
                    pbar.update(num_items_in_batch) # Advance pbar by the number of items in the failed batch
                    pbar.set_postfix_str(f"ERROR in batch starting at {i}")

                torch.cuda.empty_cache()

    print(f"Finished processing dataset {args.dataset}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    # parser.add_argument("--dataset", type=str, default="") # Removed, set in loop
    # parser.add_argument("--data_file", type=str, default="") # Removed, set in loop
    parser.add_argument("--save_dir", type=str, default="./results", help="Base directory for saving results") # Added default

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.") # Made required
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    # parser.add_argument("--output_attentions", type=bool, default=False, help="") # Removed, not needed for generation

    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")

    # parser.add_argument("--max_new_tokens", type=int, default=None, help="") # Determined by dataset2maxlen

    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")

    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default='FullKV', help='Method to use (FullKV, InfKV, etc.)') # Changed default
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")

    # --- Added head_scores_path ---
    parser.add_argument("--head_scores_path", type=str, default=None, help="Path to the JSON file containing head InfScores for InfKV.")

    # parser.add_argument("--head_choice", type=str, default='random', choices=['random', 'copy', 'reason']) # Removed, assuming InfKV uses scores path
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--temp', type=float, default=1.0)

    args = parser.parse_args()

    set_seed(args.seed)

    # --- Load Config (remains the same, but add base_capacity etc. here) ---
    print(f"[DEBUG] Loading config for {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if args.method.lower() == 'infkv':
        if hasattr(args, 'max_capacity_prompts'):
             config.base_capacity = args.max_capacity_prompts
             print(f"[DEBUG] Set config.base_capacity = {config.base_capacity}")
        if args.head_scores_path:
             config.head_scores_path = args.head_scores_path
             print(f"[DEBUG] Set config.head_scores_path = {config.head_scores_path}")
        # Set infkv_beta/infkv_temp on config *after* model loading now

    # --- Load Tokenizer (remains the same) ---
    print(f"[DEBUG] Loading tokenizer for {args.model_path}")
    # Simplified tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left",
        trust_remote_code=True # Added trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[DEBUG] Tokenizer loaded. Pad token ID: {tokenizer.pad_token_id}")


    # --- Load Model using AutoModelForCausalLM --- 
    print(f"[DEBUG] Loading model using AutoModelForCausalLM: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained( # Revert back to AutoModel
        args.model_path,
        config=config, # Pass the config with base_capacity etc.
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=True
    )
    model.eval()
    print(f"[DEBUG] Model loaded successfully onto devices: {model.hf_device_map}")
    print(f"[DEBUG] Loaded model type: {type(model)}") # Should be LlamaForCausalLM now

    # --- Set model_max_len (remains the same) --- 
    model_max_len = model.config.max_position_embeddings
    model_path_lower = args.model_path.lower()
    if model_max_len is None:
        for key, length in model2maxlen.items():
            if key in model_path_lower:
                model_max_len = length
                break
        if model_max_len is None:
            print("[WARNING] Could not determine model_max_len from config or dict, using default 4096.")
            model_max_len = 4096
    args.model_max_len = model_max_len
    print(f"[INFO] Determined model max length: {args.model_max_len}")

    # --- Apply Hijack if method is InfKV ---
    if args.method.lower() == 'infkv':
        print(f"[DEBUG] Attempting InfKV hijack for attn_implementation: {args.attn_implementation}...")
        if args.attn_implementation == "flash_attention_2":
            from headkv.InfKV_llama_hijack import inf_llama_flash_attn2_forward
            hijacked_count = 0
            for i, layer in enumerate(model.model.layers):
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'forward') and 'LlamaFlashAttention2' in type(layer.self_attn).__name__:
                    # Ensure necessary attributes are present on layer.self_attn for init_infkv
                    if not hasattr(layer.self_attn, 'layer_idx'):
                        layer.self_attn.layer_idx = i
                    
                    # num_heads and head_dim for LlamaFlashAttention2 should be available on the instance itself
                    # or can be derived from config if not directly on the instance.
                    if not hasattr(layer.self_attn, 'num_heads'):
                        # model.config.num_attention_heads might be total, self_attn.num_heads is specific
                        layer.self_attn.num_heads = layer.self_attn.num_heads if hasattr(layer.self_attn, 'num_heads') else model.config.num_attention_heads
                    if not hasattr(layer.self_attn, 'head_dim'):
                        layer.self_attn.head_dim = layer.self_attn.head_dim if hasattr(layer.self_attn, 'head_dim') else (model.config.hidden_size // model.config.num_attention_heads)
                    
                    # Also ensure the config object is passed down to the attention layer if init_infkv needs it from there
                    if not hasattr(layer.self_attn, 'config'):
                         layer.self_attn.config = model.config # Pass the main model config

                    original_forward = layer.self_attn.forward
                    layer.self_attn.forward = inf_llama_flash_attn2_forward.__get__(layer.self_attn, type(layer.self_attn))
                    print(f"  Layer {i} ({type(layer.self_attn).__name__}): Replaced self_attn.forward.")
                    # print(f"    Original: {original_forward}")
                    # print(f"    New: {layer.self_attn.forward}")
                    hijacked_count += 1
                else:
                    print(f"  Layer {i}: Skipping hijack. Type: {type(layer.self_attn).__name__}, Has self_attn: {hasattr(layer, 'self_attn')}, Has forward: {hasattr(layer.self_attn, 'forward') if hasattr(layer, 'self_attn') else False}")
            if hijacked_count > 0:
                print(f"[DEBUG] InfKV hijack for {hijacked_count} LlamaFlashAttention2 layers applied.")
            else:
                print(f"[WARNING] InfKV hijack for LlamaFlashAttention2 was requested, but no compatible layers were found or hijacked.")
        elif args.attn_implementation == "eager":
            # Placeholder for eager mode hijack
            print(f"[WARNING] Hijack for 'eager' mode (LlamaAttention) not fully implemented/tested here.")
            # from headkv.InfKV_llama_hijack import inf_llama_attention_forward # Assuming you create this
            # for i, layer in enumerate(model.model.layers):
            #     if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'forward') and 'LlamaAttention' in type(layer.self_attn).__name__:
            #         layer.self_attn.forward = inf_llama_attention_forward.__get__(layer.self_attn, type(layer.self_attn))
            #         # ... set layer_idx, num_heads, head_dim, config on layer.self_attn ...
            #         print(f"  Layer {i} ({type(layer.self_attn).__name__}): Replaced self_attn.forward.")
        else:
            print(f"[WARNING] InfKV hijack not implemented for attn_implementation: {args.attn_implementation}")
    else:
        print(f"[DEBUG] Method is not 'infkv' ({args.method}). Skipping InfKV hijack.")

    # --- Loop through datasets (remains the same) --- 
    # ... (dataset loop and call to main) ...

    # Global datasets list needs to be defined or passed correctly
    datasets_to_run = [
     "narrativeqa"
    # "qasper",
    # "multifieldqa_en",
    # "multifieldqa_zh",
    # "hotpotqa",
    # "2wikimqa",
    # "musique",
    # "dureader",
    # "gov_report",
    # "qmsum",
    # "multi_news",
    # "vcsum",
    # "trec",
    # "triviaqa",
    # "samsum",
    # "lsht",
    # "passage_count",
    # "passage_retrieval_en",
    # "passage_retrieval_zh",
    # "lcc",
    # "repobench-p",
    # "comprehension_and_reasoning",
    # "multiple_information_retrieval",
    # "timeline_reorder",
    # "computation"
    ]
    for idx, dataset_name in enumerate(datasets_to_run):
        print(f"\n===== Working on dataset {dataset_name} ({idx+1}/{len(datasets_to_run)}) =====")
        args.dataset = dataset_name
        args.data_file = f"./data/LongBench/{args.dataset}.jsonl" # Construct data file path

        # Call the main function for the current dataset
        main(args, model, tokenizer)

    print("\n===== All datasets processed. =====")