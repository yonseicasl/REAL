import os
import sys
import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Monkey patch to avoid logits.float() conversion which causes OOM
# This patches the LlamaForCausalLM.forward method to keep logits in float16
import transformers.models.llama.modeling_llama as llama_module

original_llama_forward = llama_module.LlamaForCausalLM.forward

def patched_llama_forward(self, *args, **kwargs):
    # Call original forward
    result = original_llama_forward(self, *args, **kwargs)
    # The original forward converts logits to float32 at line 1202
    # We need to intercept before that happens, so we'll patch at a lower level
    return result

# Actually, let's patch at the exact location where logits.float() is called
# We'll create a wrapper that prevents the conversion
def create_patched_forward(original_forward):
    def patched_forward(self, *args, **kwargs):
        # Temporarily store the original dtype
        output_dtype = self.lm_head.weight.dtype

        # Call original forward
        result = original_forward(self, *args, **kwargs)

        # If logits were converted to float32, convert them back to float16
        if hasattr(result, 'logits') and result.logits.dtype == torch.float32 and output_dtype == torch.float16:
            result.logits = result.logits.half()

        return result
    return patched_forward

llama_module.LlamaForCausalLM.forward = create_patched_forward(original_llama_forward)
print("[INFO] Patched LlamaForCausalLM.forward to keep logits in float16")


datasets = [
    # "narrativeqa",
    # "qasper",
     "hotpotqa",
     "2wikimqa",
     "musique",
    # "qasper",
    # "multifieldqa_en",
    #"hotpotqa",
    #"2wikimqa",
    # "musique",
    # "dureader",
    # "gov_report",
    # "multifieldqa_zh",
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
    # "new_language_translation",
    # "financial", 
    # "governmental",
    # "event_ordering",
    # "academic",
    # "detective",
    # "agent_history_qa",
    # "code_repo_qa", 
    # "literary",
    # "many_shot_learning",
    # "user_guide_qa",
    # "table_qa",
    # "multi_news",
    # "knowledge_graph_reasoning",
    # "legal",
    # "dialogue_history_qa"
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
    # LongBench v2 tasks
    "new_language_translation": 16,
    "financial": 16, 
    "governmental": 16,
    "event_ordering": 4,
    "academic": 32,
    "detective": 32,
    "agent_history_qa": 32,
    "code_repo_qa": 64, 
    "literary": 32,
    "many_shot_learning": 16,
    "user_guide_qa": 32,
    "table_qa": 4,
    "multi_news": 64,
    "knowledge_graph_reasoning": 4,
    "legal": 16,
    "dialogue_history_qa": 32 
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
    "multi_news": "You are given multiple news articles and a question about them. Please read the articles carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nNews Articles: {context}\n\nQuestion: {input}\n\nAnswer:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
    # LongBench v2 tasks - 所有都是多选题格式
    "new_language_translation": "You are given a text and a question about language translation. Please read the text carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nText: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "financial": "You are given a financial document and a question. Please read the document carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nDocument: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "governmental": "You are given a government document and a question. Please read the document carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nDocument: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "event_ordering": "You are given a text describing events and a question about their sequence or timing. Please read the text carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nText: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "academic": "You are given an academic text and a question. Please read the text carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nText: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "detective": "You are given a detective story or mystery text and a question. Please read the text carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nText: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "agent_history_qa": "You are given a text about historical events or agents and a question. Please read the text carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nText: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "code_repo_qa": "You are given code repository information and a question about the code. Please read the information carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nCode Information: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "literary": "You are given a literary text and a question about literature. Please read the text carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nText: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "many_shot_learning": "You are given examples and a question about pattern recognition or learning. Please read the examples carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nExamples: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "user_guide_qa": "You are given a user guide or manual and a question about it. Please read the guide carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nUser Guide: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "table_qa": "You are given a table or structured data and a question about it. Please read the data carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nTable/Data: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "knowledge_graph_reasoning": "You are given information from a knowledge graph and a question requiring reasoning. Please read the information carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nKnowledge Graph Information: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "legal": "You are given a legal document and a question about law. Please read the document carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nLegal Document: {context}\n\nQuestion: {input}\n\nAnswer:",
    
    "dialogue_history_qa": "You are given a dialogue history and a question about the conversation. Please read the dialogue carefully and answer the question by choosing the correct option (A, B, C, or D). Only output the letter of your chosen answer.\n\nDialogue: {context}\n\nQuestion: {input}\n\nAnswer:",
}


model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 1000,  # Restored to 1000 to improve F1 scores (may risk OOM)
    "llama-3": 1000,  # Restored to 1000 to improve F1 scores (may risk OOM)
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

# def build_prompt(prompt, dataset):
    
#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt

def main(args):
    
    print("--- GPU Diagnostics ---")
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"环境变量 CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    num_gpus = torch.cuda.device_count()
    print(f"PyTorch 可见的 GPU 数量: {num_gpus}")
    
    # if num_gpus > 0:
    #     for i in range(num_gpus):
    #         print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    #         print(f"    总显存: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    #         print(f"    已分配显存 (PyTorch): {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
    #         print(f"    已缓存显存 (PyTorch): {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
    # print("---------------------")

    print("Loading data...")
    
    test_data = []
    
    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []
    
    input_max_len = 0
    
    model_path = args.model_path.lower()

    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
            

    
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            
            
            length = example["length"]
            if length > input_max_len: input_max_len = length
            
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            
            if "llama2" in args.model_path.lower():
                prompt = build_chat(prompt)
                
            example["prompt"] = prompt
                
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    
    for example in test_data:
        
        prompts.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    print("Finish loading model and tokenizer")
    
    model_name = model_path.split("/")[-1]

    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts_ratio}", args.dataset), exist_ok=True)

    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts_ratio}", args.dataset, f"{args.method}.json"), "w")
     
    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        
        batch_prompts = prompts[i:i+args.eval_batch_size]
        batch_inputs = inputs[i:i+args.eval_batch_size]
        batch_contexts = contexts[i:i+args.eval_batch_size]
        batch_answerss = answerss[i:i+args.eval_batch_size]
        batch_lengths = lengths[i:i+args.eval_batch_size]
        
        batch_datasets = datasets[i:i+args.eval_batch_size]
        batch_languages = languages[i:i+args.eval_batch_size]
        batch_all_classess = all_classess[i:i+args.eval_batch_size]
        batch__ids = _ids[i:i+args.eval_batch_size]
        
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if len(batch_input_ids[0]) > model_max_len:
            # New truncation strategy: 'Distributed Sampling' with CPU offloading to prevent OOM.
            print(f"Warning: Input length {len(batch_input_ids[0])} is greater than model max length {model_max_len}.")
            print("Applying 'distributed sampling' truncation with CPU offload...")
            
            # Move the original tensor to CPU to free up GPU memory
            original_input_ids_cpu = batch_input_ids[0].to('cpu')
            original_len = len(original_input_ids_cpu)
            
            # The original GPU tensor can now be cleared
            del batch_input_ids
            torch.cuda.empty_cache()

            num_chunks = 10
            tokens_to_keep_per_chunk = model_max_len // num_chunks
            original_chunk_size = original_len // num_chunks

            new_input_ids_chunks = []
            
            for i in range(num_chunks):
                chunk_start_index = i * original_chunk_size
                tokens_from_chunk = original_input_ids_cpu[chunk_start_index : chunk_start_index + tokens_to_keep_per_chunk]
                new_input_ids_chunks.append(tokens_from_chunk)
            
            # Concatenate on the CPU
            new_input_ids_cpu = torch.cat(new_input_ids_chunks)

            # Move the final, truncated tensor back to the GPU
            batch_input_ids = new_input_ids_cpu.unsqueeze(0).to('cuda')

            # Create a new attention mask on the correct device
            attention_mask = torch.ones_like(batch_input_ids)

            # Update tokenized_prompts with the truncated tensors
            tokenized_prompts = {
                'input_ids': batch_input_ids,
                'attention_mask': attention_mask
            }

            print(f"Original length: {original_len}, New truncated length: {batch_input_ids.shape[1]}")

        model.model.config.window_size = 8

        # 计算max_capacity_prompts
        if args.max_capacity_prompts != -1:
            max_capacity_prompts = args.max_capacity_prompts
        elif args.max_capacity_prompts_ratio != -1:
            max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)
        else:
            max_capacity_prompts = 512  # 默认值

        print(f"[INFO] Sequence length: {batch_input_ids.shape[1]}, Ratio: {args.max_capacity_prompts_ratio}, Computed max_capacity_prompts: {max_capacity_prompts}")

        model.model.config.base_capacity = max_capacity_prompts
        model.model.config.head_choice = args.head_choice
        model.model.config.beta = args.beta
        model.model.config.temp = args.temp
        
        model.model.config.kernel_size = 7
        model.model.config.skip = 0
        model.model.config.normalize = True
        model.model.config.pooling = "maxpool"
        model.model.config.floor = 0.2


        context_length = batch_input_ids.shape[-1]

        output = model.generate(
            **tokenized_prompts,
            output_attentions = args.output_attentions,
            max_new_tokens=output_max_len,
            num_beams=1,
            
            do_sample=False,
            temperature=1.0,
            min_length=context_length+1,
            eos_token_id=[tokenizer.eos_token_id]
        )


        batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)        
        batch_generations = batch_outputs

        torch.cuda.empty_cache()
        for j in range(args.eval_batch_size):
            
            example = {}
            
            example["prompt"] = batch_prompts[j]
            example["input"] = batch_inputs[j]
            example["context"] = batch_contexts[j]
            example["answers"] = batch_answerss[j]
            example["pred"] = batch_generations[j]
            example["length"] = batch_lengths[j]
            
            example["dataset"] = batch_datasets[j]
            example["language"] = batch_languages[j]
            example["all_classes"] = batch_all_classess[j]
            example["_id"] = batch__ids[j]


            fout.write(json.dumps(example) + "\n")
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")

    parser.add_argument("--head_choice", type=str, default='random', choices=['random', 'copy', 'reason', 'sentence', 'dominant'])
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--temp', type=float, default=1.0)

    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.model_path == 'mistralai/Mistral-7B-Instruct-v0.2':
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left",
            revision='dca6e4b60aca009ed25ffa70c9bb65e46960a573',
            token=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left",
            token=True
        )

    if args.method.lower() != 'fullkv':
        from headkv.monkeypatch import replace_llama, replace_mistral 
        replace_llama(args.method)
        replace_mistral(args.method)
    
    # 手动指定device_map，将lm_head放在GPU 0上
    # 这样logits计算在GPU 0上进行，GPU 0有更多空间
    # 策略：GPU 0放前16层 + lm_head，GPU 1放后16层
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": 1,
        "lm_head": 0,  # 关键：lm_head放在GPU 0上
    }

    # 前16层放GPU 0
    for i in range(16):
        device_map[f"model.layers.{i}"] = 0

    # 后16层放GPU 1
    for i in range(16, 32):
        device_map[f"model.layers.{i}"] = 1

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation,
        token=True
    )

    print("\n[INFO] Model device map:")
    if hasattr(model, 'hf_device_map'):
        for name, device in model.hf_device_map.items():
            print(f"  {name}: {device}")

    # 检查GPU显存使用情况
    print("\n[INFO] GPU Memory after model loading:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    print()
    

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

        
    model.eval()
    
    save_dir = args.save_dir
    



        

    for idx, dataset in enumerate(datasets):
        
        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")
        print(f'base capacity: {args.max_capacity_prompts}\thead_choice:{args.head_choice}\tbeta:{args.beta}\ttemp:{args.temp}')

        args.dataset = dataset
        
        args.data_file = f"./data/LongBench/{args.dataset}.jsonl"
        
        main(args)
