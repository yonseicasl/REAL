"""
Simplified version of causality structure script - for generating model predictions and calculating F1 scores
"""

import os
import glob
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import gc
import psutil
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

class SimpleLLMTester:
    """
    Simplified test class for model loading, prediction generation, and F1 calculation
    """
    def __init__(self,
                haystack_dir="./haystack_for_detect_r2",
                model_name='',
                context_lengths_min=2048,
                context_lengths_max=2048,
                context_lengths_num_intervals=3,
                gpu_id=None,
                top_k_infscore=10,
                depths=None):
        """Initialize the tester"""
        self.similarity_by_context_length = defaultdict(list)
        self.similarity_by_depth = defaultdict(list)
        
        # Set GPU device if specified
        if gpu_id is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Using specific GPU device: cuda:{gpu_id}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"Using auto device mapping across {gpu_count} available GPUs")
                total_gpu_memory = 0
                for i in range(gpu_count):
                    total_gpu_memory += torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"Total GPU memory available: {total_gpu_memory:.2f} GB")
            else:
                print(f"Using CPU (no GPU available)")
        
        # Load Sentence Transformer model for semantic similarity
        print("Loading sentence transformer model for similarity scoring...")
        try:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2') 
            print("Sentence transformer model loaded successfully.")
            # 如果有 GPU，将模型移动到 GPU
            if self.device.type != 'cpu' and self.similarity_model:
                try:
                    self.similarity_model.to(self.device)
                    print(f"Moved sentence transformer model to device: {self.device}")
                except Exception as move_err:
                     print(f"Warning: Failed to move sentence transformer model to {self.device}: {move_err}")
        except Exception as e:
            print(f"Error loading sentence transformer model: {e}")
            print("Semantic similarity scoring will be disabled.")
            self.similarity_model = None
        
        # Load test data from file
        # Use absolute path to structure_needles.jsonl
        structure_data_file = "/home/casl/KVC/HeadKV/Important_Head/haystack_for_detect_r2/structure_needles.jsonl"
        
        if not os.path.exists(structure_data_file):
            raise FileNotFoundError(f"Data file not found: {structure_data_file}")
            
        print(f"Loading data from {structure_data_file}")
        structure_data = []
        with open(structure_data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip() # 去除首尾空白
                if not line: # 跳过空行
                    print(f"Warning: Skipped empty line at index {i} in {structure_data_file}")
                    continue
                try:
                    data = json.loads(line)
                    structure_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to decode JSON on line {i+1} in {structure_data_file}. Error: {e}. Skipping line: '{line[:100]}...'" )
        
        if not structure_data:
            raise ValueError(f"No valid JSON data loaded from {structure_data_file}")
            
        # Extract necessary data
        self.needle_list = [l.get("needle") for l in structure_data]
        self.haystack_dir = haystack_dir
        self.question_list = [l.get("question") for l in structure_data]
        self.answers_list = [l.get("answer") for l in structure_data]
        
        # 保存 top_k 值 (移到这里，在使用前赋值)
        self.top_k_infscore = top_k_infscore

        # Set context lengths
        self.context_lengths = [round(x) for x in 
            torch.linspace(context_lengths_min, context_lengths_max, context_lengths_num_intervals).tolist()]
        
        # Set depth percentages based on the depths argument or use default
        default_depths = [2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65,68,71,74,77,80,83,86,89,92,95,98]
        if depths is not None:
            valid_depths = [d for d in depths if 0 <= d <= 100]
            if valid_depths:
                self.depth_percents = sorted(list(set(valid_depths)))
                print(f"Using specified depth percentages: {self.depth_percents}")
            else:
                print("Warning: No valid depth percentages provided via --depths. Using default.")
                self.depth_percents = default_depths
        else:
            self.depth_percents = default_depths
            print(f"Using default depth percentages: {self.depth_percents}")
        
        # Print combined info
        print(f"Context lengths: {self.context_lengths}")
        print(f"Top-k for InfScore analysis: {self.top_k_infscore}") # 现在可以安全打印

        # Initialize overall similarity score tracking
        self.all_similarity_scores = []
        
        # Load model and tokenizer
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        
        # Try to load with padding token to avoid issues later
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Added padding token (set to EOS token)")
        
        # Adjust device_map based on whether a specific GPU is specified
        if gpu_id is not None and torch.cuda.is_available():
            device_map = {"": self.device}
            print("Model will be loaded entirely on a single GPU")
        else:
            device_map = "auto"
            print("Model will be distributed across available devices automatically")
            
        # Report memory before model loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)  # Free memory in GB
                print(f"GPU {i} free memory before model loading: {free_memory:.2f} GB")
        
        # Choose appropriate dtype
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using model dtype: {model_dtype}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            device_map=device_map,     # Use specific device or auto-allocation
            trust_remote_code=True,
            low_cpu_mem_usage=True,      # Reduce CPU memory usage during loading
            output_attentions=True     # Ensure attention output is enabled
        ).eval()
        
        # Get model configuration for layer and head counts
        try:
            config = self.model.config
            self.layer_num = config.num_hidden_layers
            self.head_num = config.num_attention_heads
            print(f"Model has {self.layer_num} layers and {self.head_num} attention heads per layer")
        except AttributeError:
            print("Warning: Could not determine layer and head numbers from model config.")
            self.layer_num = 0
            self.head_num = 0

        # Initialize ALL attention accumulators
        self.head_infscore_accum = defaultdict(list)
        self.head_precision_accum = defaultdict(list)
        self.head_recall_accum = defaultdict(list)
        self.head_tp_user_accum = defaultdict(list)
        self.head_tn_user_accum = defaultdict(list)
        self.head_fp_user_accum = defaultdict(list)
        self.head_fn_user_accum = defaultdict(list)
        
        # Report memory after model loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)  # Free memory in GB
                print(f"GPU {i} free memory after model loading: {free_memory:.2f} GB")
        
        print("Model loading complete")
    
    def read_context_files(self, haystack_dir):
        """Read context files"""
        context = ""
        print(f"Reading files recursively from {haystack_dir}")
        
        files_found = glob.glob(f"{haystack_dir}/**/*.txt", recursive=True)
        if not files_found:
            print(f"Warning: No .txt files found in {haystack_dir}")
            return ""
            
        for file_path in files_found:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    context += f.read()
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                
        print(f"Read {len(context)} characters as background context")
        return context
    
    def encode_and_trim(self, context, context_length):
        """Encode and trim context to specified length"""
        tokens = self.tokenizer.encode(context)
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)
    
    def insert_needle(self, context, depth_percent, context_length):
        """Insert needle at specified depth"""
        # Encode needle and context
        tokens_needle = self.tokenizer.encode(self.needle)
        tokens_context = self.tokenizer.encode(context)
        
        # Consider buffer to leave space for question and answer
        effective_context_length = context_length - 200
        
        # Ensure context length is adequate
        if len(tokens_context) + len(tokens_needle) > effective_context_length:
            tokens_context = tokens_context[:effective_context_length - len(tokens_needle)]
            
        # Calculate insertion point based on depth
        if depth_percent == 100:
            insertion_point = len(tokens_context)
        elif depth_percent == 0:
            insertion_point = 0
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            
        # Try to insert at sentence boundaries
        period_tokens = set(self.tokenizer.encode('.?!'))
        original_point = insertion_point
        
        if 0 < insertion_point < len(tokens_context):
            steps = 0
            while insertion_point > 0 and tokens_context[insertion_point-1] not in period_tokens and steps < 50:
                insertion_point -= 1
                steps += 1
                
            if insertion_point == 0 or steps == 50:
                insertion_point = original_point
            else: # Revert if no suitable boundary found nearby or if search hit limit
                insertion_point = original_point

        print(f"Inserting needle at depth {depth_percent}%, index ~{insertion_point}")

        # Record needle position *before* inserting it into the context tokens
        self.needle_start = insertion_point
        self.needle_end = insertion_point + len(tokens_needle)

        # Build new context
        tokens_new_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]
        new_context = self.tokenizer.decode(tokens_new_context)
        
        return new_context
    
    def generate_context(self, context_length, depth_percent):
        """Generate context and insert needle"""
        # Read context files
        context = self.read_context_files(self.haystack_dir)
        
        # Encode and trim to required length
        context = self.encode_and_trim(context, context_length)
        
        # Insert needle
        context = self.insert_needle(context, depth_percent, context_length)
        
        return context
    
    def generate_answer(self, input_context, max_new_tokens=100):
        """Generate model answer"""
        input_ids = self.tokenizer.encode(input_context, return_tensors="pt")
        
        # Transfer to the device being used
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        print(f"Input length: {input_length} tokens")
            
        # Generate answer
        with torch.no_grad():
            try:
                # Manually collect garbage before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    output_attentions=True,          # Ensure this is True
                    return_dict_in_generate=True, # Ensure this is True
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0
                )
                
                generated_text = ""
                attentions = None

                if hasattr(outputs, 'sequences'):
                    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                    # 新增：记录每个生成的 token（用于后续分类）
                    self.generated_token_ids = generated_ids.cpu().tolist()
                    self.generated_tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in generated_ids]
                else:
                    print("Warning: outputs object does not have 'sequences' attribute.")
                    self.generated_token_ids = []
                    self.generated_tokens = []

                if hasattr(outputs, 'attentions') and outputs.attentions:
                    attentions = outputs.attentions
                else:
                     print("Warning: outputs object does not have 'attentions' attribute or it is empty.")

                print(f"Generated {len(generated_text)} characters of response")
                return generated_text.strip(), attentions # Ensure return tuple
                
            except Exception as e:
                print(f"ERROR during generation: {e}")
                if "CUDA out of memory" in str(e):
                    for i in range(torch.cuda.device_count()):
                        free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)  # Free memory in GB
                        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        used_memory = total_memory - free_memory
                        print(f"GPU {i} memory: {used_memory:.2f} GB used / {total_memory:.2f} GB total")
                    print("Try reducing context_length with --max_len parameter")
                return "", None # Ensure return tuple on error
    
    def calculate_semantic_similarity(self, response, answer):
        """Calculate semantic similarity using sentence embeddings."""
        if self.similarity_model is None:
            print("Warning: Similarity model not loaded. Cannot calculate semantic similarity.")
            return 0.0 # Or perhaps None or -1 to indicate failure?
            
        if not response or not answer:
            return 0.0
            
        try:
            # 计算嵌入向量
            embedding1 = self.similarity_model.encode(response, convert_to_tensor=True)
            embedding2 = self.similarity_model.encode(answer, convert_to_tensor=True)

            # 计算余弦相似度
            cosine_scores = util.cos_sim(embedding1, embedding2)
            return cosine_scores.item() # 返回标量相似度
        except Exception as e:
            print(f"Error calculating semantic score: {e}")
            return 0.0 # Or handle error differently
    
    def classify_token_by_attention_behavior(self, attentions, step_idx, answer_source_indices, k=10):
        """
        根据注意力行为分类单个生成的 token

        参数：
        - answer_source_indices: Answer Source 的 token 位置集合（包括 Needle + Question）

        分类逻辑：
        1. 计算该 token 对 Answer Source 和 Context 的总注意力权重
        2. 判断是否 Answer-related（注意力主要在 Answer Source）
        3. 判断是否 Top-k（Top-k 注意力的主要位置）
        4. 映射到四个类别

        分类表（根据用户定义）：
        ┌─────────────────────┬──────────┬──────────┐
        │                     │ Top-k    │ Non-top-k│
        ├─────────────────────┼──────────┼──────────┤
        │ Answer Source       │ 🔴 RA    │ 🔵 Bias  │
        │ Non-Answer Source   │ 🟢 Dist  │ ⚫ Wide  │
        └─────────────────────┴──────────┴──────────┘
        """
        if not attentions or step_idx >= len(attentions):
            return None, {}

        step_att = attentions[step_idx]
        if not isinstance(step_att, tuple) or len(step_att) == 0:
            return None, {}

        num_layers = len(step_att)

        # 累积所有 head 的注意力权重
        total_weight_needle = 0.0
        total_weight_context = 0.0
        total_weight_topk_needle = 0.0
        total_weight_topk_context = 0.0
        total_weight_nontopk_needle = 0.0
        total_weight_nontopk_context = 0.0

        total_heads = 0

        for layer_idx in range(num_layers):
            layer_attentions = step_att[layer_idx]
            if not isinstance(layer_attentions, torch.Tensor):
                continue

            num_heads = layer_attentions.size(1)

            for head_idx in range(num_heads):
                try:
                    # 获取该 head 的注意力权重（最后一个生成位置）
                    attn_weights = layer_attentions[0, head_idx, -1, :]

                    # 计算 Top-k
                    actual_k = min(k, attn_weights.size(0))
                    values, indices = torch.topk(attn_weights, k=actual_k)
                    topk_indices_set = set(indices.cpu().tolist())

                    # 计算四个区域的权重（使用 Answer Source）
                    topk_in_answer = topk_indices_set.intersection(answer_source_indices)
                    topk_in_context = topk_indices_set - answer_source_indices
                    nontopk_in_answer = answer_source_indices - topk_indices_set

                    context_indices_set = set(range(attn_weights.size(0))) - answer_source_indices
                    nontopk_in_context = context_indices_set - topk_indices_set

                    # 累积权重
                    WRA = sum([attn_weights[i].item() for i in topk_in_answer])
                    WD = sum([attn_weights[i].item() for i in topk_in_context])
                    WR = sum([attn_weights[i].item() for i in nontopk_in_answer])
                    WN = sum([attn_weights[i].item() for i in nontopk_in_context])

                    total_weight_topk_needle += WRA
                    total_weight_topk_context += WD
                    total_weight_nontopk_needle += WR
                    total_weight_nontopk_context += WN

                    total_weight_needle += (WRA + WR)
                    total_weight_context += (WD + WN)

                    total_heads += 1

                except Exception as e:
                    continue

        if total_heads == 0:
            return None, {}

        # 归一化权重（平均每个 head）
        avg_weight_needle = total_weight_needle / total_heads
        avg_weight_context = total_weight_context / total_heads
        avg_weight_topk_needle = total_weight_topk_needle / total_heads
        avg_weight_topk_context = total_weight_topk_context / total_heads
        avg_weight_nontopk_needle = total_weight_nontopk_needle / total_heads
        avg_weight_nontopk_context = total_weight_nontopk_context / total_heads

        # 判断 1: 是否 Answer-related（注意力主要在 Answer Source）
        is_answer_related = avg_weight_needle > avg_weight_context

        # 判断 2: 是否 Top-k（Top-k 权重占主导）
        total_topk_weight = avg_weight_topk_needle + avg_weight_topk_context
        total_nontopk_weight = avg_weight_nontopk_needle + avg_weight_nontopk_context
        is_topk = total_topk_weight > total_nontopk_weight

        # 映射到四个类别（根据用户定义的分类表）
        if is_answer_related and is_topk:
            category = "🔴 Retrieval Augmented"
        elif not is_answer_related and is_topk:
            category = "🟢 Distraction"
        elif not is_answer_related and not is_topk:
            category = "⚫ Widespread"
        else:  # is_answer_related and not is_topk
            category = "🔵 Bias"

        details = {
            "category": category,
            "is_answer_related": is_answer_related,
            "is_topk": is_topk,
            "avg_weight_answer_source": avg_weight_needle,  # 重命名以反映实际含义
            "avg_weight_context": avg_weight_context,
            "avg_weight_topk": total_topk_weight,
            "avg_weight_nontopk": total_nontopk_weight,
            "total_heads": total_heads
        }

        return category, details

    def analyze_attention_infscore(self, attentions):
        """
        Analyze attention heads using the comprehensive framework.
        Calculates standard Precision, Recall, InfScore (F1) for head focus,
        and also counts based on the user-defined 2x2 attention behavior matrix:
        TP_User: Top-k attention within Needle.
        TN_User: Non-Top-k attention within Needle.
        FP_User: Top-k attention outside Needle.
        FN_User: Non-Top-k attention outside Needle.

        同时分类每个生成的 token。
        """
        if not attentions or self.layer_num == 0 or self.head_num == 0:
            print("Warning: Cannot analyze attentions. Missing attention data or model config.")
            return {}

        if not isinstance(attentions, tuple) or len(attentions) == 0:
            print("Warning: Attention structure is not a tuple or is empty.")
            return {}
        
        is_nested_tuple = all(isinstance(step_att, tuple) for step_att in attentions)
        if not is_nested_tuple or len(attentions[0]) == 0:
             print("Warning: Unexpected attention structure within steps.")
             return {}
        
        num_layers = self.layer_num
        num_heads = self.head_num
        total_steps = len(attentions)

        head_precision_weighted_steps = defaultdict(list)
        head_recall_weighted_steps = defaultdict(list)
        head_f1_weighted_steps = defaultdict(list)
        head_tp_user_steps = defaultdict(list)
        head_tn_user_steps = defaultdict(list)
        head_fp_user_steps = defaultdict(list)
        head_fn_user_steps = defaultdict(list)

        # 新增：记录每个 token 的分类
        token_classifications = []

        k = self.top_k_infscore

        context_len = -1
        first_valid_tensor = None
        for step_att in attentions:
            if step_att and isinstance(step_att, tuple) and step_att[0] is not None and hasattr(step_att[0], 'shape'):
                try:
                    context_len = step_att[0].shape[-1]
                    first_valid_tensor = step_att[0]
                    break
                except (IndexError, AttributeError, TypeError):
                    continue

        if context_len == -1:
            print("Warning: Could not reliably determine context length from attention tensors.")
            return {}
        # print(f"Context length from attention tensor: {context_len}")

        valid_start = max(0, min(getattr(self, 'needle_start', 0), context_len - 1))
        valid_end = min(getattr(self, 'needle_end', 0), context_len)
        needle_len = valid_end - valid_start

        # print(f"Analyzing needle from {valid_start} to {valid_end} (length {needle_len})")

        if needle_len <= 0:
            # print("Warning: Invalid needle length or position for analysis.")
            return {}

        needle_indices_set = set(range(valid_start, valid_end))

        # 计算 Question 的位置（Answer Source = Needle + Question）
        # Question 在输入末尾，倒推位置
        question_start = context_len - len(self.tokenizer.encode(f"answer the question: {self.question}\nAnswer:", add_special_tokens=False))
        question_end = context_len - len(self.tokenizer.encode("\nAnswer:", add_special_tokens=False))
        question_indices_set = set(range(max(0, question_start), min(context_len, question_end)))

        # Answer Source = Needle + Question
        answer_source_indices = needle_indices_set | question_indices_set

        attention_device = first_valid_tensor.device if first_valid_tensor is not None else self.device
        needle_indices_tensor = torch.tensor(list(needle_indices_set), device=attention_device, dtype=torch.long)

        for step_idx in range(total_steps):
             if not isinstance(attentions[step_idx], tuple) or len(attentions[step_idx]) != num_layers:
                 continue

             # 新增：分类当前 step 生成的 token（使用 Answer Source）
             token_category, token_details = self.classify_token_by_attention_behavior(
                 attentions, step_idx, answer_source_indices, k=k
             )
             if token_category is not None:
                 token_classifications.append({
                     "step": step_idx,
                     "category": token_category,
                     "details": token_details
                 })

             for layer_idx in range(num_layers):
                 layer_attentions = attentions[step_idx][layer_idx]
                 if not isinstance(layer_attentions, torch.Tensor) or layer_attentions.dim() != 4 or layer_attentions.size(1) != num_heads or layer_attentions.size(3) != context_len:
                     continue

                 for head_idx in range(num_heads):
                     try:
                         attn_weights = layer_attentions[0, head_idx, -1, :]
                         if attn_weights.numel() != context_len:
                             continue

                         actual_k = min(k, context_len)
                         if actual_k <= 0: continue
                         if actual_k > attn_weights.size(0): actual_k = attn_weights.size(0)

                         values, indices = torch.topk(attn_weights, k=actual_k)
                         topk_indices_set = set(indices.cpu().tolist())

                         TP_User = len(topk_indices_set.intersection(needle_indices_set))
                         FP_User = len(topk_indices_set) - TP_User
                         non_topk_in_needle = needle_indices_set - topk_indices_set
                         # Calculate FN and TN based on standard definitions
                         # FN_User_standard: Needle tokens that were NOT in top-k (missed positives)
                         FN_User_standard = len(non_topk_in_needle)
                         # TN_User_standard: Non-needle tokens that were NOT in top-k (correctly ignored negatives)
                         TN_User_standard = (context_len - needle_len) - FP_User

                         WTP, WFP = 0.0, 0.0
                         for idx_tensor, weight_tensor in zip(indices, values):
                             idx = idx_tensor.item()
                             weight = weight_tensor.item()
                             if idx in needle_indices_set: WTP += weight
                             else: WFP += weight

                         total_needle_weight = 0.0
                         if needle_indices_tensor.numel() > 0 and needle_indices_tensor.max() < attn_weights.shape[0]:
                             total_needle_weight = torch.sum(attn_weights.gather(0, needle_indices_tensor)).item()
                         
                         WFN = max(0.0, total_needle_weight - WTP)

                         precision_weighted = WTP / (WTP + WFP) if (WTP + WFP) > 1e-9 else 0.0
                         recall_weighted = WTP / (WTP + WFN) if (WTP + WFN) > 1e-9 else 0.0
                         inf_score_weighted = 2 * (precision_weighted * recall_weighted) / (precision_weighted + recall_weighted) if (precision_weighted + recall_weighted) > 1e-9 else 0.0

                         head_key = f"{layer_idx}-{head_idx}"
                         head_precision_weighted_steps[head_key].append(precision_weighted)
                         head_recall_weighted_steps[head_key].append(recall_weighted)
                         head_f1_weighted_steps[head_key].append(inf_score_weighted)
                         head_tp_user_steps[head_key].append(TP_User)
                         head_tn_user_steps[head_key].append(TN_User_standard)
                         head_fp_user_steps[head_key].append(FP_User)
                         head_fn_user_steps[head_key].append(FN_User_standard)

                     except Exception as e:
                         # print(f"Error processing head {layer_idx}-{head_idx} at step {step_idx}: {e}")
                         continue

        final_results = {}
        all_heads = set(head_f1_weighted_steps.keys())
        if not all_heads: return {}

        avg_precision_weighted = {h: np.mean(head_precision_weighted_steps.get(h, [0])) for h in all_heads}
        avg_recall_weighted = {h: np.mean(head_recall_weighted_steps.get(h, [0])) for h in all_heads}
        avg_infscore_weighted = {h: np.mean(head_f1_weighted_steps.get(h, [0])) for h in all_heads}
        avg_tp_user = {h: np.mean(head_tp_user_steps.get(h, [0])) for h in all_heads}
        avg_tn_user = {h: np.mean(head_tn_user_steps.get(h, [0])) for h in all_heads}
        avg_fp_user = {h: np.mean(head_fp_user_steps.get(h, [0])) for h in all_heads}
        avg_fn_user = {h: np.mean(head_fn_user_steps.get(h, [0])) for h in all_heads}

        def normalize_scores(scores_dict):
             if not scores_dict: return scores_dict
             values = list(scores_dict.values())
             max_val, min_val = max(values), min(values)
             range_val = max_val - min_val
             if range_val < 1e-9:
                 avg_val = np.mean(values) if values else 0.0
                 return {k: 0.5 if avg_val > 1e-9 else 0.0 for k in scores_dict}
             return {k: (v - min_val) / range_val for k, v in scores_dict.items()}

        final_results["inf_scores_weighted_normalized"] = normalize_scores(avg_infscore_weighted)
        final_results["precision_scores_weighted_normalized"] = normalize_scores(avg_precision_weighted)
        final_results["recall_scores_weighted_normalized"] = normalize_scores(avg_recall_weighted)
        final_results["avg_infscore_weighted"] = avg_infscore_weighted
        final_results["avg_precision_weighted"] = avg_precision_weighted
        final_results["avg_recall_weighted"] = avg_recall_weighted
        final_results["avg_tp_user"] = avg_tp_user
        final_results["avg_tn_user"] = avg_tn_user
        final_results["avg_fp_user"] = avg_fp_user
        final_results["avg_fn_user"] = avg_fn_user

        # 新增：添加 token 分类结果
        final_results["token_classifications"] = token_classifications

        # 统计 token 分类
        token_stats = {
            "🔴 WRA": 0,
            "🟢 WD": 0,
            "🔵 WR": 0,
            "⚫ WN": 0
        }
        for tc in token_classifications:
            token_stats[tc["category"]] += 1

        final_results["token_statistics"] = token_stats
        final_results["total_tokens"] = len(token_classifications)

        return final_results

    def evaluate_and_log(self, context_length, depth_percent):
        """Evaluate model and log results"""
        # Generate context
        print(f"\n--- Test: Length={context_length}, Depth={depth_percent}% ---")
        context = self.generate_context(context_length, depth_percent)
        template = f"Based on the content in the book, answer the question: {self.question}\nAnswer:"
        input_context = context + template
        
        # Generate answer and get attentions
        test_start_time = datetime.now()
        response, attentions = self.generate_answer(input_context) # 获取 attentions
        test_duration = (datetime.now() - test_start_time).total_seconds()
        
        # Calculate similarity score
        similarity_score = self.calculate_semantic_similarity(response, self.answer)
        # score_percent is less meaningful for cosine similarity, maybe remove or adjust interpretation
        # score_percent = similarity_score * 100 
        
        # Log results dictionary (initialize before attention analysis)
        result = {
            'model': self.model_name,
            'context_length': context_length,
            'depth_percent': depth_percent,
            'question': self.question,
            'expected_answer': self.answer,
            'model_response': response,
            'semantic_similarity': similarity_score, 
            'duration_seconds': test_duration
            # Attention scores will be added below if available
        }
        
        # Print main results first
        print(f"Question: {self.question}")
        print(f"Expected answer: {self.answer}")
        print(f"Model response: {response}")
        print(f"Semantic Similarity: {similarity_score:.4f}") 
        print(f"Duration: {test_duration:.2f} seconds")

        # Analyze attention heads using InfScore method
        if attentions: # Check if attentions were successfully generated
            print("Analyzing attention matrices with InfScore method...")
            infscore_results = self.analyze_attention_infscore(attentions)

            if infscore_results:
                # 新增：打印 token 分类统计
                if "token_statistics" in infscore_results:
                    print("\n" + "="*80)
                    print("📊 Token Classification Statistics (Based on Head Behavior)")
                    print("="*80)
                    token_stats = infscore_results["token_statistics"]
                    total_tokens = infscore_results.get("total_tokens", 0)

                    for category, count in token_stats.items():
                        percentage = (count / total_tokens * 100) if total_tokens > 0 else 0
                        print(f"{category}: {count:3d} tokens ({percentage:5.1f}%)")

                    print(f"\nTotal tokens generated: {total_tokens}")
                    print("="*80)

                    # 新增：显示带颜色标注的生成文本
                    if "token_classifications" in infscore_results and hasattr(self, 'generated_tokens'):
                        token_classifications = infscore_results["token_classifications"]
                        print("\n📝 Generated Text with Token Classification:")
                        print("-"*80)

                        # ANSI 颜色代码
                        color_codes = {
                            "🔴 WRA": "\033[91m",  # 红色
                            "🟢 WD": "\033[92m",   # 绿色
                            "🔵 WR": "\033[94m",   # 蓝色
                            "⚫ WN": "\033[90m"    # 灰色
                        }
                        reset_code = "\033[0m"

                        # 构建带颜色的文本
                        colored_text = ""
                        for i, tc in enumerate(token_classifications):
                            if i < len(self.generated_tokens):
                                token_text = self.generated_tokens[i]
                                category = tc["category"]
                                color = color_codes.get(category, "")
                                colored_text += f"{color}{token_text}{reset_code}"

                        print(colored_text)
                        print("-"*80)

                        # 显示前 30 个 token 的详细分类
                        print("\n📋 Detailed Token Classifications (First 30):")
                        print("-"*130)
                        print(f"{'Step':<6} | {'Token':<20} | {'Category':<25} | {'Answer Source?':<16} | {'Top-k?':<8} | "
                              f"{'Answer Wt':<14} | {'Context Wt':<15}")
                        print("-"*130)

                        for i, tc in enumerate(token_classifications[:30]):
                            if i < len(self.generated_tokens):
                                step = tc["step"]
                                token_text = self.generated_tokens[i]
                                category = tc["category"]
                                details = tc["details"]

                                # 截断过长的 token
                                token_display = token_text[:20] if len(token_text) > 20 else token_text

                                # 提取详细信息
                                is_answer_related = "✅ Yes" if details.get("is_answer_related", False) else "❌ No"
                                is_topk = "✅ Yes" if details.get("is_topk", False) else "❌ No"
                                answer_weight = details.get("avg_weight_answer_source", 0.0)
                                context_weight = details.get("avg_weight_context", 0.0)

                                print(f"{step:<6} | {token_display:<20} | {category:<25} | {is_answer_related:<16} | {is_topk:<8} | "
                                      f"{answer_weight:<14.4f} | {context_weight:<15.4f}")
                        print("="*130 + "\n")
                # Add attention scores to the result dictionary for this run
                result['head_infscore_weighted_norm'] = infscore_results.get("inf_scores_weighted_normalized", {})
                result['head_precision_weighted_norm'] = infscore_results.get("precision_scores_weighted_normalized", {})
                result['head_recall_weighted_norm'] = infscore_results.get("recall_scores_weighted_normalized", {})
                result['head_avg_infscore_weighted'] = infscore_results.get("avg_infscore_weighted", {})
                result['head_avg_precision_weighted'] = infscore_results.get("avg_precision_weighted", {})
                result['head_avg_recall_weighted'] = infscore_results.get("avg_recall_weighted", {})
                result['head_avg_tp_user'] = infscore_results.get("avg_tp_user", {})
                result['head_avg_tn_user'] = infscore_results.get("avg_tn_user", {})
                result['head_avg_fp_user'] = infscore_results.get("avg_fp_user", {})
                result['head_avg_fn_user'] = infscore_results.get("avg_fn_user", {})

                # Get Top-10 heads (by normalized weighted InfScore) for this run
                top_infscore_heads = sorted(
                    result.get('head_infscore_weighted_norm', {}).items(), 
                    key=lambda item: item[1], reverse=True
                )[:10]
                result['top_infscore_heads'] = top_infscore_heads

                # Print Top-5 heads for this run
                if top_infscore_heads:
                    print("\nTop-5 heads by Weighted InfScore (Normalized) for this run:")
                    # Access scores directly from infscore_results for printing consistency
                    precision_scores_w_norm = infscore_results.get("precision_scores_weighted_normalized", {})
                    recall_scores_w_norm = infscore_results.get("recall_scores_weighted_normalized", {})
                    avg_tp_user_scores = infscore_results.get("avg_tp_user", {})
                    avg_fp_user_scores = infscore_results.get("avg_fp_user", {})
                    avg_tn_user_scores = infscore_results.get("avg_tn_user", {})
                    avg_fn_user_scores = infscore_results.get("avg_fn_user", {})
                    for head, score in top_infscore_heads[:5]:
                         prec_w_norm = precision_scores_w_norm.get(head, -1)
                         rec_w_norm = recall_scores_w_norm.get(head, -1)
                         tp_u = avg_tp_user_scores.get(head, -1)
                         tn_u = avg_tn_user_scores.get(head, -1)
                         fp_u = avg_fp_user_scores.get(head, -1)
                         fn_u = avg_fn_user_scores.get(head, -1)
                         print(f"  Head {head}: InfScore(W.Norm)={score:.4f} (P={prec_w_norm:.4f}, R={rec_w_norm:.4f}) | Avg TP={tp_u:.2f}, TN={tn_u:.2f}, FP={fp_u:.2f}, FN={fn_u:.2f}")

                # Accumulate scores only if similarity threshold is met
                if similarity_score > 0.2:
                    print(f"Accumulating head scores (Similarity {similarity_score:.4f} > 0.2)")
                    for head, score in infscore_results.get("avg_infscore_weighted", {}).items(): 
                         self.head_infscore_accum[head].append(score)
                    for head, score in infscore_results.get("avg_precision_weighted", {}).items():
                         self.head_precision_accum[head].append(score)
                    for head, score in infscore_results.get("avg_recall_weighted", {}).items():
                         self.head_recall_accum[head].append(score)
                    for head, count in infscore_results.get("avg_tp_user", {}).items():
                        self.head_tp_user_accum[head].append(count)
                    for head, count in infscore_results.get("avg_tn_user", {}).items(): 
                        self.head_tn_user_accum[head].append(count)
                    for head, count in infscore_results.get("avg_fp_user", {}).items():
                        self.head_fp_user_accum[head].append(count)
                    for head, count in infscore_results.get("avg_fn_user", {}).items():
                        self.head_fn_user_accum[head].append(count)
                else:
                    print(f"Skipping head score accumulation (Similarity {similarity_score:.4f} <= 0.2)")
            else:
                print("Could not calculate InfScores for attention heads.")
        else:
            print("No attention matrices available for analysis.")

        # Track overall similarity scores (always track this)
        self.all_similarity_scores.append(similarity_score)
        self.similarity_by_context_length[context_length].append(similarity_score)
        self.similarity_by_depth[depth_percent].append(similarity_score)

        print("-" * 40)
        
        # Clear GPU memory between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result
    
    def run_test(self):
        """Run test"""
        all_results = []
        total_scenarios = len(self.needle_list)
        total_configs = len(self.context_lengths) * len(self.depth_percents)
        print(f"\nStarting test run with {total_scenarios} scenarios, {total_configs} configurations per scenario")
        print(f"Total test cases: {total_scenarios * total_configs}")
        
        for ni in range(len(self.needle_list)):
            self.needle = self.needle_list[ni]
            self.question = self.question_list[ni]
            self.answer = self.answers_list[ni]
            
            print(f"\n====== Test Scenario {ni+1}/{total_scenarios} ======")
            print(f"Needle ({len(self.needle)} chars): {self.needle[:100]}...")
            print(f"Question: {self.question}")
            print(f"Expected answer: {self.answer}")
            
            test_count = 0
            
            for context_length in self.context_lengths:
                for depth_percent in self.depth_percents:
                    test_count += 1
                    print(f"\nRunning test {test_count}/{total_configs} for this scenario...")
                    try:
                        result = self.evaluate_and_log(context_length, depth_percent)
                        all_results.append(result)
                    except Exception as e:
                        print(f"ERROR in test: {e}")
                        error_result = {
                            'model': self.model_name,
                            'context_length': context_length,
                            'depth_percent': depth_percent,
                            'question': self.question,
                            'expected_answer': self.answer,
                            'model_response': f"ERROR: {str(e)}",
                            'duration_seconds': 0,
                            'semantic_similarity': 0.0  # Use new key for errors
                        }
                        all_results.append(error_result)
                        # Continue with next test despite error
                        
                    # Clear GPU memory between tests
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
        # Save all results
        results_dir = f'results/{self.model_name.replace("/", "_")}'
        os.makedirs(results_dir, exist_ok=True)
        
        # Use 'structure' in the filename for easy identification
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(results_dir, f'predictions_structure_{timestamp}.json')
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"ERROR saving results: {e}")
            # Try to save to a backup location
            backup_file = f'structure_results_backup_{timestamp}.json'
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to backup file: {backup_file}")
        
        # Generate similarity score summary (Reports overall similarity stats)
        self.generate_similarity_summary(results_dir, timestamp)

        # --- Add Head Importance Calculation based on Filtered Accumulation --- 
        def calculate_average_scores(counter_dict):
            return {head: np.mean(scores) if scores else 0
                   for head, scores in counter_dict.items() if scores}

        print("\nCalculating final average head scores (accumulated only if similarity > 0.2)...")
        avg_infscores = calculate_average_scores(self.head_infscore_accum)
        avg_precisions = calculate_average_scores(self.head_precision_accum)
        avg_recalls = calculate_average_scores(self.head_recall_accum)
        avg_tps_user = calculate_average_scores(self.head_tp_user_accum)
        avg_tns_user = calculate_average_scores(self.head_tn_user_accum)
        avg_fps_user = calculate_average_scores(self.head_fp_user_accum)
        avg_fns_user = calculate_average_scores(self.head_fn_user_accum)

        # Prepare data for head importance file
        importance_data = {
            'filter_condition': 'semantic_similarity > 0.2', 
            'average_infscores_weighted': avg_infscores, 
            'average_precisions_weighted': avg_precisions,
            'average_recalls_weighted': avg_recalls,
            'average_tp_user': avg_tps_user,
            'average_tn_user': avg_tns_user,
            'average_fp_user': avg_fps_user,
            'average_fn_user': avg_fns_user,
            'top_infscore_weighted_heads': sorted(avg_infscores.items(), key=lambda x: x[1], reverse=True)[:20],
        }

        # Save aggregated head importance scores to a specific file
        head_scores_file = os.path.join(results_dir, f'head_importance_filtered_sim04_{timestamp}.json')
        with open(head_scores_file, 'w', encoding='utf-8') as f:
            json.dump(importance_data, f, indent=2, ensure_ascii=False)
        print(f"Aggregated head importance scores (filtered by similarity > 0.4) saved to {head_scores_file}")

        # Print top heads based on the filtered average
        print("\n==== Top-10 Heads by Average Weighted InfScore (filtered runs where similarity > 0.2) ====")
        if not importance_data['top_infscore_weighted_heads']:
            print("No head scores met the similarity threshold for accumulation.")
        else:
            for head, score in importance_data['top_infscore_weighted_heads'][:10]:
                 prec = avg_precisions.get(head, -1)
                 rec = avg_recalls.get(head, -1)
                 tp_u = avg_tps_user.get(head, -1)
                 tn_u = avg_tns_user.get(head, -1)
                 fp_u = avg_fps_user.get(head, -1)
                 fn_u = avg_fns_user.get(head, -1)
                 num_runs = len(self.head_infscore_accum.get(head,[])) # Number of runs included
                 print(f"  Head {head}: InfScore(W.Avg)={score:.4f} (P={prec:.4f}, R={rec:.4f}) | Avg TP={tp_u:.2f}, TN={tn_u:.2f}, FP={fp_u:.2f}, FN={fn_u:.2f} | Runs Incl: {num_runs}")
        # --- End of Head Importance Calculation --- 

    def generate_similarity_summary(self, results_dir, timestamp):
        """Generate and save similarity score summary statistics"""
        if not self.all_similarity_scores:
            print("No similarity scores calculated, skipping summary generation")
            return
            
        # Calculate overall average similarity score
        # Handle potential None or non-float values if error handling in calculate_semantic_similarity changes
        valid_scores = [s for s in self.all_similarity_scores if isinstance(s, (int, float))]
        if not valid_scores:
             print("No valid similarity scores found, skipping summary generation")
             return
        average_similarity = np.mean(valid_scores)
        
        # Calculate average similarity score by context length
        avg_similarity_by_context = {
            str(length): float(np.mean([s for s in scores if isinstance(s, (int, float))])) 
            for length, scores in self.similarity_by_context_length.items() if any(isinstance(s, (int, float)) for s in scores)
        }
        
        # Calculate average similarity score by depth
        avg_similarity_by_depth = {
            str(depth): float(np.mean([s for s in scores if isinstance(s, (int, float))]))
            for depth, scores in self.similarity_by_depth.items() if any(isinstance(s, (int, float)) for s in scores)
        }
        
        # Create summary dictionary
        summary = {
            'model': self.model_name,
            'overall_average_similarity': float(average_similarity), # Use new key
            'avg_similarity_by_context_length': avg_similarity_by_context, # Use new key
            'avg_similarity_by_depth': avg_similarity_by_depth, # Use new key
            'total_tests_scored': len(valid_scores), # Report based on valid scores
            'total_tests_run': len(self.all_similarity_scores), # Total attempts
            'timestamp': timestamp
        }
        
        # Save summary
        summary_file = os.path.join(results_dir, f'similarity_summary_structure_{timestamp}.json') # Use new name
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        # Print summary
        print("\n=== Similarity Score Summary ===") # Use new name
        print(f"Model: {self.model_name}")
        print(f"Overall average similarity score: {average_similarity:.4f}")
        print(f"Total tests scored: {len(valid_scores)} / {len(self.all_similarity_scores)}")
        
        # Print similarity scores by depth
        print("\nSimilarity scores by needle depth (position):")
        for depth in sorted([int(d) for d in avg_similarity_by_depth.keys()]):
            similarity = avg_similarity_by_depth[str(depth)]
            print(f"  Depth {depth}%: {similarity:.4f}")
        
        # Print similarity scores by context length
        print("\nSimilarity scores by context length:")
        for length in sorted([int(l) for l in avg_similarity_by_context.keys()]):
            similarity = avg_similarity_by_context[str(length)]
            print(f"  Length {length}: {similarity:.4f}")
            
        print(f"\nSimilarity Summary saved to {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structure test script - semantic similarity & InfScore attention analysis (filtered by similarity > 0.2)")
    
    # Parameter settings
    parser.add_argument('--model_path', type=str, required=True, help='Model path or Hugging Face identifier')
    parser.add_argument('--haystack_dir', type=str, default='./haystack_for_detect_r2', help='Directory containing test data')
    parser.add_argument('--min_len', type=int, default=2048, help='Minimum context length in tokens')
    parser.add_argument('--max_len', type=int, default=2048, help='Maximum context length (default set to 4096)')
    parser.add_argument('--context_intervals', type=int, default=3, help='Number of context length intervals')
    parser.add_argument('--gpu', type=int, default=None, help='Specify single GPU ID (e.g. 0 or 1). Omit to use all available GPUs.')
    parser.add_argument('--top_k_infscore', type=int, default=10, help='Value of K for Top-K attention analysis in InfScore calculation.')
    parser.add_argument('--depths', type=int, nargs='+', default=None, help='Specify one or more depth percentages to test (e.g., --depths 15 50 95). Overrides the default list.')

    args = parser.parse_args()
    
    print("\n======= Structure Head InfScore Test (Similarity > 0.2 Filtered Accumulation) =======")
    print(f"Model: {args.model_path}")
    print(f"Context length range: {args.min_len} to {args.max_len}")
    if args.depths:
        print(f"Depth percentages: {args.depths}")
    if args.gpu is not None:
        print(f"Note: You specified GPU {args.gpu}. For distributed execution, remove the --gpu parameter.")
    print(f"Top-K for attention analysis: {args.top_k_infscore}")
    
    try:
        tester = SimpleLLMTester(
            model_name=args.model_path,
            haystack_dir=args.haystack_dir,
            context_lengths_min=args.min_len,
            context_lengths_max=args.max_len,
            context_lengths_num_intervals=args.context_intervals,
            gpu_id=args.gpu,
            top_k_infscore=args.top_k_infscore,
            depths=args.depths
        )
        
        tester.run_test()
        print("\nTesting complete!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 