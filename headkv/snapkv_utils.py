import warnings
import os

import torch
import time
import numpy as np
import json
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union, Any,Dict
from transformers.cache_utils import Cache, DynamicCache


class DynamicCacheSplitHeadFlatten(Cache):
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            # IMPORTANT (multi-GPU): tiny_api_cuda is a custom CUDA extension and does not
            # guard the current device internally. If the current CUDA device differs from
            # the cache tensor's device, launching the kernel may cause illegal memory access.
            cache_k = self.key_cache[layer_idx]
            cache_v = self.value_cache[layer_idx]
            cache_device = cache_k.device
            if isinstance(head_lens, torch.Tensor) and head_lens.device != cache_device:
                head_lens = head_lens.to(cache_device)
            if isinstance(cu_klen, torch.Tensor) and cu_klen.device != cache_device:
                cu_klen = cu_klen.to(cache_device)
            if isinstance(key_states, torch.Tensor) and key_states.device != cache_device:
                key_states = key_states.to(cache_device)
            if isinstance(value_states, torch.Tensor) and value_states.device != cache_device:
                value_states = value_states.to(cache_device)

            import nvtx
            copy_old_rng = nvtx.start_range("copy old")
            from tiny_api_cuda import update_flatten_view
            from contextlib import nullcontext
            ctx = torch.cuda.device(cache_device) if cache_device.type == "cuda" else nullcontext()
            with ctx:
                new_key_cache = update_flatten_view(cache_k.view(-1, dim), key_states.view(-1, dim), head_lens, cu_klen)
                new_value_cache = update_flatten_view(cache_v.view(-1, dim), value_states.view(-1, dim), head_lens, cu_klen)

            nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache


        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0

        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx = None, num_hidden_layers = None, pyram_mode = False, pyram_beta = 20):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers


    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            base_capacity = self.max_capacity_prompt - self.window_size
            min_num = base_capacity // self.pyram_beta
            max_num = base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            self.max_capacity_prompt = max_num - self.layer_idx * steps + self.window_size
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, max_capacity_prompt: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states


class AdaptiveSnapKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',base_capacity=None,floor = None,skip = None,normalize=None, layer_idx = None, num_hidden_layers = None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.floor_ratio = floor
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.skip_layer_nums = skip

        self.normalize = normalize
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None


    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self,  key_states, query_states, value_states):
        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        pass
        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        if self.layer_idx >= self.skip_layer_nums:
            adaptive_attn_score = sorted_attn_score
            length = adaptive_attn_score.size(dim=-1)
            if self.normalize:
                ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
                adaptive_attn_score = adaptive_attn_score*ratio_weight
            adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads)
            sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity,dim=-1).indices
            sorted_indices = sorted_indices//length
            # floor capacity set
            head_adaptive_capacity = torch.zeros((bsz,num_heads),device=_device,dtype = sorted_indices.dtype)
            head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
            assert head_adaptive_capacity.sum().item() == num_heads*self.base_capacity
            head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
        else:
            head_adaptive_capacity = torch.ones((bsz,num_heads),device=_device,dtype = sorted_attn_score_indices.dtype) * self.base_capacity
        
        
        
        # def sparsity_forward(
        #     attention_scores: torch.LongTensor = None,
        #     method: str = "matrix"
        #     ):
            
        #     attention_scores = torch.stack(attention_scores).squeeze(1) # [layer_idx, num_heads, seq_length]
            
        #     if method== "matrix":
                
        #         softmaxed_attention_scores = F.softmax(attention_scores, dim=-1)
        #         head_sparsity = torch.exp(softmaxed_attention_scores * 10).sum(dim=-1) # [layer_idx, num_head]
        #         layer_sparsity = head_sparsity.mean(dim=-1) # [layer_idx]
        #         fixed_layer_sparsity = max(layer_sparsity) - layer_sparsity + (max(layer_sparsity) - min(layer_sparsity)) / 5
        #         return fixed_layer_sparsity
                
        # head_sparisity = sparsity_forward(attn_score)
        # head_adaptive_capacity = torch.ones((bsz,num_heads),device=_device,dtype = sorted_attn_score_indices.dtype) * head_sparisity
        
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1

        # per head
        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0


        for head_idx in range(num_heads):
            cache_index = sorted_attn_score_indices[head_idx][...,:head_adaptive_capacity[0][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states,heads_value_states



class ReasonSnapKVCluster():
    _printed_all_layer_capacities = False # Class attribute for ensuring one-time print

    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',base_capacity=None, head_choice=None, beta=None, temp=None, layer_idx = None, num_hidden_layers = None, num_attention_heads=None, model=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.beta = beta
        self.temp = temp

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if head_choice == 'random':
            raise ValueError
        elif head_choice == 'copy':
            if 'llama' in model.lower():
                path = f'{root_path}/Important_Head/head_score/Meta-Llama-3-8B-Instruct_retrieval_heads.json'
            elif 'mistral' in model.lower():
                path = f'{root_path}/Important_Head/head_score/Mistral-7B-Instruct-v0.2_retrieval_heads.json'
            else:
                raise ValueError
        elif head_choice == 'reason':
            if 'llama' in model.lower():
                path = f'{root_path}/Important_Head/head_score/Meta-Llama-3-8B-Instruct_retrieval_reasoning_heads.json'
            elif 'mistral' in model.lower():
                path = f'{root_path}/Important_Head/head_score/Mistral-7B-Instruct-v0.2_retrieval_reasoning_heads.json'
            else:
                raise ValueError
        elif head_choice == 'sentence':
            if 'llama' in model.lower():
                path = f'{root_path}/Important_Head/head_score/meta-llama_Meta-Llama-3-8B-Instruct_sentence_level_head_infscores_list_20260218_164437.json'
            else:
                raise ValueError
        elif head_choice == 'dominant':
            if 'llama' in model.lower():
                path = f'{root_path}/Important_Head/head_score/meta-llama_Meta-Llama-3-8B-Instruct_dominant_head_infscores_list_20260219_125546.json'
            else:
                raise ValueError
        with open(path, 'r') as file:
            # head_list = json.loads(file.readline())
            head_list = json.load(file)
        head_score_list = [np.mean(l[1]) for l in head_list.items()]
        head_score_list = torch.tensor(head_score_list, dtype=torch.float32)
        head_score_list = torch.pow(head_score_list, self.temp)
        self.total_attention = head_score_list.reshape(self.num_hidden_layers, self.num_attention_heads)

        # Define the small fixed constants
        epsilon_L_fixed = 0.01
        epsilon_H_fixed_budget = 0.5

        # SA is the original unnormalized head scores (InfScore_ij^temp)
        SA = self.total_attention

        # Apply epsilon_L_fixed to adjust head scores based on layer context
        # SA_prime_ij = SA_ij + epsilon_L_fixed * (SA_ij / sum_k_over_heads_in_layer(SA_ik))
        layer_sums_SA = SA.sum(dim=1, keepdim=True) + 1e-9 # layer sums, add epsilon for stability
        SA_prime = SA + epsilon_L_fixed * (SA / layer_sums_SA)

        # Normalize the adjusted scores so they sum to 1
        SA_prime_normalized = SA_prime / (SA_prime.sum() + 1e-9) # global sum, add epsilon for stability

        # Original total dynamic pool and min_num (fixed part from beta split)
        num_total_heads = self.num_hidden_layers * self.num_attention_heads
        original_total_pool_capacity = (self.base_capacity // self.beta) * num_total_heads
        original_min_num_per_head = (self.base_capacity - self.base_capacity // self.beta)

        # Adjust the dynamic pool by subtracting the total budget reserved for epsilon_H_fixed_budget
        total_budget_for_H_safety = num_total_heads * epsilon_H_fixed_budget
        pool_for_proportional_allocation = original_total_pool_capacity - total_budget_for_H_safety
        pool_for_proportional_allocation = torch.clamp(torch.as_tensor(pool_for_proportional_allocation), min=0.0) # Ensure not negative

        # Calculate head capacity: proportional part + original min_num + epsilon_H_fixed_budget
        proportional_part = SA_prime_normalized * pool_for_proportional_allocation
        capacity_before_H_fixed_add = proportional_part + original_min_num_per_head
        
        self.head_capacity = torch.round(capacity_before_H_fixed_add + epsilon_H_fixed_budget).int()
        
        # Print detailed capacity information only once per class initialization run
        if not ReasonSnapKVCluster._printed_all_layer_capacities and \
           self.num_hidden_layers is not None and \
           self.num_attention_heads is not None and \
           self.head_capacity is not None and \
           self.head_capacity.shape == (self.num_hidden_layers, self.num_attention_heads):

            print(f"\n[INFO ReasonKV] Configuration for Head Capacities (printed once for class {self.__class__.__name__}):")
            print(f"[INFO ReasonKV] Model: {model}")
            print(f"[INFO ReasonKV] Head choice strategy: {head_choice}")
            print(f"[INFO ReasonKV] Total number of hidden layers: {self.num_hidden_layers}")
            print(f"[INFO ReasonKV] Total number of attention heads per layer: {self.num_attention_heads}")
            print(f"[INFO ReasonKV] Base capacity (for compression part, after window): {self.base_capacity}")
            print(f"[INFO ReasonKV] Window size: {self.window_size}")
            print(f"[INFO ReasonKV] Beta for capacity pool allocation: {self.beta}")
            print(f"[INFO ReasonKV] Temperature for head score scaling: {self.temp}")
            print(f"[INFO ReasonKV] Per-layer total and per-head retained KV counts:")

            for i in range(self.num_hidden_layers):
                layer_cap_per_head = self.head_capacity[i]
                total_capacity_for_layer_i = layer_cap_per_head.sum().item()
                print(f"[INFO ReasonKV]   Layer {i:02d}: Total Retained KV = {total_capacity_for_layer_i:3d}, Per-Head Retained KV = {layer_cap_per_head.tolist()}")
            
            overall_total_retained_kv = self.head_capacity.sum().item()
            print(f"[INFO ReasonKV] Sum of all retained KV slots across all layers and heads: {overall_total_retained_kv}")
            ReasonSnapKVCluster._printed_all_layer_capacities = True
            print("") # Newline for readability
        
        if self.layer_idx is not None: # 确保 layer_idx 有效
            if self.head_capacity is not None and self.layer_idx < self.head_capacity.shape[0]:
                current_layer_per_head_cap = self.head_capacity[self.layer_idx]
                current_layer_total_cap = current_layer_per_head_cap.sum().item()
                # Original print: f"[DEBUG ReasonKV Init] Layer {self.layer_idx} - Specific head_capacity for this layer: {self.head_capacity[self.layer_idx]}"
                print(f"[DEBUG ReasonKV Init] Current Layer ({self.layer_idx}): Per-Head Retained KV = {current_layer_per_head_cap.tolist()}, Total for this layer = {current_layer_total_cap}")
            else:
                print(f"[DEBUG ReasonKV Init] Current Layer ({self.layer_idx}): Data not available or index out of bounds for head_capacity tensor (shape: {self.head_capacity.shape if self.head_capacity is not None else 'None'})")
            
    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self,  key_states, query_states, value_states):
        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        _,indices = attn_score.sort(dim=-1,descending=True)

        indices = indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1

        # per head
        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0


        for head_idx in range(num_heads):
            cache_index = indices[head_idx][...,:self.head_capacity[self.layer_idx][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states,heads_value_states


def init_pyramidkv(self):
    assert hasattr(self.config, 'window_size'), "window_size not set"
    assert hasattr(self.config, 'kernel_size'), "kernel_size not set"
    assert hasattr(self.config, "pooling"), "pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    if not hasattr(self.config, "pyram_beta"):
        self.config.pyram_beta = 20
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size, 
            max_capacity_prompt = self.config.base_capacity,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            pyram_mode = self.config.pyram_beta,
            )

def init_snapkv(self):

    assert hasattr(self.config, 'window_size'), "window_size not set"
    assert hasattr(self.config, 'kernel_size'), "kernel_size not set"
    assert hasattr(self.config, "pooling"), "pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size, 
            max_capacity_prompt = self.config.base_capacity,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,

            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            )
        print(f"Compress config(Snap): window_size={self.kv_cluster.window_size}, max_capacity_prompt={self.kv_cluster.max_capacity_prompt}, kernel_size={self.kv_cluster.kernel_size}, pooling={self.kv_cluster.pooling}")

def init_reason_snapkv(self):
    assert hasattr(self.config,'window_size'),"window_size not set"
    assert hasattr(self.config,'kernel_size'),"kernel_size not set"
    assert hasattr(self.config,"pooling"),"pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    assert hasattr(self.config, 'head_choice'), "head_choice not set"
    assert hasattr(self.config, 'beta'), "beta not set"
    assert hasattr(self.config, 'temp'), 'temp not set'

    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = ReasonSnapKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.base_capacity,
            head_choice=self.config.head_choice,
            beta=self.config.beta,
            temp=self.config.temp,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            model=self.config._name_or_path
            )


def init_headkv(self):
    assert hasattr(self.config,'window_size'),"window_size not set"
    assert hasattr(self.config,'kernel_size'),"kernel_size not set"
    assert hasattr(self.config,"pooling"),"pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    assert hasattr(self.config,"floor"),"floor not set"
    assert self.config.floor is not None


    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AdaptiveSnapKVCluster(
            window_size = self.config.window_size,
            base_capacity=self.config.base_capacity,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            floor= self.config.floor,
            skip = self.config.skip,
            layer_idx = self.layer_idx,
            normalize = self.config.normalize,
            num_hidden_layers = self.config.num_hidden_layers,
            )





