# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import importlib
import logging
import os
import time
import random
from copy import deepcopy
from typing import Dict, List, Counter, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.tools.base_tool import BaseTool
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, _pre_process_inputs, _repeat_interleave
import math

# 尝试导入sentence-transformers用于语义相似度计算
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, will use fallback embedding method")

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _load_tool_from_config(tool_config: DictConfig) -> BaseTool:
    """Dynamically loads a tool from its configuration."""

    logger.error(f"Failed to import config: {tool_config}")


    module_path, class_name = tool_config.class_path.rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        
        tool_class = getattr(module, class_name)
        
        tool_params = OmegaConf.to_container(tool_config.get('params', {}), resolve=True)
        
        tool_instance = tool_class(**tool_params)
        
        return tool_instance
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find class {class_name} in module {module_path}: {e}")
        raise
    except TypeError as e:
        logger.error(f"Failed to instantiate {class_name} with provided parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading tool from {tool_config.class_path}: {e}")
        raise


class vLLMRolloutWithTools(vLLMRollout):
    """
    An advanced vLLM rollout engine capable of handling multiple tools like
    code interpreters and search engines during generation.

    This class extends vLLMRollout by orchestrating a multi-step generation
    process where the language model can emit special tokens to trigger external
    tools. The tool outputs are then fed back into the model to continue
    generation.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer

        # 从配置中获取beam search相关参数
        self.initial_rollouts = self.config.get("initial_rollouts", self.config['n'])
        self.beam_size = self.config.get("beam_size", 1)
        self.branch_probability = self.config.get("branch_probability", 0.5)
        self.entropy_weight = self.config.get("entropy_weight", 0.5)
        
        # 从配置中获取工具设置
        tools_config = self.config.get("tools", OmegaConf.create({}))

        # 获取工具通用配置
        self.tool_call_limit = tools_config.get("call_limit", 5)
        self.max_tool_workers = tools_config.get("max_workers", 64)
        self.tool_timeout = tools_config.get("timeout", 120)

        # 其他可能的工具通用配置
        self.tool_retry_count = tools_config.get("retry_count", 3)
        self.tool_verbose_logging = tools_config.get("verbose_logging", False)

        
        self.tools: Dict[str, BaseTool] = {}
        if "tool_instances" in tools_config:
            for tool_name, tool_config in tools_config.tool_instances.items():
                logger.info(f"Loading tool '{tool_name}' from {tool_config.class_path}")
                try:
                    tool_instance = _load_tool_from_config(tool_config)
                    self.tools[tool_instance.trigger_tag] = tool_instance
                except Exception as e:
                    logger.error(f"Could not initialize tool '{tool_name}'. Please check your configuration. Error: {e}")
                    if tools_config.get("fail_on_error", False):
                        raise

        self.stop_sequences = [f"</{tag}>" for tag in self.tools.keys()]
        self.logprobs = 10 # entropy
        self.initial_entropy_dict = {}  # record initial entropy of active indice
        self.entropy_history = []  # record entropy history for analysis
        self.entropy_recorder = None  # external entropy recorder

        # 语义熵和噪音过滤相关配置
        self.use_semantic_entropy = self.config.get("use_semantic_entropy", True)
        self.semantic_entropy_threshold = self.config.get("semantic_entropy_threshold", 0.3)  # 语义熵阈值，低于此值不分叉
        self.noise_filter_threshold = self.config.get("noise_filter_threshold", 0.5)  # 噪音过滤阈值，相似度低于此值认为是噪音
        self.embedding_model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")  # 默认使用轻量级模型
        
        # 新的branching逻辑配置
        self.use_new_branching = self.config.get("use_new_branching", True)  # 是否使用新的branching逻辑
        self.branching_semantic_entropy_threshold = self.config.get("branching_semantic_entropy_threshold", 0.3)  # branching语义熵阈值
        self.num_candidates_per_prompt = self.config.get("num_candidates_per_prompt", 3)  # 每个prompt生成的候选回答数
        
        # 初始化embedding模型用于语义相似度计算
        self.embedding_model = None
        if self.use_semantic_entropy and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Initialized embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model {self.embedding_model_name}: {e}")
                self.embedding_model = None
                self.use_semantic_entropy = False
        
        # 存储每个分支的tool results用于噪音过滤
        self.tool_results_dict = {}  # {idx: [tool_result1, tool_result2, ...]}
        # 存储每个分支的生成文本用于语义熵计算
        self.branch_texts_dict = {}  # {orig_sample: {idx: text}} 用于计算同一原始样本的不同分支的语义熵
        # 存储每个样本的轮次信息（用于判断是否是第一轮）
        self.sample_round_dict = {}  # {idx: round_number} 记录每个样本的轮次
        # 存储每个样本的候选回答（用于新的branching逻辑）
        self.candidate_responses_dict = {}  # {orig_sample: [(prefix, response1), (prefix, response2), (prefix, response3)]}

        if not self.tools:
            logger.warning(
                "vLLMRolloutWithTools initialized, but no tools were configured.")

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_tool_workers)

    def __del__(self):
        self.executor.shutdown(wait=False)

    def _extract_content(self, text: str, tag: str) -> str:
        """Extracts content from within the last <tag>...</tag> block."""
        try:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            logger.warning(
                f"Could not extract content for tag '{tag}' from text: {text}")
            return ""

    def _execute_tool_with_retry(self, tool, content):
        retry_count = 0
        start_time = time.time()
        success = False
        
        while retry_count < self.tool_retry_count:
            try:
                logger.debug(f"Try@@@@@@@@@@@@@Tool {tool.trigger_tag} \n Tool content————{content}")

                result_text = tool.execute(content)
                if result_text:
                    success = True
                    execution_time = time.time() - start_time
                    logger.debug(f"Tool {tool.trigger_tag} executed successfully.")
                    logger.debug(f"Tool content————{content} result_text: {result_text}")
                    return {
                        "success": True,
                        "retry_count": retry_count,
                        "execution_time": execution_time,
                        "result": result_text
                    }
                else:
                    logger.warning(f"Tool({tool.trigger_tag}) returned empty output. Retrying {retry_count + 1}/{self.tool_retry_count}")
                    logger.debug(f"Tool content————{content} result_text: {result_text}")
                    retry_count += 1
            except Exception as e:
                logger.error(f"Tool({tool.trigger_tag}) execution failed. Retrying {retry_count + 1}/{self.tool_retry_count}: {e}")
                logger.debug(f"Tool content————{content} result_text: {result_text}")
                retry_count += 1
        
        execution_time = time.time() - start_time
        logger.warning(f"Tool({tool.trigger_tag}) execution failed after {self.tool_retry_count} retries. Appending EOS.")
        return {
            "success": False,
            "retry_count": retry_count,
            "execution_time": execution_time,
            "result": ""
        }

    def _calc_entropy(self, logprobs):
            if not logprobs:
                return 0.0
            p_list = [math.exp(l) for l in logprobs]
            entropy = -sum(p * l for p, l in zip(p_list, logprobs))
            return entropy

    def _calc_semantic_entropy(self, texts: List[str]) -> float:
        """
        计算多个文本之间的语义熵（基于语义相似度）
        如果文本语义相似度高，则熵低；如果语义差异大，则熵高
        
        Args:
            texts: 要比较的文本列表
            
        Returns:
            语义熵值（归一化到0-1之间）
        """
        if not texts or len(texts) < 2:
            return 0.0
        
        if not self.embedding_model:
            # 如果没有embedding模型，使用简单的文本相似度作为fallback
            logger.warning("Embedding model not available, using fallback method")
            return 0.5  # 返回中等熵值
        
        try:
            # 计算每个文本的embedding
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            
            # 计算所有文本对之间的余弦相似度
            similarities = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    # 余弦相似度
                    sim = np.dot(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            if not similarities:
                return 0.0
            
            # 语义熵：相似度越低，熵越高
            # 使用1减去平均相似度作为熵的度量
            avg_similarity = np.mean(similarities)
            semantic_entropy = 1.0 - avg_similarity
            
            return float(semantic_entropy)
        except Exception as e:
            logger.error(f"Error calculating semantic entropy: {e}")
            return 0.5  # 出错时返回中等熵值

    def _calc_pairwise_semantic_entropy(self, texts: List[str]) -> List[float]:
        """
        计算文本列表中所有文本对之间的语义熵
        
        Args:
            texts: 要比较的文本列表
            
        Returns:
            所有文本对的语义熵列表（1 - 相似度）
        """
        if not texts or len(texts) < 2:
            return []
        
        if not self.embedding_model:
            return []
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            pairwise_entropies = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    sim = np.dot(embeddings[i], embeddings[j])
                    entropy = 1.0 - sim
                    pairwise_entropies.append(entropy)
            return pairwise_entropies
        except Exception as e:
            logger.error(f"Error calculating pairwise semantic entropy: {e}")
            return []

    def _noise_filter(self, branch_text: str, tool_results: List[str]) -> bool:
        """
        噪音过滤器：判断分支是否基于tool results（grounded）
        
        Args:
            branch_text: 分支生成的文本
            tool_results: 该分支对应的所有tool results列表
            
        Returns:
            True: 分支是有效的（grounded或包含标准答案）
            False: 分支是噪音（ungrounded）
        """
        if not tool_results:
            # 如果没有tool results，无法判断，默认允许
            return True
        
        if not self.embedding_model:
            # 如果没有embedding模型，使用简单的文本匹配作为fallback
            logger.warning("Embedding model not available, using fallback method for noise filter")
            # 简单检查：如果分支文本中包含tool result中的关键词，认为grounded
            branch_lower = branch_text.lower()
            for tool_result in tool_results:
                if tool_result and len(tool_result) > 10:  # 只检查较长的tool results
                    # 提取tool result中的关键词（简单方法：取前50个字符）
                    keywords = tool_result[:50].lower().split()[:5]  # 取前5个词
                    if any(keyword in branch_lower for keyword in keywords if len(keyword) > 3):
                        return True
            return False
        
        try:
            # 计算分支文本与所有tool results的embedding相似度
            all_texts = [branch_text] + tool_results
            embeddings = self.embedding_model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)
            
            branch_embedding = embeddings[0]
            tool_embeddings = embeddings[1:]
            
            # 计算分支与每个tool result的相似度，取最大值
            similarities = [np.dot(branch_embedding, tool_emb) for tool_emb in tool_embeddings]
            max_similarity = max(similarities) if similarities else 0.0
            
            # 如果最大相似度超过阈值，认为分支是grounded的
            is_valid = max_similarity >= self.noise_filter_threshold
            
            logger.debug(f"Noise filter: branch_text length={len(branch_text)}, "
                        f"tool_results count={len(tool_results)}, "
                        f"max_similarity={max_similarity:.3f}, "
                        f"threshold={self.noise_filter_threshold}, "
                        f"is_valid={is_valid}")
            
            return is_valid
        except Exception as e:
            logger.error(f"Error in noise filter: {e}")
            return True  # 出错时默认允许

    @GPUMemoryLogger(role="vllm rollout spmd with tools", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.size(0)
        
        # 初始化工具调用统计信息
        tool_metrics = {
            "tools/total_calls": 0,
            "tools/successful_calls": 0,
            "tools/failed_calls": 0,
            "tools/total_execution_time": 0.0,
            "tools/avg_execution_time": 0.0,
            "tools/max_execution_time": 0.0,
            "tools/max_retries": 0,
            "tools/total_retries": 0,
            "tools/call_limit_reached_count": 0,
        }
        
        # 每个工具的统计信息
        calls_per_tool = Counter()
        success_per_tool = Counter()
        total_time_per_tool = Counter()

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        
        # 更新采样参数设置
        beam_size = self.beam_size
        if not do_sample:
            kwargs.update({
                'best_of': 1, 'top_p': 1.0, 'top_k': -1,
                'min_p': 0.0, 'temperature': 0, 'n': 1
            })
            beam_size = 1
        elif is_validate:
            kwargs.update({
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1  # 验证模式下使用单个样本
            })
            beam_size = 1
        
        # fix oov error
        kwargs["allowed_token_ids"] = list(self.tokenizer.get_vocab().values())

        with self.update_sampling_params(**kwargs):
            num_samples = self.sampling_params.n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

            prompt_token_ids_list = [_pre_process_inputs(self.pad_token_id, prompt) for prompt in input_ids]

            # State for each sample in the batch
            # 为每个样本创建初始rollout，数量由initial_rollouts控制
            initial_rollouts = self.initial_rollouts
            initial_rollouts = min(initial_rollouts, num_samples)  # 但不超过num_samples

            curr_inputs = []
            init_inputs = []
            result_masks = []
            call_counters = []
            active_indices = []
            
            # 初始化tool results和branch texts存储
            self.tool_results_dict = {}
            self.branch_texts_dict = {i: {} for i in range(batch_size)}
            self.sample_round_dict = {}  # 记录每个样本的轮次
            self.candidate_responses_dict = {}  # 存储候选回答
            
            # 创建初始样本
            for i, ids in enumerate(prompt_token_ids_list):
                for _ in range(initial_rollouts):
                    curr_inputs.append(ids.copy())
                    init_inputs.append(ids.copy())
                    result_masks.append([])
                    call_counters.append(0)
                    idx = len(curr_inputs) - 1
                    active_indices.append(idx)
                    # 初始化tool results存储
                    self.tool_results_dict[idx] = []
                    # 初始化轮次信息（第一轮为0）
                    self.sample_round_dict[idx] = 0
            
            # Track rollouts per original sample
            rollouts_per_sample = [initial_rollouts] * batch_size  # 每个样本初始有initial_rollouts个rollout
            # 初始时每个样本有多个索引
            sample_to_indices = {i: [i * initial_rollouts + j for j in range(initial_rollouts)] for i in range(batch_size)}

            max_len = self.config.response_length

            # 获取标准答案（从reward_model中获取，与reward_manager保持一致）
            ground_truth_answers = []
            if prompts.non_tensor_batch and "reward_model" in prompts.non_tensor_batch:
                reward_models = prompts.non_tensor_batch["reward_model"]
                if isinstance(reward_models, list):
                    # 批次数据：每个元素是一个字典，包含 {"style": "rule", "ground_truth": solution}
                    ground_truth_answers = [
                        rm.get("ground_truth", "") if isinstance(rm, dict) else "" 
                        for rm in reward_models
                    ]
                elif isinstance(reward_models, dict):
                    # 单个样本：直接获取ground_truth
                    ground_truth_answers = [reward_models.get("ground_truth", "")]
            # 兼容旧的数据格式（向后兼容）
            elif prompts.meta_info and "ground_truth" in prompts.meta_info:
                ground_truth_answers = prompts.meta_info.get("ground_truth", [])
            elif prompts.meta_info and "answers" in prompts.meta_info:
                ground_truth_answers = prompts.meta_info.get("answers", [])
            elif prompts.non_tensor_batch and "ground_truth" in prompts.non_tensor_batch:
                ground_truth_answers = prompts.non_tensor_batch.get("ground_truth", [])
            elif prompts.non_tensor_batch and "answers" in prompts.non_tensor_batch:
                ground_truth_answers = prompts.non_tensor_batch.get("answers", [])
            
            # 确保ground_truth_answers长度与batch_size一致
            while len(ground_truth_answers) < batch_size:
                ground_truth_answers.append("")

            while active_indices:
                active_prompts = [curr_inputs[i] for i in active_indices]
                logger.debug(f"rollouts_per_sample: {rollouts_per_sample}")
                logger.debug(f"active_indices: {active_indices}")
                logger.debug(f"active_prompts: {active_prompts}")

                # 判断每个active样本的轮次，决定生成多少个候选
                # 第一轮：生成1个；后续轮次：生成num_candidates_per_prompt个
                active_prompts_expanded = []
                active_indices_expanded = []
                active_round_info = []  # 记录每个prompt对应的原始索引和轮次
                
                for idx in active_indices:
                    round_num = self.sample_round_dict.get(idx, 0)
                    if round_num == 0 or not self.use_new_branching:
                        # 第一轮或未启用新branching：生成1个
                        active_prompts_expanded.append(curr_inputs[idx])
                        active_indices_expanded.append(len(active_prompts_expanded) - 1)
                        active_round_info.append({"orig_idx": idx, "round": round_num, "candidate_idx": 0})
                    else:
                        # 后续轮次：生成num_candidates_per_prompt个
                        for candidate_idx in range(self.num_candidates_per_prompt):
                            active_prompts_expanded.append(curr_inputs[idx])
                            active_indices_expanded.append(len(active_prompts_expanded) - 1)
                            active_round_info.append({"orig_idx": idx, "round": round_num, "candidate_idx": candidate_idx})

                # Update max_tokens for each active sample
                with self.update_sampling_params(
                    n=1,
                    stop=self.stop_sequences,
                    max_tokens=max(1, max((max_len - (len(curr_inputs[i]) - len(init_inputs[i])) for i in active_indices))),
                    detokenize=True,
                    logprobs = self.logprobs
                ):
                    outputs = self.inference_engine.generate(
                        prompt_token_ids=active_prompts_expanded,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )
                # ========== 处理outputs，按原始样本分组 ==========
                # 将outputs按原始样本分组，处理多个候选回答
                outputs_by_orig_idx = {}  # {orig_idx: [output1, output2, output3]}
                for i, info in enumerate(active_round_info):
                    orig_idx = info["orig_idx"]
                    if orig_idx not in outputs_by_orig_idx:
                        outputs_by_orig_idx[orig_idx] = []
                    outputs_by_orig_idx[orig_idx].append({
                        "output": outputs[i],
                        "candidate_idx": info["candidate_idx"],
                        "round": info["round"]
                    })
                
                # ========== 新的Branching逻辑：在调用tool之前进行判断 ==========
                # 对于每个原始样本，如果是后续轮次且有多个候选，进行branching判断
                selected_indices = []  # 最终选择的索引列表
                new_branches_to_add = []  # 需要添加的新分支
                
                for orig_idx, candidate_outputs in outputs_by_orig_idx.items():
                    round_num = self.sample_round_dict.get(orig_idx, 0)
                    
                    if round_num == 0 or not self.use_new_branching or len(candidate_outputs) == 1:
                        # 第一轮或未启用新branching或只有一个候选：直接使用第一个
                        selected_indices.append(orig_idx)
                        # 更新curr_inputs等
                        output = candidate_outputs[0]["output"]
                        generated_tokens = output.outputs[0].token_ids
                        curr_inputs[orig_idx].extend(generated_tokens)
                        result_masks[orig_idx].extend([1] * len(generated_tokens))
                    else:
                        # 后续轮次且有多个候选：进行branching判断
                        # 1. 提取prompt前缀和三个回答
                        prefix = curr_inputs[orig_idx].copy()  # 当前的前缀（包括之前的生成）
                        responses = []
                        for cand in candidate_outputs:
                            tokens = cand["output"].outputs[0].token_ids
                            response_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                            responses.append(response_text)
                        
                        # 2. 计算三个回答的语义熵
                        pairwise_entropies = self._calc_pairwise_semantic_entropy(responses)
                        max_pairwise_entropy = max(pairwise_entropies) if pairwise_entropies else 0.0
                        
                        # 3. 检查是否包含标准答案
                        orig_sample = None
                        for sample_idx, indices in sample_to_indices.items():
                            if orig_idx in indices:
                                orig_sample = sample_idx
                                break
                        
                        ground_truth = ground_truth_answers[orig_sample] if orig_sample is not None and orig_sample < len(ground_truth_answers) else ""
                        has_answer = False
                        if ground_truth:
                            for resp in responses:
                                if ground_truth.lower() in resp.lower():
                                    has_answer = True
                                    break
                        
                        # 4. 判断是否需要branching
                        should_branch = (max_pairwise_entropy > self.branching_semantic_entropy_threshold) and has_answer
                        
                        if should_branch:
                            # 保留语义熵差异最大的两个样本
                            # 找到语义熵最大的两个回答的索引
                            if len(pairwise_entropies) >= 2:
                                # 计算每对回答的语义熵
                                max_entropy_pair_idx = 0
                                max_entropy_val = pairwise_entropies[0]
                                for idx, entropy in enumerate(pairwise_entropies):
                                    if entropy > max_entropy_val:
                                        max_entropy_val = entropy
                                        max_entropy_pair_idx = idx
                                
                                # 将pair索引转换为response索引对
                                # pairwise_entropies的索引对应：(0,1), (0,2), (1,2)
                                if max_entropy_pair_idx == 0:
                                    selected_pair = (0, 1)
                                elif max_entropy_pair_idx == 1:
                                    selected_pair = (0, 2)
                                else:
                                    selected_pair = (1, 2)
                                
                                # 使用第一个选中的回答更新当前索引
                                selected_idx = selected_pair[0]
                                output = candidate_outputs[selected_idx]["output"]
                                generated_tokens = output.outputs[0].token_ids
                                curr_inputs[orig_idx].extend(generated_tokens)
                                result_masks[orig_idx].extend([1] * len(generated_tokens))
                                selected_indices.append(orig_idx)
                                
                                # 为第二个选中的回答创建新分支
                                second_idx = selected_pair[1]
                                second_output = candidate_outputs[second_idx]["output"]
                                second_tokens = second_output.outputs[0].token_ids
                                
                                # 创建新分支
                                new_branches_to_add.append({
                                    "prefix": prefix.copy(),
                                    "tokens": second_tokens,
                                    "orig_sample": orig_sample,
                                    "orig_idx": orig_idx
                                })
                                
                                logger.info(f"Branching: orig_idx={orig_idx}, selected pair={selected_pair}, max_entropy={max_entropy_val:.3f}")
                            else:
                                # fallback: 随机选择两个  
                                selected_pair = (0, 1) if len(candidate_outputs) >= 2 else (0, 0)
                                selected_idx = selected_pair[0]
                                output = candidate_outputs[selected_idx]["output"]
                                generated_tokens = output.outputs[0].token_ids
                                curr_inputs[orig_idx].extend(generated_tokens)
                                result_masks[orig_idx].extend([1] * len(generated_tokens))
                                selected_indices.append(orig_idx)
                        else:
                            # 不进行branching：随机选择一个
                            selected_idx = random.randint(0, len(candidate_outputs) - 1)
                            output = candidate_outputs[selected_idx]["output"]
                            generated_tokens = output.outputs[0].token_ids
                            curr_inputs[orig_idx].extend(generated_tokens)
                            result_masks[orig_idx].extend([1] * len(generated_tokens))
                            selected_indices.append(orig_idx)
                            logger.debug(f"No branching: orig_idx={orig_idx}, randomly selected candidate {selected_idx}")
                
                # 添加新分支到curr_inputs等
                for branch_info in new_branches_to_add:
                    new_idx = len(curr_inputs)
                    curr_inputs.append(branch_info["prefix"].copy())
                    curr_inputs[new_idx].extend(branch_info["tokens"])
                    init_inputs.append(init_inputs[branch_info["orig_idx"]].copy())
                    result_masks.append(result_masks[branch_info["orig_idx"]].copy())
                    result_masks[new_idx].extend([1] * len(branch_info["tokens"]))
                    call_counters.append(call_counters[branch_info["orig_idx"]])
                    self.tool_results_dict[new_idx] = self.tool_results_dict[branch_info["orig_idx"]].copy()
                    self.sample_round_dict[new_idx] = self.sample_round_dict[branch_info["orig_idx"]]
                    selected_indices.append(new_idx)
                    
                    # 更新sample_to_indices
                    orig_sample = branch_info["orig_sample"]
                    sample_to_indices.setdefault(orig_sample, []).append(new_idx)
                    rollouts_per_sample[orig_sample] += 1
                
                # ========== Entropy Variation Monitoring ==========
                vocab_size = len(self.tokenizer.get_vocab())
                entropy_norm_factor = math.log(vocab_size)
                current_entropy_dict = {}
                current_semantic_entropy_dict = {}  # 存储语义熵
                for out_idx in selected_indices:
                    # 简化处理：使用默认熵值
                    current_entropy_dict[out_idx] = 0.0
                # ============================

                tool_requests: Dict[str, List[Dict]] = {tag: [] for tag in self.tools}
                next_active_indices = []

                for out_idx in selected_indices:
                    finish_reason = None
                    stop_reason = None
                    is_tool_call = False
                    
                    # 检查是否是tool call（通过检查最后生成的token）
                    if len(curr_inputs[out_idx]) > len(init_inputs[out_idx]):
                        # 获取最后生成的文本
                        generated_part = curr_inputs[out_idx][len(init_inputs[out_idx]):]
                        generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=False)
                        
                        # 检查是否包含tool stop sequence
                        for stop_seq in self.stop_sequences:
                            if stop_seq in generated_text:
                                is_tool_call = True
                                stop_reason = stop_seq
                                finish_reason = 'stop'
                                break
                        
                        if not is_tool_call:
                            # 检查是否达到长度限制
                            response_len = len(generated_part)
                            if response_len >= max_len:
                                finish_reason = 'length'
                            else:
                                finish_reason = 'stop'  # EOS
                    
                    # Debug information
                    logger.debug(f"  Sample {out_idx} finish_reason: {finish_reason}, is_tool_call: {is_tool_call}")

                    if is_tool_call:
                        tag = stop_reason.strip("</>")
                        if call_counters[out_idx] < self.tool_call_limit:
                            call_counters[out_idx] += 1
                            full_text = self.tokenizer.decode(curr_inputs[out_idx])
                            content = self._extract_content(full_text, tag)
                            if content:
                                tool_requests[tag].append({"index": out_idx, "content": content})
                                next_active_indices.append(out_idx)
                                # 更新工具调用计数统计
                                tool_metrics["tools/total_calls"] += 1
                                calls_per_tool[tag] += 1
                            # 更新轮次
                            self.sample_round_dict[out_idx] = self.sample_round_dict.get(out_idx, 0) + 1
                        else:
                            logger.warning(f"Tool call limit reached for sample {out_idx}. Appending EOS.")
                            curr_inputs[out_idx].append(eos_token_id)
                            result_masks[out_idx].append(1)
                            tool_metrics["tools/call_limit_reached_count"] += 1

                    elif finish_reason == 'length':
                        if len(curr_inputs[out_idx]) - len(init_inputs[out_idx]) < max_len:
                            next_active_indices.append(out_idx)

                    elif finish_reason == 'stop':  # EOS
                        pass

                if any(tool_requests.values()):
                    logger.info(f"Processing tool requests: {sum(len(reqs) for reqs in tool_requests.values())} total requests")
                    futures = {}
                    for tag, requests in tool_requests.items():
                        if not requests:
                            continue
                        logger.debug(f"Processing {len(requests)} requests for tool '{tag}'")
                        tool = self.tools[tag]
                        for req in requests:
                            logger.debug(f"Submitting tool request: tool={tag}, idx={req['index']}, content={req['content']}")
                            future = self.executor.submit(self._execute_tool_with_retry, tool, req["content"])
                            futures[future] = {"index": req["index"], "tag": tag}

                    total_futures = len(futures)
                    completed_futures = 0
                    logger.debug(f"Submitted {total_futures} tool requests for execution")
                    for future in concurrent.futures.as_completed(futures):
                        completed_futures += 1
                        fut_info = futures[future]
                        idx = fut_info["index"]
                        tag = fut_info["tag"]
                        try:
                            result = future.result(timeout=self.tool_timeout)
                            # 解析工具执行结果
                            success = result["success"]
                            retry_count = result["retry_count"]
                            execution_time = result["execution_time"]
                            result_text = result["result"]
                            
                            # 更新统计信息
                            if success:
                                tool_metrics["tools/successful_calls"] += 1
                                success_per_tool[tag] += 1
                                logger.info(f"Tool({tag}) for sample {idx} completed successfully in {execution_time:.2f}s, result length: {len(result_text)}")
                            else:
                                tool_metrics["tools/failed_calls"] += 1
                                result_text = f"Tool({tag}) returned empty output."
                                logger.warning(f"Tool({tag}) for sample {idx} failed after {retry_count} retries, execution time: {execution_time:.2f}s")
                            
                            tool_metrics["tools/total_execution_time"] += execution_time
                            tool_metrics["tools/max_execution_time"] = max(tool_metrics["tools/max_execution_time"], execution_time)
                            tool_metrics["tools/total_retries"] += retry_count
                            tool_metrics["tools/max_retries"] = max(tool_metrics["tools/max_retries"], retry_count)
                            
                            # 更新每个工具的时间统计
                            total_time_per_tool[tag] += execution_time
                            
                            if not result_text:
                                result_text = f"Tool({tag}) returned empty output."
                                logger.warning(f"Tool({tag}) for sample {idx} returned empty output, execution time: {execution_time:.2f}s")
                            else:
                                logger.debug(f"Tool({tag}) result: {result_text}")
                                
                        except Exception as e:
                            logger.error(f"Tool({tag}) execution failed for sample {idx}: {e}")
                            result_text = f"Error: Tool({tag}) execution failed with message: {e}"
                            tool_metrics["tools/failed_calls"] += 1
                        
                        logger.debug(f"Tool completion progress: {completed_futures}/{total_futures} ({completed_futures/total_futures*100:.1f}%)")
                        formatted_result = f" <result>\n{result_text}\n</result>"
                        result_tokens = self.tokenizer.encode(formatted_result)
                        logger.debug(f"Result for tool({tag}), sample {idx} tokenized to {len(result_tokens)} tokens")
                        curr_inputs[idx].extend(result_tokens)
                        result_masks[idx].extend([0] * len(result_tokens))

                # 新的branching逻辑已经在tool调用之前完成，这里只需要更新active_indices
                # 旧的branching逻辑已移除，因为新的逻辑在生成后、tool调用前就完成了
                active_indices = []
                for idx in next_active_indices:
                    response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                    if response_len < max_len:
                        active_indices.append(idx)
                
                # Add non-active samples that still need more rollouts
                # 构建active_by_sample映射，用于判断哪些样本是活跃的
                active_by_sample = {}
                for idx in active_indices:
                    # Find which original sample this index belongs to
                    orig_sample = None
                    for sample_idx, indices in sample_to_indices.items():
                        if idx in indices:
                            orig_sample = sample_idx
                            break
                    
                    if orig_sample is not None:
                        if orig_sample not in active_by_sample:
                            active_by_sample[orig_sample] = []
                        active_by_sample[orig_sample].append(idx)
                
                # 为不活跃样本补充新的rollout分支
                new_inputs = []
                new_init_inputs = []
                new_result_masks = []
                new_call_counters = []
                new_sample_origins = []
                
                for orig_sample in range(batch_size):
                    if orig_sample not in active_by_sample and rollouts_per_sample[orig_sample] < num_samples:
                        # 对于不活跃样本，每次只新增一个branch
                        branches_to_add = min(1, num_samples - rollouts_per_sample[orig_sample])
                        if branches_to_add <= 0:
                            continue
                            
                        # Use first index of this sample as template
                        source_idx = sample_to_indices[orig_sample][0]
                        
                        # Create new branches
                        for _ in range(branches_to_add):
                            new_inputs.append(init_inputs[source_idx].copy())
                            new_init_inputs.append(init_inputs[source_idx].copy())
                            new_result_masks.append([])
                            new_call_counters.append(0)
                            new_sample_origins.append(orig_sample)  # 记录原始样本
                            rollouts_per_sample[orig_sample] += 1
                
                # Add new branches to existing lists
                if new_inputs:
                    start_idx = len(curr_inputs)
                    curr_inputs.extend(new_inputs)
                    init_inputs.extend(new_init_inputs)
                    result_masks.extend(new_result_masks)
                    call_counters.extend(new_call_counters)
                    
                    # 初始化新分支的相关数据结构
                    for i, new_idx in enumerate(range(start_idx, start_idx + len(new_inputs))):
                        orig_sample = new_sample_origins[i]
                        # 更新sample_to_indices
                        sample_to_indices.setdefault(orig_sample, []).append(new_idx)
                        # 初始化tool results存储
                        self.tool_results_dict[new_idx] = []
                        # 初始化轮次信息
                        self.sample_round_dict[new_idx] = 0
                        # 添加到active_indices
                        active_indices.append(new_idx)
                
                # 注释掉旧的branching逻辑，因为新的branching逻辑已经在tool调用之前完成
                # Apply beam search: split active samples into multiple branches (OLD LOGIC - DISABLED)
                '''
                final_active_indices = []
                for idx in next_active_indices:
                    response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                    if response_len < max_len:
                        final_active_indices.append(idx)
                
                # Apply beam search: split active samples into multiple branches
                new_indices = []
                new_inputs = []
                new_init_inputs = []
                new_result_masks = []
                new_call_counters = []
                new_sample_origins = []  # 记录每个新分支对应的原始样本
                
                # Map from original sample index to its active rollouts
                '''
                
                '''
                示例:
                    假设：
                    batch_size = 2
                    initial_rollouts = 3
                    sample_to_indices = {0: [0, 1, 2], 1: [3, 4, 5]}
                    final_active_indices = [0, 2, 3, 5]（索引1和4已完成）
                    执行后：
                    active_by_sample = {    
                    0: [0, 2],  # 原始样本0的活跃rollouts：索引0和2    
                    1: [3, 5]   # 原始样本1的活跃rollouts：索引3和5
                    }
                '''

                '''
                active_by_sample = {}
                for idx in final_active_indices:
                    # Find which original sample this index belongs to
                    orig_sample = None
                    for sample_idx, indices in sample_to_indices.items():
                        if idx in indices:
                            orig_sample = sample_idx
                            break
                    
                    if orig_sample is not None:
                        if orig_sample not in active_by_sample:
                            active_by_sample[orig_sample] = []
                        active_by_sample[orig_sample].append(idx)
                
        
                for orig_sample, active_idxs in active_by_sample.items():
                    remaining_slots = num_samples - rollouts_per_sample[orig_sample]
                    if remaining_slots <= 0:
                        continue
                    branches_created = 0
                    for source_idx in active_idxs:
                        branches_per_idx = min(beam_size - 1, remaining_slots - branches_created)
                        if branches_per_idx <= 0:
                            break
                        for _ in range(branches_per_idx):
                            # ==== Entropy-based Adaptive Beaming ====
                            
                            entropy_now = current_entropy_dict.get(source_idx, 0.0)
                            entropy_init = self.initial_entropy_dict.get(source_idx, 0.0)
                            entropy_delta = entropy_now - entropy_init
                            prob = random.random() - self.entropy_weight * entropy_delta
                    
                            prob = max(0.0, min(1.0, prob))
                            if prob > self.branch_probability: 
                                continue
                            # ==== END ====
                            new_inputs.append(curr_inputs[source_idx].copy())
                            new_init_inputs.append(init_inputs[source_idx].copy())
                            new_result_masks.append(result_masks[source_idx].copy())
                            new_call_counters.append(call_counters[source_idx])
                            new_sample_origins.append(orig_sample)
                            rollouts_per_sample[orig_sample] += 1
                            branches_created += 1
                        if branches_created >= remaining_slots:
                            break


                # Add non-active samples that still need more rollouts
                for orig_sample in range(batch_size):
                    if orig_sample not in active_by_sample and rollouts_per_sample[orig_sample] < num_samples:
                        # 对于不活跃样本，每次只新增一个branch
                        branches_to_add = min(1, num_samples - rollouts_per_sample[orig_sample])
                        if branches_to_add <= 0:
                            continue
                            
                        # Use first index of this sample as template
                        source_idx = sample_to_indices[orig_sample][0]
                        
                        # Create new branches
                        for _ in range(branches_to_add):
                            new_inputs.append(init_inputs[source_idx].copy())
                            new_init_inputs.append(init_inputs[source_idx].copy())
                            new_result_masks.append([])
                            new_call_counters.append(0)
                            new_sample_origins.append(orig_sample)  # 记录原始样本
                            rollouts_per_sample[orig_sample] += 1
                
                # Add new branches to existing lists
                if new_inputs:
                    start_idx = len(curr_inputs)
                    curr_inputs.extend(new_inputs)
                    init_inputs.extend(new_init_inputs)
                    result_masks.extend(new_result_masks)
                    call_counters.extend(new_call_counters)
                    
                    # Add new indices to active list
                    final_active_indices.extend(range(start_idx, start_idx + len(new_inputs)))
                    
                    # 使用正确的原始样本信息更新映射
                    for i, new_idx in enumerate(range(start_idx, start_idx + len(new_inputs))):
                        orig_sample = new_sample_origins[i]
                        sample_to_indices.setdefault(orig_sample, []).append(new_idx)
                
                active_indices = final_active_indices
                '''

            # 确保所有序列不超过max_len
            for idx in range(len(curr_inputs)):
                response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                if response_len > max_len:
                    offset = len(init_inputs[idx])
                    curr_inputs[idx] = curr_inputs[idx][:offset + max_len]
                    result_masks[idx] = result_masks[idx][:max_len]
            
            # Reorganize outputs to match original batch structure and select up to num_samples per sample
            output_sequences = []
            output_result_masks = []
            for i in range(batch_size):
                # Get all indices for this sample
                sample_indices = sample_to_indices.get(i, [])
                # Ensure we have exactly num_samples outputs per sample
                selected_indices = sample_indices[:num_samples]
                
                # If we have fewer rollouts than requested, duplicate the last one
                while len(selected_indices) < num_samples:
                    if selected_indices:
                        selected_indices.append(selected_indices[-1])
                    else:
                        break  # Should not happen but just in case
                        
                # Extract outputs for selected indices
                for idx in selected_indices:
                    output_sequences.append(curr_inputs[idx][len(prompt_token_ids_list[i]):])
                    output_result_masks.append(result_masks[idx])

            padded_response_list = []
            padded_result_mask_list = []
            for output_ids, result_mask in zip(output_sequences, output_result_masks):
                logger.debug(f"len(output_ids): {len(output_ids)}, len(result_mask): {len(result_mask)}, output_ids: {output_ids}, result_mask: {result_mask}")
                
                assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
                
                response = torch.tensor(output_ids)
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                
                result_mask_tensor = torch.tensor(result_mask)
                result_mask_tensor = pad_sequence_to_length(result_mask_tensor, self.config.response_length, 0)
                
                padded_response_list.append(response)
                padded_result_mask_list.append(result_mask_tensor)
            
            response = torch.stack(padded_response_list, dim=0).to(input_ids.device)
            loss_mask = torch.stack(padded_result_mask_list, dim=0).to(input_ids.device)
            
            non_tensor_batch = deepcopy(prompts.non_tensor_batch)
            if num_samples > 1 and do_sample:
                input_ids = _repeat_interleave(input_ids, num_samples)
                attention_mask = _repeat_interleave(attention_mask, num_samples)
                position_ids = _repeat_interleave(position_ids, num_samples)
                if non_tensor_batch:
                    for key, value in non_tensor_batch.items():
                        if isinstance(value, np.ndarray):
                            non_tensor_batch[key] = np.repeat(value, num_samples, axis=0)
                        elif isinstance(value, list):
                            non_tensor_batch[key] = [item for item in value for _ in range(num_samples)]

            final_batch_size = input_ids.size(0)
            seq = torch.cat([input_ids, response], dim=-1)

            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).expand(final_batch_size, -1)

            if position_ids.dim() == 3:  # for RoPE scaling like qwen2vl mrope
                delta_position_id = delta_position_id.view(final_batch_size, 1, -1).expand(final_batch_size, position_ids.size(1), -1)
                response_position_ids = position_ids[..., -1:].expand(-1, position_ids.size(1), -1) + delta_position_id
            else:
                response_position_ids = position_ids[..., -1:] + delta_position_id

            final_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            loss_mask = loss_mask * response_attention_mask

            # 计算平均执行时间
            if tool_metrics["tools/total_calls"] > 0:
                tool_metrics["tools/avg_execution_time"] = tool_metrics["tools/total_execution_time"] / tool_metrics["tools/total_calls"]
                
            # 计算每个工具的平均执行时间和成功率
            tool_specific_metrics = {}
            for tag in self.tools.keys():
                calls = calls_per_tool[tag]
                if calls > 0:
                    tool_specific_metrics[f"tools/{tag}/calls"] = calls
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = total_time_per_tool[tag] / calls
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = success_per_tool[tag] / calls
                else:
                    tool_specific_metrics[f"tools/{tag}/calls"] = 0
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = 0
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = 0

            batch = TensorDict({
                "prompts": input_ids,
                "responses": response,
                "input_ids": seq,
                "attention_mask": final_attention_mask,
                "loss_mask": loss_mask,
                "position_ids": final_position_ids,
            }, batch_size=final_batch_size)

        if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
            
        # 合并所有metrics
        all_metrics = {**tool_metrics, **tool_specific_metrics}
        
        # 将metrics添加到meta_info中
        meta_info = deepcopy(prompts.meta_info) if prompts.meta_info else {}
        meta_info["metrics"] = all_metrics

        data_proto = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

        return data_proto