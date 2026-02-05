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
from transformers import AutoModelForCausalLM
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

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, worker=None, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.worker = worker  # 保存 worker 引用，用于访问 ref 模型

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

        # ========== Information-Gain Driven Diverse Branching (IGD) 配置 ==========
        igd_config = self.config.get("igd", OmegaConf.create({}))
        self.enable_igd = igd_config.get("enable", True)    # 总开关
        self.mi_threshold_low = igd_config.get("mi_threshold_low", 0.0)  # MI低阈值，用于触发反思性分支
        self.mi_threshold_high = igd_config.get("mi_threshold_high", 0.5)  # MI高阈值，用于触发正交分解分支
        self.mi_lambda = igd_config.get("lambda", 0.1)  # Reward中的MI权重
        self.enable_negative_constraint = igd_config.get("enable_negative_constraint", True)  # 策略一开关
        self.enable_reflective_branching = igd_config.get("enable_reflective_branching", True)  # 策略二开关
        self.enable_orthogonal_branching = igd_config.get("enable_orthogonal_branching", True)  # 策略三开关
        
        # 查询历史追踪：每个样本索引 -> 查询历史列表
        self.query_history = {}  # Dict[int, List[str]] - 记录每个rollout的查询历史
        
        # MI分数记录：每个样本索引 -> MI分数列表
        self.mi_scores = {}  # Dict[int, List[float]] - 记录每次工具调用后的MI分数
        
        # 策略应用记录：每个样本索引 -> 最后应用的策略列表
        self.applied_strategies = {}  # Dict[int, List[str]] - 记录每个rollout应用的策略（用于策略优先级控制）
        
        # MI计算模型（使用训练过程中的模型，避免重复加载）
        # 优先使用 actor 模型，如果没有则使用 ref 模型
        # self.mi_model = None
        # self.mi_model_is_fsdp = False  # 标记是否是 FSDP 模型
        # if self.enable_igd:
        #     try:
        #         if self.worker is not None:
        #             # 优先使用 actor 模型（更常用）
        #             if hasattr(self.worker, 'actor_module_fsdp') and self.worker.actor_module_fsdp is not None:
        #                 self.mi_model = self.worker.actor_module_fsdp
        #                 self.mi_model_is_fsdp = True
        #                 self.mi_model.eval()  # 设置为评估模式
        #                 logger.info("MI evaluation model loaded from worker's actor_module_fsdp (FSDP model) successfully")
        #             # 如果没有 actor 模型，尝试使用 ref 模型
        #             elif hasattr(self.worker, 'ref_module_fsdp') and self.worker.ref_module_fsdp is not None:
        #                 self.mi_model = self.worker.ref_module_fsdp
        #                 self.mi_model_is_fsdp = True
        #                 self.mi_model.eval()  # 设置为评估模式
        #                 logger.info("MI evaluation model loaded from worker's ref_module_fsdp (FSDP model) successfully")
        #             else:
        #                 # 如果 worker 中没有可用模型，回退到从路径加载
        #                 from transformers import AutoModelForCausalLM
        #                 logger.info(f"No model available in worker, loading MI evaluation model from path: {model_path}")
        #                 self.mi_model = AutoModelForCausalLM.from_pretrained(
        #                     model_path,
        #                     device_map="auto",
        #                     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        #                 )
        #                 self.mi_model_is_fsdp = False
        #                 self.mi_model.eval()  # 设置为评估模式
        #                 logger.info("MI evaluation model loaded from path successfully")
        #         else:
        #             # 如果没有 worker，从路径加载
        #             from transformers import AutoModelForCausalLM
        #             logger.info(f"No worker available, loading MI evaluation model from path: {model_path}")
        #             self.mi_model = AutoModelForCausalLM.from_pretrained(
        #                 model_path,
        #                 device_map="auto",
        #                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        #             )
        #             self.mi_model_is_fsdp = False
        #             self.mi_model.eval()  # 设置为评估模式
        #             logger.info("MI evaluation model loaded from path successfully")
        #     except Exception as e:
        #         logger.warning(f"Failed to load MI evaluation model: {e}. MI calculation will return 0.0")
        #         self.mi_model = None
        #         self.mi_model_is_fsdp = False


        # 方案 B: 显存不够，放 CPU (最稳，慢一点，不会死锁，不会 OOM)
        # 既然是算标量 Reward，CPU 其实完全能接受
        # self.mi_model = AutoModelForCausalLM.from_pretrained(
        #     "/mnt/zhongwenlin/model/Qwen/Qwen2.5-3B-Instruct",
        #     torch_dtype=torch.float32,
        #     device_map="cpu", 
        #     attn_implementation="eager" # CPU 必须用 eager
        # )
        # self.mi_model.eval()

        # 方案 B: 显存不够，放 CPU (最稳，慢一点，不会死锁，不会 OOM)
        # 既然是算标量 Reward，CPU 其实完全能接受
        self.mi_model = AutoModelForCausalLM.from_pretrained(
            "/mnt/zhongwenlin/model/Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float16, # 必须，否则显存翻倍
            device_map="cuda",         # 简单写法，如果环境简单这也行
            attn_implementation="eager" # 强烈建议加上
        )
        
        self.mi_model.eval()



        
        # ========== END IGD配置 ==========

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

    def _calculate_mi_score(self, context: str, formatted_tool_result: str, ground_truth: str) -> float:
        """
        计算互信息(MI)分数: Score = log P(A*|Context, O) - log P(A*|Context)
        参考 show_mi_1.py 的实现，使用加载的模型计算loss
        
        Args:
            context: 包含所有查询的历史上下文
            formatted_tool_result: 工具结果 O
            ground_truth: 标准答案 A*
            
        Returns:
            MI分数，正值表示工具结果有效，负值表示误导，接近0表示无关
        """
        if not ground_truth or not context:
            return 0.0
        
        # 如果没有加载MI评估模型，返回0.0
        if self.mi_model is None:
            logger.debug("MI evaluation model not loaded. Returning 0.0 (neutral score).")
            return 0.0
        
        # 构造 Prior 和 Posterior prompts
        # Prior: 只有上下文，不知道工具结果
        prompt_prior = f"{context}<answer>"
        # Posterior: 上下文 + 工具结果
        prompt_posterior = f"{context}{formatted_tool_result}<answer>"
        
        try:
            # 定义计算 Sequence Loss 的辅助函数
            def get_sequence_loss(prompt: str, answer: str) -> float:
                """
                计算给定prompt下生成answer的loss
                参考 show_mi_1.py 的实现
                """
                # 拼接完整序列：Prompt + Answer
                full_text = prompt + answer
                
                # 使用当前的tokenizer编码
                inputs = self.tokenizer(full_text, return_tensors="pt")
                
                

                device = next(self.mi_model.parameters()).device
                
                input_ids = inputs["input_ids"].to(device)
                
                # 找到 Answer 开始的位置（Label Masking）
                # 我们只计算 Answer 部分的概率，不计算 Prompt 部分的
                prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                prompt_len = prompt_ids.shape[1]
                
                # 构造 Labels：Prompt 部分设为 -100 (忽略)，Answer 部分保留原 ID
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100
                
                # 计算loss
                with torch.no_grad():
                    outputs = self.mi_model(input_ids=input_ids, labels=labels)
                    
                    # HuggingFace 默认返回的 loss 就是 CrossEntropyLoss (平均值)
                    # 这里直接用 outputs.loss (Mean NLL)
                    return outputs.loss.item()
            
            # 计算两次 Loss
            loss_prior = get_sequence_loss(prompt_prior, ground_truth)
            loss_posterior = get_sequence_loss(prompt_posterior, ground_truth)
            
            # 计算 PMI (Score)
            # Score = log P(post) - log P(prior)
            # 因为 Loss = - log P
            # 所以 Score = (- Loss_post) - (- Loss_prior) = Loss_prior - Loss_post
            pmi_score = loss_prior - loss_posterior
            
            logger.debug(f"MI calculation: loss_prior={loss_prior:.4f}, loss_posterior={loss_posterior:.4f}, pmi_score={pmi_score:.4f}")
            
            return pmi_score
            
        except Exception as e:
            logger.warning(f"Failed to calculate MI score: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return 0.0
    
    def _check_duplicate_query(self, sample_idx: int, new_query: str) -> bool:
        """
        检查查询是否重复（用于策略一：负向约束分支）
        
        Args:
            sample_idx: 样本索引
            new_query: 新的查询字符串
            
        Returns:
            如果查询重复返回True
        """
        if sample_idx not in self.query_history:
            return False
        
        # 检查是否与历史查询完全相同（可以扩展为语义相似度检查）
        for prev_query in self.query_history[sample_idx]:
            if prev_query.strip().lower() == new_query.strip().lower():
                return True
        return False
    
    def _get_negative_constraint_prompt(self, previous_query: str) -> str:
        """
        生成负向约束提示（策略一）
        
        Args:
            previous_query: 之前搜索过的查询
            
        Returns:
            约束提示字符串
        """
        return f"\n<hint>You have previously searched for '{previous_query}'. This did not yield the answer. You must generate a semantically distinct query now.</hint>\n"
    
    def _get_reflective_thought_prompt(self) -> str:
        """
        生成反思性思考提示（策略二：反思性分支）
        
        Returns:
            thought标签内容
        """
        return "\n<hint>Analyze the gap between the current tool result and the final goal. What is missing? Based on the analysis, generate a new search query that targets the missing information.</hint>\n"
    
    def _get_orthogonal_hint_prompt(self) -> str:
        """
        生成正交分解提示（策略三：正交分解分支）
        
        Returns:
            hint提示字符串
        """
        return "\n<hint>You have now found key information about one aspect of this question. If the above information supports your direct answer, you should answer directly; otherwise, you should consider the other aspect of this question.</hint>\n"

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
        
        # ========== 初始化IGD数据结构（每次generate_sequences调用时重置）==========
        if self.enable_igd:
            self.query_history = {}
            self.mi_scores = {}
            self.applied_strategies = {}
            # 注意：mi_scores 会在获取 num_samples 后进一步初始化
        # ========== END IGD初始化 ==========


        # 获取标准答案（从reward_model中获取，与reward_manager保持一致）
        ground_truth_answers = []
        if prompts.non_tensor_batch and "reward_model" in prompts.non_tensor_batch:
            reward_models = prompts.non_tensor_batch["reward_model"]
            
            # 处理 numpy array 或 list 类型
            # 检查是否是可迭代的序列类型（list, numpy array等），但不是字符串或字典
            if isinstance(reward_models, (list, np.ndarray)) or (hasattr(reward_models, '__iter__') and not isinstance(reward_models, (str, dict, bytes))):
                # 批次数据：每个元素是一个字典，包含 {"style": "rule", "ground_truth": [...]}
                # ground_truth 本身是一个列表，如 ['Hakeem Olajuwon']
                try:
                    for rm in reward_models:
                        if isinstance(rm, dict):
                            gt = rm.get("ground_truth", "")
                            # 如果 ground_truth 是列表，取第一个元素；如果是字符串，直接使用
                            if isinstance(gt, list):
                                # 如果列表不为空，取第一个元素；如果为空，使用空字符串
                                ground_truth_answers.append(gt[0] if len(gt) > 0 else "")
                            elif isinstance(gt, str):
                                ground_truth_answers.append(gt)
                            else:
                                ground_truth_answers.append("")
                        else:
                            ground_truth_answers.append("")
                except (TypeError, AttributeError) as e:
                    logger.warning(f"Error iterating over reward_models: {e}, treating as single dict")
                    # 如果迭代失败，尝试作为单个字典处理
                    if isinstance(reward_models, dict):
                        gt = reward_models.get("ground_truth", "")
                        if isinstance(gt, list):
                            ground_truth_answers.append(gt[0] if len(gt) > 0 else "")
                        elif isinstance(gt, str):
                            ground_truth_answers.append(gt)
                        else:
                            ground_truth_answers.append("")
                    else:
                        ground_truth_answers.append("")
            elif isinstance(reward_models, dict):
                # 单个样本：直接获取ground_truth
                gt = reward_models.get("ground_truth", "")
                if isinstance(gt, list):
                    ground_truth_answers.append(gt[0] if len(gt) > 0 else "")
                elif isinstance(gt, str):
                    ground_truth_answers.append(gt)
                else:
                    ground_truth_answers.append("")

        
        # 确保ground_truth_answers长度与batch_size一致
        while len(ground_truth_answers) < batch_size:
            ground_truth_answers.append("")
        # ========== END Ground Truth获取 ==========
        
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
            num_samples = self.sampling_params.n

            # ========== 根据最终response长度预初始化IGD数据结构 ==========
            if self.enable_igd:
                # 最终的response数量是 batch_size * num_samples
                # 预初始化 mi_scores 为足够大的字典（虽然索引可能不连续，但可以确保每个位置都有对应的值）
                # 实际使用时，会根据 rollout 索引动态添加，但这里预分配可以避免长度不匹配的问题
                final_response_count = batch_size * num_samples
                # 预分配一个足够大的字典，键为 0 到 final_response_count - 1，值为空列表
                # 虽然实际的 rollout 索引可能不是连续的，但这样可以确保每个位置都有对应的值
                self.mi_scores = {idx: [] for idx in range(final_response_count)}
                logger.debug(f"Pre-initialized mi_scores with {final_response_count} entries (batch_size={batch_size}, num_samples={num_samples})")
            # ========== END 预初始化 ==========

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
            
            # 创建初始样本
            for i, ids in enumerate(prompt_token_ids_list):
                for _ in range(initial_rollouts):
                    curr_inputs.append(ids.copy())
                    init_inputs.append(ids.copy())
                    result_masks.append([])
                    call_counters.append(0)
                    active_indices.append(len(curr_inputs) - 1)
            
            # Track rollouts per original sample
            rollouts_per_sample = [initial_rollouts] * batch_size  # 每个样本初始有initial_rollouts个rollout
            # 初始时每个样本有多个索引
            sample_to_indices = {i: [i * initial_rollouts + j for j in range(initial_rollouts)] for i in range(batch_size)}

            max_len = self.config.response_length

            while active_indices:
                active_prompts = [curr_inputs[i] for i in active_indices]
                logger.debug(f"rollouts_per_sample: {rollouts_per_sample}")
                logger.debug(f"active_indices: {active_indices}")
                logger.debug(f"active_prompts: {active_prompts}")

                # Update max_tokens for each active sample
                with self.update_sampling_params(
                    n=1,
                    stop=self.stop_sequences,
                    max_tokens=max(1, max((max_len - (len(curr_inputs[i]) - len(init_inputs[i])) for i in active_indices))),
                    detokenize=True,
                    logprobs = self.logprobs
                ):
                    outputs = self.inference_engine.generate(
                        prompt_token_ids=active_prompts,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )
                # ========== Entropy Variation Monitoring ==========
                vocab_size = len(self.tokenizer.get_vocab())
                entropy_norm_factor = math.log(vocab_size)
                current_entropy_dict = {}
                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    logprobs = []
                    tokens = output.outputs[0].token_ids
                    for j in range(min(20, len(tokens))):
                        try:
                            logprob_info = output.outputs[0].logprobs[j]
                        except Exception:
                            logprob_info = output.outputs[0].logprobs[-1]
                        token_list = list(logprob_info.values())
                        token_logprobs = [token.logprob for token in token_list]
                        logprobs.extend(token_logprobs)
                    if logprobs:
                        entropy = self._calc_entropy(logprobs) / entropy_norm_factor
                    else:
                        entropy = 0.0
                    current_entropy_dict[out_idx] = entropy
                    is_after_tool_call = False
                    tool_call_round = 0
                    
                    # 判断是否是工具调用后的生成
                    if out_idx < len(call_counters):
                        tool_call_round = call_counters[out_idx]
                        is_after_tool_call = tool_call_round > 0
                    
                    if out_idx not in self.initial_entropy_dict:
                        self.initial_entropy_dict[out_idx] = entropy
                        # 记录初始熵值（第一轮，工具调用前）
                        if hasattr(self, 'entropy_history'):
                            self.entropy_history.append({
                                'sample_idx': out_idx,
                                'round': 0,
                                'entropy': entropy,
                                'is_after_tool_call': False
                            })
                        if hasattr(self, 'entropy_recorder') and self.entropy_recorder:
                            self.entropy_recorder.record_entropy(out_idx, entropy, is_after_tool_call=False)
                    elif is_after_tool_call:
                        # 记录工具调用后的熵值
                        if hasattr(self, 'entropy_history'):
                            self.entropy_history.append({
                                'sample_idx': out_idx,
                                'round': tool_call_round,
                                'entropy': entropy,
                                'is_after_tool_call': True
                            })
                        if hasattr(self, 'entropy_recorder') and self.entropy_recorder:
                            self.entropy_recorder.record_entropy(out_idx, entropy, is_after_tool_call=True)
                # ============================

                tool_requests: Dict[str, List[Dict]] = {tag: [] for tag in self.tools}
                next_active_indices = []

                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    generated_tokens = output.outputs[0].token_ids

                    curr_inputs[out_idx].extend(generated_tokens)
                    result_masks[out_idx].extend([1] * len(generated_tokens))

                    finish_reason = output.outputs[0].finish_reason
                    stop_reason = output.outputs[0].stop_reason

                    is_tool_call = finish_reason == 'stop' and stop_reason in self.stop_sequences
                    
                    # Debug information
                    decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    logger.debug(f"  Sample {out_idx} output:")
                    logger.debug(f"  Token IDs: {generated_tokens}")
                    logger.debug(f"  Text: {decoded_text}")
                    logger.debug(f"  Finish reason: {finish_reason}")
                    logger.debug(f"  Stop reason: {stop_reason}")
                    logger.debug(f"  Is tool call: {is_tool_call}")
                    logger.debug(f"  Tool: {stop_reason.strip('</>') if is_tool_call else 'No tool call'}")

                    

                    if is_tool_call:
                        tag = stop_reason.strip("</>")
                        if call_counters[out_idx] < self.tool_call_limit:
                            call_counters[out_idx] += 1
                            full_text = self.tokenizer.decode(curr_inputs[out_idx])
                            content = self._extract_content(full_text, tag)
                            if content:
                                
                                # ========== 记录查询历史（用于策略一：负向约束分支）==========
                                if self.enable_igd and self.enable_negative_constraint and tag == "search":
                                    if out_idx not in self.query_history:
                                        self.query_history[out_idx] = []
                                    # 先记录查询历史（重复检测在工具结果返回后通过检查历史记录完成）
                                    self.query_history[out_idx].append(content)
                                # ========== END 查询历史记录 ==========
                                
                                tool_requests[tag].append({"index": out_idx, "content": content})
                                next_active_indices.append(out_idx)
                                # 更新工具调用计数统计
                                tool_metrics["tools/total_calls"] += 1
                                calls_per_tool[tag] += 1
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
                        
                        # ========== 策略一预处理：检测是否触发策略一（在添加result之前） ==========
                        should_empty_result = False
                        if self.enable_igd and self.enable_negative_constraint and tag == "search":
                            if idx in self.query_history and len(self.query_history[idx]) >= 2:
                                current_query = self.query_history[idx][-1]
                                # 检查当前查询是否与之前的任何查询重复
                                for prev_query in self.query_history[idx][:-1]:
                                    if current_query.strip().lower() == prev_query.strip().lower():
                                        should_empty_result = True
                                        logger.info(f"Sample {idx}: Strategy 1 triggered, will empty tool result (duplicate query detected: '{current_query}' matches '{prev_query}')")
                                        break
                        
                        # 如果策略一触发，将result内容挖空
                        if should_empty_result:
                            formatted_result = f" <result>\n\n</result>"
                        else:
                            formatted_result = f" <result>\n{result_text}\n</result>"
                        
                        result_tokens = self.tokenizer.encode(formatted_result)
                        logger.debug(f"Result for tool({tag}), sample {idx} tokenized to {len(result_tokens)} tokens")
                        curr_inputs[idx].extend(result_tokens)
                        result_masks[idx].extend([0] * len(result_tokens))
                        
                        # ========== Information-Gain Driven Diverse Branching (IGD) ==========
                        if self.enable_igd:
                            # 找到原始样本索引
                            orig_sample = None
                            for sample_idx, indices in sample_to_indices.items():
                                if idx in indices:
                                    orig_sample = sample_idx
                                    break
                            
                            if orig_sample is not None and orig_sample < len(ground_truth_answers):
                                ground_truth = ground_truth_answers[orig_sample]
                                
                                # 获取当前上下文（到工具结果之前的内容）
                                context_text = self.tokenizer.decode(curr_inputs[idx][:len(curr_inputs[idx]) - len(result_tokens)], skip_special_tokens=True)
                                
                                # breakpoint()
                                # 计算MI分数（注意：如果策略一触发，formatted_result已经被挖空）
                                mi_score = self._calculate_mi_score(context_text, formatted_result, ground_truth)
                                
                                # 记录MI分数
                                if idx not in self.mi_scores:
                                    self.mi_scores[idx] = []
                                self.mi_scores[idx].append(mi_score)
                                
                                logger.debug(f"Sample {idx}, Tool {tag}: MI score = {mi_score:.4f}")
                                
                                # 应用分支策略（优先级：策略一 > 策略二 > 策略三）
                                branching_prompt_tokens = []
                                strategy_applied = None  # 记录应用的策略
                                
                                # 策略一：负向约束分支（检测重复查询）- 最高优先级
                                if self.enable_negative_constraint and tag == "search":
                                    if idx in self.query_history and len(self.query_history[idx]) >= 2:
                                        # 获取当前查询（最新的）和之前的查询
                                        current_query = self.query_history[idx][-1]
                                        # 检查当前查询是否与之前的任何查询重复
                                        for prev_query in self.query_history[idx][:-1]:
                                            if current_query.strip().lower() == prev_query.strip().lower():
                                                # 发现重复查询，添加负向约束提示
                                                constraint_prompt = self._get_negative_constraint_prompt(prev_query)
                                                constraint_tokens = self.tokenizer.encode(constraint_prompt)
                                                branching_prompt_tokens.extend(constraint_tokens)
                                                strategy_applied = "negative_constraint"
                                                logger.info(f"Sample {idx}: Applied negative constraint branching (duplicate query detected: '{current_query}' matches '{prev_query}')")
                                                break  # 只需要添加一次提示
                                
                                # 策略二：反思性分支（MI低时）- 仅在策略一未触发时考虑
                                if strategy_applied is None and self.enable_reflective_branching and mi_score < self.mi_threshold_low:
                                    reflective_prompt = self._get_reflective_thought_prompt()
                                    reflective_tokens = self.tokenizer.encode(reflective_prompt)
                                    branching_prompt_tokens.extend(reflective_tokens)
                                    strategy_applied = "reflective"
                                    logger.info(f"Sample {idx}: Applied reflective branching (MI={mi_score:.4f} < {self.mi_threshold_low})")
                                
                                # 策略三：正交分解分支（MI高时）- 仅在策略一、二未触发时考虑
                                # 注意：这个策略在分支创建时处理，这里只做标记
                                if strategy_applied is None and self.enable_orthogonal_branching and mi_score > self.mi_threshold_high:
                                    # 标记需要应用正交分解分支（在分支创建时处理）
                                    strategy_applied = "orthogonal"
                                    logger.debug(f"Sample {idx}: Will apply orthogonal branching (MI={mi_score:.4f} > {self.mi_threshold_high})")
                                
                                # 记录应用的策略（用于后续策略优先级控制）
                                if idx not in self.applied_strategies:
                                    self.applied_strategies[idx] = []
                                # 如果三个策略都没有应用，记录为 "Default policy"
                                if strategy_applied:
                                    self.applied_strategies[idx].append(strategy_applied)
                                else:
                                    self.applied_strategies[idx].append("Default policy")
                                
                                # 添加分支策略提示到输入
                                if branching_prompt_tokens:
                                    curr_inputs[idx].extend(branching_prompt_tokens)
                                    result_masks[idx].extend([0] * len(branching_prompt_tokens))
                        # ========== END IGD ==========

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
                new_source_indices = []  # 记录每个新分支对应的父分支索引（用于复制IGD数据）
                
                # Map from original sample index to its active rollouts
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
                        
                        # ========== 策略三：正交分解分支（MI高时） ==========
                        # 仅在策略一、二未应用时执行（检查最近应用的策略）
                        orthogonal_branch_applied = False
                        should_apply_orthogonal = False
                        if self.enable_igd and self.enable_orthogonal_branching and source_idx in self.mi_scores:
                            # 检查是否已经应用了策略一或策略二
                            has_applied_strategy_1_or_2 = False
                            if source_idx in self.applied_strategies and self.applied_strategies[source_idx]:
                                last_applied = self.applied_strategies[source_idx][-1]
                                if last_applied in ["negative_constraint", "reflective"]:
                                    has_applied_strategy_1_or_2 = True
                            
                            # 获取最后一次MI分数
                            last_mi_score = self.mi_scores[source_idx][-1] if self.mi_scores[source_idx] else 0.0
                            # 仅在策略一、二未应用且MI高时应用策略三
                            if not has_applied_strategy_1_or_2 and last_mi_score > self.mi_threshold_high and branches_per_idx >= 2:
                                should_apply_orthogonal = True
                        
                        if should_apply_orthogonal:
                            # MI高时，创建两个分支：Follow-up 和 Parallel Hint
                            orthogonal_branch_applied = True
                            
                            # Branch 1 (Follow-up): 顺着当前结果继续搜索（不添加额外提示）
                            # 当前路径自己
                            
                            # Branch 2 (Parallel Hint): 添加hint提示
                            hint_prompt = self._get_orthogonal_hint_prompt()
                            hint_tokens = self.tokenizer.encode(hint_prompt)
                            
                            branch2_input = curr_inputs[source_idx].copy()
                            branch2_input.extend(hint_tokens)
                            branch2_mask = result_masks[source_idx].copy()
                            branch2_mask.extend([0] * len(hint_tokens))
                            
                            new_inputs.append(branch2_input)
                            new_init_inputs.append(init_inputs[source_idx].copy())
                            new_result_masks.append(branch2_mask)
                            new_call_counters.append(call_counters[source_idx])
                            new_sample_origins.append(orig_sample)
                            new_source_indices.append(source_idx)  # 记录父分支索引
                            rollouts_per_sample[orig_sample] += 1
                            branches_created += 1
                            
                            logger.info(f"Sample {source_idx}: Applied orthogonal branching (MI={last_mi_score:.4f} > {self.mi_threshold_high})")
                        # ========== END 策略三 ==========
                        
                        # if not orthogonal_branch_applied:
                        #     # 常规分支创建（原有的Entropy-based Adaptive Beaming）
                        #     for _ in range(branches_per_idx):
                        #         # ==== Entropy-based Adaptive Beaming ====
                                
                        #         entropy_now = current_entropy_dict.get(source_idx, 0.0)
                        #         entropy_init = self.initial_entropy_dict.get(source_idx, 0.0)
                        #         entropy_delta = entropy_now - entropy_init
                        #         prob = random.random() - self.entropy_weight * entropy_delta
                        
                        #         prob = max(0.0, min(1.0, prob))
                        #         if prob > self.branch_probability: 
                        #             continue
                        #         # ==== END ====
                        #         new_inputs.append(curr_inputs[source_idx].copy())
                        #         new_init_inputs.append(init_inputs[source_idx].copy())
                        #         new_result_masks.append(result_masks[source_idx].copy())
                        #         new_call_counters.append(call_counters[source_idx])
                        #         new_sample_origins.append(orig_sample)
                        #         new_source_indices.append(source_idx)  # 记录父分支索引
                        #         rollouts_per_sample[orig_sample] += 1
                        #         branches_created += 1
                        
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
                            new_source_indices.append(source_idx)  # 记录父分支索引
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
                    
                    # 使用正确的原始样本信息更新映射，并复制IGD相关数据（查询历史、MI分数、应用策略）
                    for i, new_idx in enumerate(range(start_idx, start_idx + len(new_inputs))):
                        orig_sample = new_sample_origins[i]
                        sample_to_indices.setdefault(orig_sample, []).append(new_idx)
                        
                        # ========== 复制父分支的IGD数据到新分支 ==========
                        if self.enable_igd and i < len(new_source_indices):
                            source_idx = new_source_indices[i]  # 获取父分支索引
                            
                            # 复制查询历史
                            if source_idx in self.query_history:
                                self.query_history[new_idx] = self.query_history[source_idx].copy()
                            
                            # 复制MI分数
                            if source_idx in self.mi_scores:
                                self.mi_scores[new_idx] = self.mi_scores[source_idx].copy()
                            
                            # 复制应用策略
                            if source_idx in self.applied_strategies:
                                self.applied_strategies[new_idx] = self.applied_strategies[source_idx].copy()
                            
                            logger.debug(f"Copied IGD data from parent branch {source_idx} to new branch {new_idx}: "
                                       f"query_history_len={len(self.query_history.get(new_idx, []))}, "
                                       f"mi_scores_len={len(self.mi_scores.get(new_idx, []))}, "
                                       f"applied_strategies_len={len(self.applied_strategies.get(new_idx, []))}")
                        # ========== END IGD数据复制 ==========
                
                active_indices = final_active_indices

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
        
        # breakpoint()
        if vllm_version in ('0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
            
        # 合并所有metrics
        all_metrics = {**tool_metrics, **tool_specific_metrics}
        
        logger.debug(f"self.mi_scores.len: {len(self.mi_scores)}")
        if self.mi_scores:
            # 安全地访问第一个键（如果存在）
            first_key = next(iter(self.mi_scores))
            logger.debug(f"self.mi_scores first key={first_key}, value={self.mi_scores[first_key]}")
        # breakpoint()

        # ========== 添加IGD相关数据到meta_info ==========
        # Reward公式: R_total = R_outcome + λ * mean(Score_i)
        # 计算MI部分的reward增量：λ * mean(Score_i)
        if self.enable_igd:
            # 为每个样本计算MI分数总和和reward增量
            igd_data = {}
            mi_reward_increments = []  # 存储每个rollout的MI reward增量（顺序与output_sequences一致）
            
            # 按照output_sequences的顺序收集MI reward增量
            # 注意：output_sequences的长度是 batch_size * num_samples，即 final_batch_size
            # 必须确保mi_reward_increments的长度与output_sequences完全一致
            for i in range(batch_size):
                sample_indices = sample_to_indices.get(i, [])
                # 确保我们按照与output_sequences相同的顺序处理
                selected_indices = sample_indices[:num_samples]
                # 如果选中的rollouts少于num_samples，可能需要重复最后一个
                while len(selected_indices) < num_samples:
                    if selected_indices:
                        selected_indices.append(selected_indices[-1])
                    else:
                        # 如果selected_indices为空，与output_sequences生成逻辑保持一致
                        # output_sequences生成时，如果selected_indices为空，不会添加任何元素
                        # 但为了确保长度一致，我们需要添加num_samples个0.0
                        for _ in range(num_samples):
                            mi_reward_increments.append(0.0)
                        break
                
                # 如果selected_indices为空，已经在上面处理了，跳过
                if not selected_indices:
                    continue
                
                # 收集该样本所有rollout的MI分数（用于metrics）
                sample_mi_scores = []
                sample_query_histories = []
                
                for idx in selected_indices:
                    if idx in self.mi_scores and len(self.mi_scores[idx]) > 0:
                        # 计算该rollout的MI分数均值
                        rollout_mi_scores = self.mi_scores[idx]
                        rollout_mi_avg = (sum(rollout_mi_scores) / len(rollout_mi_scores)) if rollout_mi_scores else 0.0
                        # 计算该rollout的MI reward增量：λ * mean(Score_i)
                        rollout_mi_reward = self.mi_lambda * rollout_mi_avg
                        mi_reward_increments.append(rollout_mi_reward)
                        
                        sample_mi_scores.extend(rollout_mi_scores)
                    else:
                        # 如果该rollout没有MI分数，reward增量为0
                        mi_reward_increments.append(0.0)
                    
                    if idx in self.query_history:
                        sample_query_histories.append(self.query_history[idx])
                
                # 计算MI分数总和（用于metrics）
                mi_sum = sum(sample_mi_scores) if sample_mi_scores else 0.0
                mi_count = len(sample_mi_scores)
                mi_avg = mi_sum / mi_count if mi_count > 0 else 0.0
                
                igd_data[f"mi_scores_{i}"] = sample_mi_scores
                igd_data[f"mi_sum_{i}"] = mi_sum
                igd_data[f"mi_avg_{i}"] = mi_avg
                igd_data[f"query_history_{i}"] = sample_query_histories
            
            # 将MI reward增量添加到non_tensor_batch（在收集完成之后）
            # 这样mi_reward_increment也会与其他non_tensor_batch元素保持一致的格式
            if mi_reward_increments:
                if non_tensor_batch is None:
                    non_tensor_batch = {}
                mi_reward_array = np.array(mi_reward_increments, dtype=np.float32)
                
                # 验证长度：mi_reward_array应该对应每个response
                # response的实际长度是 len(output_sequences)，即 batch_size * num_samples
                expected_length = len(output_sequences)  # 这是实际的response数量
                actual_length = len(mi_reward_array)
                
                if actual_length != expected_length:
                    logger.error(f"MI reward increment length mismatch: got {actual_length}, expected {expected_length}. "
                               f"batch_size={batch_size}, num_samples={num_samples}, final_batch_size={final_batch_size}, "
                               f"do_sample={do_sample}, response.size(0)={response.size(0)}, "
                               f"output_sequences length={len(output_sequences)}, "
                               f"mi_reward_increments collected={len(mi_reward_increments)}. "
                               f"This indicates a bug in the collection logic.")
                    raise ValueError(f"MI reward increment length mismatch: got {actual_length}, expected {expected_length} "
                                   f"(response.size(0) = {response.size(0)}, "
                                   f"batch_size * num_samples = {batch_size} * {num_samples} = {batch_size * num_samples})")
                
                # mi_reward_increment的长度已经是 batch_size * num_samples，与response的第一维长度一致
                # 如果 num_samples > 1 and do_sample，其他non_tensor_batch元素已经被重复了
                # 但mi_reward_increment不需要重复，因为它已经是最终的长度
                non_tensor_batch["mi_reward_increment"] = mi_reward_array
                logger.debug(f"Added MI reward increments to non_tensor_batch: shape={mi_reward_array.shape}, "
                           f"batch_size={batch_size}, num_samples={num_samples}, response_batch_dim={response.size(0)}, "
                           f"final_batch_size={final_batch_size}, lambda={self.mi_lambda}, total_increment={mi_reward_array.sum():.4f}")
            else:
                # 如果没有收集到任何MI reward增量，创建一个全0数组，长度与response一致
                if len(output_sequences) > 0:
                    if non_tensor_batch is None:
                        non_tensor_batch = {}
                    mi_reward_array = np.zeros(len(output_sequences), dtype=np.float32)
                    non_tensor_batch["mi_reward_increment"] = mi_reward_array
                    logger.debug(f"No MI reward increments collected, created zero array with shape={mi_reward_array.shape}")
            
            all_metrics["igd/lambda"] = self.mi_lambda
            all_metrics["igd/mi_threshold_high"] = self.mi_threshold_high
            all_metrics["igd/mi_threshold_low"] = self.mi_threshold_low
            all_metrics.update(igd_data)
        # ========== END IGD数据添加 ==========
        
        # 将metrics添加到meta_info中
        meta_info = deepcopy(prompts.meta_info) if prompts.meta_info else {}
        meta_info["metrics"] = all_metrics

        data_proto = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

        return data_proto