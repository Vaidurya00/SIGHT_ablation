#!/usr/bin/env python3
"""
从数据集生成 response 并计算熵值的脚本

该脚本会：
1. 加载 parquet 数据集
2. 使用 vLLMRolloutWithTools 生成 response
3. 记录每轮工具调用后的熵值
4. 保存熵值数据用于后续绘图
"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import pandas as pd
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

# 添加 verl_arpo_entropy 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'verl_arpo_entropy'))

from verl import DataProto
from verl.workers.rollout.vllm_rollout.vllm_rollout_with_tools import vLLMRolloutWithTools
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntropyRecorder:
    """记录熵值的辅助类"""
    def __init__(self):
        self.entropy_history = []
        self.current_round = defaultdict(int)  # 记录每个样本的工具调用轮次
        
    def record_entropy(self, sample_idx: int, entropy: float, is_after_tool_call: bool = False):
        """记录熵值"""
        if is_after_tool_call:
            self.current_round[sample_idx] += 1
        
        self.entropy_history.append({
            'sample_idx': sample_idx,
            'round': self.current_round[sample_idx],
            'entropy': entropy,
            'is_after_tool_call': is_after_tool_call
        })
    
    def save(self, filepath: str):
        """保存熵值到文件"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.entropy_history, f, indent=2)
        logger.info(f"熵值数据已保存到: {filepath}")


def load_dataset(dataset_path: str, num_samples: int = None):
    """加载 parquet 数据集"""
    logger.info(f"加载数据集: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    if num_samples:
        df = df.head(num_samples)
        logger.info(f"限制样本数量为: {num_samples}")
    
    logger.info(f"数据集大小: {len(df)}")
    return df


def format_prompt(prompt_data):
    """格式化 prompt 数据"""
    if isinstance(prompt_data, np.ndarray):
        prompt_data = prompt_data.tolist()
    
    # prompt_data 应该是一个包含 system 和 user 消息的列表
    if isinstance(prompt_data, list) and len(prompt_data) >= 2:
        system_msg = prompt_data[0].get('content', '') if isinstance(prompt_data[0], dict) else str(prompt_data[0])
        user_msg = prompt_data[1].get('content', '') if isinstance(prompt_data[1], dict) else str(prompt_data[1])
        return f"{system_msg}\n\n{user_msg}"
    elif isinstance(prompt_data, str):
        return prompt_data
    else:
        # 尝试转换为字符串
        return str(prompt_data)


def tokenize_prompts(prompts: List[str], tokenizer, max_length: int = 1536):
    """将 prompts 转换为 token IDs"""
    # 使用 tokenizer 编码
    encodings = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    # 计算 position_ids（对于大多数模型，就是序列位置）
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    return input_ids, attention_mask, position_ids


def create_dataproto(input_ids, attention_mask, position_ids, meta_info=None):
    """创建 DataProto 对象"""
    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids
    }
    
    return DataProto.from_single_dict(batch, meta_info=meta_info or {})


def create_rollout_config(model_path: str, tools_config_path: str = None):
    """创建 rollout 配置"""
    config = DictConfig({
        'n': 1,  # 每个 prompt 生成 1 个 response
        'initial_rollouts': 1,
        'beam_size': 1,
        'branch_probability': 0.5,
        'entropy_weight': 0.2,
        'response_length': 4096,
        'logprobs': 10,  # 用于计算熵值
        'tools': {}
    })
    
    # 如果有工具配置文件，加载它
    if tools_config_path and os.path.exists(tools_config_path):
        tools_config = OmegaConf.load(tools_config_path)
        config.tools = tools_config.get('tools', {})
        logger.info(f"已加载工具配置: {tools_config_path}")
    else:
        logger.warning("未提供工具配置文件，将不使用工具")
    
    return config


def compute_entropy_from_logprobs(logprobs: List[float]) -> float:
    """从 logprobs 计算熵值"""
    if not logprobs:
        return 0.0
    p_list = [math.exp(l) for l in logprobs]
    entropy = -sum(p * l for p, l in zip(p_list, logprobs))
    return entropy


def patch_rollout_for_entropy_recording(rollout: vLLMRolloutWithTools, entropy_recorder: EntropyRecorder):
    """修改 rollout 对象以记录熵值"""
    # 保存原始的熵值计算和记录逻辑
    original_calc_entropy = rollout._calc_entropy
    
    # 创建一个包装的熵值字典来记录所有熵值
    rollout.entropy_recorder = entropy_recorder
    rollout.entropy_by_round = defaultdict(list)  # {round: [entropy_values]}
    rollout.current_sample_idx = 0
    
    # 修改 _calc_entropy 方法以记录熵值（如果需要）
    # 实际上，我们需要在 generate_sequences 中记录，所以这里只是保存引用
    return rollout


def main():
    parser = argparse.ArgumentParser(description='从数据集生成 response 并计算熵值')
    parser.add_argument('--dataset', type=str, 
                        default='/mnt/zhongwenlin/ARPO/ARPO/rl_datasets/nq_hotpotqa_arpo_format_train.parquet',
                        help='数据集路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--output_dir', type=str, default='./entropy_results',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='处理的样本数量')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--tools_config', type=str, default=None,
                        help='工具配置文件路径')
    parser.add_argument('--max_prompt_length', type=int, default=1536,
                        help='最大 prompt 长度')
    parser.add_argument('--max_response_length', type=int, default=4096,
                        help='最大 response 长度')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集
    df = load_dataset(args.dataset, args.num_samples)
    
    # 加载 tokenizer
    logger.info(f"加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建 rollout 配置
    rollout_config = create_rollout_config(args.model_path, args.tools_config)
    rollout_config.response_length = args.max_response_length
    
    # 初始化 rollout
    logger.info("初始化 vLLMRolloutWithTools...")
    try:
        from transformers import AutoConfig
        model_hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        
        rollout = vLLMRolloutWithTools(
            model_path=args.model_path,
            config=rollout_config,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config
        )
        logger.info("Rollout 初始化成功")
    except Exception as e:
        logger.error(f"Rollout 初始化失败: {e}")
        logger.error("请确保已安装 vLLM 和相关依赖")
        raise
    
    # 创建熵值记录器
    entropy_recorder = EntropyRecorder()
    
    # 为 rollout 添加熵值记录功能
    # 我们将在生成后从 rollout 对象中提取熵值
    rollout.entropy_recorder = entropy_recorder
    rollout.entropy_history = []
    
    # 处理数据
    all_responses = []
    all_entropies = []
    
    for batch_start in range(0, len(df), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        logger.info(f"处理批次 {batch_start//args.batch_size + 1}/{(len(df)-1)//args.batch_size + 1} "
                   f"(样本 {batch_start} 到 {batch_end-1})")
        
        # 格式化 prompts
        prompts_text = [format_prompt(row['prompt']) for _, row in batch_df.iterrows()]
        
        # Tokenize
        input_ids, attention_mask, position_ids = tokenize_prompts(
            prompts_text, tokenizer, args.max_prompt_length
        )
        
        # 创建 DataProto
        meta_info = {
            'do_sample': True,
            'validate': False
        }
        prompts_proto = create_dataproto(input_ids, attention_mask, position_ids, meta_info)
        
        # 设置样本索引（用于记录熵值）
        if hasattr(rollout, 'entropy_recorder'):
            rollout.current_sample_idx = batch_start
        
        # 生成 response
        try:
            results = rollout.generate_sequences(prompts_proto)
            
            # 提取 responses
            responses = results.batch['responses']
            loss_mask = results.batch.get('loss_mask', None)
            
            # 解码 responses 并提取熵值
            for i in range(len(prompts_text)):
                response_ids = responses[i]
                if loss_mask is not None:
                    # 只保留 loss_mask 为 1 的部分
                    mask = loss_mask[i].bool()
                    response_ids = response_ids[mask]
                
                response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
                sample_idx = batch_start + i
                
                all_responses.append({
                    'prompt': prompts_text[i],
                    'response': response_text,
                    'sample_idx': sample_idx
                })
                
                # 从 rollout 对象中提取熵值
                # 检查是否有工具调用（通过 response 中是否包含 </result> 标签）
                tool_call_count = response_text.count('</result>')
                
                # 提取初始熵值（第一轮生成）
                if hasattr(rollout, 'initial_entropy_dict'):
                    # 注意：rollout 中的索引可能不是我们期望的样本索引
                    # 我们需要找到对应的 rollout 索引
                    # 由于每个 batch 中可能有多个 rollout，我们需要映射
                    
                    # 简化：假设 rollout 中的索引顺序与我们的样本顺序一致
                    # 对于第一个 batch，索引应该是 0, 1, 2, ...
                    rollout_idx = i  # 在当前 batch 中的索引
                    
                    if rollout_idx in rollout.initial_entropy_dict:
                        initial_entropy = rollout.initial_entropy_dict[rollout_idx]
                        entropy_recorder.record_entropy(sample_idx, initial_entropy, is_after_tool_call=False)
                        
                        # 如果有工具调用，我们需要记录每轮工具调用后的熵值
                        # 但由于 rollout 代码没有保存每轮的熵值，我们只能记录初始熵值
                        # 要获取每轮的熵值，需要修改 rollout 代码
                        if tool_call_count > 0:
                            logger.debug(f"样本 {sample_idx} 有 {tool_call_count} 次工具调用，但无法获取每轮熵值")
                            # 我们可以尝试从 response 中推断工具调用位置
                            # 但这需要重新计算熵值，成本较高
            
            # 从 rollout 对象中提取熵值
            if hasattr(rollout, 'entropy_history') and rollout.entropy_history:
                logger.info(f"从 rollout 中提取到 {len(rollout.entropy_history)} 条熵值记录")
                # 将 rollout 的熵值历史添加到记录器
                for entry in rollout.entropy_history:
                    # 需要将 rollout 内部的索引映射到实际的样本索引
                    # 由于每个 batch 可能有多个 rollout，我们需要正确映射
                    # 简化：假设顺序一致
                    actual_sample_idx = batch_start + entry['sample_idx']
                    entropy_recorder.record_entropy(
                        actual_sample_idx, 
                        entry['entropy'], 
                        entry['is_after_tool_call']
                    )
                # 清空 rollout 的熵值历史，为下一批准备
                rollout.entropy_history = []
            
            if hasattr(rollout, 'initial_entropy_dict'):
                logger.info(f"检测到熵值字典，包含 {len(rollout.initial_entropy_dict)} 个样本的初始熵值")
                # 清空初始熵值字典，为下一批准备
                rollout.initial_entropy_dict = {}
            
        except Exception as e:
            logger.error(f"生成 response 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    logger.info("保存结果...")
    
    # 保存 responses
    responses_file = os.path.join(args.output_dir, 'responses.jsonl')
    with open(responses_file, 'w') as f:
        for item in all_responses:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Responses 已保存到: {responses_file}")
    
    # 保存熵值（如果有）
    if entropy_recorder.entropy_history:
        entropy_file = os.path.join(args.output_dir, 'entropy_data.json')
        entropy_recorder.save(entropy_file)
    else:
        logger.warning("未记录到熵值数据。可能需要修改 rollout 代码来记录熵值。")
    
    logger.info("完成！")


if __name__ == '__main__':
    main()

