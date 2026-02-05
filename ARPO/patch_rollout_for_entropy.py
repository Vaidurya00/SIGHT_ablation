"""
为 vllm_rollout_with_tools.py 添加熵值记录功能的补丁
"""

import math
from collections import defaultdict


def patch_rollout_for_entropy_recording(rollout_module):
    """
    为 rollout 模块添加熵值记录功能
    
    使用方法:
        import verl.workers.rollout.vllm_rollout.vllm_rollout_with_tools as rollout_module
        patch_rollout_for_entropy_recording(rollout_module)
    """
    original_generate = rollout_module.vLLMRolloutWithTools.generate_sequences
    
    def generate_sequences_with_entropy_recording(self, prompts, **kwargs):
        """带熵值记录的生成方法"""
        # 初始化熵值记录
        if not hasattr(self, 'entropy_history'):
            self.entropy_history = []
        if not hasattr(self, 'sample_to_round'):
            self.sample_to_round = {}
        
        # 调用原始生成方法
        result = original_generate(self, prompts, **kwargs)
        
        # 在生成过程中，熵值已经被计算
        # 我们需要从 self.initial_entropy_dict 和工具调用信息中提取
        
        return result
    
    # 替换方法
    rollout_module.vLLMRolloutWithTools.generate_sequences = generate_sequences_with_entropy_recording
    
    # 修改熵值计算部分，添加记录逻辑
    original_calc_entropy = rollout_module.vLLMRolloutWithTools._calc_entropy
    
    def _calc_entropy_with_recording(self, logprobs):
        """带记录的熵值计算"""
        entropy = original_calc_entropy(self, logprobs)
        
        # 如果设置了记录器，记录熵值
        if hasattr(self, 'entropy_recorder'):
            # 这里需要知道当前是哪个样本和哪一轮
            # 但由于在 while 循环中，我们需要在调用处记录
            pass
        
        return entropy
    
    rollout_module.vLLMRolloutWithTools._calc_entropy = _calc_entropy_with_recording
    
    return rollout_module

