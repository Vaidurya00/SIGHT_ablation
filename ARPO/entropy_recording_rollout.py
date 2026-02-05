"""
修改版本的 vLLMRolloutWithTools，用于记录熵值
"""

import math
from collections import defaultdict
from verl.workers.rollout.vllm_rollout.vllm_rollout_with_tools import vLLMRolloutWithTools


class EntropyRecordingRollout(vLLMRolloutWithTools):
    """扩展 vLLMRolloutWithTools 以记录熵值"""
    
    def __init__(self, *args, entropy_recorder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_recorder = entropy_recorder
        self.entropy_by_round = defaultdict(list)  # {round: [entropy_values]}
        self.sample_to_round = {}  # {sample_idx: current_round}
        
    def generate_sequences(self, prompts, **kwargs):
        """重写生成方法以记录熵值"""
        # 调用父类方法
        result = super().generate_sequences(prompts, **kwargs)
        
        # 在生成过程中，熵值已经被计算并存储在 self.initial_entropy_dict 和 current_entropy_dict 中
        # 但我们需要在工具调用后记录熵值
        
        # 由于熵值计算在父类的 while 循环中，我们需要在父类代码中插入记录逻辑
        # 但更好的方法是在这里分析生成的 response 来推断工具调用
        
        return result
    
    def _record_entropy_after_tool_call(self, sample_idx, entropy, round_num):
        """记录工具调用后的熵值"""
        if self.entropy_recorder:
            self.entropy_recorder.record_entropy(sample_idx, entropy, is_after_tool_call=True)
        self.entropy_by_round[round_num].append(entropy)
        self.sample_to_round[sample_idx] = round_num

