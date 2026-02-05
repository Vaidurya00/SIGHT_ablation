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

from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", global_step=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.global_step = global_step  # Store global step for format-only mode
    
    def set_global_step(self, global_step):
        """Update the global step for format-only mode."""
        self.global_step = global_step

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        # 获取MI reward增量数组（如果存在）
        # mi_reward_increment是在rollout阶段添加到non_tensor_batch中的，是一个与batch size相同的数组
        mi_reward_increment_array = None
        if "mi_reward_increment" in data.non_tensor_batch:
            mi_reward_increment_array = data.non_tensor_batch["mi_reward_increment"]

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            # add tokenizer to extra_info if not exists
            if extra_info is None or extra_info.get("tokenizer") is None:
                extra_info = {}
            if "tokenizer" not in extra_info:
                extra_info["tokenizer"] = self.tokenizer
            # add global_step to extra_info if available
            if self.global_step is not None and "global_step" not in extra_info:
                extra_info["global_step"] = self.global_step
            
            # 获取当前样本的MI reward增量
            if mi_reward_increment_array is not None:
                if isinstance(mi_reward_increment_array, (list, tuple, np.ndarray)):
                    # 如果是数组，取当前索引的值
                    if i < len(mi_reward_increment_array):
                        extra_info["mi_reward_increment"] = float(mi_reward_increment_array[i])
                    else:
                        extra_info["mi_reward_increment"] = 0.0
                elif isinstance(mi_reward_increment_array, (int, float)):
                    # 如果是单个数值（理论上不应该发生，但为了兼容性保留）
                    extra_info["mi_reward_increment"] = float(mi_reward_increment_array)
                else:
                    extra_info["mi_reward_increment"] = 0.0

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                # 直接使用 compute_score 返回的 score，不再进行额外计算
                # score 已经在 deep_research_IGD.py 中计算完成，包含所有必要的加权和 bonus
                reward = score.get("score", 0)
                if "score" not in score:
                    print(f"Warning: 'score' key not found in score dict, using 0 as default")
                
                # Store the information including original reward and individual scores
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
                # 如果不是字典，设置默认值
                reward_extra_info["score_format"].append(0)
                reward_extra_info["score_accuracy"].append(0)
                reward_extra_info["score_mi"].append(0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
