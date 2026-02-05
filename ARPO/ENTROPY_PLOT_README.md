# ARPO 熵增图生成指南

本指南说明如何生成 ARPO 论文中展示的熵增图，该图展示了每轮工具调用反馈后生成的初始 token 的熵值变化。

## 快速开始

### 方法 1: 使用模拟数据（演示）

```bash
cd /mnt/zhongwenlin/ARPO/ARPO
python plot_entropy.py --simulate --num_rounds 5 --num_samples 100 --output entropy_increase.png
```

### 方法 2: 从实际训练数据提取（需要修改代码记录熵值）

## 从实际训练数据提取熵值

由于当前代码在 rollout 过程中计算熵值但没有保存，我们需要修改代码来记录熵值。

### 步骤 1: 修改 rollout 代码以记录熵值

在 `verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py` 中，找到熵值计算的部分（约第 293-315 行），添加熵值记录功能：

```python
# 在 generate_sequences 方法中，找到熵值计算部分
# 添加以下代码来保存熵值

# 在类初始化中添加
self.entropy_history = []  # 记录每轮的熵值

# 在熵值计算后（约第 315 行后）添加
if out_idx not in self.initial_entropy_dict:
    self.initial_entropy_dict[out_idx] = entropy
    
# 记录熵值和工具调用轮次
tool_call_round = call_counters.get(out_idx, 0) + 1
self.entropy_history.append({
    'sample_idx': out_idx,
    'round': tool_call_round,
    'entropy': entropy,
    'is_after_tool_call': is_tool_call  # 标记是否在工具调用后
})
```

### 步骤 2: 保存熵值到文件

在 `generate_sequences` 方法的最后，添加保存熵值的代码：

```python
# 在返回 data_proto 之前添加
if hasattr(self, 'entropy_history') and self.entropy_history:
    entropy_file = os.path.join(self.config.get('entropy_save_dir', '.'), 
                                f'entropy_step_{global_step}.json')
    os.makedirs(os.path.dirname(entropy_file), exist_ok=True)
    with open(entropy_file, 'w') as f:
        json.dump(self.entropy_history, f, indent=2)
    self.entropy_history = []  # 清空历史
```

### 步骤 3: 从保存的熵值文件生成图表

修改 `plot_entropy.py` 脚本，添加从 JSON 文件读取熵值的功能：

```python
def load_entropy_from_json(json_file: str) -> Dict[int, List[float]]:
    """从 JSON 文件加载熵值数据"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    entropy_by_round = {}
    for entry in data:
        if entry.get('is_after_tool_call', False):
            round_num = entry['round']
            if round_num not in entropy_by_round:
                entropy_by_round[round_num] = []
            entropy_by_round[round_num].append(entry['entropy'])
    
    return entropy_by_round
```

然后运行：

```bash
python plot_entropy.py --entropy_file path/to/entropy_step_*.json --output entropy_increase.png
```

## 脚本参数说明

```bash
python plot_entropy.py [选项]

选项:
  --log_file LOG_FILE        训练日志文件路径
  --rollout_dir ROLLOUT_DIR  Rollout 数据目录路径
  --entropy_file ENTROPY_FILE 熵值 JSON 文件路径
  --output OUTPUT            输出图片路径（默认: entropy_increase.png）
  --simulate                 使用模拟数据（用于演示）
  --num_rounds NUM_ROUNDS    工具调用轮次数（用于模拟，默认: 5）
  --num_samples NUM_SAMPLES  每轮样本数（用于模拟，默认: 100）
```

## 示例

### 示例 1: 使用模拟数据生成演示图

```bash
python plot_entropy.py --simulate --num_rounds 5 --num_samples 100 --output demo.png
```

### 示例 2: 从实际数据生成（需要先修改代码记录熵值）

```bash
# 1. 修改代码记录熵值（见步骤 1-2）
# 2. 运行训练，熵值会保存到文件
# 3. 从熵值文件生成图表
python plot_entropy.py --entropy_file checkpoint_save_dir/entropy_step_100.json --output entropy_increase.png
```

## 图表说明

生成的熵增图展示了以下信息：

1. **X 轴**: 工具调用轮次（Tool Call Round）
2. **Y 轴**: 归一化熵值（Normalized Entropy，范围 0-1）
3. **数据点**: 每轮工具调用后初始 token 的熵值分布
4. **误差棒**: 显示熵值的标准差

根据 ARPO 论文，每轮工具调用反馈后生成的初始 token 应该表现出较高的熵值（通常 > 0.6），这表明外部工具调用引入了不确定性到 LLM 的推理过程中。

## 注意事项

1. **字体问题**: 如果系统没有中文字体，脚本会自动使用英文标签
2. **数据来源**: 当前代码默认不保存熵值，需要修改代码才能从实际训练数据中提取
3. **模拟数据**: 使用 `--simulate` 参数可以快速生成演示图表，但数据是模拟的

## 故障排除

### 问题 1: 中文字体警告

**解决方案**: 脚本会自动处理，使用英文标签。如果需要中文，请安装中文字体：
- Ubuntu/Debian: `sudo apt-get install fonts-wqy-microhei`
- CentOS/RHEL: `sudo yum install wqy-microhei-fonts`

### 问题 2: 无法从日志提取熵值

**原因**: 当前代码不记录熵值到日志

**解决方案**: 
1. 使用模拟数据（`--simulate`）
2. 或按照步骤 1-2 修改代码来记录熵值

### 问题 3: 图片显示不正常

**解决方案**: 确保安装了 matplotlib 和 numpy：
```bash
pip install matplotlib numpy
```

## 联系

如有问题，请参考 ARPO 项目的主 README 或提交 issue。

