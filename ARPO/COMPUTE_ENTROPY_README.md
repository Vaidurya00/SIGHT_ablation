# 从数据集计算熵值使用指南

本指南说明如何使用 `compute_entropy_from_dataset.py` 脚本从数据集中生成 response 并计算熵值。

## 功能说明

该脚本会：
1. 加载 parquet 数据集
2. 使用 vLLMRolloutWithTools 生成 response（支持工具调用）
3. 在生成过程中自动记录每轮工具调用后的熵值
4. 保存 responses 和熵值数据用于后续分析

## 使用方法

### 基本用法

```bash
cd /mnt/zhongwenlin/ARPO/ARPO
python compute_entropy_from_dataset.py \
    --model_path /path/to/your/model \
    --dataset rl_datasets/nq_hotpotqa_arpo_format_train.parquet \
    --output_dir ./entropy_results \
    --num_samples 100 \
    --batch_size 8
```

### 完整参数说明

```bash
python compute_entropy_from_dataset.py [选项]

必需参数:
  --model_path MODEL_PATH    模型路径（必需）

可选参数:
  --dataset DATASET          数据集路径（默认: rl_datasets/nq_hotpotqa_arpo_format_train.parquet）
  --output_dir OUTPUT_DIR    输出目录（默认: ./entropy_results）
  --num_samples NUM_SAMPLES  处理的样本数量（默认: 100）
  --batch_size BATCH_SIZE    批次大小（默认: 8）
  --tools_config TOOLS_CONFIG 工具配置文件路径（可选）
  --max_prompt_length LENGTH 最大 prompt 长度（默认: 1536）
  --max_response_length LENGTH 最大 response 长度（默认: 4096）
```

### 示例

#### 示例 1: 基本使用

```bash
python compute_entropy_from_dataset.py \
    --model_path /mnt/zhongwenlin/model/Qwen/Qwen2.5-3B-Instruct \
    --num_samples 50 \
    --batch_size 4
```

#### 示例 2: 使用工具配置

```bash
python compute_entropy_from_dataset.py \
    --model_path /mnt/zhongwenlin/model/Qwen/Qwen2.5-3B-Instruct \
    --tools_config verl_arpo_entropy/verl/workers/rollout/tools/config_example.yaml \
    --num_samples 100 \
    --output_dir ./entropy_results_with_tools
```

## 输出文件

脚本会在输出目录中生成以下文件：

1. **responses.jsonl**: 包含所有生成的 responses
   ```json
   {
     "prompt": "原始 prompt",
     "response": "生成的 response",
     "sample_idx": 0
   }
   ```

2. **entropy_data.json**: 包含所有熵值记录
   ```json
   [
     {
       "sample_idx": 0,
       "round": 0,
       "entropy": 0.75,
       "is_after_tool_call": false
     },
     {
       "sample_idx": 0,
       "round": 1,
       "entropy": 0.82,
       "is_after_tool_call": true
     }
   ]
   ```

## 使用熵值数据生成图表

生成熵值数据后，可以使用 `plot_entropy.py` 脚本生成熵增图：

```bash
python plot_entropy.py \
    --entropy_file entropy_results/entropy_data.json \
    --output entropy_increase.png
```

## 代码修改说明

为了支持熵值记录，我已经修改了以下文件：

1. **verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py**
   - 添加了 `entropy_history` 和 `entropy_recorder` 属性
   - 在熵值计算时自动记录熵值
   - 区分工具调用前后的熵值

## 注意事项

1. **模型路径**: 确保模型路径正确，且模型支持工具调用
2. **工具配置**: 如果使用工具，需要提供正确的工具配置文件
3. **内存**: 生成 response 需要足够的 GPU 内存
4. **批次大小**: 根据 GPU 内存调整批次大小
5. **样本数量**: 建议先用少量样本测试（如 10-20 个）

## 故障排除

### 问题 1: 模型加载失败

**错误**: `Rollout 初始化失败`

**解决方案**:
- 检查模型路径是否正确
- 确保已安装 vLLM: `pip install vllm`
- 检查 GPU 是否可用

### 问题 2: 工具调用失败

**错误**: `Tool execution failed`

**解决方案**:
- 检查工具配置文件是否正确
- 确保工具 API 密钥已配置
- 查看日志了解具体错误

### 问题 3: 内存不足

**错误**: `CUDA out of memory`

**解决方案**:
- 减小 `--batch_size`
- 减小 `--max_response_length`
- 使用更小的模型

### 问题 4: 未记录到熵值

**原因**: 可能是样本没有触发工具调用

**解决方案**:
- 检查生成的 response 是否包含工具调用
- 确保工具配置正确
- 增加样本数量

## 后续步骤

1. 使用生成的熵值数据绘制熵增图
2. 分析不同轮次的熵值变化
3. 比较不同模型或配置的熵值模式

## 联系

如有问题，请参考 ARPO 项目的主 README 或提交 issue。

