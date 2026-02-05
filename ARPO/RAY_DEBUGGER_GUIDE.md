# Ray Distributed Debugger 使用指南

本指南说明如何使用 Ray Distributed Debugger 来调试 ARPO 项目。

## 📋 前置要求

### 1. 安装必要的软件包

确保已安装以下依赖：

```bash
# 确保 Ray 版本 >= 2.39
pip install "ray[default]>=2.39"

# 安装 debugpy
pip install "debugpy>=1.8.0"
```

### 2. 安装 VSCode 扩展

1. 在 VSCode 中安装 **Ray Distributed Debugger** 扩展
   - 打开 VSCode
   - 进入 Extensions (Ctrl+Shift+X)
   - 搜索 "Ray Distributed Debugger"
   - 点击安装

## 🔧 配置步骤

### 步骤 1: 设置环境变量

在训练脚本中添加调试相关的环境变量。修改你的训练脚本（如 `ARPO_7B_Reasoning_1node.sh`），在环境变量设置部分添加：

```bash
# ============================ Environment Setup ============================
# Set basic environment variables
export PYTHONUNBUFFERED=1            
export HYDRA_FULL_ERROR=1           
export VLLM_ATTENTION_BACKEND=XFORMERS 
export VERL_LOGGING_LEVEL=DEBUG
export MKL_SERVICE_FORCE_INTEL=1    
export MKL_THREADING_LAYER=GNU       
export RAY_memory_usage_threshold=0.8  
export RAY_memory_monitor_refresh_ms=0 

# ========== Ray Distributed Debugger 配置 ==========
# 启用 post-mortem 调试
export RAY_DEBUG_POST_MORTEM=1

# 重要：确保移除旧的调试标志（如果存在）
# 不要设置 RAY_DEBUG=legacy
# 不要使用 --ray-debugger-external 参数
```

### 步骤 2: 启动 Ray 集群（单机模式）

对于单机训练，Ray 会自动初始化。但如果你想手动控制，可以：

**选项 A: 使用自动初始化（推荐用于单机）**
- 直接运行训练脚本，代码会自动调用 `ray.init()`

**选项 B: 手动启动 Ray 集群（用于多节点或需要 Dashboard）**
```bash
# 启动 head 节点
ray start --head --dashboard-host=0.0.0.0 --port=6379

# 查看 Ray 状态
ray status

# 获取 Dashboard 地址（通常是 http://<head_node_ip>:8265）
```

### 步骤 3: 在代码中添加断点

在你想调试的 `@ray.remote` 函数中添加 `breakpoint()`。例如：

#### 示例 1: 在 TaskRunner 中添加断点

编辑 `/mnt/zhongwenlin/ARPO/ARPO/verl_arpo_entropy/verl/trainer/main_ppo.py`:

```python
@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        # 在这里添加断点
        breakpoint()  # 调试训练开始时的配置
        
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        
        # 在这里添加另一个断点
        breakpoint()  # 调试模型路径加载
        
        # ... 其余代码
```

#### 示例 2: 在 Worker 类中添加断点

如果你想调试具体的 worker 逻辑，需要在相应的 worker 类中添加断点。例如，在 rollout worker 中：

```python
# 在 verl/workers/fsdp_workers.py 或相关文件中
@ray.remote
class ActorRolloutRefWorker:
    def some_method(self, ...):
        breakpoint()  # 在这里添加断点
        # ... 你的代码
```

### 步骤 4: 在 VSCode 中配置 Ray 集群连接

1. 打开 VSCode
2. 点击左侧边栏的 **Ray Distributed Debugger** 图标
3. 点击 **"Add Cluster"** 或 **"+"** 按钮
4. 输入 Ray Dashboard 地址：
   - 单机模式：`http://localhost:8265`（如果 Ray 已启动）
   - 多节点模式：`http://<head_node_ip>:8265`
   - 如果使用自动初始化，可能需要先启动 Ray Dashboard

### 步骤 5: 运行训练脚本

直接运行你的训练脚本（**不要使用 launch.json**）：

```bash
cd /mnt/zhongwenlin/ARPO/ARPO/scripts
conda activate arpo
bash ARPO_7B_Reasoning_1node.sh
```

或者直接运行 Python 命令：

```bash
python3 -m verl.trainer.main_ppo \
    --config-path=/mnt/zhongwenlin/ARPO/ARPO/scripts/config \
    --config-name=ppo_trainer.yaml \
    # ... 其他参数
```

### 步骤 6: 附加调试器

1. 当代码执行到 `breakpoint()` 时，程序会暂停
2. 在 VSCode 中，点击 **Ray Distributed Debugger** 侧边栏图标
3. 你会看到检测到的断点信息
4. 点击 **"Attach"** 或断点旁边的连接按钮来附加调试器
5. 现在你可以在 VSCode 中：
   - 查看变量值
   - 单步执行
   - 查看调用栈
   - 修改变量值（如果支持）

### 步骤 7: 处理多个断点

如果代码中有多个 `breakpoint()`：

1. 当第一个断点触发时，附加调试器
2. 调试完成后，**先断开当前调试会话**
3. 继续执行，当下一个断点触发时，再次附加调试器

## 📝 完整示例脚本

创建一个支持调试的训练脚本 `ARPO_7B_Reasoning_1node_debug.sh`:

```bash
#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "Switched to parent directory: $PARENT_DIR"

# ============================ Environment Setup ============================
# Set basic environment variables
export PYTHONUNBUFFERED=1            
export HYDRA_FULL_ERROR=1           
export VLLM_ATTENTION_BACKEND=XFORMERS 
export VERL_LOGGING_LEVEL=DEBUG
export MKL_SERVICE_FORCE_INTEL=1    
export MKL_THREADING_LAYER=GNU       
export RAY_memory_usage_threshold=0.8  
export RAY_memory_monitor_refresh_ms=0 

# ========== Ray Distributed Debugger 配置 ==========
export RAY_DEBUG_POST_MORTEM=1

# Set Python path
export PYTHONPATH="/mnt/zhongwenlin/ARPO/ARPO"/verl_arpo_entropy:$PYTHONPATH

# ============================ Basic Configuration ============================
PROJECT_NAME="reasoning_tasks"
EXPERIMENT_NAME="ARPO_debug_test"
CONFIG_PATH="/mnt/zhongwenlin/ARPO/ARPO/scripts/config"
CONFIG_NAME="ppo_trainer.yaml"
NNODES=1                            
N_GPUS_PER_NODE=4                   

# ============================ Data Configuration ============================
PROMPT_KEY="prompt"
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=16
MAX_PROMPT_LENGTH=1536
MAX_RESPONSE_LENGTH=4096
TRAIN_FILES="/mnt/zhongwenlin/ARPO/ARPO/rl_datasets/hard_search_1k.parquet"
VALID_FILES="/mnt/zhongwenlin/ARPO/ARPO/rl_datasets/gaia_test.parquet"

# ============================ Model Configuration ============================
ACTOR_MODEL_PATH="/mnt/zhongwenlin/model/Qwen/Qwen2.5-3B-Instruct"

# ============================ Rollout Configuration ==========================
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync_with_tool"
ROLLOUT_N=16
INITIAL_ROLLOUTS=8
BEAM_SIZE=2
BRANCH_PROBABILITY=0.5
Entropy_weight=0.2
SEARCH_CACHE_PATH="/mnt/zhongwenlin/ARPO/ARPO/search_cache/search_cache.json"

# ============================ Reward Model Configuration ==========================
REWARD_MANAGER="naive"
CUSTOM_REWARD_FUNCTION_PATH="/mnt/zhongwenlin/ARPO/ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============================ Training Configuration ============================
TOTAL_EPOCHS=2
SAVE_FREQ=5
TEST_FREQ=5

# ============================ Path Configuration ============================
SAVE_PATH="/mnt/zhongwenlin/ARPO/ARPO/checkpoint_save_dir/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============================ Preparation ============================
if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

# ============================ Start Training (with Debug) ============================
echo "Starting training with Ray Distributed Debugger..."
echo "Make sure to attach the debugger in VSCode when breakpoints are hit!"

python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${Entropy_weight} \
    +actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log
```

## ⚠️ 重要注意事项

1. **断点位置限制**：
   - 断点只能在 `@ray.remote` 装饰的函数内部使用
   - 不能在主进程（非 remote 函数）中使用 `breakpoint()`

2. **环境变量**：
   - 必须设置 `RAY_DEBUG_POST_MORTEM=1`
   - 不要使用旧的 `RAY_DEBUG=legacy` 标志
   - 不要使用 `--ray-debugger-external` 参数

3. **Ray 版本**：
   - 确保 Ray 版本 >= 2.39
   - 如果版本过低，请升级：`pip install --upgrade "ray[default]>=2.39"`

4. **多节点调试**：
   - 对于多节点训练，需要先手动启动 Ray 集群
   - 确保所有节点都能访问 Dashboard 地址

5. **性能影响**：
   - 调试模式可能会影响训练性能
   - 建议只在需要调试时启用

## 🔍 调试技巧

1. **查看 Ray Dashboard**：
   - 访问 `http://localhost:8265` 查看 Ray 集群状态
   - 可以查看任务执行情况和资源使用

2. **日志查看**：
   - 训练日志会输出到 `${SAVE_PATH}/run.log`
   - 可以在终端实时查看日志

3. **逐步调试**：
   - 从简单的断点开始，逐步深入
   - 使用 VSCode 的调试控制（继续、单步、查看变量等）

## 📚 参考资源

- [Ray Distributed Debugger 官方文档](https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html)
- [VERL 多节点训练文档](../verl_arpo_entropy/docs/start/multinode.rst)

## 🐛 常见问题

**Q: 断点没有触发？**
A: 确保：
- 代码在 `@ray.remote` 函数内
- 设置了 `RAY_DEBUG_POST_MORTEM=1`
- Ray 版本 >= 2.39
- VSCode 扩展已正确安装

**Q: 无法连接到 Ray 集群？**
A: 检查：
- Ray Dashboard 是否可访问
- 防火墙设置
- 网络连接

**Q: 调试器附加失败？**
A: 确保：
- `debugpy` 已安装
- Ray 集群正常运行
- 没有使用旧的调试标志

