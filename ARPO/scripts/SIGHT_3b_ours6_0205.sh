SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "Switched to parent directory: $PARENT_DIR"


# ============================ Environment Setup ============================
# Set basic environment variables
export PYTHONUNBUFFERED=1            
export HYDRA_FULL_ERROR=1           
export VLLM_ATTENTION_BACKEND=XFORMERS 
export VERL_LOGGING_LEVEL=INFO   # DEBUG 时日志很大，正常跑用 INFO
export MKL_SERVICE_FORCE_INTEL=1    
export MKL_THREADING_LAYER=GNU       
export RAY_memory_usage_threshold=0.8  
export RAY_memory_monitor_refresh_ms=0 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # TODO
export RAY_HEAD_NODE_PORT=6380
export RAY_OBJECT_STORE_PORT=12346


# ========== Ray Distributed Debugger 配置 ==========
# 启用 post-mortem 调试（已注释，避免程序卡住等待调试器）
# export RAY_DEBUG_POST_MORTEM=1

# 重要：确保移除旧的调试标志（如果存在）
# 不要设置 RAY_DEBUG=legacy
# 不要使用 --ray-debugger-external 参数



# Set Python path
export PYTHONPATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/yangjinluan/project/zwl_project/SIGHT_ablation/ARPO"/verl_arpo_entropy:$PYTHONPATH

# ============================ Basic Configuration ============================
# Experiment name and project
PROJECT_NAME="reasoning_tasks" # Modify experiment group
EXPERIMENT_NAME="ARPO_7B_ours6_0205" # Modify experiment name

# Configuration file path
CONFIG_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/yangjinluan/project/zwl_project/SIGHT_ablation/ARPO/scripts/config" # Modify the absolute path of the config folder, relative path is not recommended
CONFIG_NAME="ppo_trainer.yaml"

# Distributed training settings
NNODES=1                            
N_GPUS_PER_NODE=8              # TODO      

# ============================ Data Configuration ============================
# Data parameters
PROMPT_KEY="prompt"                 # Prompt field name
TRAIN_BATCH_SIZE=128                # Training batch size
PPO_MINI_BATCH_SIZE=32              # PPO mini-batch size
MAX_PROMPT_LENGTH=1536              # Maximum prompt length
MAX_RESPONSE_LENGTH=4096            # Maximum response length

# Data file paths
TRAIN_FILES="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/yangjinluan/project/zwl_project/SIGHT_ablation/ARPO/rl_datasets/SIGHT/train.parquet"  # 
VALID_FILES="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/yangjinluan/project/zwl_project/SIGHT_ablation/ARPO/rl_datasets/SIGHT/test.parquet"  # 

# ============================ Model Configuration ============================
# Actor model path
ACTOR_MODEL_PATH="/mnt/zhongwenlin/model/Qwen/Qwen2.5-3B-Instruct" # TODO

# ============================ Rollout Configuration ==========================
# Rollout settings
ROLLOUT_NAME="vllm"                 # Use vllm engine
ROLLOUT_MODE="sync_with_tool"       # Synchronous mode with tool support
# 用哪个 rollout 模块（对应 verl/workers/rollout/vllm_rollout/ 下的文件名，不含 .py）
# 可选: vllm_rollout_with_tools | vllm_rollout_with_tools_IGD_1230
ROLLOUT_WITH_TOOLS_MODULE="vllm_rollout_with_tools_IGD_1230"
export ROLLOUT_WITH_TOOLS_MODULE

ROLLOUT_N=16                         # Number of responses generated per sample
INITIAL_ROLLOUTS=8                 # Initial rollout number
BEAM_SIZE=2                        # Beam size
BRANCH_PROBABILITY=0.5             # Branch probability
Entropy_weight=0.2
# ============================ Rollout Tools Configuration ==========================

# ============================ Reward Model Configuration ==========================
# Reward model settings
REWARD_MANAGER="naive"              # Reward manager type
CUSTOM_REWARD_FUNCTION_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/yangjinluan/project/zwl_project/SIGHT_ablation/ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research_SIGHT_wo_ses_reward0204.py" # 
CUSTOM_REWARD_FUNCTION_NAME="compute_score"
FORMAT_ONLY_STEPS=15                # Only use format reward for the first N steps, other rewards set to 0
export FORMAT_ONLY_STEPS            # Export to environment variable for Python code

# ============================ Training Configuration ============================
# Training parameters
TOTAL_EPOCHS=3                      # Total training epochs
SAVE_FREQ=25                        # Save frequency
TEST_FREQ=1000                        # Test frequency

# ============================ Path Configuration ============================
# Save path
SAVE_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/yangjinluan/project/verl_ckpt/${EXPERIMENT_NAME}" # Modify save path
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============================ WandB Configuration ============================
# WandB settings
WANDB_API_KEY="02432d171dfdeec4e0a105a9f2d3d17fe25b0a28" # Modify your wandb key

# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

# Create save directory
if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

# Create rollout save directory
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

# ============================ Start Training ============================
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
    actor_rollout_ref.rollout.with_tools_module=${ROLLOUT_WITH_TOOLS_MODULE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${Entropy_weight} \
    actor_rollout_ref.rollout.multi_turn.enable=${ENABLE_MULTI_TURN} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console]" \
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

'''

tmux new -s arpo-env
tmux attach -t arpo-env


cd /mnt/zhongwenlin/ours2/ARPO/scripts
conda activate arpo

'''
