set -x

kill -9 $(lsof -t -i :8000);

cur_iter=3
prev_iter=2

tp=2
dp=4
gpus=8
batch_per_gpu=2
rollout_memory_utilization=0.5

challenger_algorithm=grpo_batch
challenger_grpo_group_size=1
challenger_reward_group_size=5

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

algorithm=grpo
grpo_group_size=5
model=Qwen/Qwen2.5-3B-Instruct
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

CONFIG_PATH="./config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

CHALLENGER_NAME="challenger_iter${cur_iter}_ratio${hop_ratio}_${challenger_algorithm}_group${challenger_grpo_group_size}-${challenger_reward_group_size}_${model_name}"
SOLVER_NAME="solver_iter${cur_iter}_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}"

TRAIN_DATA="./data/zero_${CHALLENGER_NAME}.parquet"
VAL_DATA="./data/test_1200.parquet"
VAL_DATA_DIR="./data/${SOLVER_NAME}"

RESUME_PATH="./checkpoints/dr-zero/solver_iter${prev_iter}_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}/solver_iter${prev_iter}_hf"


source activate llm;

python search/retrieval_server.py \
    --index_path='./corpus/e5_Flat.index' \
    --corpus_path='./corpus/wiki-18.jsonl' \
    --retriever_model='intfloat/e5-base-v2' \
    --retriever_name='e5' \
    --faiss_gpu \
    --topk 3 &

python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=${algorithm} \
    actor_rollout_ref.model.path=$RESUME_PATH \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=${grpo_group_size} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.logger='["wandb", "console"]' \
    trainer.project_name="dr-zero" \
    trainer.experiment_name=$SOLVER_NAME \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.val_before_train=False \
    trainer.validation_data_dir=${VAL_DATA_DIR} \
    trainer.total_epochs=1 $@