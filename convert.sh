source activate llm;

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ./checkpoints/dr-zero/solver_iter1_ratio4321_grpo_group5_qwen2.5-3b-instruct/global_step_50/actor \
    --target_dir ./checkpoints/dr-zero/solver_iter1_ratio4321_grpo_group5_qwen2.5-3b-instruct/solver_iter1_hf