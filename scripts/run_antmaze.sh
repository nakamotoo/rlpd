export CUDA_VISIBLE_DEVICES=3
export WANDB_DISABLED=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false 
export D4RL_SUPPRESS_IMPORT_ERROR=1

# env=antmaze-large-play-v2
# env=antmaze-large-diverse-v2
# env=antmaze-medium-diverse-v2
env=antmaze-medium-play-v2

for seed in 42 43 44 45 46
do
python train_finetuning.py --env_name=$env \
                --seed=$seed \
                --utd_ratio=20 \
                --start_training 5000 \
                --max_steps 500000 \
                --config=configs/rlpd_config.py \
                --config.backup_entropy=False \
                --config.hidden_dims="(256, 256, 256)" \
                --config.num_min_qs=2 \
                --project_name=rlpd_antmaze
done