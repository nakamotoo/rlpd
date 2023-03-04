export CUDA_VISIBLE_DEVICES=5
# export WANDB_DISABLED=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false 

# env=antmaze-large-play-v2
# env=antmaze-large-diverse-v2
# env=antmaze-medium-diverse-v2
# env=pen-binary-v0
# env=door-binary-v0
env=relocate-binary-v0

# for seed in 42 43 44
for seed in 1 2 3
do
python train_finetuning.py --env_name=$env \
                --seed=$seed \
                --utd_ratio=20 \
                --start_training 5000 \
                --max_steps 3000000 \
                --config=configs/rlpd_config.py \
                --config.backup_entropy=False \
                --config.hidden_dims="(256, 256, 256)" \
                --project_name=rlpd_adroit
done