#!/bin/bash

export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

# Update paths to match your parquet dataset
DATA_DIR=/fsx-project/philippehansen/datasets/MiraData/try_data/parquet_dataset_distributed
VALIDATION_DIR=/fsx-project/philippehansen/datasets/MiraData/try_data/parquet_dataset_distributed  # Using same data for now
NUM_GPUS=8

# Checkpoint path (uncomment to resume from checkpoint)
# CHECKPOINT_PATH="$DATA_DIR/outputs/wan_finetune/checkpoint-50"

# Training with 8 GPUs - adjusted parameters for your setup
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    -m fastvideo.v1.training.wan_training_pipeline \
    --model_path /fsx-project/philippehansen/models/Wan2.1-T2V-1.3B-Diffusers \
    --inference_mode False \
    --pretrained_model_name_or_path /fsx-project/philippehansen/models/Wan2.1-T2V-1.3B-Diffusers \
    --cache_dir "/fsx-project/philippehansen/.cache" \
    --data_path "$DATA_DIR" \
    --validation_prompt_dir "$VALIDATION_DIR" \
    --train_batch_size=1 \
    --num_latent_t 20 \
    --sp_size $NUM_GPUS \
    --tp_size $NUM_GPUS \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000 \
    --learning_rate=1e-6 \
    --mixed_precision="bf16" \
    --checkpointing_steps=100 \
    --validation_steps 50 \
    --gradient_checkpointing \
    --validation_sampling_steps "2,4,8" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --output_dir="/fsx-project/philippehansen/datasets/MiraData/try_data/outputs/wan_finetune" \
    --tracker_project_name wan_mira_finetune \
    --num_height 480 \
    --num_width 832 \
    --num_frames 81 \
    --shift 3 \
    --validation_guidance_scale "1.0" \
    --num_euler_timesteps 50 \
    --multi_phased_distill_schedule "4000-1" \
    --weight_decay 0.01 \
    --not_apply_cfg_solver \
    --master_weight_type "fp32" \
    --max_grad_norm 1.0

# Uncomment to resume from checkpoint
# --resume_from_checkpoint "$CHECKPOINT_PATH"
