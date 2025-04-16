#!/bin/bash

# Запускаем обучение через accelerate

accelerate launch --config_file /home/jovyan/nkiselev/kazachkovda/2025-project-DiffModels/accelerate_config.yaml tutorial_train_plus.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --data_json_file="/home/jovyan/nkiselev/kazachkovda/2025-project-DiffModels/data/metadata.jsonl" \
  --data_root_path="/home/jovyan/nkiselev/kazachkovda/2025-project-DiffModels/dataset" \
  --mixed_precision="bf16" \
  --resolution=512 \
  --train_batch_size=64 \
  --dataloader_num_workers=8 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --num_train_epochs=50 \
  --output_dir="/home/jovyan/nkiselev/kazachkovda/2025-project-DiffModels/output_dir" \
  --save_steps=250 \
  --report_to="wandb"
