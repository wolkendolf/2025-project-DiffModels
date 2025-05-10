CUDA_VISIBLE_DEVICES=2 python3 train_ip_adapter_selfattention.py \
    --images_number 100 \
    --save_steps 10 \
    --num_train_epochs 10 \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --dir_data_json_files "/home/kazachkovda/2025-project-DiffModels/metadata_jsons" \
    --output_dir "/home/kazachkovda/2025-project-DiffModels/weights" \
    --data_root_path "/data/kazachkovda/2025_ipAdap_image/" \
    --image_encoder_path "/home/kazachkovda/2025-project-DiffModels/weights/IP-Adapter/models/image_encoder"