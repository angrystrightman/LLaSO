#!/bin/bash
#--model_name_or_path /code/syr/Llama-3.2-3B-Instruct - replace with your local path. It is the LLM backbone.
#--audio_tower /code/syr/whisper-large-v3 - replace it with your local path for the audio encoder.
#At present we default audio select layer is the last layer of the encoder.

deepspeed   /llaso/train/train_mem.py \
    --deepspeed  /llaso/scripts/zero2.json \
    --model_name_or_path /code/syr/Llama-3.2-3B-Instruct \
    --version llama32_audio_v1_mmtag \
    --data_path ./LLaSO-Align/text_audio/ASR_stage1_train.json \
    --audio_tower /code/syr/whisper-large-v3 \
    --audio_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --tune_mm_audio_projector True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_audio_start_end True \
    --bf16 True \
    --output_dir /llaso/llaso_aligned_3b_ckpts \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 20 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.0015 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False\
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --mix_va False \
    --group_by_modality_length False \
    --max_grad_norm 1.0 \