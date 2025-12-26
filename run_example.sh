#!/bin/bash
PORT_ID=$(expr $RANDOM + 1000)

# create a directory for the cache
mkdir -p /tmp/triton_cache
# set the cache
export TRITON_CACHE_DIR=/tmp/triton_cache

# python train.py \
deepspeed --num_gpus=1 --master_port $PORT_ID train.py \
    --model_name_or_path anferico/bert-for-patents \
    --train_dir data/preprocessed_data/ \
    --output_dir results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 512 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_seq_length 512 \
    --logging_steps=10 \
    --pooler_type cls \
    --mlp_only_train \
    --temperature 0.05 \
    --do_eval_flag \
    --fp16 \
    --deepspeed "./ds_config.json" \
    --do_train \
    --data_augmentation dropout section_pair \
    --additional_views claim \


