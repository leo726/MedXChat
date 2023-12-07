#!/bin/bash

version=T2V_v11
savepath="./save/$version"

dataset='./data/chatgpt_dialogue.json'
delta_file='/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/save/VQA_v11/checkpoints/checkpoint_epoch5_step9893_val_loss0.766605.pth'

python -u train.py \
    --llm_use_lora True \
    --dataset $dataset \
    --delta_file $delta_file \
    --lora_inference False \
    --batch_size 4 \
    --val_batch_size 12 \
    --max_length 240 \
    --num_workers 8 \
    --learning_rate 0.00002 \
    --devices 4 \
    --accelerator gpu \
    --precision bf16-mixed \
    --num_nodes 1 \
    --strategy ddp \
    --max_epochs 500 \
    --accumulate_grad_batches 2 \
    --num_sanity_val_steps 0 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --savedmodel_path ${savepath} \
    2>&1 |tee -a ${savepath}/log.txt

