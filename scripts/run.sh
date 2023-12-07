#!/bin/bash

version=VQA_v11_RG
savepath="./save/$version"

dataset='/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/mimic_cxr_tiny.json'
# dataset='/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/data/mimic_cxr/my_mimic_anno.json'
# delta_file='/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/save/VQA_v11/checkpoints/checkpoint_epoch5_step9893_val_loss0.766605.pth'
delta_file="/apdcephfs/share_733425/vinnylywang/zhanyuwang/Code/xray_chat/save/T2V_v11/checkpoints/checkpoint_epoch20_step4914_val_loss0.717129.pth"

python -u train.py \
    --llm_use_lora True \
    --dataset $dataset \
    --delta_file $delta_file \
    --lora_inference False \
    --batch_size 4 \
    --val_batch_size 16 \
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
    --num_sanity_val_steps 2 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.3 \
    --savedmodel_path ${savepath} \
    2>&1 |tee -a ${savepath}/log.txt

