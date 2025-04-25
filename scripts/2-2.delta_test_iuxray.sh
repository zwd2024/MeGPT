#!/bin/bash

dataset="iu_xray"
annotation="data/iu_xray/test.json"
base_dir="./data/iu_xray/images2"
delta_file="/home/zhengweidong/projects/R2GenGPT/save/iu_xray/v1_delta/checkpoints/checkpoint_epoch0_step0_bleu0.000000_cider0.014359.pth"

version="v1_delta"
savepath="./save/$dataset/$version"
export CUDA_VISIBLE_DEVICES=3
python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 16 \
    --freeze_vm True \
    --vis_use_lora True \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt