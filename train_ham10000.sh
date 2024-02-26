#!/bin/bash

l2d=$1
p_out=$2
mode=$3
seed=$4

train_batch_size=128
lr_wrn=1e-2
lr_other=1e-3
weight_decay=5e-4
dataset=ham10000
val_batch_size=8
test_batch_size=1
warmstart=true
n_cntx_pts=140
epochs=100
depth_embed=6
depth_reject=4
lr_finetune="1e-2 1e-3"
scoring_rule=sys_acc
decouple=false

flags=''
if [ "${warmstart}" = true ]; then
    flags="${flags} --warmstart"
fi
if [ "${decouple}" = true ]; then
    flags="${flags} --decouple"
fi

command="python main.py --l2d=${l2d} --p_out=${p_out} --mode=${mode} --seed=${seed}\
                        --train_batch_size=${train_batch_size} --lr_wrn=${lr_wrn} --lr_other=${lr_other} --weight_decay=${weight_decay}\
                        --dataset=${dataset} --val_batch_size=${val_batch_size} --test_batch_size=${test_batch_size}\
                        --n_cntx_pts=${n_cntx_pts} --epochs=${epochs} --depth_embed=${depth_embed}\
                        --depth_reject=${depth_reject} --lr_finetune ${lr_finetune} --scoring_rule=${scoring_rule} ${flags}"
echo ${command}

eval $command
