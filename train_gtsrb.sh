#!/bin/bash

l2d=$1
p_out=$2
mode=$3
seed=$4
flags=''
# if [ ! -z "$5" ]; then
#     flags="${flags} --n_steps_maml=${5}"
# fi
# if [ ! -z "$6" ]; then
#     flags="${flags} --lr_maml=${6}"
# fi
flags="${flags} --n_steps_maml=5"
flags="${flags} --lr_maml=1e-2"

train_batch_size=64
lr_wrn=1e-2
lr_other=1e-3
epochs=150
weight_decay=1e-3
dataset=gtsrb
val_batch_size=8
test_batch_size=1
warmstart=false
n_cntx_pts=50
depth_embed=5
depth_reject=3
norm_type=batchnorm
loss_type=softmax
decouple=false

if [ "${warmstart}" = true ]; then
    flags="${flags} --warmstart"
fi
if [ "${decouple}" = true ]; then
    flags="${flags} --decouple"
fi

command="python main.py --l2d=${l2d} --p_out=${p_out} --mode=${mode} --seed=${seed}\
                        --train_batch_size=${train_batch_size} --lr_wrn=${lr_wrn} --lr_other=${lr_other} --weight_decay=${weight_decay}\
                        --dataset=${dataset} --val_batch_size=${val_batch_size} --test_batch_size=${test_batch_size}\
                        --epochs=${epochs} --n_cntx_pts=${n_cntx_pts} --depth_embed=${depth_embed}\
                        --depth_reject=${depth_reject} --norm_type=${norm_type} --loss_type=${loss_type} ${flags}"
echo ${command}

eval $command
