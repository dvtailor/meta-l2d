#!/bin/bash

l2d=$1
p_out=$2
mode=$3
seed=$4
flags=''
if [ ! -z "$5" ]; then
    flags="${flags} --n_steps_maml=${5}"
fi
if [ ! -z "$6" ]; then
    flags="${flags} --lr_maml=${6}"
fi

train_batch_size=128
lr_wrn=1e-2
lr_other=1e-3
weight_decay=5e-4
dataset=cifar10
val_batch_size=8
test_batch_size=1
warmstart=true
epochs=100
n_cntx_pts=50
depth_embed=6
depth_reject=4
norm_type=frn

if [ "${warmstart}" = true ]; then
    flags="${flags} --warmstart"
fi

command="python main.py --l2d=${l2d} --p_out=${p_out} --mode=${mode} --seed=${seed}\
                        --train_batch_size=${train_batch_size} --lr_wrn=${lr_wrn} --lr_other=${lr_other} --weight_decay=${weight_decay}\
                        --dataset=${dataset} --val_batch_size=${val_batch_size} --test_batch_size=${test_batch_size}\
                        --epochs=${epochs} --n_cntx_pts=${n_cntx_pts} --depth_embed=${depth_embed}\
                        --depth_reject=${depth_reject} --norm_type=${norm_type} ${flags}"
echo ${command}

eval $command
