#!/bin/bash

l2d=$1
p_out=$2
mode=$3
seed=$4

train_batch_size=128
lr_wrn=1e-1
lr_other=1e-2
dataset=cifar10
val_batch_size=8
test_batch_size=1
warmstart=true
warmstart_epochs=100
n_cntx_pts=50
depth_embed=6
depth_reject=4


flags=''
if [ "${warmstart}" = true ]; then
    flags="${flags} --warmstart"
fi

command="python main.py --l2d=${l2d} --p_out=${p_out} --mode=${mode} --seed=${seed}\
                        --train_batch_size=${train_batch_size} --lr_wrn=${lr_wrn} --lr_other=${lr_other}\
                        --dataset=${dataset} --val_batch_size=${val_batch_size} --test_batch_size=${test_batch_size}\
                        --warmstart_epochs=${warmstart_epochs} --n_cntx_pts=${n_cntx_pts} --depth_embed=${depth_embed}\
                        --depth_reject=${depth_reject} ${flags}"
echo ${command}

eval $command
