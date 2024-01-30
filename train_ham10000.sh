#!/bin/bash

l2d=$1
p_out=$2
mode=$3
seed=$4

train_batch_size=128
lr_wrn=1e-2
lr_other=1e-3
dataset=ham10000
val_batch_size=8
test_batch_size=1
warmstart=false
n_cntx_pts=140
epochs=200


flags=''
if [ "${warmstart}" = true ]; then
    flags="${flags} --warmstart"
fi

command="python main.py --l2d=${l2d} --p_out=${p_out} --mode=${mode} --seed=${seed}\
                        --train_batch_size=${train_batch_size} --lr_wrn=${lr_wrn} --lr_other=${lr_other}\
                        --dataset=${dataset} --val_batch_size=${val_batch_size} --test_batch_size=${test_batch_size}\
                        --n_cntx_pts=${n_cntx_pts} --epochs=${epochs} ${flags}"
echo ${command}

eval $command
