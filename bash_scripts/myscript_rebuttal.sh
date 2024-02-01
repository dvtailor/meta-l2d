#!/bin/bash

readarray -t seed_arr < <(grep -vE "^#" seeds.txt)

source $HOME/miniconda3/bin/activate l2d
cd $HOME/workspace/L2D/meta-l2d/

for seed in ${seed_arr[@]}; do
    command="python main_cifar.py --mode=eval --seed=${seed} --p_out=0.1 --l2d=pop"
    echo ${command}
    eval $command
done
