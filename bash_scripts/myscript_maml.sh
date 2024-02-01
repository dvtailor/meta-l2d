#!/bin/bash

readarray -t seed_arr < <(grep -vE "^#" seeds.txt)
# NB: careful when iterating over array of strings that contain spaces, should loop over indices
arr_settings=(
	"--lr_maml=1e-1 --n_steps_maml=1"
	"--lr_maml=1e-1 --n_steps_maml=2"
	"--lr_maml=1e-1 --n_steps_maml=5"
	"--lr_maml=1e-2 --n_steps_maml=1"
	"--lr_maml=1e-2 --n_steps_maml=2"
)
# "--lr_maml=1e-2 --n_steps_maml=5"

source $HOME/miniconda3/bin/activate l2d
cd $HOME/workspace/L2D/meta-l2d/

for ((i = 0; i < ${#arr_settings[@]}; i++)); do
    for seed in ${seed_arr[@]}; do
        command="python main_cifar.py --mode=eval --seed=${seed} --p_out=0.1 --l2d=single_maml ${arr_settings[$i]}"
        echo ${command}
        eval $command
    done
done
