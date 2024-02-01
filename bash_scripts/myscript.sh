#!/bin/bash

readarray -t pout_arr < <(grep -vE "^#" p_out.txt) # ignore commented lines
readarray -t seed_arr < <(grep -vE "^#" seeds.txt)
readarray -t l2d_arr < <(grep -vE "^#" l2d_types.txt)

runs_arr=("runs_06" "runs_05")

source $HOME/miniconda3/bin/activate l2d
cd $HOME/workspace/L2D/meta-l2d/

for runs in ${runs_arr[@]}; do
    for l2d in ${l2d_arr[@]}; do
        for pout in ${pout_arr[@]}; do
            for seed in ${seed_arr[@]}; do
                command="python main.py --mode=eval --seed=${seed} --p_out=${pout} --l2d=${l2d} --runs=${runs}"
                echo ${command}
                eval $command
            done
        done
    done
done
