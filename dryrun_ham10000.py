import subprocess

seeds = [
    1071,
    6637,
    3918,
    4420,
    5251,
]
l2d_type_lst = ['pop', 'single']
p_out_lst = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95]

for seed in seeds:
    subprocess.check_output(f"python train_classifier_ham10000.py --epochs 3 --seed {seed}", shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
    for l2d_type in l2d_type_lst:
        for p_out in p_out_lst:
            #TODO: generate docker calls
            print(f"Seed: {seed}, Alg: {l2d_type}, p_out: {p_out}")
            subprocess.check_output(f"python main_ham10000.py --warmstart_epochs 3 --seed {seed} --p_out {p_out} --l2d {l2d_type}", shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
