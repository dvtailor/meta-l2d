# Learning to Defer to a Population: A Meta-Learning Approach

This is a PyTorch implementation of the following paper:

<table>
    <tr>
        <td>
            <strong>Learning to Defer to a Population: A Meta-Learning Approach</strong><br>
            Dharmesh Tailor, Aditya Patra, Rajeev Verma, Putra Manggala, Eric Nalisnick<br>
            <strong>27th International Conference on Artificial Intelligence and Statistics (AISTATS 2014)</strong><br>
            <a href="https://github.com/dvtailor/meta-l2d"><img alt="Paper" src="https://img.shields.io/badge/-Paper-gray"></a>
            <a href="https://arxiv.org/abs/2403.02683"><img alt="arxiv" src="https://img.shields.io/badge/-arxiv-gray" ></a>
        </td>
    </tr>
</table>

## Environment setup
To create a conda environment `l2d` with all necessary dependencies run: `conda env create -f environment.yml` or use the following explicit instructions:

```
conda create --name l2d python=3.9
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install numpy scipy matplotlib jupyterlab jupyter_console jupyter_client scikit-learn
pip install attrdict
```

## Usage

To reproduce figure 3 on *varying population diversity on image classification tasks*:

* `DATASET` {gtsrb/cifar10/ham10000}
    * *gtsrb* is traffic sign detection; *cifar10* is image recognition, and *ham10000* is skin lesion diagnosis
    * For *ham10000* follow the instructions in `/data/HAM10000/README.md` to setup dataset
* `L2D` {single/pop}
    * *single* is "single-L2D" (which also runs "L2D-Pop (finetune)", and *pop* is "L2D-Pop (NP)"
* `EXPERT_OVERLAP_PROB` {0.1/0.2/0.4/0.6/0.8/0.95}
    * This controls the expert overlap probability varying from specialized experts (p=0.1) to near-identical experts (p=0.95)
* `SEED`

In the case of *cifar10* and *ham10000*, the networks are warmstarted and so we first need to train a stand-alone classifier: 
`python train_classifier.py --seed=[SEED] --dataset=[DATASET]`

Then run `bash train_[DATASET].sh [L2D] [EXPERT_OVERLAP_PROB] train [SEED]`

To reproduce figure 4 on CIFAR-20 which also has an additional method using conditional neural process with attention mechanism:

* `L2D` {single/pop/pop_attn}
    * *pop_attn* is "L2D-Pop (NP+attention)"
* `EXPERT_OVERLAP_PROB` {0.1/0.2/0.4/0.6/0.8/0.95/1.0}
    * This experiment evaluates on an additional setting p=1.0

Again we need to pretrain a stand-alone classifier: `python train_classifier.py --seed=[SEED] --dataset=cifar20_100`

Then run: `bash train_cifar20_100.sh [L2D] [EXPERT_OVERLAP_PROB] train [SEED]`


## Acknowledgements

This codebase is largely an extension of the codebases of [OvA-L2D [Verma & Nalisnick]](https://github.com/rajevv/OvA-L2D) and [learn-to-defer [Mozannar & Sontag]](https://github.com/clinicalml/learn-to-defer). We also acknowledge code related to attention mechanism from [TNP-pytorch [Nguyen & Grover]](https://github.com/tung-nd/TNP-pytorch/) and [bnp [Lee et. al.]](https://github.com/juho-lee/bnp).

## Troubleshooting

Please open an issue in this repository or contact [Dharmesh](mailto:d.v.tailor@uva.nl).

## Citation

Please consider citing our conference paper
```bibtex
@inproceedings{tailor2024learning,
  title           = {{Learning to Defer to a Population: A Meta-Learning Approach}},
  booktitle       = {Proceedings of the 27th International Conference on Artificial Intelligence and Statistics},
  author          = {Tailor, Dharmesh and Patra, Aditya and Verma, Rajeev and Manggala, Putra and Nalisnick, Eric},
  year            = {2024}
}
```
