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
