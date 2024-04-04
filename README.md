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

Environment setup:
```
conda create --name l2d python=3.9
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install numpy scipy matplotlib jupyterlab jupyter_console jupyter_client scikit-learn
pip install attrdict
```