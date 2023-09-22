# meta-l2d

Environment setup:
```
conda create --name l2d python=3.9
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install numpy scipy matplotlib jupyterlab jupyter_console jupyter_client scikit-learn
pip install attrdict
```