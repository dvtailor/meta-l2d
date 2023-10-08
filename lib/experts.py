# Experts for CIFAR-10
from attrdict import AttrDict
import random
import numpy as np
import torch

from lib.datasets import MyVisionDataset


# expert correct in class_oracle with prob. p_in; correct on other classes with prob. p_out
class SyntheticExpertOverlap():
    def __init__(self, class_oracle=None, n_classes=10, p_in=1.0, p_out=0.1):
        self.expert_static = True
        self.class_oracle = class_oracle
        if self.class_oracle is None:
            self.class_oracle = random.randint(0, n_classes-1)
            self.expert_static = False
        self.n_classes = n_classes
        self.p_in = p_in
        self.p_out = p_out

    def resample(self):
        if not self.expert_static:
            self.class_oracle = random.randint(0, self.n_classes-1)

    def __call__(self, images, labels):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i].item() == self.class_oracle:
                coin_flip = np.random.binomial(1, self.p_in)
                if coin_flip == 1:
                    outs[i] = labels[i].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes-1)
            else:
                coin_flip = np.random.binomial(1, self.p_out)
                if coin_flip == 1:
                    outs[i] = labels[i].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes-1)
        return outs
