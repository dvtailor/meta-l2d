# Experts for CIFAR-10
from attrdict import AttrDict
import random
import numpy as np
import torch

from lib.datasets import MyVisionDataset, coarse2sparse

# TODO: consider creating parent class of expert generator

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

    # def resample(self):
    #     if not self.expert_static:
    #         self.class_oracle = random.randint(0, self.n_classes-1)

    def __call__(self, images, labels, labels_sparse=None):
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


class Cifar20SyntheticExpert():
    def __init__(self, classes_coarse=None, n_classes=20, p_in=1.0, p_out=0.1, n_oracle_subclass=4, n_oracle_superclass=5):
        '''
        n_oracle_subclass : number of sparse/sub-classes expert is oracle at for a given coarse/super-class
        '''
        self.expert_static = True
        self.classes_coarse = classes_coarse
        self.n_oracle_subclass = n_oracle_subclass
        self.n_oracle_superclass = n_oracle_superclass
        self.n_classes = n_classes
        if self.classes_coarse is None:
            self.classes_coarse = np.random.choice(np.arange(self.n_classes), size=self.n_oracle_superclass, replace=False)
            self.expert_static = False
        # Ensures we select n_oracle_subclass per coarse class
        indices_sparse = np.vstack([np.random.choice(np.arange(5), size=self.n_oracle_subclass, replace=False) for _ in range(len(self.classes_coarse))])
        self.sparse_classes_oracle = coarse2sparse(self.classes_coarse)[np.arange(len(self.classes_coarse))[:,None], indices_sparse].flatten()

        self.p_in = p_in
        self.p_out = p_out

    # def resample(self):
    #     if not self.expert_static:
    #         self.classes_coarse = np.random.choice(np.arange(self.n_classes), size=self.n_oracle_superclass, replace=False)
    #         indices_sparse = np.vstack([np.random.choice(np.arange(5), size=self.n_oracle_subclass, replace=False) for _ in range(len(self.classes_coarse))])
    #         self.sparse_classes_oracle = coarse2sparse(self.classes_coarse)[np.arange(len(self.classes_coarse))[:,None], indices_sparse].flatten()            

    def __call__(self, images, labels, labels_sparse):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i].item() in self.classes_coarse:
                if labels_sparse[i].item() in self.sparse_classes_oracle:
                    coin_flip = np.random.binomial(1, self.p_in)
                    if coin_flip == 1:
                        outs[i] = labels[i].item()
                    if coin_flip == 0:
                        outs[i] = random.randint(0, self.n_classes-1) # NB: could make this anti-oracle by excluding true class
                else:
                    coin_flip = np.random.binomial(1, self.p_out)
                    if coin_flip == 1:
                        outs[i] = labels[i].item()
                    if coin_flip == 0:
                        outs[i] = random.randint(0, self.n_classes-1)
            else: # predict randomly if not in superclasses, OR should this be anti-oracle (exclude true class) ?
                outs[i] = random.randint(0, self.n_classes-1)
        return outs
