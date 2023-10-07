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


# def extract_expert_cntxt_pts(images, labels, transform, expert_fn_lst, n_experts=10, n_classes=10, n_cntx_per_class=5, device='cpu'):
# 	'''
# 	Returns
# 		expert_cntx : AttrDict with
# 			xc : Tensor [E,Nc,3,32,32]
# 			yc : Tensor [E,Nc]
# 			mc : Tensor [E,Nc]
# 		train_data_new  : remaining data
# 	'''
# 	# here we make a choice to sample a different {x,y} for each expert
# 	indices_by_class = np.vstack([np.random.choice(np.where(labels==c)[0], size=n_cntx_per_class*n_experts, replace=False) \
# 								   for c in range(n_classes)])
# 	cntxt_xc = []
# 	cntxt_yc = []
# 	cntxt_mc = []
# 	jj=0
# 	for idx_exp in range(n_experts):
# 		expert_fn = expert_fn_lst[idx_exp]
# 		indices = indices_by_class[:,jj:jj+n_cntx_per_class].flatten()
# 		np.random.shuffle(indices)
# 		cntxt_xc.append(torch.tensor(np.vstack([transform(img)[None,:] for img in images[indices]]), device=device).unsqueeze(0))
# 		cntxt_yc.append(torch.tensor(labels[indices], device=device).unsqueeze(0))
# 		cntxt_mc.append(torch.tensor(expert_fn(cntxt_xc[idx_exp].squeeze(),cntxt_yc[idx_exp].squeeze()), device=device).unsqueeze(0))
# 		jj += n_cntx_per_class

# 	indices_rest = np.setdiff1d(np.arange(len(images)), indices_by_class.flatten())
# 	train_data_new = MyVisionDataset(images[indices_rest], labels[indices_rest], transform)

# 	expert_cntxt = AttrDict()
# 	expert_cntxt.xc = torch.vstack(cntxt_xc)
# 	expert_cntxt.yc = torch.vstack(cntxt_yc)
# 	expert_cntxt.mc = torch.vstack(cntxt_mc)
    
# 	return expert_cntxt, train_data_new
