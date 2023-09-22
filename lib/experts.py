# Experts for CIFAR-10
from attrdict import AttrDict
import random
import numpy as np
import torch

from lib.datasets import MyVisionDataset


# expert correct in class_oracle with prob. p_in; correct on other classes with prob. p_out
def synthetic_expert_overlap(images, labels, class_oracle=None, n_classes=10, p_in=1.0, p_out=0.1, return_oracle=False):
	if class_oracle is None:
		class_oracle = random.randint(0, n_classes-1)
	batch_size = labels.size()[0]
	outs = [0] * batch_size
	for i in range(0, batch_size):
		if labels[i].item() == class_oracle:
			coin_flip = np.random.binomial(1, p_in)
			if coin_flip == 1:
				outs[i] = labels[i].item()
			if coin_flip == 0:
				outs[i] = random.randint(0, n_classes-1)
		else:
			coin_flip = np.random.binomial(1, p_out)
			if coin_flip == 1:
				outs[i] = labels[i].item()
			if coin_flip == 0:
				outs[i] = random.randint(0, n_classes-1)
	if return_oracle: # needed to identify expert when oracle sampled
		return outs, class_oracle
	else:
		return outs


def extract_expert_cntxt_pts(images, labels, transform, expert_fn_lst, n_experts=10, n_classes=10, n_cntx_per_class=5, device='cpu'):
	'''
	Returns
		expert_cntx_lst : list of tuples of tensors (xc,yc,mc) corresponding to each expert
		train_data_new  : remaining data
	'''
	# here we make a choice to sample a different {x,y} for each expert
	indices_by_class = np.vstack([np.random.choice(np.where(labels==c)[0], size=n_cntx_per_class*n_experts, replace=False) \
								   for c in range(n_classes)])
	expert_cntx_lst = [] # list of tuples of tensors (xc,yc,mc)
	jj=0
	for idx_exp in range(n_experts):
		expert_fn = expert_fn_lst[idx_exp]
		indices = indices_by_class[:,jj:jj+n_cntx_per_class].flatten()
		np.random.shuffle(indices)
		cntxt = AttrDict()
		cntxt.xc = torch.tensor(np.vstack([transform(img)[None,:] for img in images[indices]]), device=device)
		cntxt.yc = torch.tensor(labels[indices], device=device)
		cntxt.mc = torch.tensor(expert_fn(cntxt.xc,cntxt.yc), device=device)
		expert_cntx_lst.append(cntxt)
		jj += n_cntx_per_class

	indices_rest = np.setdiff1d(np.arange(len(images)), indices_by_class.flatten())
	train_data_new = MyVisionDataset(images[indices_rest], labels[indices_rest], transform)
	
	return expert_cntx_lst, train_data_new
