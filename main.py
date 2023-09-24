from attrdict import AttrDict
import math
import random
import argparse
import os
import shutil
import time
import json
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# local imports
from lib.utils import AverageMeter, accuracy, get_logger
from lib.losses import cross_entropy
from lib.experts import synthetic_expert_overlap
from lib.wideresnet import WideResNetBase, Classifier, ClassifierRejectorWithContextEmbedder
from lib.datasets import load_cifar10, ContextSampler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)


def evaluate(model,
			expert_fn,
			cntx_sampler,
			n_classes,
			data_loader,
			config,
			logger):
	'''
	Computes metrics for deferal
	-----
	Arguments:
	net: model
	expert_fn: expert model
	n_classes: number of classes
	loader: data loader
	'''
	correct = 0
	correct_sys = 0
	exp = 0
	exp_total = 0
	total = 0
	real_total = 0
	clf_alone_correct = 0
	exp_alone_correct = 0
	losses = []
	model.eval() # Crucial for networks with batchnorm layers!
	with torch.no_grad():
		for data in data_loader:
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			batch_size = len(images)

			if config["l2d"] == 'pop':
				# sample expert predictions for context
				expert_cntx = cntx_sampler.sample(n_experts=1)
				exp_preds = torch.tensor(expert_fn(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze()), device=device)
				expert_cntx.mc = exp_preds.unsqueeze(0)
				
				outputs = model(images, expert_cntx).squeeze(0)
			else:
				outputs = model(images)

			outputs = F.softmax(outputs, dim=-1)
			_, predicted = torch.max(outputs.data, -1)
			
			# sample expert predictions for evaluation data and evaluate costs
			exp_prediction = expert_fn(images, labels)
			m = [0]*batch_size
			for j in range(0, batch_size):
				if exp_prediction[j] == labels[j].item():
					m[j] = 1
				else:
					m[j] = 0
			m = torch.tensor(m)
			m = m.to(device)

			loss = cross_entropy(outputs, m, labels, n_classes) # single-expert L2D loss
			losses.append(loss.item())

			for i in range(0, batch_size):
				r = (predicted[i].item() == n_classes)
				prediction = predicted[i]
				if predicted[i] == n_classes:
					max_idx = 0
					# get second max
					for j in range(0, n_classes):
						if outputs.data[i][j] >= outputs.data[i][max_idx]:
							max_idx = j
					prediction = max_idx
				else:
					prediction = predicted[i]
				clf_alone_correct += (prediction == labels[i]).item()
				exp_alone_correct += (exp_prediction[i] == labels[i].item())
				if r == 0:
					total += 1
					correct += (predicted[i] == labels[i]).item()
					correct_sys += (predicted[i] == labels[i]).item()
				if r == 1:
					exp += (exp_prediction[i] == labels[i].item())
					correct_sys += (exp_prediction[i] == labels[i].item())
					exp_total += 1
				real_total += 1
	cov = str(total) + str("/") + str(real_total)
	metrics = {"cov": cov, "sys_acc": 100 * correct_sys / real_total,
				"exp_acc": 100 * exp / (exp_total + 0.0002),
				"clf_acc": 100 * correct / (total + 0.0001),
				"exp_acc_alone": 100 * exp_alone_correct / real_total,
				"clf_acc_alone": 100 * clf_alone_correct / real_total,
				"val_loss": np.average(losses)}
	to_print = ""
	for k,v in metrics.items():
		if type(v)==str:
			to_print += f"{k} {v} "
		else:
			to_print += f"{k} {v:.6f} "
	logger.info(to_print)
	return metrics


def train_epoch(iters,
				train_loader,
				model,
				optimizer,
				scheduler,
				epoch,
				expert_fns_train,
				cntx_sampler,
				n_classes,
				config,
				logger):
	""" Train for one epoch """
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	model.train()
	end = time.time()

	epoch_train_loss = []

	for i, (input, target) in enumerate(train_loader):
		target = target.to(device)
		input = input.to(device)
		n_experts = len(expert_fns_train)

		if config["l2d"] == 'pop':
			expert_cntx = cntx_sampler.sample(n_experts=n_experts)

			# sample expert predictions for context
			exp_preds_cntx = []
			for idx_exp, expert_fn in enumerate(expert_fns_train):
				preds = torch.tensor(expert_fn(expert_cntx.xc[idx_exp], expert_cntx.yc[idx_exp]), device=device)
				exp_preds_cntx.append(preds.unsqueeze(0))
			expert_cntx.mc = torch.vstack(exp_preds_cntx)

			logits = model(input,expert_cntx) # [E,B,K+1]
		else:
			logits = model(input) # [B,K+1]
			logits = logits.unsqueeze(0).repeat(n_experts,1,1) # [E,B,K+1]

		output = F.softmax(logits, dim=-1) # [E,B,K+1]
		loss = 0
		for idx_exp, expert_fn in enumerate(expert_fns_train):
			m = torch.tensor(expert_fn(input, target), device=device)
			costs = (m==target).int()
			loss += cross_entropy(output[idx_exp], costs, target, n_classes) # loss per expert
		loss /= len(expert_fns_train)
		epoch_train_loss.append(loss.item())

		# measure accuracy and record loss
		prec1 = accuracy(logits.data[0,:,:10], target, topk=(1,))[0] # just measures clf accuracy
		losses.update(loss.data.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		iters+=1

		# if i % 10 == 0:
		if i % 50 == 0:
			logger.info('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				loss=losses, top1=top1))

	return iters, np.average(epoch_train_loss)


def train(model,
		train_dataset,
		validation_dataset,
		expert_fns_train, 
		expert_fn_eval, 
		cntx_sampler,
		config):
	logger = get_logger(os.path.join(config["ckp_dir"], "train.log"))
	logger.info(f"p_out={config['p_out']}  seed={config['seed']}")
	logger.info(config)
	logger.info('No. of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
	n_classes = config["n_classes"]
	kwargs = {'num_workers': 0, 'pin_memory': True}
	train_loader = torch.utils.data.DataLoader(train_dataset,
											   batch_size=config["train_batch_size"], shuffle=True, **kwargs) # drop_last=True
	valid_loader = torch.utils.data.DataLoader(validation_dataset,
											   batch_size=config["val_batch_size"], shuffle=False, **kwargs) # shuffle=True, drop_last=True
	model = model.to(device)
	cudnn.benchmark = True
	optimizer = torch.optim.SGD(model.parameters(), config["lr"],
								momentum=0.9, nesterov=True,
								weight_decay=config["weight_decay"])
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * config["epochs"])
	# best_validation_loss = np.inf
	# patience = 0
	iters = 0
	lrate = config["lr"]

	for epoch in range(0, config["epochs"]):
		iters, train_loss = train_epoch(iters, 
										train_loader, 
										model, 
										optimizer, 
										scheduler, 
										epoch,
										expert_fns_train,
										cntx_sampler,
										n_classes,
										config,
										logger)
		metrics = evaluate(model,
							expert_fn_eval,
							cntx_sampler,
							n_classes,
							valid_loader,
							config,
							logger)

		validation_loss = metrics["val_loss"]

		# if validation_loss < best_validation_loss:
			# best_validation_loss = validation_loss
		# logger.info("Saving the model with classifier accuracy {}".format(metrics['clf_acc']))
		torch.save(model.state_dict(), os.path.join(config["ckp_dir"], config["experiment_name"] + ".pt"))
		# Additionally save the whole config dict
		with open(os.path.join(config["ckp_dir"], config["experiment_name"] + ".json"), "w") as f:
			json.dump(config, f)
		# 	patience = 0
		# else:
		# 	patience += 1

		# if patience >= config["patience"]:
		# 	print("Early Exiting Training.", flush=True)
		# 	break	


def eval(model, test_data, expert_fn_eval, cntx_sampler, config):
	model.load_state_dict(torch.load(os.path.join(config["ckp_dir"], config["experiment_name"] + ".pt"), map_location=device))
	model = model.to(device)
	kwargs = {'num_workers': 0, 'pin_memory': True}
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["test_batch_size"], shuffle=False, **kwargs)
	logger = get_logger(os.path.join(config["ckp_dir"], "eval.log"))
	evaluate(model, expert_fn_eval, cntx_sampler, config["n_classes"], test_loader, config, logger)


def main(config):
	set_seed(config["seed"])
	config["ckp_dir"] = f"./runs/gradual_overlap/l2d_{config['l2d']}/p{str(config['p_out'])}_seed{str(config['seed'])}"
	os.makedirs(config["ckp_dir"], exist_ok=True)
	train_data, val_data, test_data = load_cifar10(data_aug=False, seed=config["seed"])
	config["n_classes"] = 10

	wrnbase = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0)
	if config["l2d"] == "pop":
		model = ClassifierRejectorWithContextEmbedder(wrnbase, num_classes=int(config["n_classes"])+1, n_features=wrnbase.nChannels)
	else:
		model = Classifier(wrnbase, num_classes=int(config["n_classes"])+1, n_features=wrnbase.nChannels)

	expert_fns_train = []
	for i in range(config["n_classes"]):
		expert_fn = functools.partial(synthetic_expert_overlap, class_oracle=i, \
						n_classes=config["n_classes"], p_in=1.0, p_out=config['p_out'])
		expert_fns_train.append(expert_fn)
	# oracle class sampled every time
	expert_fn_eval = functools.partial(synthetic_expert_overlap, \
						n_classes=config["n_classes"], p_in=1.0, p_out=config['p_out'])
	
	# Context set (x,y) sampler (always from train set, even during evaluation)
	images_train = train_data.dataset.data[train_data.indices]
	labels_train = np.array(train_data.dataset.targets)[train_data.indices]
	transform_train = train_data.dataset.transform # Assuming without data augmentation
	kwargs = {'num_workers': 0, 'pin_memory': True}
	cntx_sampler = ContextSampler(images_train, labels_train, transform_train, cntx_pts_per_class=config["n_cntx_per_class"], \
									n_classes=config["n_classes"], device=device, **kwargs)
	
	if config["mode"] == 'train':
		train(model, train_data, val_data, expert_fns_train, expert_fn_eval, cntx_sampler, config)
		cntx_sampler.reset()
		eval(model, test_data, expert_fn_eval, cntx_sampler, config)
	else: # evaluation on test data
		eval(model, test_data, expert_fn_eval, cntx_sampler, config)

	# ##### DEBUGGING
	# cntxt_xc = []
	# cntxt_yc = []
	# cntxt_mc = []
	# for expert_cntx in expert_cntx_lst:
	# 	cntxt_xc.append(expert_cntx.xc[None,:])
	# 	cntxt_yc.append(expert_cntx.yc[None,:])
	# 	cntxt_mc.append(expert_cntx.mc[None,:])
	# cntxt = AttrDict()
	# cntxt.xc = torch.vstack(cntxt_xc) # [E,Nc,3,32,32]
	# cntxt.yc = torch.vstack(cntxt_yc) # [E,Nc]
	# cntxt.mc = torch.vstack(cntxt_mc) # [E,Nc]

	# kwargs = {'num_workers': 0, 'pin_memory': True}
	# train_loader = torch.utils.data.DataLoader(train_data_new, batch_size=config["batch_size"], shuffle=True, **kwargs) # drop_last=True
	# model = model.to(device)
	# model.train()
	# batches = [(X.to(device), y.to(device)) for X, y in train_loader]
	# input, target = batches[0]
	# model(input, cntxt)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--train_batch_size", type=int, default=128)
	parser.add_argument("--epochs", type=int, default=200)
	# parser.add_argument("--patience", type=int, default=50, 
	# 						help="number of patience steps for early stopping the training.")
	parser.add_argument("--lr", type=float, default=0.1,
							help="learning rate.")
	parser.add_argument("--weight_decay", type=float, default=5e-4)
	parser.add_argument("--experiment_name", type=str, default="default",
							help="specify the experiment name. Checkpoints will be saved with this name.")
	## NEW args
	parser.add_argument('--mode', choices=['train', 'eval'], default='train')
	parser.add_argument("--p_out", type=float, default=0.2) # [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
	parser.add_argument("--n_cntx_per_class", type=int, default=5)
	parser.add_argument('--l2d', choices=['single', 'pop'], default='pop')
	parser.add_argument("--val_batch_size", type=int, default=8)
	parser.add_argument("--test_batch_size", type=int, default=1)
	
	config = parser.parse_args().__dict__
	main(config)
