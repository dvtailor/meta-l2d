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
from lib.losses import cross_entropy, ova
from lib.experts import SyntheticExpertOverlap
from lib.wideresnet import ClassifierRejector, ClassifierRejectorWithContextEmbedder, WideResNetBase
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
            expert,
            loss_fn,
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
    expert: expert model
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
                exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze()), device=device)
                expert_cntx.mc = exp_preds.unsqueeze(0)
                
                outputs = model(images, expert_cntx).squeeze(0)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs.data, -1)
            
            # sample expert predictions for evaluation data and evaluate costs
            exp_prediction = expert(images, labels)
            m = [0]*batch_size
            for j in range(0, batch_size):
                if exp_prediction[j] == labels[j].item():
                    m[j] = 1
                else:
                    m[j] = 0
            m = torch.tensor(m)
            m = m.to(device)

            loss = loss_fn(outputs, m, labels, n_classes) # single-expert L2D loss
            losses.append(loss.item())
            expert.resample() # sample new expert

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
                optimizer_lst,
                scheduler_lst,
                epoch,
                experts_train,
                loss_fn,
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
        n_experts = len(experts_train)

        if config["l2d"] == 'pop':
            expert_cntx = cntx_sampler.sample(n_experts=n_experts)

            # sample expert predictions for context
            exp_preds_cntx = []
            for idx_exp, expert in enumerate(experts_train):
                preds = torch.tensor(expert(expert_cntx.xc[idx_exp], expert_cntx.yc[idx_exp]), device=device)
                exp_preds_cntx.append(preds.unsqueeze(0))
            expert_cntx.mc = torch.vstack(exp_preds_cntx)

            outputs = model(input,expert_cntx) # [E,B,K+1]
        else:
            outputs = model(input) # [B,K+1]
            outputs = outputs.unsqueeze(0).repeat(n_experts,1,1) # [E,B,K+1]
        
        loss = 0
        for idx_exp, expert in enumerate(experts_train):
            m = torch.tensor(expert(input, target), device=device)
            costs = (m==target).int()
            loss += loss_fn(outputs[idx_exp], costs, target, n_classes) # loss per expert
        loss /= len(experts_train)
        epoch_train_loss.append(loss.item())

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data[0,:,:10], target, topk=(1,))[0] # just measures clf accuracy
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        for optimizer in optimizer_lst:
            optimizer.zero_grad()
        loss.backward()
        for optimizer, scheduler in zip(optimizer_lst,scheduler_lst):
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
          loss_fn,
          experts_train,
          expert_eval,
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

    if config["warmstart"]:
        epochs = config["warmstart_epochs"]
        lr_wrn = config["warmstart_lr"]
        lr_clf_rej = config["warmstart_lr"]
    else:
        epochs = config["epochs"]
        lr_wrn = config["lr_wrn"]
        lr_clf_rej = config["lr_other"]
    optimizer_base = torch.optim.SGD(model.params.base.parameters(), 
                        lr=lr_wrn,
                        momentum=0.9, 
                        nesterov=True,
                        weight_decay=config["weight_decay"])
    scheduler_base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, len(train_loader) * epochs)

    parameter_group = [{'params': model.params.clf.parameters()}]
    if config["l2d"] == "pop":
        parameter_group += [{'params': model.params.rej.parameters()}]
    optimizer_new = torch.optim.Adam(parameter_group, lr=lr_clf_rej)
    scheduler_new = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_new, len(train_loader) * epochs)

    optimizer_lst = [optimizer_base, optimizer_new]
    scheduler_lst = [scheduler_base, scheduler_new]

    # best_validation_loss = np.inf
    # patience = 0
    iters = 0

    for epoch in range(0, epochs):
        iters, train_loss = train_epoch(iters, 
                                        train_loader, 
                                        model, 
                                        optimizer_lst, 
                                        scheduler_lst, 
                                        epoch,
                                        experts_train,
                                        loss_fn,
                                        cntx_sampler,
                                        n_classes,
                                        config,
                                        logger)
        metrics = evaluate(model,
                           expert_eval,
                           loss_fn,
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


def eval(model, test_data, loss_fn, expert_eval, cntx_sampler, config):
    model.load_state_dict(torch.load(os.path.join(config["ckp_dir"], config["experiment_name"] + ".pt"), map_location=device))
    model = model.to(device)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["test_batch_size"], shuffle=False, **kwargs)
    logger = get_logger(os.path.join(config["ckp_dir"], "eval.log"))
    evaluate(model, expert_eval, loss_fn, cntx_sampler, config["n_classes"], test_loader, config, logger)


def main(config):
    set_seed(config["seed"])
    # NB: consider extending export dir with loss_type, n_context_pts if this comparison becomes prominent
    config["ckp_dir"] = f"./runs/gradual_overlap/l2d_{config['l2d']}/p{str(config['p_out'])}_seed{str(config['seed'])}"
    os.makedirs(config["ckp_dir"], exist_ok=True)
    train_data, val_data, test_data = load_cifar10(data_aug=False, seed=config["seed"])
    config["n_classes"] = 10

    with_softmax = False
    if config["loss_type"] == 'softmax':
        loss_fn = cross_entropy
        with_softmax = True
    else: # ova
        loss_fn = ova

    with_attn=False
    if len(config["l2d"].split("_"))==2:
        with_attn=True
        config["l2d"] = "pop"

    wrnbase = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0)
    if config["warmstart"]:
        warmstart_path = f"./pretrained/seed{str(config['seed'])}/default.pt"
        if not os.path.isfile(warmstart_path):
            raise FileNotFoundError('warmstart model checkpoint not found')
        wrnbase.load_state_dict(torch.load(warmstart_path, map_location=device))
        wrnbase = wrnbase.to(device)
    
    if config["l2d"] == "pop":
        model = ClassifierRejectorWithContextEmbedder(wrnbase, num_classes=int(config["n_classes"]), n_features=wrnbase.nChannels, \
                                                      with_attn=with_attn, with_softmax=with_softmax)
    else:
        model = ClassifierRejector(wrnbase, num_classes=int(config["n_classes"]), n_features=wrnbase.nChannels, with_softmax=with_softmax)

    experts_train = []
    for i in range(config["n_classes"]):
        expert = SyntheticExpertOverlap(class_oracle=i, n_classes=config["n_classes"], p_in=1.0, p_out=config['p_out'])
        experts_train.append(expert)
    # oracle class sampled every time
    expert_eval = SyntheticExpertOverlap(n_classes=config["n_classes"], p_in=1.0, p_out=config['p_out'])
    
    # Context set (x,y) sampler (always from train set, even during evaluation)
    images_train = train_data.dataset.data[train_data.indices]
    labels_train = np.array(train_data.dataset.targets)[train_data.indices]
    transform_train = train_data.dataset.transform # Assuming without data augmentation
    kwargs = {'num_workers': 0, 'pin_memory': True}
    cntx_sampler = ContextSampler(images_train, labels_train, transform_train, cntx_pts_per_class=config["n_cntx_per_class"], \
                                    n_classes=config["n_classes"], device=device, **kwargs)
    
    if config["mode"] == 'train':
        train(model, train_data, val_data, loss_fn, experts_train, expert_eval, cntx_sampler, config)
        cntx_sampler.reset()
        eval(model, test_data, loss_fn, expert_eval, cntx_sampler, config)
    else: # evaluation on test data
        eval(model, test_data, loss_fn, expert_eval, cntx_sampler, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1071)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    # parser.add_argument("--patience", type=int, default=50, 
    # 						help="number of patience steps for early stopping the training.")
    parser.add_argument("--lr_wrn", type=float, default=0.1, help="learning rate for wrn.")
    parser.add_argument("--lr_other", type=float, default=1e-2, help="learning rate for non-wrn model components.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--experiment_name", type=str, default="default",
                            help="specify the experiment name. Checkpoints will be saved with this name.")
    
    ## NEW experiment setup
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument("--p_out", type=float, default=0.1) # [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    parser.add_argument("--n_cntx_per_class", type=int, default=5)
    parser.add_argument('--l2d', choices=['single', 'pop', 'pop_attn'], default='pop_attn')
    parser.add_argument('--loss_type', choices=['softmax', 'ova'], default='ova')

    ## NEW train args
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument('--warmstart', action='store_true')
    parser.set_defaults(warmstart=False)
    parser.add_argument("--warmstart_epochs", type=int, default=50)
    parser.add_argument("--warmstart_lr", type=float, default=1e-4)
    
    config = parser.parse_args().__dict__
    main(config)
