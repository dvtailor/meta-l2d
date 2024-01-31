from attrdict import AttrDict
import math
import random
import argparse
import os
import shutil
import time
import json
import functools
import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# local imports
from lib.utils import AverageMeter, accuracy, get_logger
from lib.losses import cross_entropy, ova
from lib.experts import SyntheticExpertOverlap, Cifar20SyntheticExpert
from lib.wideresnet import ClassifierRejector, ClassifierRejectorWithContextEmbedder, WideResNetBase
from lib.resnet224 import ResNet34
from lib.datasets import load_cifar, load_ham10000, ContextSampler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate(model,
            experts_test,
            loss_fn,
            cntx_sampler,
            n_classes,
            data_loader,
            config,
            logger=None,
            budget=1.0,
            n_finetune_steps=0,
            lr_finetune=1e-1,
            p_cntx_inclusion=1.0):
    '''
    data loader : assumed to be instantiated with shuffle=False
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
    if config["l2d"] == 'single_maml':
        model.train()
    is_finetune = ((config["l2d"] == 'single') or (config["l2d"] == 'single_maml')) and (n_finetune_steps > 0)
    if is_finetune:
        model_state_dict = model.state_dict()
        model_backup = copy.deepcopy(model)
    # with torch.no_grad():
    confidence_diff = []
    is_rejection = []
    clf_predictions = []
    exp_predictions = []
    for data in data_loader:
        if len(data) == 2:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            labels_sparse = None
        else:
            images, labels, labels_sparse = data
            images, labels, labels_sparse = images.to(device), labels.to(device), labels_sparse.to(device)
        
        choice = random.randint(0, len(experts_test)-1)
        expert = experts_test[choice]

        # sample expert predictions for context
        expert_cntx = cntx_sampler.sample(n_experts=1)
        cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse.squeeze(0)
        exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_sparse), device=device)
        expert_cntx.mc = exp_preds.unsqueeze(0)

        if is_finetune:
            model.train()
            images_cntx = expert_cntx.xc.squeeze(0)
            targets_cntx = expert_cntx.yc.squeeze(0)
            costs = (exp_preds==targets_cntx).int()
            # NB: could freeze base network like finetuning in train_epoch()
            for _ in range(n_finetune_steps):
                outputs_cntx = model(images_cntx)
                loss = loss_fn(outputs_cntx, costs, targets_cntx, n_classes)
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for param in model.params.clf.parameters():
                        new_param = param - lr_finetune * param.grad
                        param.copy_(new_param)
            if config["l2d"] == 'single': # finetuning on single-expert, use running batch statistics for eval
                model.eval()
        
        with torch.no_grad():
            # removes expert context based on coin flip
            coin_flip = np.random.binomial(1, p_cntx_inclusion)
            if coin_flip == 0:
                expert_cntx = None

            if config["l2d"] == 'pop':
                outputs = model(images, expert_cntx).squeeze(0)
            else:
                outputs = model(images)

            if config["loss_type"] == "ova":
                probs = F.sigmoid(outputs)
            else:
                probs = outputs
            
            clf_probs, clf_preds = probs[:,:n_classes].max(dim=-1)
            exp_probs = probs[:,n_classes]
            confidence_diff.append(clf_probs - exp_probs)
            clf_predictions.append(clf_preds)
            # defer if rejector logit strictly larger than (max of) classifier logits
            # since max() returns index of first maximal value (different from paper (geq))
            _, predicted = outputs.max(dim=-1)
            is_rejection.append((predicted==n_classes).int())

            # sample expert predictions for evaluation data and evaluate costs
            exp_pred = torch.tensor(expert(images, labels, labels_sparse)).to(device)
            m = (exp_pred==labels).int()
            exp_predictions.append(exp_pred)

            loss = loss_fn(outputs, m, labels, n_classes) # single-expert L2D loss
            losses.append(loss.item())

            if is_finetune: # restore model on single-expert
                model = model_backup
                model.load_state_dict(copy.deepcopy(model_state_dict))
                if config["l2d"] == 'single':
                    model.eval()
                else:
                    model.train()

    confidence_diff = torch.cat(confidence_diff)
    indices_order = confidence_diff.argsort()

    is_rejection = torch.cat(is_rejection)[indices_order]
    clf_predictions = torch.cat(clf_predictions)[indices_order]
    exp_predictions = torch.cat(exp_predictions)[indices_order]

    kwargs = {'num_workers': 0, 'pin_memory': True}
    data_loader_new = torch.utils.data.DataLoader(torch.utils.data.Subset(data_loader.dataset, indices=indices_order),
                                                    batch_size=data_loader.batch_size, shuffle=False, **kwargs)
    
    max_defer = math.floor(budget * len(data_loader.dataset))

    for data in data_loader_new:
        if len(data) == 2:
            images, labels = data
        else:
            images, labels, _ = data
        images, labels = images.to(device), labels.to(device)
        batch_size = len(images)

        for i in range(0, batch_size):
            defer_running = is_rejection[:real_total].sum().item()
            if defer_running >= max_defer:
                r = 0
            else:
                r = is_rejection[real_total].item()
            prediction = clf_predictions[real_total].item()
            exp_prediction = exp_predictions[real_total].item()

            clf_alone_correct += (prediction == labels[i]).item()
            exp_alone_correct += (exp_prediction == labels[i].item())
            if r == 0:
                total += 1
                correct += (prediction == labels[i]).item()
                correct_sys += (prediction == labels[i]).item()
            if r == 1:
                exp += (exp_prediction == labels[i].item())
                correct_sys += (exp_prediction == labels[i].item())
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
    if logger is not None:
        logger.info(to_print)
    else:
        print(to_print)
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
                logger,
                n_steps_maml=5,
                lr_maml=1e-1):
    """ Train for one epoch """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    epoch_train_loss = []

    for i, data in enumerate(train_loader):
        if len(data) == 2:
            input, target = data
            input, target = input.to(device), target.to(device)
            target_sparse = None
        else:
            input, target, target_sparse = data # ignore additional labels
            input, target, target_sparse = input.to(device), target.to(device), target_sparse.to(device)
        n_experts = len(experts_train)

        # For MAML: need to do backprop once at start to initialize grads
        if (i==0) and (config["l2d"] == 'single_maml'):
            outputs = model(input)
            loss = loss_fn(outputs, torch.zeros_like(target, device=target.device), target, n_classes) # loss per expert
            loss.backward()
            model.zero_grad()
        
        if (config["l2d"] == 'pop') or (config["l2d"] == 'single_maml'):
            expert_cntx = cntx_sampler.sample(n_experts=n_experts)

            # sample expert predictions for context
            exp_preds_cntx = []
            for idx_exp, expert in enumerate(experts_train):
                cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse[idx_exp]
                preds = torch.tensor(expert(expert_cntx.xc[idx_exp], expert_cntx.yc[idx_exp], cntx_yc_sparse), device=device)
                exp_preds_cntx.append(preds.unsqueeze(0))
            expert_cntx.mc = torch.vstack(exp_preds_cntx)

        if config["l2d"] == 'pop':
            outputs = model(input,expert_cntx) # [E,B,K+1]
        elif config["l2d"] == 'single':
            outputs = model(input) # [B,K+1]
            outputs = outputs.unsqueeze(0).repeat(n_experts,1,1) # [E,B,K+1]

        if config["l2d"] == 'single_maml':
            for optimizer in optimizer_lst:
                optimizer.zero_grad()

            loss_cum = 0
            for idx_exp, expert in enumerate(experts_train): 
                local_model = copy.deepcopy(model)
                local_model.train()

                # freeze base network in finetuning
                for param in local_model.params.base.parameters():
                    param.requires_grad = False

                local_optim = torch.optim.SGD(local_model.parameters(), lr=lr_maml)
                local_optim.zero_grad()

                images_cntx = expert_cntx.xc[idx_exp]
                targets_cntx = expert_cntx.yc[idx_exp]
                exp_preds_cntx = expert_cntx.mc[idx_exp]
                costs = (exp_preds_cntx==targets_cntx).int()
                for _ in range(n_steps_maml):
                    outputs = local_model(images_cntx)
                    loss = loss_fn(outputs, costs, targets_cntx, n_classes)
                    loss.backward()
                    local_optim.step()
                    local_optim.zero_grad()

                # unfreeze base network for global update
                for param in local_model.params.base.parameters():
                    param.requires_grad = True

                # NOTE: repeated above so should be combined.
                m = torch.tensor(expert(input, target, target_sparse), device=device)
                costs = (m==target).int()

                outputs = local_model(input)
                loss = loss_fn(outputs, costs, target, n_classes) / len(experts_train)
                loss.backward()

                for p_global, p_local in zip(model.parameters(), local_model.parameters()):
                    p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model

                loss_cum += loss
            
            epoch_train_loss.append(loss_cum.item())

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data[:,:n_classes], target, topk=(1,))[0] # just measures clf accuracy
            losses.update(loss_cum.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            for optimizer, scheduler in zip(optimizer_lst,scheduler_lst):
                optimizer.step()
                scheduler.step()

        else: # l2d=single,pop
            loss = 0
            for idx_exp, expert in enumerate(experts_train):
                m = torch.tensor(expert(input, target, target_sparse), device=device)
                costs = (m==target).int()
                loss += loss_fn(outputs[idx_exp], costs, target, n_classes) # loss per expert
            loss /= len(experts_train)
            epoch_train_loss.append(loss.item())

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data[0,:,:n_classes], target, topk=(1,))[0] # just measures clf accuracy
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
          experts_test,
          cntx_sampler_train, 
          cntx_sampler_eval,
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
        lr_wrn = config["lr_wrn"]/10
        lr_clf_rej = config["lr_other"]/10
    else:
        epochs = config["epochs"]
        lr_wrn = config["lr_wrn"]
        lr_clf_rej = config["lr_other"]
    # assuming epochs >= 50
    if epochs > 100:
        milestone_epoch = epochs - 50    
    else:
        milestone_epoch = 50
    optimizer_base = torch.optim.SGD(model.params.base.parameters(), 
                        lr=lr_wrn,
                        momentum=0.9, 
                        nesterov=True,
                        weight_decay=config["weight_decay"])
    scheduler_base_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, len(train_loader)*milestone_epoch, eta_min=lr_wrn/1000)
    scheduler_base_constant = torch.optim.lr_scheduler.ConstantLR(optimizer_base, factor=1., total_iters=0)
    scheduler_base_constant.base_lrs = [lr_wrn/1000 for _ in optimizer_base.param_groups]
    scheduler_base = torch.optim.lr_scheduler.SequentialLR(optimizer_base, [scheduler_base_cosine,scheduler_base_constant], 
                                                           milestones=[len(train_loader)*milestone_epoch])

    parameter_group = [{'params': model.params.clf.parameters()}]
    if config["l2d"] == "pop":
        parameter_group += [{'params': model.params.rej.parameters()}]
    optimizer_new = torch.optim.Adam(parameter_group, lr=lr_clf_rej)    
    scheduler_new_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_new, len(train_loader)*milestone_epoch, eta_min=lr_clf_rej/1000)
    scheduler_new_constant = torch.optim.lr_scheduler.ConstantLR(optimizer_new, factor=1., total_iters=0)
    scheduler_new_constant.base_lrs = [lr_clf_rej/1000 for _ in optimizer_new.param_groups]
    scheduler_new = torch.optim.lr_scheduler.SequentialLR(optimizer_new, [scheduler_new_cosine,scheduler_new_constant], 
                                                          milestones=[len(train_loader)*milestone_epoch])

    optimizer_lst = [optimizer_base, optimizer_new]
    scheduler_lst = [scheduler_base, scheduler_new]

    best_validation_loss = np.inf
    # patience = 0
    iters = 0

    n_finetune_steps_eval = config['n_steps_maml'] if config['l2d']=='single_maml' else 0
    for epoch in range(0, epochs):
        iters, train_loss = train_epoch(iters, 
                                        train_loader, 
                                        model, 
                                        optimizer_lst, 
                                        scheduler_lst, 
                                        epoch,
                                        experts_train,
                                        loss_fn,
                                        cntx_sampler_train,
                                        n_classes,
                                        config,
                                        logger,
                                        config['n_steps_maml'],
                                        config['lr_maml'])
        metrics = evaluate(model,
                           experts_test,
                           loss_fn,
                           cntx_sampler_eval,
                           n_classes,
                           valid_loader,
                           config,
                           logger,
                           n_finetune_steps=n_finetune_steps_eval,
                           lr_finetune=config['lr_maml'])

        validation_loss = metrics["val_loss"]

        if (epoch+1 > milestone_epoch) and (validation_loss < best_validation_loss): # NOTE: tweak for HAM10000
            best_validation_loss = validation_loss
            # logger.info("Saving the model with system accuracy {}".format(metrics['sys_acc']))
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


def eval(model, val_data, test_data, loss_fn, experts_test, val_cntx_sampler, test_cntx_sampler, config):
    '''val_data and val_cntx_sampler are only used for single-expert finetuning'''
    model_state_dict = torch.load(os.path.join(config["ckp_dir"], config["experiment_name"] + ".pt"), map_location=device)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["val_batch_size"], shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["test_batch_size"], shuffle=False, **kwargs)

    for budget in config["budget"]: # budget=1.0
        test_cntx_sampler.reset()
        logger = get_logger(os.path.join(config["ckp_dir"], "eval{}.log".format(budget)))
        model.load_state_dict(copy.deepcopy(model_state_dict))
        evaluate(model, experts_test, loss_fn, test_cntx_sampler, config["n_classes"], test_loader, config, logger, budget)

        if (config["l2d"] == 'single_maml') or ((config["l2d"] == 'single') and config["finetune_single"]):
            logger = get_logger(os.path.join(config["ckp_dir"], "eval{}_finetune.log".format(budget)))
            
            n_finetune_steps_lst = [n_steps for n_steps in config["n_finetune_steps"] if n_steps >= config["n_steps_maml"]] \
                                    if (config["l2d"] == 'single_maml') else config["n_finetune_steps"]
            lr_finetune_lst = [config["lr_maml"]] if (config["l2d"] == 'single_maml') else config["lr_finetune"]

            steps_lr_comb = list(itertools.product(n_finetune_steps_lst, lr_finetune_lst))
            val_losses = []
            for (n_steps, lr) in steps_lr_comb:
                print(f'no. finetune steps: {n_steps}  step size: {lr}')
                val_cntx_sampler.reset()
                model.load_state_dict(copy.deepcopy(model_state_dict))
                metrics = evaluate(model, experts_test, loss_fn, val_cntx_sampler, config["n_classes"], val_loader, config, None, budget, \
                                n_steps, lr)
                val_losses.append(metrics['val_loss'])
            idx = np.argmin(np.array(val_losses))
            best_finetune_steps, best_lr = steps_lr_comb[idx]
            test_cntx_sampler.reset()
            model.load_state_dict(copy.deepcopy(model_state_dict))
            metrics = evaluate(model, experts_test, loss_fn, test_cntx_sampler, config["n_classes"], test_loader, config, logger, budget, \
                                best_finetune_steps, best_lr)

    # # Only for l2d=pop
    # for budget in config["budget"]:
    #     for p_cntx_inclusion in config["p_cntx_inclusion"]:
    #         test_cntx_sampler.reset()
    #         logger = get_logger(os.path.join(config["ckp_dir"], "eval{}_pc{}.log".format(budget,p_cntx_inclusion)))
    #         model.load_state_dict(copy.deepcopy(model_state_dict))
    #         evaluate(model, experts_test, loss_fn, test_cntx_sampler, config["n_classes"], test_loader, config, logger, budget, p_cntx_inclusion=p_cntx_inclusion)


def main(config):
    set_seed(config["seed"])
    # NB: consider extending export dir with loss_type, n_context_pts if this comparison becomes prominent
    config["ckp_dir"] = f"./runs/{config['dataset']}/{config['loss_type']}/l2d_{config['l2d']}/p{str(config['p_out'])}_seed{str(config['seed'])}"
    # config["ckp_dir"] = f"./runs/{config['dataset']}/{config['loss_type']}/l2d_{config['l2d']}_lr{config['lr_maml']}_s{config['n_steps_maml']}/p{str(config['p_out'])}_seed{str(config['seed'])}"
    os.makedirs(config["ckp_dir"], exist_ok=True)
    if config["dataset"] == 'cifar20_100':
        config["n_classes"] = 20
        config["decouple"] = True
        train_data, val_data, test_data = load_cifar(variety='20_100', data_aug=True, seed=config["seed"])
        resnet_base = WideResNetBase(depth=28, n_channels=3, widen_factor=4, dropRate=0.0)
        n_features = resnet_base.nChannels
    elif config["dataset"] == 'cifar10':
        config["n_classes"] = 10
        config["decouple"] = False
        train_data, val_data, test_data = load_cifar(variety='10', data_aug=False, seed=config["seed"])
        resnet_base = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0)
        n_features = resnet_base.nChannels
    elif config["dataset"] == 'ham10000':
        config["n_classes"] = 7
        config["decouple"] = False
        train_data, val_data, test_data = load_ham10000()
        resnet_base = ResNet34()
        n_features = resnet_base.n_features
    else:
        raise ValueError('dataset unrecognised')

    with_softmax = False
    if config["loss_type"] == 'softmax':
        loss_fn = cross_entropy
        with_softmax = True
    else: # ova
        loss_fn = ova

    with_attn=False
    config_tokens = config["l2d"].split("_")
    if (len(config_tokens) > 1) and (config_tokens[0] == 'pop'):
        if config_tokens[1] == 'attn':
            with_attn = True
        config["l2d"] = "pop"

    if config["warmstart"]:
        warmstart_path = f"./pretrained/{config['dataset']}/seed{str(config['seed'])}/default.pt"
        if not os.path.isfile(warmstart_path):
            raise FileNotFoundError('warmstart model checkpoint not found')
        resnet_base.load_state_dict(torch.load(warmstart_path, map_location=device))
        resnet_base = resnet_base.to(device)

    if config["l2d"] == "pop":
        model = ClassifierRejectorWithContextEmbedder(resnet_base, num_classes=int(config["n_classes"]), n_features=n_features, \
                                                      with_attn=with_attn, with_softmax=with_softmax, decouple=config["decouple"], \
                                                      depth_embed=config["depth_embed"], depth_rej=config["depth_reject"])
    else:
        model = ClassifierRejector(resnet_base, num_classes=int(config["n_classes"]), n_features=n_features, with_softmax=with_softmax, \
                                   decouple=config["decouple"])
    
    config["n_experts"] = 10 # assume exactly divisible by 2
    experts_train = []
    experts_test = []
    if config["dataset"] == 'cifar20_100':
        n_oracle_superclass = 4
        n_oracle_subclass = 3 # 3 or 4 here. Affects gap between {single,pop,pop_attn}
        for _ in range(config["n_experts"]): # n_experts
            # this specifies "superset" of subclasses expert is oracle at
            classes_coarse = np.random.choice(np.arange(config["n_classes"]), size=n_oracle_superclass, replace=False)
            expert = Cifar20SyntheticExpert(classes_coarse, n_classes=config["n_classes"], p_in=1.0, p_out=config['p_out'], \
                                            n_oracle_subclass=n_oracle_subclass)
            experts_train.append(expert)
        experts_test += experts_train[:config["n_experts"]//2]
        for _ in range(config["n_experts"]//2):
            classes_coarse = np.random.choice(np.arange(config["n_classes"]), size=n_oracle_superclass, replace=False)
            expert = Cifar20SyntheticExpert(classes_coarse, n_classes=config["n_classes"], p_in=1.0, p_out=config['p_out'], \
                                            n_oracle_subclass=n_oracle_subclass)
            experts_test.append(expert)
    else:
        for _ in range(config["n_experts"]): # train
            class_oracle = random.randint(0, config["n_classes"]-1)
            expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=config['p_out'])
            experts_train.append(expert)
        experts_test += experts_train[:config["n_experts"]//2] # pick 50% experts from experts_train (order not matter)
        for _ in range(config["n_experts"]//2): # then sample 50% new experts
            class_oracle = random.randint(0, config["n_classes"]-1)
            expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=1.0, p_out=config['p_out'])
            experts_test.append(expert)
    
    # Context sampler train-time: just take from full train set (potentially with data augmentation)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    cntx_sampler_train = ContextSampler(train_data.data, train_data.targets, train_data.transform, train_data.targets_sparse, \
                                        n_cntx_pts=config["n_cntx_pts"], device=device, **kwargs)
    
    # Context sampler val/test-time: partition val/test sets
    prop_cntx = 0.2
    val_cntx_size = int(prop_cntx * len(val_data))
    val_data_cntx, val_data_trgt = torch.utils.data.random_split(val_data, [val_cntx_size, len(val_data)-val_cntx_size], \
                                                                 generator=torch.Generator().manual_seed(config["seed"]))
    test_cntx_size = int(prop_cntx * len(test_data))
    test_data_cntx, test_data_trgt = torch.utils.data.random_split(test_data, [test_cntx_size, len(test_data)-test_cntx_size], \
                                                                 generator=torch.Generator().manual_seed(config["seed"]))
    cntx_sampler_val = ContextSampler(images=val_data_cntx.dataset.data[val_data_cntx.indices], 
                                      labels=val_data_cntx.dataset.targets[val_data_cntx.indices], 
                                      transform=val_data.transform, 
                                      labels_sparse=val_data_cntx.dataset.targets_sparse[val_data_cntx.indices] if config["dataset"]=='cifar20_100' else None,
                                      n_cntx_pts=config["n_cntx_pts"], device=device, **kwargs)
    cntx_sampler_test = ContextSampler(images=test_data_cntx.dataset.data[test_data_cntx.indices], 
                                      labels=np.array(test_data_cntx.dataset.targets)[test_data_cntx.indices], 
                                      transform=test_data.transform, 
                                      labels_sparse=test_data_cntx.dataset.targets_sparse[test_data_cntx.indices] if config["dataset"]=='cifar20_100' else None,
                                      n_cntx_pts=config["n_cntx_pts"], device=device, **kwargs)
    
    if config["mode"] == 'train':
        train(model, train_data, val_data_trgt, loss_fn, experts_train, experts_test, cntx_sampler_train, cntx_sampler_val, config)
        eval(model, val_data_trgt, test_data_trgt, loss_fn, experts_test, cntx_sampler_val, cntx_sampler_test, config)
    else: # evaluation on test data
        eval(model, val_data_trgt, test_data_trgt, loss_fn, experts_test, cntx_sampler_val, cntx_sampler_test, config) # NOTE
        # eval(model, val_data_trgt, val_data_trgt, loss_fn, experts_test, cntx_sampler_val, cntx_sampler_val, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1071)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    # parser.add_argument("--patience", type=int, default=50, 
    # 						help="number of patience steps for early stopping the training.")
    parser.add_argument("--lr_wrn", type=float, default=1e-1, help="learning rate for wrn.")
    parser.add_argument("--lr_other", type=float, default=1e-2, help="learning rate for non-wrn model components.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--experiment_name", type=str, default="default",
                            help="specify the experiment name. Checkpoints will be saved with this name.")
    
    ## NEW experiment setup
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument("--p_out", type=float, default=0.1) # [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    # parser.add_argument("--n_cntx_per_class", type=int, default=5) # moved to main()
    parser.add_argument('--l2d', choices=['single', 'single_maml', 'pop', 'pop_attn'], default='single')
    parser.add_argument('--loss_type', choices=['softmax', 'ova'], default='softmax')
    parser.add_argument("--n_cntx_pts", type=int, default=50)

    ## NEW train args
    parser.add_argument("--dataset", choices=["cifar10", "cifar20_100", "ham10000"], default="cifar10")
    parser.add_argument("--val_batch_size", type=int, default=8) # 32 maml
    parser.add_argument("--test_batch_size", type=int, default=1) # 32 maml
    parser.add_argument('--warmstart', action='store_true')
    parser.set_defaults(warmstart=False)
    parser.add_argument("--warmstart_epochs", type=int, default=100)
    parser.add_argument("--depth_embed", type=int, default=6)
    parser.add_argument("--depth_reject", type=int, default=3)
    ## NEW maml
    parser.add_argument('--n_steps_maml', type=int, default=5)
    parser.add_argument('--lr_maml', type=float, default=1e-1)

    ## EVAL
    # parser.add_argument('--budget', nargs='+', type=float, default=[0.01,0.02,0.05,0.1,0.2,0.5]) # 1.0
    parser.add_argument('--budget', nargs='+', type=float, default=[1.0])
    # parser.add_argument('--p_cntx_inclusion', nargs='+', type=float, default=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) # 1.0 # NB: rebuttal

    parser.add_argument('--finetune_single', action='store_true')
    parser.set_defaults(finetune_single=True)
    parser.add_argument('--n_finetune_steps', nargs='+', type=int, default=[1,2,5,10,20])
    parser.add_argument('--lr_finetune', nargs='+', type=float, default=[1e-1,1e-2])

    # # Hack (remove after)
    # parser.add_argument("--runs", type=str, default="runs")
    
    config = parser.parse_args().__dict__
    main(config)
