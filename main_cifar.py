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
from lib.experts import SyntheticExpertOverlap, Cifar20SyntheticExpert
from lib.wideresnet import ClassifierRejector, ClassifierRejectorWithContextEmbedder, WideResNetBase
from lib.datasets import load_cifar, ContextSampler


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
            logger,
            budget=1.0):
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
    with torch.no_grad():
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
            
            if config["l2d"] == 'pop':
                # sample expert predictions for context
                expert_cntx = cntx_sampler.sample(n_experts=1)
                cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse.squeeze(0)
                exp_preds = torch.tensor(expert(expert_cntx.xc.squeeze(0), expert_cntx.yc.squeeze(), cntx_yc_sparse), device=device)
                expert_cntx.mc = exp_preds.unsqueeze(0)
                
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

    for i, data in enumerate(train_loader):
        if len(data) == 2:
            input, target = data
            input, target = input.to(device), target.to(device)
            target_sparse = None
        else:
            input, target, target_sparse = data # ignore additional labels
            input, target, target_sparse = input.to(device), target.to(device), target_sparse.to(device)
        n_experts = len(experts_train)

        if config["l2d"] == 'pop':
            expert_cntx = cntx_sampler.sample(n_experts=n_experts)

            # sample expert predictions for context
            exp_preds_cntx = []
            for idx_exp, expert in enumerate(experts_train):
                cntx_yc_sparse = None if expert_cntx.yc_sparse is None else expert_cntx.yc_sparse[idx_exp]
                preds = torch.tensor(expert(expert_cntx.xc[idx_exp], expert_cntx.yc[idx_exp], cntx_yc_sparse), device=device)
                exp_preds_cntx.append(preds.unsqueeze(0))
            expert_cntx.mc = torch.vstack(exp_preds_cntx)

            outputs = model(input,expert_cntx) # [E,B,K+1]
        else:
            outputs = model(input) # [B,K+1]
            outputs = outputs.unsqueeze(0).repeat(n_experts,1,1) # [E,B,K+1]
        
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
                                        logger)
        metrics = evaluate(model,
                           experts_test,
                           loss_fn,
                           cntx_sampler_eval,
                           n_classes,
                           valid_loader,
                           config,
                           logger)

        validation_loss = metrics["val_loss"]

        if validation_loss < best_validation_loss:
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


def eval(model, test_data, loss_fn, experts_test, cntx_sampler, config):
    model.load_state_dict(torch.load(os.path.join(config["ckp_dir"], config["experiment_name"] + ".pt"), map_location=device))
    model = model.to(device)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["test_batch_size"], shuffle=False, **kwargs)
    for budget in config["budget"]:
        logger = get_logger(os.path.join(config["ckp_dir"], "eval{}.log".format(budget)))
        cntx_sampler.reset()
        evaluate(model, experts_test, loss_fn, cntx_sampler, config["n_classes"], test_loader, config, logger, budget)


def main(config):
    set_seed(config["seed"])
    # NB: consider extending export dir with loss_type, n_context_pts if this comparison becomes prominent
    config["ckp_dir"] = f"./runs/cifar{config['cifar']}/{config['loss_type']}/l2d_{config['l2d']}/p{str(config['p_out'])}_seed{str(config['seed'])}"
    os.makedirs(config["ckp_dir"], exist_ok=True)
    if config["cifar"] == '20_100':
        config["n_classes"] = 20
        config["data_aug"] = True
        config["wrn_widen_factor"] = 4
        config["n_cntx_pts"] = 100
    else:
        config["n_classes"] = 10
        config["data_aug"] = False
        config["wrn_widen_factor"] = 2
        config["n_cntx_pts"] = 50
    train_data, val_data, test_data = load_cifar(variety=config["cifar"], data_aug=config["data_aug"], seed=config["seed"])

    with_softmax = False
    if config["loss_type"] == 'softmax':
        loss_fn = cross_entropy
        with_softmax = True
    else: # ova
        loss_fn = ova

    with_cross_attn=False
    with_self_attn=False
    if len(config["l2d"].split("_")) > 2: # pop_attn_sa
        with_self_attn=True
    if len(config["l2d"].split("_")) > 1: # pop_attn
        with_cross_attn=True
        config["l2d"] = "pop"

    wrnbase = WideResNetBase(depth=28, n_channels=3, widen_factor=config["wrn_widen_factor"], dropRate=0.0)

    if config["warmstart"]:
        warmstart_path = f"./pretrained/cifar{config['cifar']}/seed{str(config['seed'])}/default.pt"
        if not os.path.isfile(warmstart_path):
            raise FileNotFoundError('warmstart model checkpoint not found')
        wrnbase.load_state_dict(torch.load(warmstart_path, map_location=device))
        wrnbase = wrnbase.to(device)
    
    decouple = True # NOTE
    if config["l2d"] == "pop":
        model = ClassifierRejectorWithContextEmbedder(wrnbase, num_classes=int(config["n_classes"]), n_features=wrnbase.nChannels, \
                                                      with_cross_attn=with_cross_attn, with_self_attn=with_self_attn, with_softmax=with_softmax, \
                                                      decouple=decouple)
    else:
        model = ClassifierRejector(wrnbase, num_classes=int(config["n_classes"]), n_features=wrnbase.nChannels, with_softmax=with_softmax, \
                                   decouple=decouple)
    
    config["n_experts"] = 10 # assume exactly divisible by 2
    experts_train = []
    experts_test = []
    if config["cifar"] == '20_100':
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
                                      labels_sparse=None if config["cifar"]=='10' else val_data_cntx.dataset.targets_sparse[val_data_cntx.indices],
                                      n_cntx_pts=config["n_cntx_pts"], device=device, **kwargs)
    cntx_sampler_test = ContextSampler(images=test_data_cntx.dataset.data[test_data_cntx.indices], 
                                      labels=np.array(test_data_cntx.dataset.targets)[test_data_cntx.indices], 
                                      transform=test_data.transform, 
                                      labels_sparse=None if config["cifar"]=='10' else test_data_cntx.dataset.targets_sparse[test_data_cntx.indices],
                                      n_cntx_pts=config["n_cntx_pts"], device=device, **kwargs)
    
    if config["mode"] == 'train':
        train(model, train_data, val_data_trgt, loss_fn, experts_train, experts_test, cntx_sampler_train, cntx_sampler_val, config)
        eval(model, test_data_trgt, loss_fn, experts_test, cntx_sampler_test, config)
    else: # evaluation on test data
        eval(model, test_data_trgt, loss_fn, experts_test, cntx_sampler_test, config)


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
    parser.add_argument('--l2d', choices=['single', 'pop', 'pop_attn', 'pop_attn_sa'], default='single')
    parser.add_argument('--loss_type', choices=['softmax', 'ova'], default='softmax')

    ## NEW train args
    parser.add_argument("--cifar", choices=["10", "20_100"], default="20_100")
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument('--warmstart', action='store_true')
    parser.set_defaults(warmstart=True)
    parser.add_argument("--warmstart_epochs", type=int, default=100)

    ## EVAL
    parser.add_argument('--budget', nargs='+', type=float, default=[0.01,0.05,0.1,0.2,1.0])

    # # Hack (remove after)
    # parser.add_argument("--runs", type=str, default="runs")
    
    config = parser.parse_args().__dict__
    main(config)
