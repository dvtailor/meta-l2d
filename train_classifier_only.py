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
from lib.wideresnet import Classifier #WideResNetBase
from lib.models_fixup import WideResNetBase
from lib.datasets import load_cifar10


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate(model,
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
    real_total = 0
    clf_alone_correct = 0
    losses = []
    model.eval() # Crucial for networks with batchnorm layers!
    criterion = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            batch_size = len(images)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, -1)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            for i in range(0, batch_size):
                prediction = predicted[i]
                clf_alone_correct += (prediction == labels[i]).item()
                real_total += 1
    metrics = {"clf_acc": 100 * clf_alone_correct / real_total,
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

    criterion = nn.CrossEntropyLoss(reduction='mean')
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)
        
        outputs = model(input) # [B,K]
        loss = criterion(outputs, target)
        epoch_train_loss.append(loss.item())

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, target, topk=(1,))[0] # just measures clf accuracy
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
          config):
    logger = get_logger(os.path.join(config["ckp_dir"], "train.log"))
    logger.info(f"seed={config['seed']}")
    logger.info(config)
    logger.info('No. of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    n_classes = config["n_classes"]
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["train_batch_size"], shuffle=True, **kwargs) # drop_last=True
    valid_loader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=config["train_batch_size"], shuffle=False, **kwargs) # shuffle=True, drop_last=True
    model = model.to(device)
    cudnn.benchmark = True

    parameters_bias = [p[1] for p in model.named_parameters() if 'bias' in p[0]]
    parameters_scale = [p[1] for p in model.named_parameters() if 'scale' in p[0]]
    parameters_others = [p[1] for p in model.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
    optimizer = torch.optim.SGD(
        [{'params': parameters_bias, 'lr': config["lr"]/10.}, 
        {'params': parameters_scale, 'lr': config["lr"]/10.}, 
        {'params': parameters_others}], 
        lr=config["lr"],
        momentum=0.9, 
        nesterov=True,
        weight_decay=config["weight_decay"])

    # optimizer = torch.optim.SGD(model.parameters(), config["lr"], momentum=0.9, nesterov=True, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * config["epochs"])

    iters = 0
    lrate = config["lr"]

    for epoch in range(0, config["epochs"]):
        iters, train_loss = train_epoch(iters, 
                                        train_loader, 
                                        model, 
                                        optimizer, 
                                        scheduler, 
                                        epoch,
                                        n_classes,
                                        config,
                                        logger)
        metrics = evaluate(model,
                           n_classes,
                           valid_loader,
                           config,
                           logger)

        validation_loss = metrics["val_loss"]

        # torch.save(model.state_dict(), os.path.join(config["ckp_dir"], config["experiment_name"] + ".pt"))
        # Additionally save the whole config dict
        # with open(os.path.join(config["ckp_dir"], config["experiment_name"] + ".json"), "w") as f:
        #     json.dump(config, f)


def eval(model, test_data, config):
    # model.load_state_dict(torch.load(os.path.join(config["ckp_dir"], config["experiment_name"] + ".pt"), map_location=device))
    # model = model.to(device)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["train_batch_size"], shuffle=False, **kwargs)
    logger = get_logger(os.path.join(config["ckp_dir"], "eval.log"))
    evaluate(model, config["n_classes"], test_loader, config, logger)


def main(config):
    set_seed(config["seed"])
    
    config["ckp_dir"] = f"./pretrained/seed{str(config['seed'])}"
    os.makedirs(config["ckp_dir"], exist_ok=True)
    train_data, val_data, test_data = load_cifar10(data_aug=False, seed=config["seed"])
    config["n_classes"] = 10

    wrnbase = WideResNetBase(depth=28, n_channels=3, widen_factor=2, dropRate=0.0)
    model = Classifier(wrnbase, num_classes=int(config["n_classes"]), n_features=wrnbase.nChannels, with_softmax=False)
    
    train(model, train_data, val_data, config)
    eval(model, test_data, config)

    # save only WideResNet part
    torch.save(wrnbase.state_dict(), os.path.join(config["ckp_dir"], config["experiment_name"] + ".pt"))
    with open(os.path.join(config["ckp_dir"], config["experiment_name"] + ".json"), "w") as f:
        json.dump(config, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1,
                            help="learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--experiment_name", type=str, default="default",
                            help="specify the experiment name. Checkpoints will be saved with this name.")
    
    config = parser.parse_args().__dict__
    main(config)
