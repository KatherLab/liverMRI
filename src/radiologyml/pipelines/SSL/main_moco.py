#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

from .ctran import ctranspath, tiny_vit, pvt_v2_b2_li, convnextv2

from .moco import builder
from .moco import loader
from .moco import optimizer

from . import vits


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base', 'swin', 'ctp'] + torchvision_model_names


def main(args):

    if args["seed"] is not None:
        random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args["gpu"] is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args["dist_url"] == "env://" and args["world_size"] == -1:
        args["world_size"] = int(os.environ["WORLD_SIZE"])

    args["distributed"] = args["world_size"] > 1 or args["multiprocessing_distributed"]

    ngpus_per_node = torch.cuda.device_count()
    if args["multiprocessing_distributed"]:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args["world_size"] = ngpus_per_node * args["world_size"]
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args["gpu"], ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args["gpu"] = gpu

    # suppress printing if not first GPU on each node
    if args["multiprocessing_distributed"] and (args["gpu"] != 0 or args["rank"] != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args["gpu"] is not None:
        print("Use GPU: {} for training".format(args["gpu"]))

    if args["distributed"]:
        if args["dist_url"] == "env://" and args["rank"] == -1:
            args["rank"] = int(os.environ["RANK"])
        if args["multiprocessing_distributed"]:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args["rank"] = args["rank"] * ngpus_per_node + gpu
        dist.init_process_group(backend=args["dist_backend"], init_method=args["dist_url"],
                                world_size=args["world_size"], rank=args["rank"])
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args["arch"]))
    if args["arch"].startswith('vit'):
        model = builder.MoCo_ViT(
            partial(vits.__dict__[args["arch"]], stop_grad_conv1=args["stop_grad_conv1"]),
            args["moco_dim"], args["moco_mlp_dim"], args["moco_t"])
    elif args["arch"].startswith("swin"):
        model = builder.MoCo_ViT(ctranspath,
            args["moco_dim"], args["moco_mlp_dim"], args["moco_t"],swin=True,CTP=False)
    elif args["arch"].startswith("ctp"):
        model = builder.MoCo_ViT(ctranspath,
            args["moco_dim"], args["moco_mlp_dim"], args["moco_t"],swin=True,CTP=True)
    elif args["arch"].startswith("pvt"):
        model = builder.MoCo_ViT(pvt_v2_b2_li,
            args["moco_dim"], args["moco_mlp_dim"], args["moco_t"],swin=True,CTP=False)
    elif args["arch"].startswith("convnext"):
        model = builder.MoCo_ResNet(convnextv2,
            args["moco_dim"], args["moco_mlp_dim"], args["moco_t"],swin=True,resnext=True)
    elif args["arch"].startswith("tiny_vit"):
        model = builder.MoCo_ViT(tiny_vit,
            args["moco_dim"], args["moco_mlp_dim"], args["moco_t"],swin=True,CTP=False)
    else:
        model = builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args["arch"]], zero_init_residual=True), 
            args["moco_dim"], args["moco_mlp_dim"], args["moco_t"])
    print(f"# base-model params: {sum(p.numel() for p in model.base_encoder.parameters())}")
    # infer learning rate before changing batch size
    args["lr"] = args["lr"] * args["batch_size"] / 256

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args["distributed"]:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args["gpu"] is not None:
            torch.cuda.set_device(args["gpu"])
            model.cuda(args["gpu"])
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args["batch_size"] = int(args["batch_size"] / args["world_size"])
            args["workers"] = int((args["workers"] + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args["gpu"]])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args["gpu"] is not None:
        torch.cuda.set_device(args["gpu"])
        model = model.cuda(args["gpu"])
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    if args["optimizer"] == 'lars':
        optimizer = optimizer.LARS(model.parameters(), args["lr"],
                                        weight_decay=args["weight_decay"],
                                        momentum=args["momentum"])
    elif args["optimizer"] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args["lr"],
                                weight_decay=args["weight_decay"])
        
    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter() if args["rank"] == 0 else None

    # optionally resume from a checkpoint
    if args["resume"]:
        if os.path.isfile(args["resume"]):
            print("=> loading checkpoint '{}'".format(args["resume"]))
            if args["gpu"] is None:
                checkpoint = torch.load(args["resume"])
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args["gpu"])
                checkpoint = torch.load(args["resume"], map_location=loc)
            args["start_epoch"] = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args["resume"], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args["resume"]))

    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args["data"], 'train')
    traindir = args["data"]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(args["image_size"], scale=(args["crop_min"], 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(args["image_size"], scale=(args["crop_min"], 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                                      transforms.Compose(augmentation2)))

    print(f"number of training samples = {len(train_dataset.imgs)}")
    
    if args["distributed"]:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args["batch_size"], shuffle=(train_sampler is None),
        num_workers=args["workers"], pin_memory=True, sampler=train_sampler, drop_last=True)

    epoch_time = 0
    for epoch in range(args["start_epoch"], args["epochs"]):
        
        if args["distributed"]:
            train_sampler.set_epoch(epoch)
        e_start = time.time()
        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)
        epoch_time+=time.time()-e_start
        #with open(f"/raid/tim/logs/moco-time-log-brca-{args['arch']}.log",'a') as f:
        #   f.write(f"{epoch_time/((epoch+1)*len(train_loader)):.5f}s\n")
        if not args["multiprocessing_distributed"] or (args["multiprocessing_distributed"]
                and args["rank"] == 0): # only the first GPU saves checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args["arch"],
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=f'{args["model_out_path"]}/{args["arch"]}-moco-liver-checkpoint_%04d.pth.tar' % (epoch+1))

    if args["rank"] == 0:
        summary_writer.close()

def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args["moco_m"]
    
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args["moco_m_cos"]:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args["gpu"] is not None:
            images[0] = images[0].cuda(args["gpu"], non_blocking=True)
            images[1] = images[1].cuda(args["gpu"], non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].size(0))
        if args["rank"] == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args["print_freq"] == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args["warmup_epochs"]:
        lr = args["lr"] * epoch / args["warmup_epochs"] 
    else:
        lr = args["lr"] * 0.5 * (1. + math.cos(math.pi * (epoch - args["warmup_epochs"]) / (args["epochs"] - args["warmup_epochs"])))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args["epochs"])) * (1. - args["moco_m"])
    return m


if __name__ == '__main__':
    main()
