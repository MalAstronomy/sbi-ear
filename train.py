#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import wandb

from dawgz import job, schedule
from itertools import chain, islice
from pathlib import Path
from torch import Tensor
from tqdm import tqdm

from lampe.data import H5Dataset
from lampe.inference import NPE, NPELoss
from lampe.nn import ResMLP
from lampe.nn.flows import NAF
from lampe.utils import GDStep

from ees import Simulator, LOWER, UPPER


scratch = os.environ.get('SCRATCH', '')
datapath = Path(scratch) / 'eac/data'
savepath = Path(scratch) / 'eac/runs'


@job(array=3, cpus=2, gpus=1, ram='8GB', time='1-00:00:00')
def train(i: int):
    # Run
    run = wandb.init(project='eac')

    # Simulator
    simulator = Simulator(noisy=False)

    def noisy(x: Tensor) -> Tensor:
        return x + simulator.sigma * torch.randn_like(x)

    l, u = torch.tensor(LOWER), torch.tensor(UPPER)

    # Estimator
    embedding = ResMLP(379, 64, hidden_features=[512] * 2 + [256] * 3 + [128] * 5, activation='ELU')
    estimator = NPE(
        16, 64,
        moments=((l + u) / 2, (u - l) / 2),
        transforms=3,
        build=NAF,
        signal=16,
        hidden_features=[512] * 5,
        activation='ELU',
    )

    embedding.cuda(), estimator.cuda()

    # Optimizer
    loss = NPELoss(estimator)
    optimizer = optim.AdamW(chain(embedding.parameters(), estimator.parameters()), lr=1e-3, weight_decay=1e-2)
    scheduler = sched.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5, patience=32, threshold=1e-2)
    step = GDStep(optimizer, clip=1)

    # Data
    trainset = H5Dataset(datapath / 'train.h5', batch_size=2048, shuffle=True)
    validset = H5Dataset(datapath / 'valid.h5', batch_size=2048, shuffle=True)

    # Training
    def pipe(theta: Tensor, x: Tensor) -> Tensor:
        theta, x = theta.cuda(), x.cuda()
        x = noisy(x)
        x = embedding(x)
        return loss(theta, x)

    for epoch in tqdm(range(512), unit='epoch'):
        embedding.train(), estimator.train()
        start = time.time()

        losses = torch.stack([
            step(pipe(theta, x))
            for theta, x in islice(trainset, 1024)
        ]).cpu().numpy()

        end = time.time()
        embedding.eval(), estimator.eval()

        with torch.no_grad():
            losses_val = torch.stack([
                pipe(theta, x)
                for theta, x in islice(validset, 256)
            ]).cpu().numpy()

        run.log({
            'lr': optimizer.param_groups[0]['lr'],
            'loss': np.nanmean(losses),
            'loss_val': np.nanmean(losses_val),
            'nans': np.isnan(losses).mean(),
            'nans_val': np.isnan(losses_val).mean(),
            'speed': len(losses) / (end - start),
        })

        scheduler.step(np.nanmean(losses_val))

        if optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
            break

    runpath = savepath / run.name
    runpath.mkdir(parents=True, exist_ok=True)

    torch.save({
        'embedding': embedding.state_dict(),
        'estimator': estimator.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, runpath / 'states.pth')

    run.finish()


if __name__ == '__main__':
    schedule(
        train,
        name='Training',
        backend='slurm',
        env=[
            'source ~/.bashrc',
            'conda activate eac',
            'export WANDB_SILENT=true',
        ]
    )
