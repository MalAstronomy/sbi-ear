#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import wandb

from dawgz import job, after, ensure, schedule
from itertools import chain, islice
from pathlib import Path
from torch import Tensor
from tqdm import tqdm
from typing import *

from lampe.data import H5Dataset
from lampe.distributions import BoxUniform
from lampe.inference import NPE, NPELoss
from lampe.nn import ResMLP
from lampe.nn.flows import NAF, NSF
from lampe.plots import nice_rc, corner, rank_ecdf
from lampe.utils import GDStep

from ees import Simulator, LOWER, UPPER, LABELS, pt_profile


scratch = os.environ['SCRATCH']
datapath = Path(scratch) / 'ees/data_379'
savepath = Path(scratch) / 'ees/sweep'


CONFIGS = {
    'embedding': ['shallow', 'deep'],
    'flow': ['NAF', 'NSF'],
    'transforms': [3, 5],
    'signal': [16, 32],  # not important- the autoregression network output 
    'hidden_features': [256, 512], # hidden layers of the autoregression network
    'activation': ['ELU'],
    'optimizer': ['AdamW'],
    'init_lr': [1e-3, 5e-4, 1e-4, 1e-5],
    'weight_decay': [0, 1e-4, 1e-3, 1e-2],
    'scheduler': ['ReduceLROnPlateau', 'CosineAnnealingLR'],
    'min_lr': [1e-6, 1e-5],
    'patience': [8, 16, 32],
    'epochs': [256, 512, 1024],
    'stop_criterion': ['early', 'late'],
    'batch_size': [256, 512, 1024, 2048, 4096],
}


@job(array=2**3, cpus=2, gpus=1, ram='8GB', time='1-00:00:00')
def experiment(index: int) -> None:
    # Config
    config = {
        key: random.choice(values)
        for key, values in CONFIGS.items()
    }
    
    run = wandb.init(project='ees-sweep', config=config)
    
    # Simulator
    simulator = Simulator(noisy=False)
    
    def noisy(x: Tensor) -> Tensor:
        return x + simulator.sigma * torch.randn_like(x)
    
    l, u = torch.tensor(LOWER), torch.tensor(UPPER)
    
    # Estimator
    if config['embedding'] == 'shallow':
        embedding = ResMLP(379, 64, hidden_features=[512] * 2 + [256] * 3 + [128] * 5, activation='ELU')
    else:
        embedding = ResMLP(379, 128, hidden_features=[512] * 3 + [256] * 5 + [128] * 7, activation='ELU')

    if config['flow'] == 'NSF':
        estimator = NPE(
            16, embedding.out_features,
            moments=((l + u) / 2, (l - u) / 2),
            transforms=config['transforms'],
            build=NSF,
            bins=config['signal'],
            hidden_features=[config['hidden_features']] * 5,
            activation=config['activation'],
        )
    else:
        estimator = NPE(
            16, embedding.out_features,
            moments=((l + u) / 2, (l - u) / 2),
            transforms=config['transforms'],
            build=NAF,
            signal=config['signal'],
            hidden_features=[config['hidden_features']] * 5,
            activation=config['activation'],
        )
    
    # print(embedding, flush=True)
    # print(estimator, flush=True)

    embedding.cuda(), estimator.cuda()

    # Optimizer
    loss = NPELoss(estimator)
    optimizer = optim.AdamW(chain(embedding.parameters(), estimator.parameters()), lr=config['init_lr'], weight_decay=config['weight_decay'])
    scheduler = sched.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=config['min_lr'], patience=config['patience'], threshold=1e-2, threshold_mode='abs')
    step = GDStep(optimizer, clip=1)

    # Data
    trainset = H5Dataset(datapath / 'train.h5', batch_size=config['batch_size'], shuffle=True)
    validset = H5Dataset(datapath / 'valid.h5', batch_size=config['batch_size'], shuffle=True)

    # Training
    def pipe(theta: Tensor, x: Tensor) -> Tensor:
        theta, x = theta.cuda(), x.cuda()
        x = noisy(x)
        x = embedding(x)
        return loss(theta, x)

    for epoch in tqdm(range(config['epochs']), unit='epoch'):
        embedding.train(), estimator.train()
        
        start = time.time()
        losses = torch.stack([
            step(pipe(theta, x))
            for theta, x in islice(trainset, 2**10)
        ]).cpu().numpy()
        end = time.time()
        
        embedding.eval(), estimator.eval()
        
        with torch.no_grad():            
            losses_val = torch.stack([
                pipe(theta, x)
                for theta, x in islice(validset, 2**8)
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

        if config['stop_criterion'] == 'early' and optimizer.param_groups[0]['lr'] <= config['min_lr']:
            break

    runpath = savepath / f'{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    torch.save({
        'embedding': embedding.state_dict(),
        'estimator': estimator.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, runpath / 'weights.pth')

    # Evaluation
    plt.rcParams.update(nice_rc(latex=True))

    ## Coverage
    testset = H5Dataset(datapath / 'test.h5', batch_size=2**4)

    ranks = []

    with torch.no_grad():
        for theta, x in tqdm(islice(testset, 2**8)):
            theta, x = theta.cuda(), x.cuda()
            x = noisy(x)
            x = embedding(x)

            posterior = estimator.flow(x)
            samples = posterior.sample((2**10,))
            log_p = posterior.log_prob(theta)
            log_p_samples = posterior.log_prob(samples)

            ranks.append((log_p_samples < log_p).float().mean(dim=0).cpu())

    ranks = torch.cat(ranks)
    ecdf_fig = rank_ecdf(ranks)
    ecdf_fig.savefig(runpath / 'ecdf.pdf')

    ## Corner
    dataset = H5Dataset(datapath / 'event.h5')
    theta_star, x_star = dataset[1]

    with torch.no_grad():
        x = embedding(x_star.cuda())
        theta = torch.cat([
            estimator.sample(x, (2**14,)).cpu()
            for _ in range(2**6)
        ])

    corner_fig = corner(
        theta,
        smooth=2,
        bounds=(LOWER, UPPER),
        labels=LABELS,
        legend=r'$p_{\phi}(\theta | x^*)$',
        markers=[theta_star],
        figsize=(12, 12),
    )
    corner_fig.savefig(runpath / 'corner.pdf')
    
    ## NumPy
    theta_star, x_star = theta_star.double().numpy(), x_star.double().numpy()
    theta = theta[:2**8].double().numpy()

    ## PT profile
    pt_fig, ax = plt.subplots(figsize=(4.8, 4.8))

    pressures = simulator.atmosphere.press / 1e6
    temperatures = pt_profile(theta, pressures)

    for q in [0.997, 0.95, 0.68]:
        left, right = np.quantile(temperatures, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax.fill_betweenx(pressures, left, right, color='C0', alpha=0.25, linewidth=0)

    ax.plot(pt_profile(theta_star, pressures), pressures, color='k', linestyle='--')

    ax.set_xlabel(r'Temperature $[\mathrm{K}]$')
    ax.set_xlim(0, 4000)
    ax.set_ylabel(r'Pressure $[\mathrm{bar}]$')
    ax.set_ylim(1e-2, 1e1)
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.grid()

    pt_fig.savefig(runpath / 'pt_profile.pdf')

    ## Residuals
    res_fig, ax = plt.subplots(figsize=(4.8, 4.8))

    x = np.stack([simulator(t) for t in tqdm(theta)])
    mask = ~np.isnan(x).any(axis=-1)
    theta, x = theta[mask], x[mask]

    wlength = np.linspace(0.95, 2.45, x.shape[-1])
    
    for q in [0.997, 0.95, 0.68]:
        lower, upper = np.quantile(x, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax.fill_between(wlength, lower, upper, color='C0', alpha=0.25, linewidth=0)

    ax.plot(wlength, x_star, color='k', linestyle=':')
    
    ax.set_xlabel(r'Wavelength $[\mu\mathrm{m}]$')
    ax.set_ylabel(r'Flux $[\mathrm{W} \, \mathrm{m}^{-2} \, \mu\mathrm{m}^{-1}]$')
    ax.grid()

    res_fig.savefig(runpath / 'residuals.pdf')

    run.log({
        'ecdf': wandb.Image(ecdf_fig),
        'corner': wandb.Image(corner_fig),
        'pt_profile': wandb.Image(pt_fig),
        'res_fig': wandb.Image(res_fig),
    })
    run.finish()


if __name__ == '__main__':
    schedule(
        experiment,
        name='EES sweep',
        backend='slurm',
        env=[
            'source ~/.bashrc',
            'conda activate lampe',
            'export WANDB_SILENT=true',
        ]
    )
