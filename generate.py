#!/usr/bin/env python

import numpy as np
import os
import torch

from dawgz import job, after, ensure, schedule
from itertools import starmap
from pathlib import Path
from typing import *

from lampe.data import JointLoader, H5Dataset
from lampe.distributions import BoxUniform

from ees import Simulator, LOWER, UPPER


scratch = os.environ.get('SCRATCH', '')
path = Path(scratch) / 'eac/data'
path.mkdir(parents=True, exist_ok=True)


@ensure(lambda i: (path / f'samples_{i:06d}.h5').exists())
@job(array=1024, cpus=1, ram='4GB', time='1-00:00:00')
def simulate(i: int):
    prior = BoxUniform(torch.tensor(LOWER), torch.tensor(UPPER))
    simulator = Simulator(noisy=False)
    loader = JointLoader(prior, simulator, batch_size=16, numpy=True)

    def filter_nan(theta, x):
        mask = ~torch.any(torch.isnan(x), dim=-1)
        return theta[mask], x[mask]

    H5Dataset.store(
        starmap(filter_nan, loader),
        path / f'samples_{i:06d}.h5',
        size=4096,
    )


@after(simulate)
@job(cpus=1, ram='4GB', time='01:00:00')
def aggregate():
    files = list(path.glob('samples_*.h5'))
    length = len(files)

    i = int(0.9 * length)
    splits = {
        'train': files[:i],
        'valid': files[i:-1],
        'test': files[-1:],
    }

    def filter_large(theta, x):
        mask = x.mean(dim=-1) < 6
        return theta[mask], x[mask]

    for name, files in splits.items():
        dataset = H5Dataset(*files, batch_size=4096)

        H5Dataset.store(
            starmap(filter_large, dataset),
            path / f'{name}.h5',
            size=len(dataset) // 2,
        )


@ensure(lambda: (path / 'event.h5').exists())
@job(cpus=1, ram='4GB', time='05:00')
def event():
    simulator = Simulator(noisy=False)

    theta_star = np.array([0.55, 0., -5., -0.86, -0.65, 3., 8.5, 2., 3.75, 1., 1063.6, 0.26, 0.29, 0.32, 1.39, 0.48])
    x_star = simulator(theta_star)

    theta = theta_star[None].repeat(256, axis=0)
    x = x_star[None].repeat(256, axis=0)

    noise = simulator.sigma * np.random.standard_normal(x.shape)
    noise[0] = 0

    H5Dataset.store(
        [(theta, x + noise)],
        path / 'event.h5',
        size=256,
    )


if __name__ == '__main__':
    schedule(
        aggregate, event,
        name='Data generation',
        backend='slurm',
        prune=True,
        env=[
            'source ~/.bashrc',
            'conda activate eac',
        ]
    )
