r"""Exoplanet emission spectrum (EES) simulator.

The simulator computes an emission spectrum based on disequilibrium carbon chemistry,
equilibrium clouds and a spline temperature-pressure profile of the exoplanet atmosphere.

References:
    Retrieving scattering clouds and disequilibrium chemistry in the atmosphere of HR 8799e
    (Mollière et al., 2020)
    https://arxiv.org/abs/2006.09394

Shapes:
    theta: :math:`(16,)`
    x: :math:`(379,)`
"""

import numpy as np
import os

os.environ['pRT_input_data_path'] = os.path.join(os.getcwd(), 'input_data')

import petitRADTRANS as prt
import petitRADTRANS.retrieval.models as models
import petitRADTRANS.retrieval.parameter as prm

from functools import partial
from joblib import Memory
from numpy import ndarray as Array
from typing import *


MEMORY = Memory(os.getcwd(), mmap_mode='c', verbose=0)

LABELS, LOWER, UPPER = zip(*[
    [                  r'${\rm C/O}$',  0.1,   1.6],  # C/O
    [    r'$\left[{\rm Fe/H}\right]$', -1.5,   1.5],  # [Fe/H]
    [        r'$\log P_{\rm quench}$',  -6.,    3.],  # log P_quench
    [            r'$\log X_{\rm Fe}$', -2.3,    1.],  # log X_Fe
    [       r'$\log X_{\rm MgSiO_3}$', -2.3,    1.],  # log X_MgSiO3
    [                r'$f_{\rm sed}$',   0.,   10.],  # f_sed
    [                r'$\log K_{zz}$',   5.,   13.],  # log K_zz
    [                   r'$\sigma_g$', 1.05,    3.],  # sigma_g
    [                     r'$\log g$',   2.,   5.5],  # log g
    [                        r'$R_P$',  0.9,    2.],  # R_P / R_Jupyter
    [                        r'$T_0$', 300., 2300.],  # T_0
    [r'$\frac{T_3}{T_{\rm connect}}$',   0.,    1.],  # ∝ T_3 / T_connect
    [            r'$\frac{T_2}{T_3}$',   0.,    1.],  # ∝ T_2 / T_3
    [            r'$\frac{T_1}{T_2}$',   0.,    1.],  # ∝ T_1 / T_2
    [                     r'$\alpha$',   1.,    2.],  # alpha
    [ r'$\frac{\log \delta}{\alpha}$',   0.,    1.],  # ∝ log delta / alpha
])


class Simulator(object):
    r"""Creates a EES simulator.

    Arguments:
        noisy: Whether noise is added to spectra or not.
        kwargs: Simulator settings and constants (e.g. planet distance, pressures, ...).
    """

    def __init__(self, noisy: bool = True, **kwargs):
        super().__init__()

        # Constants
        default = {
            'D_pl': 41.2925 * prt.nat_cst.pc,
            'pressure_scaling': 10,
            'pressure_simple': 100,
            'pressure_width': 3,
            'scale': 1e16,
        }

        self.constants = {
            k: kwargs.get(k, v)
            for k, v in default.items()
        }
        self.scale = self.constants.pop('scale')

        self.atmosphere = MEMORY.cache(prt.Radtrans)(
            line_species=[
                'H2O_HITEMP_R_400',
                'CO_all_iso_R_400',
                'CH4_R_400',
                'NH3_R_400',
                'CO2_R_400',
                'H2S_R_400',
                'VO_R_400',
                'TiO_all_Exomol_R_400',
                'PH3_R_400',
                'Na_allard_R_400',
                'K_allard_R_400',
            ],
            cloud_species=['MgSiO3(c)_cd', 'Fe(c)_cd'],
            rayleigh_species=['H2', 'He'],
            continuum_opacities=['H2-H2', 'H2-He'],
            wlen_bords_micron=[0.95, 2.45],
            do_scat_emis=True,
        )

        levels = (
            self.constants['pressure_simple'] + len(self.atmosphere.cloud_species) *
            (self.constants['pressure_scaling'] - 1) * self.constants['pressure_width']
        )

        self.atmosphere.setup_opa_structure(np.logspace(-6, 3, levels))

        # Noise
        self.noisy = noisy
        self.sigma = 1.25754e-17 * self.scale

    def __call__(self, theta: Array) -> Array:
        x = emission_spectrum(self.atmosphere, theta, **self.constants)
        x = self.process(x)

        if self.noisy:
            x = x + self.sigma * np.random.standard_normal(x.shape)

        return x

    def process(self, x: Array) -> Array:
        r"""Processes spectra into network-friendly inputs."""

        return x * self.scale


def emission_spectrum(
    atmosphere: prt.Radtrans,
    theta: Array,
    **kwargs,
) -> Array:
    r"""Simulates the emission spectrum of an exoplanet."""

    names = [
        'C/O', 'Fe/H', 'log_pquench', 'log_X_cb_Fe(c)', 'log_X_cb_MgSiO3(c)',
        'fsed', 'log_kzz', 'sigma_lnorm', 'log_g', 'R_pl',
        'T_int', 'T3', 'T2', 'T1', 'alpha', 'log_delta',
    ]

    kwargs.update(dict(zip(names, theta)))
    kwargs['R_pl'] = kwargs['R_pl'] * prt.nat_cst.r_jup_mean

    parameters = {
        k: prm.Parameter(name=k, value=v, is_free_parameter=False)
        for k, v in kwargs.items()
    }

    _, spectrum = models.emission_model_diseq(atmosphere, parameters, AMR=True)

    return spectrum


@partial(np.vectorize, signature='(m),(n)->(n)')
def pt_profile(theta: Array, pressures: Array) -> Array:
    r"""Returns the pressure-temperature profile."""

    CO, FeH, *_, T_int, T3, T2, T1, alpha, log_delta = theta

    T3 = ((3 / 4 * T_int ** 4 * (0.1 + 2 / 3)) ** (1 / 4)) * (1 - T3)
    T2 = T3 * (1 - T2)
    T1 = T2 * (1 - T1)
    delta = (1e6 * 10 ** (-3 + 5 * log_delta)) ** (-alpha)

    return models.PT_ret_model(
        np.array([T1, T2, T3]),
        delta,
        alpha,
        T_int,
        pressures,
        FeH,
        CO,
    )
