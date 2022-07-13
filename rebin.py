#!/usr/bin/env python

import numpy as np
import os

os.environ['pRT_input_data_path'] = os.path.join(os.getcwd(), 'input_data')

from molmass import Formula
from petitRADTRANS import Radtrans


if __name__ == '__main__':
    species = [
        'H2O_HITEMP',
        'CO_all_iso_HITEMP',
        'CH4',
        'NH3',
        'CO2',
        'H2S',
        'VO',
        'TiO_all_Exomol',
        'PH3',
        'Na_allard',
        'K_allard',
    ]

    masses = {
        s: Formula(s).isotope.massnumber
        for s in map(lambda s: s.split('_')[0], species)
    }

    path = os.path.join(os.environ['pRT_input_data_path'], 'opacities/lines/corr_k')

    atmosphere = Radtrans(line_species=species, wlen_bords_micron=[0.1, 251.])
    atmosphere.write_out_rebin(400, path=path, species=species, masses=masses)
