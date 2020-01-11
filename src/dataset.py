import torch
from torch.utils.data import Dataset

import warnings
import numpy
from numpy import array, concatenate, arange, unwrap, angle
from scipy.constants import c, pi
from .compact_simulator import simulate_scintillation
fL1 = 1.57542e9
fL2 = 1.2276e9
fL5 = 1.17645e9


class ScintillationDataset(Dataset):

    def __init__(self, length, set_index):
        self.length = length
        self.set_index = set_index

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        U, mu0, p1, p2, roveff = 1.2, 1, 2.6, 3.2, 0.9
        freqs = array([fL1, fL2, fL5])
        Nt = 2**12
        Nf = len(freqs)
        dt = 0.01

        alpha = array(list(map(lambda f: 40.308e16 / f**2, freqs)))

        warnings.filterwarnings('ignore')
        psi, phase_screens = simulate_scintillation(U, mu0, p1, p2, roveff, freqs, Nt, dt, self.set_index * self.length + idx)
        TEC = numpy.mean(phase_screens[:, :] * c * freqs[:, None] / 40.308e16 / (2 * pi), axis=0)
        
        phases = array([unwrap(angle(psi[k, :])) for k in range(Nf)])
        amplitudes = array([abs(psi[k, :]) for k in range(Nf)])
        
        x = concatenate((phases, amplitudes), axis=0)
        y = TEC
        return torch.FloatTensor(x), torch.FloatTensor(y)