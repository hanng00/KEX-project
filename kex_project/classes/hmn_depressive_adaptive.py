import nest
import numpy as np
from .hmn_network import HMN_network

from .hmn_adaptive import HMNAdaptive
from .hmn_depressive import HMNDepressive


class HMNDepressiveAdaptive(HMNAdaptive, HMNDepressive):
    def __init__(self, J_E=0.8, J_I=-0.8, t_ref=0.5, verbose=False) -> None:
        super().__init__(J_E, J_I, t_ref, verbose)