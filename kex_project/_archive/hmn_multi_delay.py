import nest
import numpy as np
from .hmn_network import HMN_network


class HMNMultiDelay(HMN_network):
    def __init__(self, J_E=0.8, J_I=-0.8, verbose=False) -> None:
        super().__init__(J_E, J_I, verbose)

        self.within_delay = 2.0
        self.across_delay = 5.0

        self.delay = self.within_delay
