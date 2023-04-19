import nest
import numpy as np
from .hmn_network import HMN_network


class HMNAdaptive(HMN_network):
    def __init__(self, J_E=0.8, J_I=-0.8, verbose=False) -> None:
        super().__init__(J_E, J_I, verbose)

        self.inh_neuron_model = "aeif_cond_exp"
        print(f"Using {self.inh_neuron_model}-model for inhibitory neurons")
