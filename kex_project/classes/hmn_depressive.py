import nest
import numpy as np
from .hmn_network import HMN_network


class HMNDepressive(HMN_network):
    def __init__(self, J_E=0.8, J_I=-0.8, t_ref=0.5, verbose=False) -> None:
        super().__init__(J_E, J_I, t_ref, verbose)

    def _setup_synapses(self):
        print("Using depressing synapses.")
        """
        U - probability of release increment (U1) [0,1], default=0.

        u - Maximum probability of release (U_se) [0,1], default=0.

        x - current scaling factor of the weight, default=U

        tau_rec - time constant for depression in ms, default=800 ms

        tau_fac - time constant for facilitation in ms, default=0 (off)
        
        """

        Tau_rec = 20.0  # recovery time
        Tau_fac = 0.0  # facilitation time
        U = 0.5  # facilitation parameter U
        A = 250.0  # PSC weight in pA
        u = 0.5
        x = 10

        nest.CopyModel(
            "static_synapse",
            "excitatory_synapse",
            {
                "weight": self.J_E,
            },
        )

        nest.CopyModel(
            "static_synapse",
            "input_synapse",
            {
                "weight": self.J_E,
            },
        )

        nest.CopyModel(
            "tsodyks2_synapse",
            "inhibitory_synapse",
            {
                "tau_rec": Tau_rec,
                "tau_fac": Tau_fac,
                "U": U,
                "weight": A,
                "u": u,
                "x": x,
            },
        )
