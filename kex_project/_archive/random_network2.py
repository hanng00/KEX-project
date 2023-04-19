import nest

"""
The following input led to 500ms sustained activity:
------

from classes.random_network2 import RandomNetwork2

strength = 10
params_1 = {
    "J_E": 0.6 * strength,
    "J_I": -8.0 * strength,
    "p_rate": 50 * 1000,
}
poisson_config = {
    "rate": 2 * 1000,
    "start": 0,
    "stop": 200,
}

random_network_1 = RandomNetwork2(**params_1)
random_network_1.build()
random_network_1.run(
    simtime=1000.0,
    # poisson_config=poisson_config,
)
random_network_1.plot()


"""


class RandomNetwork2:
    def __init__(
        self, J_E: float = 4.0, J_I: float = -51.0, p_rate: float = 20_000
    ) -> None:
        nest.ResetKernel()
        n = 4  # number of threads'
        self.dt = 0.1

        nest.SetKernelStatus(
            {"local_num_threads": n, "resolution": self.dt, "overwrite_files": True}
        )

        self.N_total = 10_000

        self.N_E = 8_000  # Number of excitatory neurons
        self.N_I = 2_000  # Number of excitatory neurons
        self.N_rec = 8_000  # Number of neurons to record

        self.synapse_model = "static_synapse"
        self.neuron_model = "iaf_cond_exp"

        self.neuron_params = {
            "V_m": -60.0,
            "E_L": -60.0,
            "V_th": -50.0,
            "V_reset": -60.0,
            "t_ref": 5.0,
            "E_ex": 0.0,
            "E_in": -80.0,
            "C_m": 200.0,
            "g_L": 10.0,
            "I_e": 20.0,
            "tau_syn_ex": 5.0,
            "tau_syn_in": 10.0,
        }

        # Synapse parameters
        self.P_0 = 0.02  # Connection probability
        self.delay = 1.0

        self.J_E = J_E
        self.J_I = J_I

        # Stimulation
        self.p_rate = p_rate

    def run(self, simtime: float):
        print("Starting simulation")
        nest.Simulate(simtime)

    def plot(self):
        nest.raster_plot.from_device(self.spikes_E, hist=True)

    def _setup_synapses(self):
        nest.CopyModel(
            "static_synapse",
            "excitatory_synapse",
            {
                "weight": self.J_E,
                "delay": self.delay,
            },
        )

        nest.CopyModel(
            "static_synapse",
            "inhibitory_synapse",
            {
                "weight": self.J_I,
                "delay": self.delay,
            },
        )

    def build(self):
        # Set up neurons
        self.nodes = nest.Create(
            self.neuron_model, self.N_total, params=self.neuron_params
        )
        self.nodes_E = self.nodes[: self.N_E]
        self.nodes_I = self.nodes[self.N_E :]

        nest.SetStatus(self.nodes, "V_m", -60.0)

        # Connect hte network
        self._setup_synapses()
        prob_conn_dict = {"rule": "pairwise_bernoulli", "p": self.P_0}

        nest.Connect(
            self.nodes_E,
            self.nodes,
            syn_spec={"synapse_model": "excitatory_synapse"},
            conn_spec=prob_conn_dict,
        )
        nest.Connect(
            self.nodes_I,
            self.nodes,
            syn_spec={"synapse_model": "inhibitory_synapse"},
            conn_spec=prob_conn_dict,
        )

        # STIMULATION
        # 1. Poisson generator
        noise = nest.Create(
            "poisson_generator", 1, {"rate": self.p_rate, "start": 0, "stop": 200}
        )

        # 2. Noise generator
        noise_generator = nest.Create(
            "noise_generator",
            {"mean": 0.0, "std": 200.0, "start": 0, "stop": 200},
        )
        nest.Connect(
            noise_generator,
            self.nodes,
            {"rule": "pairwise_bernoulli", "p": 0.6},
        )

        # 3. AC Generator
        """ 
        drive = nest.Create(
            "ac_generator",
            {"amplitude": 50000.0, "frequency": 50_000.0, "start": 0, "stop": 200},
        )
        nest.Connect(drive, self.nodes_E)
        """
        # DEVICES
        spikes = nest.Create("spike_recorder", 1, [{"label": "va-py-ex"}])
        self.spikes_E = spikes[:1]

        nest.Connect(self.nodes_E[: self.N_rec], self.spikes_E)
        # nest.Connect(self.nodes_I, self.spikes_E)

        """ nest.Connect(
            noise,
            self.nodes_E,
            {"rule": "pairwise_bernoulli", "p": 0.3},
            # {"synapse_model": "excitatory_synapse"},
        ) """
        print("BUILasdasdT!")
