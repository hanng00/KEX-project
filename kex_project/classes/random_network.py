import nest


class RandomNetwork:
    def __init__(self, J_E: float = 4.0, J_I: float = -51.0) -> None:
        nest.ResetKernel()
        print("STARTING")

        self.N_total = 10_000

        self.N_E = 8000
        self.N_I = 2000  # Number of excitatory neurons
        self.N_rec = 8000  # Number of neurons to record

        self.synapse_model = "static_synapse"
        self.neuron_model = "iaf_cond_exp"

        time_constant = 20  # ms
        membrane_resistance = 100  # Mohm
        C_m = time_constant / membrane_resistance * 1000  # pF

        self.neuron_params = {
            "V_m": nest.random.uniform(min=-60.0, max=-55.0),
            "E_L": -60.0,
            "V_th": -50.0,
            "V_reset": -60.0,
            "t_ref": 5.0,
            "E_ex": 0.0,
            "E_in": -80.0,
            "C_m": 200.0,
            "g_L": 10.0,
            # "I_e": 20.0,
            "tau_syn_ex": 5.0,
            "tau_syn_in": 10.0,
        }

        # Synapse parameters
        self.P_0 = 0.02  # Connection probability
        self.delay = 2.0

        self.J_E = J_E
        self.J_I = J_I

    def run(
        self,
        simtime: float,
        poisson_config={},
    ):
        print("Starting simulation")
        # Add stimulation and spike detector
        noise = nest.Create(
            "poisson_generator",
            1,
            poisson_config,
        )

        spikes = nest.Create("spike_recorder", 1, [{"label": "va-py-ex"}])
        self.spikes_E = spikes[:1]

        nest.Connect(self.nodes_E[: self.N_rec], self.spikes_E)
        nest.Connect(noise, self.nodes)

        nest.Simulate(simtime)

    def plot(self):
        nest.raster_plot.from_device(self.spikes_E, hist=True)

    def build(self):
        # Set up neurons
        self.nodes = nest.Create(
            self.neuron_model, self.N_total, params=self.neuron_params
        )
        self.nodes_E = self.nodes[: self.N_E]
        self.nodes_I = self.nodes[self.N_E :]

        # nest.SetStatus(self.nodes, "V_m", -60.0)

        # Set up synapses
        exc_syn_dict = {
            "synapse_model": "static_synapse",
            "weight": self.J_E,
            "delay": self.delay,
        }
        inh_syn_dict = {
            "synapse_model": "static_synapse",
            "weight": self.J_I,
            "delay": self.delay,
        }

        # Connect hte network
        prob_conn_dict = {"rule": "pairwise_bernoulli", "p": self.P_0}

        nest.Connect(
            self.nodes_E,
            self.nodes,
            syn_spec=exc_syn_dict,
            conn_spec=prob_conn_dict,
        )
        nest.Connect(
            self.nodes_I,
            self.nodes,
            syn_spec=inh_syn_dict,
            conn_spec=prob_conn_dict,
        )
