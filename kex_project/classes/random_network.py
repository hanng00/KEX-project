import nest


class RandomNetwork:
    def __init__(
        self, J_E: float = 4.0, J_I: float = -51.0, p_rate: float = 20_000
    ) -> None:
        nest.ResetKernel()

        self.N_total = 10_000

        self.N_E = int(self.N_total * (4 / 5))  # Number of excitatory neurons
        self.N_I = int(self.N_total * (1 / 5))  # Number of excitatory neurons
        self.N_rec = self.N_E  # Number of neurons to record

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
            "C_m": 100.0,
            "g_L": 10.0,
            # "I_e": 200.0,
            "tau_syn_ex": 5.0,
            "tau_syn_in": 10.0,
        }

        # Synapse parameters
        self.P_0 = 0.01  # Connection probability
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

    def build(self):
        # Set up neurons
        nodes = nest.Create(self.neuron_model, self.N_total, params=self.neuron_params)
        nodes_E = nodes[: self.N_E]
        nodes_I = nodes[self.N_E :]

        nest.SetStatus(nodes, "V_m", -60.0)

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
            nodes_E,
            nodes,
            syn_spec=exc_syn_dict,
            conn_spec=prob_conn_dict,
        )
        nest.Connect(
            nodes_I,
            nodes,
            syn_spec=inh_syn_dict,
            conn_spec=prob_conn_dict,
        )

        # Add stimulation and spike detector
        noise = nest.Create(
            "poisson_generator", 1, {"rate": self.p_rate, "start": 0, "stop": 200}
        )

        spikes = nest.Create("spike_recorder", 1, [{"label": "va-py-ex"}])
        self.spikes_E = spikes[:1]

        nest.Connect(nodes_E[: self.N_rec], self.spikes_E)
        nest.Connect(noise, nodes)
