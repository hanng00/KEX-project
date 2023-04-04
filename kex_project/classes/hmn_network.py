import nest
import numpy as np


class HMN_network:
    def __init__(self, J_E=0.8, J_I=-0.8, verbose=False) -> None:
        nest.ResetKernel()
        n = 4  # number of threads'
        self.dt = 0.1

        self.verbose = verbose

        nest.SetKernelStatus(
            {"local_num_threads": n, "resolution": self.dt, "overwrite_files": True}
        )

        self.N_total = 10_000
        self.n_submodules = 16

        self.E_perc = 4 / 5
        self.N_E = int(self.N_total * self.E_perc)  # Number of excitatory neurons
        self.N_I = int(self.N_total * (1 - self.E_perc))  # Number of excitatory neurons

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
            # "I_e": 50.0,
            "tau_syn_ex": 5.0,
            "tau_syn_in": 10.0,
        }

        # Synapse parameters
        self.P_0 = 0.01  # Connection probability
        self.delay = 1.0

        self.J_E = J_E
        self.J_I = J_I

        # Rewiring parameters
        self.R_ex = 0.99
        self.R_inh = 1

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
        self.nodes = nest.Create(
            self.neuron_model, self.N_total, params=self.neuron_params
        )
        self.submodule_dict = self._generate_base_network(
            self.nodes, self.n_submodules, E_perc=self.E_perc
        )
        self._connect_submodules(self.submodule_dict, self.P_0, self.R_ex, self.E_perc)
        print("Network fully built.")

    def simulate(self, simtime, noise_hz, stimulate_module_ids, record_module_ids):
        noise = nest.Create(
            "poisson_generator",
            1,
            {
                "rate": noise_hz,
                "start": 0,
                "stop": 200,
            },
        )
        nodes_stimulate = self._get_exc_nodes(self.submodule_dict, stimulate_module_ids)
        nest.Connect(noise, nodes_stimulate)

        spikes = nest.Create("spike_recorder", 1, [{"label": "va-py-ex"}])
        self.spikes_E = spikes[:1]
        nodes_record = self._get_exc_nodes(self.submodule_dict, record_module_ids)
        nest.Connect(nodes_record, self.spikes_E)

        nest.Simulate(simtime)

    def plot(self):
        nest.raster_plot.from_device(self.spikes_E, hist=True)

    def _get_exc_nodes_from_submodule(self, submodule):
        N_E = int(self.E_perc * len(submodule))
        return submodule[:N_E]

    def _get_exc_nodes(self, submodule_dict, submodule_ids):
        node_collection = nest.NodeCollection()
        for idx in submodule_ids:
            exc_nodes = self._get_exc_nodes_from_submodule(submodule_dict[idx])
            node_collection += exc_nodes
        return node_collection

    def _generate_base_network(self, nodes, n_modules, E_perc):
        n_of_nodes = len(nodes)
        len_submodule = int(n_of_nodes // n_modules)
        print(
            f"Generating submodules of size {len_submodule}, total network size {n_of_nodes}"
        )

        submodule_dict = {}
        for submodule_idx, node_idx in enumerate(range(0, n_of_nodes, len_submodule)):
            submodule = nodes[node_idx : node_idx + len_submodule]
            submodule_dict[submodule_idx + 1] = submodule

        print("Base network configured")

        return submodule_dict

    def _get_local_connection_density(self, P_0, R_ex, neuron_type):
        if neuron_type == "excitatory":
            return (0.8 * P_0) * (1 + R_ex) ** 4
        elif neuron_type == "inhibitory":
            return 0.2 * P_0 * 2**4
        else:
            raise (f"{neuron_type} is not a valid neuron")

    def _get_inter_connection_density_by_level(self, P_0, R_ex, l):
        return (0.8 * P_0) * (1 + R_ex) ** (l - 1) * (1 - R_ex)

    def _find_smallest_common_interval(self, source_id, target_id, interval):
        half_idx = int(len(interval) / 2)
        first_half = interval[:half_idx]
        second_half = interval[half_idx:]
        smallest_common_interval = interval

        for sub_interval in [first_half, second_half]:
            if source_id in sub_interval and target_id in sub_interval:
                if len(sub_interval) == 1:
                    return sub_interval
                # Nodes exists in a smaller module
                smallest_common_interval = self._find_smallest_common_interval(
                    source_id=source_id,
                    target_id=target_id,
                    interval=sub_interval,
                )

        return smallest_common_interval

    def _get_inter_connection_density(
        self, source_id, target_id, n_submodules, P_0, R_ex
    ):
        """Computes the level where the nodes closest common parent lives."""
        submodules_ids = list(range(1, n_submodules + 1))
        smallest_common_interval = self._find_smallest_common_interval(
            source_id, target_id, submodules_ids
        )
        level = np.log2(n_submodules / len(smallest_common_interval))
        connection_density = self._get_inter_connection_density_by_level(
            P_0, R_ex, l=level
        )
        return connection_density

    def _connect_nodes(
        self,
        source_nodes,
        target_nodes,
        exc_connection_density,
        inh_connection_density,
        E_perc,
    ):
        N_E = int(len(source_nodes) * E_perc)
        source_nodes_E = source_nodes[:N_E]
        source_nodes_I = source_nodes[N_E:]

        if exc_connection_density > 0:
            exc_conn_dict = {"rule": "pairwise_bernoulli", "p": exc_connection_density}
            nest.Connect(
                source_nodes,
                target_nodes,
                syn_spec={"synapse_model": "excitatory_synapse"},
                conn_spec=exc_conn_dict,
            )
        if inh_connection_density > 0:
            inh_conn_dict = {"rule": "pairwise_bernoulli", "p": inh_connection_density}
            nest.Connect(
                source_nodes,
                target_nodes,
                syn_spec={"synapse_model": "inhibitory_synapse"},
                conn_spec=inh_conn_dict,
            )

    def _connect_submodules(self, submodule_dict, P_0, R_ex, E_perc):
        n_submodules = len(submodule_dict)

        for source_idx, source_nodes in submodule_dict.items():
            if self.verbose:
                print(f"Connecting source: {source_idx}...")
            for target_idx, target_nodes in submodule_dict.items():
                exc_connection_density = 0
                inh_connection_density = 0
                if target_idx == source_idx:
                    exc_connection_density = self._get_local_connection_density(
                        P_0=P_0, R_ex=R_ex, neuron_type="excitatory"
                    )
                    inh_connection_density = self._get_local_connection_density(
                        P_0=P_0, R_ex=R_ex, neuron_type="inhibitory"
                    )
                else:
                    exc_connection_density = self._get_inter_connection_density(
                        source_id=source_idx,
                        target_id=target_idx,
                        n_submodules=n_submodules,
                        P_0=P_0,
                        R_ex=R_ex,
                    )

                self._connect_nodes(
                    source_nodes=source_nodes,
                    target_nodes=target_nodes,
                    exc_connection_density=exc_connection_density,
                    inh_connection_density=inh_connection_density,
                    E_perc=E_perc,
                )
                if self.verbose:
                    print(
                        f"- Connected to target: {target_idx}, exc_dens: {exc_connection_density}, inh_dens: {inh_connection_density}"
                    )
