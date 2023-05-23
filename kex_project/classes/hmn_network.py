import nest
import numpy as np
from typing import List
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd


class HMN_network:
    def __init__(self, J_E=0.8, J_I=-0.8, t_ref=5.0, verbose=False) -> None:
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

        self.exc_neuron_model = "iaf_cond_exp"
        self.inh_neuron_model = self.exc_neuron_model

        self.neuron_params = {
            "V_m": nest.random.uniform(min=-60.0, max=-55.0),
            "E_L": -60.0,
            "V_th": -50.0,
            "V_reset": -60.0,
            "t_ref": t_ref,
            "E_ex": 0.0,
            "E_in": -80.0,
            "C_m": 200.0,
            "g_L": 10.0,
            "I_e": 10.0,
            "tau_syn_ex": 5.0,
            "tau_syn_in": 10.0,
        }

        # Synapse parameters
        self.P_0 = 0.01  # Connection probability
        self.within_delay = 2.0
        self.across_delay = 5.0

        self.J_E = J_E
        self.J_I = J_I

        # Rewiring parameters
        self.R_ex = 0.99
        self.R_inh = 1

        # Setup synapses
        self._setup_synapses()

    def _setup_synapses(self):
        print("Using static synapses.")

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
            "static_synapse",
            "inhibitory_synapse",
            {
                "weight": self.J_I,
            },
        )

    def build(self):
        """self.nodes = nest.Create(
            self.neuron_model, self.N_total, params=self.neuron_params
        )"""
        self.submodule_dict = self._generate_submodule_dict(
            self.N_total, self.n_submodules, E_perc=self.E_perc
        )
        self._connect_submodules(self.submodule_dict, self.P_0, self.R_ex, self.E_perc)
        print("Network fully built.")

    def _add_background_noise(self, noise_config):
        noise_generator = nest.Create(
            "noise_generator",
            {
                "mean": 0.0,
                "std": 50.0,
                "dt": 10,
                "stop": noise_config.get("stop", 10_000),
            },
        )
        node_collections_E = self._get_exc_nodes(
            self.submodule_dict, noise_config["module_ids"]
        )

        noise_conn_dict = {"rule": "pairwise_bernoulli", "p": noise_config["p"]}

        for nodes in self.submodule_dict.values():
            nest.Connect(
                noise_generator,
                nodes,
                noise_conn_dict,
            )

    def simulate(
        self,
        simtime,
        poisson_config,
        stimulate_module_ids,
        record_module_ids,
        stimulate_module_ratio,
        noise_config,
    ):
        self._add_background_noise(noise_config)

        noise = nest.Create(
            "poisson_generator",
            1,
            poisson_config,
        )

        # self.vm = nest.Create("voltmeter")
        # nest.Connect(self.vm, self.nodes)

        nodes_stimulate_collection = self._get_exc_nodes(
            self.submodule_dict, stimulate_module_ids
        )

        for nodes_stimulate in nodes_stimulate_collection:
            nest.Connect(
                noise,
                nodes_stimulate,
                {"rule": "pairwise_bernoulli", "p": stimulate_module_ratio},
                "input_synapse",
            )

        spikes = nest.Create("spike_recorder", 1, [{"label": "va-py-ex"}])
        self.spikes_E = spikes[:1]
        nodes_record_collection = self._get_exc_nodes(
            self.submodule_dict, record_module_ids
        )

        for nodes_record in nodes_record_collection:
            nest.Connect(nodes_record, self.spikes_E)

        nest.Simulate(simtime)

    def plot(self, title=""):
        nest.raster_plot.from_device(self.spikes_E, hist=True, title=title)

    def _get_exc_nodes_from_submodule(self, submodule):
        N_E = int(self.E_perc * len(submodule))
        return submodule[:N_E]

    def _get_exc_nodes(
        self, submodule_dict, submodule_ids
    ) -> List[nest.NodeCollection]:
        node_collections = []
        for idx in submodule_ids:
            exc_nodes = self._get_exc_nodes_from_submodule(submodule_dict[idx])

            node_collections.append(exc_nodes)

        return node_collections

    def _generate_submodule_dict(self, N_total, n_modules, E_perc):
        len_submodule = int(N_total // n_modules)
        print(
            f"Generating submodules of size {len_submodule}, total network size {N_total}"
        )

        submodule_dict = {}
        for submodule_idx, node_idx in enumerate(range(0, N_total, len_submodule)):
            n_exc_nodes = int(len_submodule * E_perc)
            exc_nodes = nest.Create(
                self.exc_neuron_model, n_exc_nodes, params=self.neuron_params
            )
            inh_nodes = nest.Create(
                self.inh_neuron_model,
                len_submodule - n_exc_nodes,
                params=self.neuron_params,
            )
            submodule_dict[submodule_idx + 1] = exc_nodes + inh_nodes

        print("Base network configured")

        return submodule_dict

    def _get_local_connection_density(self, P_0, R_ex, neuron_type):
        if neuron_type == "excitatory":
            return (P_0) * (1 + R_ex) ** 4
        elif neuron_type == "inhibitory":
            # return 0.2 * P_0 * 2**4 + 0.05
            return P_0 * 2**4

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
        is_within_module,
    ):
        N_E = int(len(source_nodes) * E_perc)
        source_nodes_E = source_nodes[:N_E]
        source_nodes_I = source_nodes[N_E:]

        delay = self.across_delay
        if is_within_module:
            delay = self.within_delay

        if exc_connection_density > 0:
            exc_conn_dict = {"rule": "pairwise_bernoulli", "p": exc_connection_density}
            nest.Connect(
                source_nodes_E,
                target_nodes,
                syn_spec={
                    "synapse_model": "excitatory_synapse",
                    "delay": delay,
                },
                conn_spec=exc_conn_dict,
            )
        if inh_connection_density > 0:
            inh_conn_dict = {
                "rule": "pairwise_bernoulli",
                "p": inh_connection_density,
            }
            nest.Connect(
                source_nodes_I,
                target_nodes,
                syn_spec={
                    "synapse_model": "inhibitory_synapse",
                    "delay": delay,
                },
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
                is_within_module = target_idx == source_idx
                if is_within_module:
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
                    is_within_module=is_within_module,
                )
                if self.verbose:
                    print(
                        f"- Connected to target: {target_idx}, exc_dens: {exc_connection_density}, inh_dens: {inh_connection_density}"
                    )

    def _generate_connection_density(self, submodule_dict):
        connection_density = {i: [] for i, _ in submodule_dict.items()}

        for source_idx, source_nodes in submodule_dict.items():
            print("Source node:", source_idx)
            for target_idx, target_nodes in submodule_dict.items():
                if source_idx > target_idx:
                    continue
                print(target_idx, end=" ")

                n_connections = len(nest.GetConnections(source_nodes, target_nodes))
                n_connections_possible = len(source_nodes) * len(target_nodes)
                connection_density[source_idx] += [
                    n_connections / n_connections_possible
                ]
                if source_idx != target_idx:
                    connection_density[target_idx] += [
                        n_connections / n_connections_possible
                    ]

        return pd.DataFrame(connection_density).iloc[::-1]

    def plot_connection_density(self):
        connection_density = self._generate_connection_density(self.submodule_dict)
        sns.heatmap(
            connection_density,
            norm=LogNorm(vmin=10**-5, vmax=2 * 10**-1),
            cmap="gray",
        )
        return connection_density
