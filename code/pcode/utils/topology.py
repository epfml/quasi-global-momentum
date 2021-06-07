# -*- coding: utf-8 -*-
import functools
import itertools

from abc import abstractmethod, ABC
import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs
from scipy.sparse import coo_matrix
import networkx

import torch
import torch.distributed as dist

from pcode.utils.auxiliary import dict_parser


def configure_gpu(world_conf):
    # the logic of world_conf follows "a,b,c,d,e" where:
    # the block range from 'a' to 'b' with interval 'c' (and each integer will repeat for 'd' time);
    # the block will be repeated for 'e' times.
    start, stop, interval, local_repeat, block_repeat = [
        int(x) for x in world_conf.split(",")
    ]
    _block = [
        [x] * local_repeat for x in range(start, stop + 1, interval)
    ] * block_repeat
    world_list = functools.reduce(lambda a, b: a + b, _block)
    return world_list


class PhysicalLayout(object):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        self._n_mpi_process = n_mpi_process
        self._n_sub_process = n_sub_process
        self._world = self.configure_world(world, world_conf=None)
        self._comm_device = (
            torch.device("cpu")
            if comm_device == "cpu" or comm_device is None or not on_cuda
            else torch.device("cuda")
        )
        self._rank = rank
        self._on_cuda = on_cuda
        self.rng_state = kwargs["rng_state"]
        self.topology_conf = (
            dict_parser(kwargs["topology_conf"])
            if kwargs["topology_conf"] is not None
            else dict()
        )

    def configure_world(self, world, world_conf):
        if world is not None:
            world_list = world.split(",")
            assert self._n_mpi_process <= len(world_list)
        elif world_conf is not None:
            # the logic of world_conf follows "a,b,c,d,e" where:
            # the block range from 'a' to 'b' with interval 'c' (and each integer will repeat for 'd' time);
            # the block will be repeated for 'e' times.
            return configure_gpu(world_conf)
        else:
            raise RuntimeError(
                "you should at least make sure world or world_conf is not None."
            )
        return [int(l) for l in world_list]

    @property
    def device(self):
        return self.world[
            self._rank * self._n_sub_process : (self._rank + 1) * self._n_sub_process
        ]

    @property
    def on_cuda(self):
        return self._on_cuda

    @property
    def comm_device(self):
        return self._comm_device

    @property
    def rank(self):
        return self._rank

    @property
    def ranks(self):
        return list(range(self.n_nodes))

    @property
    def world(self):
        return self._world

    @property
    def simulated_delay(self):
        return (
            self._simulated_delay[self._rank]
            if self._simulated_delay is not None
            else 0
        )

    @property
    def in_neighborhood(self):
        row = self._mixing_matrix[self._rank, :]

        return {
            c: v for c, v in zip(range(len(row)), row) if (v != 0 or c == self._rank)
        }

    @property
    def out_neighborhood(self):
        column = self._mixing_matrix[:, self._rank]

        return {
            c: v
            for c, v in zip(range(len(column)), column)
            if (v != 0 or c == self._rank)
        }

    @property
    def n_nodes(self):
        return self._n_mpi_process

    @property
    def n_edges(self):
        # count it in a directed way.
        return np.sum(
            (self._mixing_matrix != 0).astype(float)
            - np.eye(*self._mixing_matrix.shape),
            dtype=int,
        )

    @property
    def rho(self):
        if self._n_mpi_process > 3:
            # Find largest real part
            eigenvalues, _ = eigs(self._mixing_matrix, k=2, which="LR")
            lambda2 = min(abs(i.real) for i in eigenvalues)

            # Find smallest real part
            eigenvalues, _ = eigs(self._mixing_matrix, k=1, which="SR")
            lambdan = eigenvalues[0].real
        else:
            eigenvals = sorted(
                np.linalg.eigvals(self._mixing_matrix.toarray()), reverse=True
            )
            lambda2 = eigenvals[1]
            lambdan = eigenvals[-1]
        return 1 - max(abs(lambda2), abs(lambdan))

    @property
    def scaling(self):
        return len(self.in_neighborhood)

    @property
    def matrix(self):
        return self._mixing_matrix


class CompleteGraph(PhysicalLayout):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(CompleteGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )
        self._mixing_matrix = np.ones((n_mpi_process, n_mpi_process)) / n_mpi_process


class RingGraph(PhysicalLayout):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(RingGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )
        _mixing_matrix, self._rho = self._compute_mixing_matrix_and_rho(n_mpi_process)
        self._mixing_matrix = _mixing_matrix.toarray()

    def _compute_mixing_matrix_and_rho(self, n):
        assert n > 2

        # create ring matrix
        diag_rows = np.array(
            [
                [1 / 3 for _ in range(n)],
                [1 / 3 for _ in range(n)],
                [1 / 3 for _ in range(n)],
            ]
        )
        positions = [-1, 0, 1]
        mixing_matrix = sp.sparse.spdiags(diag_rows, positions, n, n).tolil()

        mixing_matrix[0, n - 1] = 1 / 3
        mixing_matrix[n - 1, 0] = 1 / 3
        mixing_matrix = mixing_matrix.tocsr()

        if n > 3:
            # Find largest real part
            eigenvalues, _ = eigs(mixing_matrix, k=2, which="LR")
            lambda2 = min(abs(i.real) for i in eigenvalues)

            # Find smallest real part
            eigenvalues, _ = eigs(mixing_matrix, k=1, which="SR")
            lambdan = eigenvalues[0].real
        else:
            eigenvals = sorted(np.linalg.eigvals(mixing_matrix.toarray()), reverse=True)
            lambda2 = eigenvals[1]
            lambdan = eigenvals[-1]

        return mixing_matrix, 1 - max(abs(lambda2), abs(lambdan))


class TorusGraph(PhysicalLayout):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(TorusGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )
        self._mixing_matrix = self._define_graph(n_mpi_process)

    def _define_graph(self, n_mpi_process):
        # get proper _width and _height.
        m = int(np.sqrt(n_mpi_process))
        assert m * m == n_mpi_process

        while True:
            if n_mpi_process % m == 0:
                n = int(n_mpi_process / m)
                break
            else:
                m -= 1

        # define the graph.
        graph = networkx.generators.lattice.grid_2d_graph(m, n, periodic=True)

        # get the mixing matrix.
        mixing_matrix = networkx.adjacency_matrix(graph).toarray()
        for i in range(0, mixing_matrix.shape[0]):
            mixing_matrix[i][i] = 1
        mixing_matrix = mixing_matrix / np.sum(mixing_matrix, axis=1, keepdims=True)
        return mixing_matrix


class ExpanderGraph(PhysicalLayout):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(ExpanderGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )
        self._mixing_matrix = self._define_graph(n_mpi_process)

    def _define_graph(self, n_mpi_process):
        # define the graph.
        def modulo_inverse(i, p):
            for j in range(1, p):
                if (j * i) % p == 1:
                    return j

        # the number of processes needs to be a prime number.
        graph = networkx.generators.classic.cycle_graph(n_mpi_process)

        # get the mixing matrix.
        mixing_matrix = networkx.adjacency_matrix(graph).toarray()
        # for i in range(0, mixing_matrix.shape[0]):
        #     mixing_matrix[i][i] = 1
        mixing_matrix[0][0] = 1

        # connect with the inverse modulo p node.
        for i in range(1, mixing_matrix.shape[0]):
            mixing_matrix[i][modulo_inverse(i, n_mpi_process)] = 1

        mixing_matrix = mixing_matrix / 3
        return mixing_matrix


class MargulisExpanderGraph(PhysicalLayout):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(MargulisExpanderGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )
        self._mixing_matrix = self._define_graph(n_mpi_process)

    def _define_graph(self, n_mpi_process):
        base = int(np.sqrt(n_mpi_process))
        assert (base * base) == n_mpi_process

        # the degree of 8.
        graph = networkx.generators.expanders.margulis_gabber_galil_graph(base)

        # get the mixing matrix.
        mixing_matrix = networkx.adjacency_matrix(graph).toarray().astype(np.float)
        mixing_matrix[mixing_matrix > 1] = 1

        degrees = mixing_matrix.sum(axis=1)
        mixing_matrix = mixing_matrix.astype(np.float)
        for node in np.argsort(degrees)[::-1]:
            mixing_matrix[:, node][mixing_matrix[:, node] == 1] = 1.0 / degrees[node]
            mixing_matrix[node, :][mixing_matrix[node, :] == 1] = 1.0 / degrees[node]
            mixing_matrix[node, node] = (
                1 - np.sum(mixing_matrix[node, :]) + mixing_matrix[node, node]
            )
        return mixing_matrix


class SocialNetworkGraph(PhysicalLayout):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(SocialNetworkGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )

        # create mixing matrix.
        assert "mixing_stochastic" in self.topology_conf
        self._mixing_stochastic = self.topology_conf["mixing_stochastic"]
        self._graph_topology = self.topology_conf["graph_topology"]
        self._is_directed = (
            "directed" in self._graph_topology
            and "undirected" not in self._graph_topology
        )
        self._mixing_matrix = self._define_graph(n_mpi_process)

    def _define_graph(self, n_mpi_process):
        # define the graph.
        if "florentine_families" in self._graph_topology:
            assert n_mpi_process == 15
            graph = networkx.florentine_families_graph()
        elif "davis_southern_women" in self._graph_topology:
            assert n_mpi_process == 32
            graph = networkx.davis_southern_women_graph()
        elif "karate_club" in self._graph_topology:
            assert n_mpi_process == 34
            graph = networkx.karate_club_graph()
        elif "les_miserables" in self._graph_topology:
            assert n_mpi_process == 77
            graph = networkx.les_miserables_graph()

        # get the mixing matrix.
        adjacency_matrix = networkx.adjacency_matrix(graph).toarray().astype(
            np.float
        ) + np.eye(self.n_nodes)

        if not self._is_directed:
            degrees = adjacency_matrix.sum(axis=1)
            for node in np.argsort(degrees)[::-1]:
                adjacency_matrix[:, node][adjacency_matrix[:, node] == 1] = (
                    1.0 / degrees[node]
                )
                adjacency_matrix[node, :][adjacency_matrix[node, :] == 1] = (
                    1.0 / degrees[node]
                )
                adjacency_matrix[node, node] = (
                    1 - np.sum(adjacency_matrix[node, :]) + adjacency_matrix[node, node]
                )
            self._mixing_matrix = adjacency_matrix
        else:
            self._mixing_matrix = adjacency_matrix / np.sum(
                adjacency_matrix,
                axis=1 if self._mixing_stochastic == "row" else 0,
                keepdims=True,
            )
        return self._mixing_matrix


class RingExtGraph(PhysicalLayout):
    """
    Ring graph with skip connections to the most distant point in the graph.
    """

    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(RingExtGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )
        self._mixing_matrix = self._define_graph(n_mpi_process)

    def _define_graph(self, n_mpi_process):
        assert n_mpi_process > 3 and n_mpi_process % 2 == 0

        # define the graph.
        graph = networkx.generators.classic.cycle_graph(n_mpi_process)

        # get the mixing matrix.
        mixing_matrix = networkx.adjacency_matrix(graph).toarray()
        for i in range(0, mixing_matrix.shape[0]):
            mixing_matrix[i][i] = 1

        # connect with the most distant node.
        for i in range(0, mixing_matrix.shape[0]):
            mixing_matrix[i][(i + n_mpi_process // 2) % n_mpi_process] = 1

        mixing_matrix = mixing_matrix / 4
        return mixing_matrix


class UndirectedTimeVaryingSGPGraph(PhysicalLayout):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(UndirectedTimeVaryingSGPGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )

        # init the mixing matrix.
        self.p = itertools.cycle(
            [x for x in range(int(np.log(self.n_nodes - 1) / np.log(2)) + 1)]
        )
        self.update_topology()

    def update_topology(self):
        next_hop = 2 ** next(self.p)
        undirected_pairs = functools.reduce(
            lambda a, b: a + b,
            [
                [
                    (idx2, idx1)
                    for idx2 in range(self.n_nodes)
                    if idx2 == (idx1 + next_hop) % self.n_nodes
                ]
                for idx1 in range(self.n_nodes)
            ],
        )
        undirected_pairs += [(pair[1], pair[0]) for pair in undirected_pairs]
        undirected_pairs = list(set(undirected_pairs))
        self._mixing_matrix = (
            coo_matrix(
                ([1.0] * len(undirected_pairs), zip(*undirected_pairs)),
                shape=(self.n_nodes, self.n_nodes),
            )
        ).toarray() + np.eye(self.n_nodes)
        self._mixing_matrix /= np.sum(self._mixing_matrix, axis=1)
        return self._mixing_matrix


class TimeVaryingDynamicRandomRingGraph(RingGraph):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(TimeVaryingDynamicRandomRingGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )

        _mixing_matrix, self._rho = self._compute_mixing_matrix_and_rho(n_mpi_process)
        self._mixing_matrix = _mixing_matrix.toarray()
        self._original_mixing_matrix = self._mixing_matrix

        # init some pytorch specific group configuration.
        self.update_topology()

    def update_topology(self):
        indices = np.arange(self._mixing_matrix.shape[0])
        random_indices = (
            self.rng_state.permutation(indices)
            if self.rng_state is not None
            else np.random.permutation(indices)
        )
        _W_diag = np.diag(np.diag(self._original_mixing_matrix))
        _W = (self._original_mixing_matrix - _W_diag)[random_indices, :]
        self._mixing_matrix = _W + _W_diag
        return self._mixing_matrix


class TimeVaryingRandomRingGraph(RingGraph):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(TimeVaryingRandomRingGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )

        _mixing_matrix, self._rho = self._compute_mixing_matrix_and_rho(n_mpi_process)
        self._mixing_matrix = _mixing_matrix.toarray()
        self._original_mixing_matrix = self._mixing_matrix

        # init some pytorch specific group configuration.
        self.node_indices = list(range(self.n_nodes))
        self.update_topology()

    def update_topology(self):
        # get random indices and build mapping.
        indices = np.arange(self._original_mixing_matrix.shape[0])
        random_indices = (
            self.rng_state.permutation(indices)
            if self.rng_state is not None
            else np.random.permutation(indices)
        )
        mapping = dict(zip(indices, random_indices))

        # convert to sparse matrix.
        sparse_W = coo_matrix(self._original_mixing_matrix)
        values = []
        for row, col, data in zip(sparse_W.row, sparse_W.col, sparse_W.data):
            values.append((row, col, data))

        # build the mapping relationship
        mapping = list(map(lambda x: (mapping[x[0]], mapping[x[1]], x[2]), values))

        # create new mixing matrix.
        indices_tobe_assigned = [(x[0], x[1]) for x in mapping]
        new_values = [x[2] for x in mapping]
        self._mixing_matrix = coo_matrix(
            (new_values, zip(*indices_tobe_assigned)),
            shape=(self.n_nodes, self.n_nodes),
        ).toarray()
        return self._mixing_matrix


class TimeVaryingBipartiteExponentialOnePeerGraph(RingGraph):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(TimeVaryingBipartiteExponentialOnePeerGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )

        # init the mixing matrix.
        assert self.n_nodes % 2 == 0
        self.p = itertools.cycle(
            [x for x in range(1, int(np.log(self.n_nodes - 1) / np.log(2)) + 1)]
        )
        self.update_topology()

    def update_topology(self):
        next_hop = 2 ** next(self.p) - 1
        undirected_pairs = functools.reduce(
            lambda a, b: a + b,
            [
                [
                    (idx2, idx1)
                    for idx2 in range(self.n_nodes)
                    if idx2 == (idx1 + next_hop) % self.n_nodes
                ]
                for idx1 in range(self.n_nodes)
                if idx1 % 2 == 1
            ],
        )
        undirected_pairs += [(pair[1], pair[0]) for pair in undirected_pairs]
        undirected_pairs = list(set(undirected_pairs))
        self._mixing_matrix = (
            coo_matrix(
                ([1.0] * len(undirected_pairs), zip(*undirected_pairs)),
                shape=(self.n_nodes, self.n_nodes),
            )
        ).toarray() + np.eye(self.n_nodes)
        self._mixing_matrix /= np.sum(self._mixing_matrix, axis=1)
        return self._mixing_matrix


class TimeVaryingRandomPairGraph(RingGraph):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(TimeVaryingRandomPairGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )

        _mixing_matrix, self._rho = self._compute_mixing_matrix_and_rho(n_mpi_process)
        self._mixing_matrix = _mixing_matrix.toarray()
        self._original_mixing_matrix = self._mixing_matrix

        # init some pytorch specific group configuration.
        self.node_indices = list(range(self.n_nodes))
        self.update_topology()

    def update_topology(self):
        # get the shuffled node.
        shuffled_nodes = (
            self.rng_state.permutation(self.node_indices)
            if self.rng_state is not None
            else np.random.permutation(self.node_indices)
        )

        # create a list of tuple based on the tuple_size and shuffled_nodes.
        created_chunks = chunks(shuffled_nodes, n=2)

        # create the mixing matrix.
        # the elements in each chunk can communicate with each other.
        indices_tobe_assigned = functools.reduce(
            lambda a, b: a + b,
            [[(i, j) for i in chunk for j in chunk] for chunk in created_chunks],
        )
        self._mixing_matrix = coo_matrix(
            ([1.0 / 2] * len(indices_tobe_assigned), zip(*indices_tobe_assigned)),
            shape=(self.n_nodes, self.n_nodes),
        ).toarray()
        return self._mixing_matrix


class DirectedTimeVaryingOnePeerGraph(PhysicalLayout):
    """it is the topology used in the main experiments of
    Stochastic gradient push for distributed deep learning.
    """

    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(DirectedTimeVaryingOnePeerGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )
        assert "mixing_stochastic" in self.topology_conf
        self._mixing_stochastic = self.topology_conf["mixing_stochastic"]

        # init the mixing matrix.
        self.p = itertools.cycle(
            [x for x in range(int(np.log(self.n_nodes - 1) / np.log(2)) + 1)]
        )
        self.update_topology()

    def rotate_edge_direction(self, directed_pairs):
        if self._mixing_stochastic == "row":
            return [(b, a) for a, b in directed_pairs]
        else:
            return directed_pairs

    def update_topology(self):
        next_hop = 2 ** next(self.p)
        directed_pairs = functools.reduce(
            lambda a, b: a + b,
            [
                [
                    (idx2, idx1)
                    for idx2 in range(self.n_nodes)
                    if idx2 == (idx1 + next_hop) % self.n_nodes
                ]
                for idx1 in range(self.n_nodes)
            ],
        )
        directed_pairs = list(set(directed_pairs))

        self._mixing_matrix = (
            coo_matrix(
                ([1.0] * len(directed_pairs), zip(*directed_pairs)),
                shape=(self.n_nodes, self.n_nodes),
            )
        ).toarray() + np.eye(self.n_nodes)
        self._mixing_matrix /= np.sum(
            self._mixing_matrix,
            axis=1 if self._mixing_stochastic == "row" else 0,
            keepdims=True,
        )
        return self._mixing_matrix


class DirectedTimeVaryingTwoPeerGraph(PhysicalLayout):
    """it is the topology used (but not the main one) in the experiments of
    Stochastic gradient push for distributed deep learning.
    """

    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(DirectedTimeVaryingTwoPeerGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )
        assert "mixing_stochastic" in self.topology_conf
        self._mixing_stochastic = self.topology_conf["mixing_stochastic"]

        # init the mixing matrix.
        self.p = itertools.cycle(
            [x for x in range(int(np.log(self.n_nodes - 1) / np.log(2)) + 1)]
        )
        self.update_topology()

    def update_topology(self):
        next_hop = 2 ** next(self.p)
        next_next_hop = 2 * next_hop
        directed_pairs = functools.reduce(
            lambda a, b: a + b,
            [
                [
                    (idx2, idx1)
                    for idx2 in range(self.n_nodes)
                    if idx2 == (idx1 + next_hop) % self.n_nodes
                ]
                for idx1 in range(self.n_nodes)
            ],
        )
        directed_pairs = list(set(directed_pairs))
        directed_pairs += list(
            set(
                functools.reduce(
                    lambda a, b: a + b,
                    [
                        [
                            (idx2, idx1)
                            for idx2 in range(self.n_nodes)
                            if idx2 == (idx1 + next_next_hop) % self.n_nodes
                        ]
                        for idx1 in range(self.n_nodes)
                    ],
                )
            )
        )

        self._mixing_matrix = (
            coo_matrix(
                ([1.0] * len(directed_pairs), zip(*directed_pairs)),
                shape=(self.n_nodes, self.n_nodes),
            )
        ).toarray() + np.eye(self.n_nodes)
        self._mixing_matrix /= np.sum(
            self._mixing_matrix,
            axis=1 if self._mixing_stochastic == "row" else 0,
            keepdims=True,
        )
        return self._mixing_matrix


class DirectedOnePeerGraph(PhysicalLayout):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(DirectedOnePeerGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )
        assert "mixing_stochastic" in self.topology_conf
        self._mixing_stochastic = self.topology_conf["mixing_stochastic"]

        self.hop = int(
            self.topology_conf["graph_topology"].replace("directed_one_peer_", "")
        )
        assert self.hop >= 1

        # init the mixing matrix.
        self.update_topology()

    def update_topology(self):
        next_hop = self.hop
        directed_pairs = functools.reduce(
            lambda a, b: a + b,
            [
                [
                    (idx2, idx1)
                    for idx2 in range(self.n_nodes)
                    if idx2 == (idx1 + next_hop) % self.n_nodes
                ]
                for idx1 in range(self.n_nodes)
            ],
        )
        directed_pairs = list(set(directed_pairs))

        self._mixing_matrix = (
            coo_matrix(
                ([1.0] * len(directed_pairs), zip(*directed_pairs)),
                shape=(self.n_nodes, self.n_nodes),
            )
        ).toarray() + np.eye(self.n_nodes)
        self._mixing_matrix /= np.sum(
            self._mixing_matrix,
            axis=1 if self._mixing_stochastic == "row" else 0,
            keepdims=True,
        )
        return self._mixing_matrix


class DirectedRandomPeerGraph(PhysicalLayout):
    def __init__(
        self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
    ):
        super(DirectedRandomPeerGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank, **kwargs
        )
        assert "mixing_stochastic" in self.topology_conf
        self._mixing_stochastic = self.topology_conf["mixing_stochastic"]
        self.degree = int(
            self.topology_conf["graph_topology"].replace("directed_random_peer_", "")
        )
        assert self.degree >= 1

        # init the mixing matrix.
        self.update_topology()

    def update_topology(self):
        directed_pairs = []

        # different node has different node degree.
        for idx1 in range(self.n_nodes):
            cur_node_degree = self.rng_state.randint(1, self.degree)
            for _ in range(cur_node_degree):
                next_hop = self.rng_state.randint(1, self.n_nodes)

                idx2 = (idx1 + next_hop) % self.n_nodes

                # idx1 send to idx2.
                directed_pairs += [(idx2, idx1)] if idx2 != idx1 else []

        directed_pairs = list(set(directed_pairs))

        adjacent_matrix = (
            coo_matrix(
                ([1.0] * len(directed_pairs), zip(*directed_pairs)),
                shape=(self.n_nodes, self.n_nodes),
            )
        ).toarray() + np.eye(self.n_nodes)
        assert networkx.algorithms.components.is_strongly_connected(
            networkx.convert_matrix.from_numpy_matrix(adjacent_matrix).to_directed()
        )

        self._mixing_matrix = adjacent_matrix / np.sum(
            adjacent_matrix,
            axis=1 if self._mixing_stochastic == "row" else 0,
            keepdims=True,
        )
        return self._mixing_matrix


"""others."""


class Edge(object):
    def __init__(self, dest, src):
        self.src = src
        self.dest = dest
        self.process_group = dist.new_group([src, dest])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


"""entry level."""


def safety_check(graph_topology, optimizer_name, mixing_stochastic):
    if "directed" in graph_topology and "undirected" not in graph_topology:
        assert optimizer_name != "decentralized_sgd" and (
            (
                "decentralized_push_sum" in optimizer_name
                and "column" == mixing_stochastic
            )
            or (
                "decentralized_pull_sum" in optimizer_name
                and "row" == mixing_stochastic
            )
        )


def define_graph_topology(
    graph_topology,
    world,
    n_mpi_process,
    n_sub_process,
    comm_device,
    on_cuda,
    rank,
    rng_state,
    topology_conf,
):
    if graph_topology == "complete":
        graph_class = CompleteGraph
    elif graph_topology == "ring":
        graph_class = RingGraph
    elif graph_topology == "ringext":
        graph_class = RingExtGraph
    elif graph_topology == "torus":
        graph_class = TorusGraph
    elif graph_topology == "expander":
        graph_class = ExpanderGraph
    elif graph_topology == "margulis_expander":
        graph_class = MargulisExpanderGraph
    elif "social" in graph_topology:
        graph_class = SocialNetworkGraph
    elif graph_topology == "undirected_time_varying":
        graph_class = UndirectedTimeVaryingSGPGraph
    elif graph_topology == "time_varying_dynamic_random_ring":
        graph_class = TimeVaryingDynamicRandomRingGraph
    elif graph_topology == "time_varying_random_ring":
        graph_class = TimeVaryingRandomRingGraph
    elif graph_topology == "time_varying_random_pair":
        graph_class = TimeVaryingRandomPairGraph
    elif graph_topology == "time_varying_bipartite_one_peer":
        graph_class = TimeVaryingBipartiteExponentialOnePeerGraph
    elif graph_topology == "directed_time_varying_one_peer":
        graph_class = DirectedTimeVaryingOnePeerGraph
    elif graph_topology == "directed_time_varying_two_peer":
        graph_class = DirectedTimeVaryingTwoPeerGraph
    elif "directed_one_peer_" in graph_topology:
        graph_class = DirectedOnePeerGraph
    elif "directed_random_peer_" in graph_topology:
        graph_class = DirectedRandomPeerGraph
    else:
        raise NotImplementedError

    graph = graph_class(
        n_mpi_process=n_mpi_process,
        n_sub_process=n_sub_process,
        world=world,
        comm_device=comm_device,
        on_cuda=on_cuda,
        rank=rank,
        rng_state=rng_state,
        topology_conf=f"graph_topology={graph_topology},{topology_conf}"
        if topology_conf is not None
        else f"graph_topology={graph_topology}",
    )
    return graph
