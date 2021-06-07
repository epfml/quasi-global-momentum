# -*- coding: utf-8 -*-
import math
import collections

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.auxiliary import dict_parser


"""slightly improve the original implementation of D2,
by considering the changing learning rate (it does not include the momentum).
"""


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        conf=None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.dampening = dampening

        # store the whole training arguments.
        self.conf = conf

        # define the aggregator.
        self.decentralized_aggregator = comm.get_aggregators(
            aggregator_type="decentralized", graph=conf.graph
        )
        self.world_aggregator = comm.get_aggregators(
            aggregator_type="centralized", graph=conf.graph
        )

        # define sorted param names.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )

        # init the momentum buffer.
        params, _ = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params)
        self.momentum_buffer = torch.zeros_like(flatten_params.buffer)
        self.virtual_seq_buffer = torch.zeros_like(flatten_params.buffer)

        # init the conf for slow buffer.
        self.conf.slow_buffer_conf_ = dict_parser(conf.slow_buffer_conf)

        # init the dictionary.
        self.conf.tracking_dict = collections.defaultdict(list)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        assert not self.conf.is_centralized
        with kargs["timer"]("sync.get_data", epoch=self.conf.epoch_):
            # first get and flatten all params.
            params, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=False
            )
            grads, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=True
            )
            flatten_params = TensorBuffer(params)
            flatten_grads = TensorBuffer(grads)

        with kargs["timer"]("sync.apply_grads", epoch=self.conf.epoch_):
            # add weight decay.
            flatten_grads.buffer.add_(flatten_params.buffer, alpha=self.weight_decay)

            # D2 update step.
            prev_flatten_params = flatten_params.buffer.clone()

            if self.conf.local_index == 0:
                flatten_params.buffer.add_(
                    flatten_grads.buffer, alpha=-self.param_groups[0]["lr"]
                )
            else:
                flatten_params.buffer = flatten_params.buffer - self.param_groups[0][
                    "lr"
                ] * (self.params_diff + flatten_grads.buffer - self.prev_flatten_grads)
            self.prev_flatten_grads = flatten_grads.buffer.clone()

        # sync.
        with kargs["timer"]("sync.sync_params", epoch=self.conf.epoch_):
            # prepare the sync.
            if self.conf.comm_device == "cpu":
                flatten_params.buffer.cpu().detach_()

            # then sync.
            if "time_varying" in self.conf.graph_topology:
                self.conf.graph.update_topology()
                self.decentralized_aggregator.update_neighbors_info()
            flatten_params.buffer = self.decentralized_aggregator._agg(
                flatten_params.buffer, op="weighted"
            )
            self.params_diff = (
                prev_flatten_params - flatten_params.buffer
            ) / self.param_groups[0]["lr"]
            flatten_params.unpack(params)

        # get the # of transmitted bits.
        n_bits = get_n_bits(flatten_params.buffer)
        return n_bits * int(
            len(self.decentralized_aggregator.out_neighbors_info)
        ) + n_bits * math.ceil(math.log2(self.conf.graph.n_nodes))
