# -*- coding: utf-8 -*-
import math
import collections

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer


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
        self.rank = conf.graph.rank

        # define the aggregator.
        self.decentralized_aggregator = comm.get_aggregators(
            aggregator_type="decentralized", graph=conf.graph
        )
        self.world_aggregator = comm.get_aggregators(
            aggregator_type="centralized", graph=conf.graph
        )

        # define reducer.
        self.backend = conf.backend

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

        # init the dictionary.
        self.conf.tracking_dict = collections.defaultdict(list)

        # init for the evaluator.
        self.cosine_sim_fn = torch.nn.CosineSimilarity()

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        # first sync gradients and then apply the aggregated graidents.
        assert self.conf.is_centralized
        with kargs["timer"]("sync.get_data", epoch=self.conf.epoch_):
            # Get data.
            params, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=False
            )
            grads, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=True
            )
            flatten_params = TensorBuffer(params)
            flatten_grads = TensorBuffer(grads)

        with kargs["timer"]("sync.sync", epoch=self.conf.epoch_):
            # aggregate the gradients.
            flatten_grads.buffer = self.world_aggregator._agg(
                flatten_grads.buffer, op="avg", distributed=self.conf.distributed
            )

        with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
            # add weight decay.
            flatten_grads.buffer.add_(flatten_params.buffer, alpha=self.weight_decay)

            if self.momentum != 0:
                # apply momentum via the slow momentum buffer.
                momentum_buffer = self.momentum_buffer
                momentum_buffer.mul_(self.momentum).add_(
                    flatten_grads.buffer, alpha=1 - self.dampening
                )
                if self.nesterov:
                    to_be_applied = flatten_grads.buffer.add(
                        momentum_buffer, alpha=self.momentum
                    )
                else:
                    to_be_applied = momentum_buffer
            else:
                to_be_applied = flatten_grads.buffer

            # apply on the model params.
            flatten_params.buffer.add_(to_be_applied, alpha=-self.param_groups[0]["lr"])
            flatten_params.unpack(params)

        # get the # of transmitted bits.
        n_bits = get_n_bits(flatten_grads.buffer)
        return 1.0 * n_bits * math.ceil(math.log2(self.conf.graph.n_nodes))
