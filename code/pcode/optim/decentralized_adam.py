# -*- coding: utf-8 -*-
import math
import torch
from torch.optim.optimizer import Optimizer, required

import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer


class Adam(Optimizer):
    def __init__(
        self, params, lr=required, momentum=0, nesterov=False, weight_decay=0, conf=None
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.rank = conf.graph.rank

        self.weight_decay = self.conf.weight_decay
        self.beta1 = self.conf.adam_beta_1
        self.beta2 = self.conf.adam_beta_2
        self.eps = self.conf.adam_eps

        # init parameters.
        for group in self.param_groups:
            group.setdefault("betas", [self.conf.adam_beta_1, self.conf.adam_beta_2])
            group.setdefault("eps", self.conf.adam_eps)

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

        # initialize few-step-back params for slowmo.
        params, _ = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params)
        self.momentum_buffer1 = torch.zeros_like(flatten_params.buffer)
        self.momentum_buffer2 = torch.zeros_like(flatten_params.buffer)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)

    def step(self, closure=None, **kargs):
        assert not self.conf.is_centralized
        with kargs["timer"]("sync.get_data", epoch=self.conf.epoch_):
            # Get parameters.
            params, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=False
            )
            flatten_params = TensorBuffer(params)

            # Get gradients
            grads, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=True
            )
            flatten_grads = TensorBuffer(grads)

        # update the bias correction terms.
        bias_correction1 = 1 - self.beta1 ** (self.conf.local_index + 1)
        bias_correction2 = 1 - self.beta2 ** (self.conf.local_index + 1)

        # add weight decay for adam.
        flatten_grads.buffer.add_(flatten_params.buffer, alpha=self.weight_decay)

        # decay the first and second moment running average coefficient
        exp_avg = self.momentum_buffer1
        exp_avg_sq = self.momentum_buffer2
        exp_avg.mul_(self.beta1).add_(flatten_grads.buffer, alpha=1 - self.beta1)
        exp_avg_sq.mul_(self.beta2).addcmul_(
            flatten_grads.buffer, flatten_grads.buffer, value=1 - self.beta2
        )
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

        # apply the gradients
        step_size = self.param_groups[0]["lr"] / bias_correction1
        flatten_params.buffer.addcdiv_(exp_avg, denom, value=-step_size)

        # periodic sync.
        with kargs["timer"]("sync.sync", epoch=self.conf.epoch_):
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
            flatten_params.unpack(params)

        # get the # of transmitted bits.
        n_bits = get_n_bits(flatten_params.buffer)
        return n_bits * int(len(self.decentralized_aggregator.out_neighbors_info))
