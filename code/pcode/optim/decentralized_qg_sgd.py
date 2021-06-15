# -*- coding: utf-8 -*-
import collections

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.auxiliary import dict_parser


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
            tmp_flatten_params = flatten_params.buffer.clone()

        with kargs["timer"]("sync.apply_grads", epoch=self.conf.epoch_):
            # add weight decay.
            flatten_grads.buffer.add_(flatten_params.buffer, alpha=self.weight_decay)

            if self.momentum != 0:
                # apply momentum via the slow momentum buffer.
                momentum_buffer = self.momentum_buffer.clone()
                momentum_buffer.mul_(self.momentum).add_(
                    flatten_grads.buffer, alpha=1 - self.dampening
                )
                if self.nesterov:
                    to_be_applied = flatten_grads.buffer.add(
                        momentum_buffer, alpha=self.momentum
                    )
                else:
                    to_be_applied = momentum_buffer

            # apply on the model params (and we may clip the update).
            flatten_params.buffer.add_(-to_be_applied * self.param_groups[0]["lr"])

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
            flatten_params.unpack(params)

        with kargs["timer"]("sync.sync_buffer", epoch=self.conf.epoch_):
            # update the progress buffer on the virtual sequence.
            self.virtual_seq_buffer.add_(
                (tmp_flatten_params - flatten_params.buffer)
                / self.param_groups[0]["lr"]
            )

            # update the gossip buffer.
            local_step = get_slow_buffer_step(
                self.conf.slow_buffer_conf_, current_epoch=self.conf.epoch_
            )
            assert local_step > 0
            if local_step > 0 and self.conf.local_index % local_step == 0:
                opp_mu = (
                    0.1
                    if "factor" not in self.conf.slow_buffer_conf_
                    else self.conf.slow_buffer_conf_["factor"]
                )
                self.momentum_buffer.mul_(1 - opp_mu).add_(
                    self.virtual_seq_buffer / local_step, alpha=opp_mu
                )
                self.virtual_seq_buffer = torch.zeros_like(self.virtual_seq_buffer)

        # get the # of transmitted bits.
        n_bits = get_n_bits(flatten_params.buffer)
        return n_bits * int(len(self.decentralized_aggregator.out_neighbors_info))


def get_slow_buffer_step(slow_buffer_conf, current_epoch) -> int:
    if slow_buffer_conf is None or "local_step" not in slow_buffer_conf:
        return 1
    else:
        return int(slow_buffer_conf["local_step"])
