# -*- coding: utf-8 -*-
import copy
import math

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
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

        # initialize few-step-back params for slowmo.
        params, _ = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        self.slowmo_params = TensorBuffer(params)
        self.slowmo_buffer = copy.deepcopy(self.slowmo_params)
        self.slowmo_buffer.buffer = torch.zeros_like(self.slowmo_buffer.buffer)

        # init for slowmo steps.
        if conf.slowmo_step_conf is not None and conf.slowmo_step_conf != "None":
            self.conf.slowmo_step_conf_ = dict_parser(conf.slowmo_step_conf)
        else:
            self.conf.slowmo_step_conf_ = None

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
            utils.apply_gradient(
                self.param_groups, self.state, apply_grad_to_model=True
            )

        with kargs["timer"]("sync.get_data", epoch=self.conf.epoch_):
            # first get and flatten all params.
            params, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=False
            )
            flatten_params = TensorBuffer(params)

        # sync.
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

        # use SlowMo.
        with kargs["timer"]("sync.use_slowmo", epoch=self.conf.epoch_):
            # in the case of using slowMo.
            slowmo_tau = get_slowmo_step(
                self.conf.slowmo_step_conf_, current_epoch=self.conf.epoch_
            )
            if slowmo_tau > 0 and self.conf.local_index % slowmo_tau == 0:
                # get exact averaged params.
                flatten_params = TensorBuffer(params)
                complete_sync_params = self.world_aggregator._agg(
                    flatten_params.buffer, op="avg", distributed=self.conf.distributed
                )

                # update slow momentum.
                self.slowmo_buffer.buffer = (
                    self.conf.slowmo_step_conf_["slowmo_beta"]
                    * self.slowmo_buffer.buffer
                    + (self.slowmo_params.buffer - complete_sync_params)
                    / self.param_groups[0]["lr"]
                )

                # update outer iterates.
                self.slowmo_params.buffer = (
                    self.slowmo_params.buffer
                    - (
                        self.conf.slowmo_step_conf_["slowmo_alpha"]
                        if "slowmo_alpha" in self.conf.slowmo_step_conf_
                        else 1.0
                    )
                    * self.param_groups[0]["lr"]
                    * self.slowmo_buffer.buffer
                )

                # update model in place.
                self.slowmo_params.unpack(params)

        # get the # of transmitted bits.
        n_bits = get_n_bits(flatten_params.buffer)
        return n_bits * int(len(self.decentralized_aggregator.out_neighbors_info))


def get_slowmo_step(slowmo_conf_, current_epoch) -> int:
    if slowmo_conf_ is None or "slowmo_tau" not in slowmo_conf_:
        return 0
    else:
        return int(slowmo_conf_["slowmo_tau"])
