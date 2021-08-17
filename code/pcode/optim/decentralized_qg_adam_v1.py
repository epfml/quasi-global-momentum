# -*- coding: utf-8 -*-
import copy
import math

import torch
from torch.optim.optimizer import Optimizer, required

from pcode.utils.auxiliary import dict_parser
import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer


"""original implementation."""


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
        self.momentum_buffer1 = copy.deepcopy(flatten_params)
        self.momentum_buffer1.buffer = torch.zeros_like(flatten_params.buffer)
        self.momentum_buffer2 = copy.deepcopy(flatten_params)
        self.momentum_buffer2.buffer = torch.zeros_like(flatten_params.buffer)

        # init the conf for slow buffer.
        self.conf.slow_buffer_conf_ = dict_parser(conf.slow_buffer_conf)

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
            tmp_flatten_params = flatten_params.buffer.clone()

            # Get gradients.
            grads, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=True
            )
            flatten_grads = TensorBuffer(grads)

            # Normalize gradients.
            flatten_grads.buffer = self.normalize_tensor(
                flatten_grads, normalize_style="filter_wise"
            )

        with kargs["timer"]("sync.apply", epoch=self.conf.epoch_):
            exp_avg = self.momentum_buffer1.buffer.clone()
            exp_avg_sq = self.momentum_buffer2.buffer.clone()

            # update the bias correction terms.
            bias_correction1 = 1 - self.beta1 ** (self.conf.local_index + 1)
            bias_correction2 = 1 - self.beta2 ** (self.conf.local_index + 1)

            # add weight decay for adam.
            flatten_grads.buffer.add_(flatten_params.buffer, alpha=self.weight_decay)

            # decay the first and second moment running average coefficient
            exp_avg.mul_(self.beta1).add_(flatten_grads.buffer, alpha=1 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(
                flatten_grads.buffer, flatten_grads.buffer, value=1 - self.beta2
            )
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

            # apply the gradients
            step_size = self.param_groups[0]["lr"] / bias_correction1
            flatten_params.buffer.addcdiv_(exp_avg, denom, value=-step_size)

        # sync model params.
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

        # update the momentum through model update.
        with kargs["timer"]("sync.sync_buffer", epoch=self.conf.epoch_):
            # update the progress buffer on the virtual sequence.
            flatten_updates = copy.deepcopy(flatten_grads)
            flatten_updates.buffer = tmp_flatten_params - flatten_params.buffer
            flatten_updates_buffer = self.normalize_tensor(
                flatten_updates, normalize_style="global_wise"
            )

            self.momentum_buffer1.buffer.mul_(self.beta1).add_(
                flatten_updates_buffer, alpha=1 - self.beta1
            )
            self.momentum_buffer2.buffer.mul_(self.beta2).add_(
                flatten_updates_buffer * flatten_updates_buffer,
                alpha=1 - self.beta2,
            )

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_grads.buffer)
        return n_bits

    def get_tensor_norm(self, tensor, norm_type=2, norm_scheme="layer_wise"):
        if norm_scheme == "layer_wise":
            return tensor.norm(p=norm_type)
        elif norm_scheme == "filter_wise":
            if len(tensor.shape) > 2:  # for conv layer: each output channel
                tensor_norm = tensor.norm(p=2, dim=[2, 3], keepdim=True).norm(
                    p=2, dim=1, keepdim=True
                )
            elif len(tensor.shape) == 2:  # for linear layer: each output channel.
                tensor_norm = tensor.norm(p=2, dim=0, keepdim=True)
            else:
                tensor_norm = tensor.norm(p=2)
            return tensor_norm

    def normalize_tensor(self, flatten_grads, normalize_style):
        # maybe normalize gradient.
        eps = (
            self.conf.slow_buffer_conf_["normalization_eps"]
            if "normalization_eps" in self.conf.slow_buffer_conf_
            else 1e-8
        )

        if normalize_style == "global_wise":
            flatten_grads_buffer = flatten_grads.buffer / (
                flatten_grads.buffer.norm(p=2) + eps
            )
        elif (normalize_style == "layer_wise") or (normalize_style == "filter_wise"):
            for grad in flatten_grads:
                grad_norm = self.get_tensor_norm(
                    grad,
                    norm_type=2,
                    norm_scheme=normalize_style,
                )
                grad.copy_(grad.data / (grad_norm + eps))
            flatten_grads_buffer = flatten_grads.buffer
        else:
            flatten_grads_buffer = flatten_grads.buffer
        return flatten_grads_buffer


def get_slow_buffer_step(slow_buffer_conf, current_epoch) -> int:
    if slow_buffer_conf is None or "local_step" not in slow_buffer_conf:
        return 1
    else:
        return int(slow_buffer_conf["local_step"])
