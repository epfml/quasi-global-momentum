# -*- coding: utf-8 -*-
import copy
import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
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
        model=None,
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

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        r"""here we want to use gradient tracking techniques.
            \xx_{t+1}^i & = \sum_{j=1}^n w_{ij} \xx_t^j - \eta \yy_t^i \,,
            \yy_{t+1}^i & = \sum_{j=1}^n w_{ij} \yy_t^j + \nabla f_i (\xx_{t+1}^i) - \nabla f_i(\xx_t^i) \,.

        In case of using momentum, the \yy will be first aggregated before applying the momentum.
        """
        assert not self.conf.is_centralized

        # get params.
        with kargs["timer"]("sync.get_params", epoch=self.conf.epoch_):
            # first get and flatten all params.
            params, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=False
            )
            flatten_params = TensorBuffer(params)

        # get grads.
        with kargs["timer"]("sync.get_grads", epoch=self.conf.epoch_):
            # only apply the weight decay.
            utils.apply_gradient(
                self.param_groups,
                self.state,
                apply_grad_to_model=False,
                apply_weight_decay=True,
                only_apply_weight_decay=True,
            )
            # first get and flatten all grads.
            grads, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=True
            )
            flatten_grads = TensorBuffer(grads)

            # create or update \yy_t.
            if not hasattr(self, "flatten_y"):
                self.flatten_y = copy.deepcopy(flatten_grads)
            else:
                self.flatten_y.buffer += (
                    flatten_grads.buffer - self.flatten_old_g.buffer
                )
            self.flatten_old_g = copy.deepcopy(flatten_grads)

        # sync.
        with kargs["timer"]("sync.sync", epoch=self.conf.epoch_):
            # prepare the sync.
            tensor_to_be_sync = torch.cat(
                [flatten_params.buffer, self.flatten_y.buffer]
            )

            # then sync.
            if "time_varying" in self.conf.graph_topology:
                self.conf.graph.update_topology()
                self.decentralized_aggregator.update_neighbors_info()

            tensor_to_be_sync = self.decentralized_aggregator._agg(
                tensor_to_be_sync, op="weighted"
            )

            # split the synced tensor.
            tensor_size = int(len(tensor_to_be_sync) / 2)
            flatten_params.buffer = tensor_to_be_sync[:tensor_size]
            update_y = tensor_to_be_sync[tensor_size:]

        # update tensor in place.
        with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
            # update params in place.
            flatten_params.unpack(params)
            # update grads using 'previous y' in place.
            self.flatten_y.unpack(grads)
            # apply grads to params (w/o applying weight decay).
            utils.apply_gradient(
                self.param_groups,
                self.state,
                apply_grad_to_model=True,
                apply_weight_decay=False,
                only_apply_weight_decay=False,
            )
            # update y using aggregated y from neighbors and will be used for next round.
            self.flatten_y.buffer = update_y.clone()

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_params.buffer) * 2
        return 1.0 * n_bits * int(len(self.decentralized_aggregator.out_neighbors_info))
