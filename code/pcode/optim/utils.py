# -*- coding: utf-8 -*-
import math
import copy
import threading

import torch

from pcode.utils.tensor_buffer import TensorBuffer
import pcode.utils.communication as comm


"""common utilities"""


def apply_gradient(
    param_groups,
    state,
    apply_grad_to_model=True,
    apply_lr_to_grad=False,
    apply_weight_decay=True,
    only_apply_weight_decay=False,
):
    for group in param_groups:
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        nesterov = group["nesterov"]

        for p in group["params"]:
            if p.grad is None:
                continue
            d_p = p.grad.data

            # get param_state
            param_state = state[p]

            # add weight decay.
            if weight_decay != 0 and apply_weight_decay:
                d_p.add_(p.data, alpha=weight_decay)
            if only_apply_weight_decay:
                continue

            # apply the momentum.
            if momentum != 0:
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            if apply_grad_to_model:
                p.data.add_(d_p, alpha=-group["lr"])
            elif apply_lr_to_grad:
                p.grad.data = d_p * group["lr"]
            else:
                p.grad.data = d_p


def apply_adapative_gradients(param_groups, state):
    for group in param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError(
                    "Adam does not support sparse gradients, please consider SparseAdam instead"
                )
            param_state = state[p]

            # State initialization
            if len(param_state) == 0:
                param_state["step"] = 0
                # Exponential moving average of gradient values
                param_state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                param_state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            exp_avg, exp_avg_sq = param_state["exp_avg"], param_state["exp_avg_sq"]
            beta1, beta2 = group["betas"]

            param_state["step"] += 1
            bias_correction1 = 1 - beta1 ** param_state["step"]
            bias_correction2 = 1 - beta2 ** param_state["step"]

            if group["weight_decay"] != 0:
                grad = grad.add(p, alpha=group["weight_decay"])

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

            # get adaptive stepsize.
            step_size = group["lr"] / bias_correction1

            # apply adaptive gradients to model parameters.
            p.data.addcdiv_(exp_avg, denom, value=-step_size)


def recover_params(
    param_groups, param_names, rank=None, neighbor_hat_params=None, get_hat_params=True
):
    # get flattened params.
    params, _ = comm.get_data(param_groups, param_names, is_get_grad=False)
    flatten_params = TensorBuffer(params)

    if get_hat_params:
        assert neighbor_hat_params is not None and rank is not None
        # recover the hat_params.
        flatten_hat_params = TensorBuffer(params)
        flatten_hat_params.buffer.data[:] = neighbor_hat_params[rank].buffer
        return params, flatten_params, flatten_hat_params
    else:
        return params, flatten_params


def update_params_from_neighbor(
    neighbor_hat_params, flatten_params, consensus_stepsize, self_rank
):
    flatten_params.buffer += consensus_stepsize * (
        neighbor_hat_params["memory"].buffer - neighbor_hat_params[self_rank].buffer
    )


def get_cosine_similarity(cosine_sim_fn, flatten_grads, momentum_buffer):
    # init.
    _momentum_buffer = copy.deepcopy(flatten_grads)
    _momentum_buffer.buffer = momentum_buffer.clone()

    # compute the cosine similarity over tensors.
    values = []
    for grad, momentum in zip(flatten_grads, _momentum_buffer):
        value = cosine_sim_fn(grad.view(1, -1), momentum.view(1, -1))
        values.append(value.item())
    return values


"""utilities for parallel choco."""


class HelperThread(threading.Thread):
    def __init__(self, name, func, *args, **kargs):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func

        # task-related.
        self.args = args
        self.kargs = kargs

    def run(self):
        self.func(**self.kargs)


def join_thread(thread):
    if thread is None:
        return False
    thread.join()
    return True
