# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcode.utils.misc import onehot


"""borrowed from https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py """


def _is_long(x):
    if hasattr(x, "data"):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy(
    inputs,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    smooth_eps=None,
    smooth_dist=None,
    from_logits=True,
):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(
                inputs, target, weight, ignore_index=ignore_index, reduction=reduction
            )
        else:
            return F.nll_loss(
                inputs, target, weight, ignore_index=ignore_index, reduction=reduction
            )

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1.0 - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(
        self,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        smooth_eps=None,
        smooth_dist=None,
        from_logits=True,
    ):
        super(CrossEntropyLoss, self).__init__(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_eps=None, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        if smooth_eps is None:
            smooth_eps = self.smooth_eps
        return cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            smooth_eps=smooth_eps,
            smooth_dist=smooth_dist,
            from_logits=self.from_logits,
        )


def binary_cross_entropy(
    inputs, target, weight=None, reduction="mean", smooth_eps=None, from_logits=False
):
    """cross entropy loss, with support for label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target = target.float()
        target.add_(smooth_eps).div_(2.0)
    if from_logits:
        return F.binary_cross_entropy_with_logits(
            inputs, target, weight=weight, reduction=reduction
        )
    else:
        return F.binary_cross_entropy(
            inputs, target, weight=weight, reduction=reduction
        )


def binary_cross_entropy_with_logits(
    inputs, target, weight=None, reduction="mean", smooth_eps=None, from_logits=True
):
    return binary_cross_entropy(
        inputs, target, weight, reduction, smooth_eps, from_logits
    )


class BCELoss(nn.BCELoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        smooth_eps=None,
        from_logits=False,
    ):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.smooth_eps = smooth_eps
        self.from_logits = from_logits

    def forward(self, input, target):
        return binary_cross_entropy(
            input,
            target,
            weight=self.weight,
            reduction=self.reduction,
            smooth_eps=self.smooth_eps,
            from_logits=self.from_logits,
        )


class BCEWithLogitsLoss(BCELoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        smooth_eps=None,
        from_logits=True,
    ):
        super(BCEWithLogitsLoss, self).__init__(
            weight,
            size_average,
            reduce,
            reduction,
            smooth_eps=smooth_eps,
            from_logits=from_logits,
        )
