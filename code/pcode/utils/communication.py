# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist

"""some auxiliary functions for communication."""


def global_average(sum, count, on_cuda=True):
    def helper(array):
        array = torch.FloatTensor(array)
        array = array.cuda() if on_cuda else array
        dist.all_reduce(array, op=dist.ReduceOp.SUM)
        all_sum, all_count = array
        if all_count == 0:
            return 0
        else:
            return all_sum / all_count

    avg = helper([sum, count])
    return avg


def elementwise_min(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return tensor


def broadcast(tensor, src):
    return dist.broadcast(tensor, src=src)


"""some aggregation functions."""


def _get_data(param_groups, idx, is_get_grad):
    # Define the function to get the data.
    # when we create the param_group, each group only has one param.
    if is_get_grad:
        return param_groups[idx]["params"][0].grad
    else:
        if param_groups[idx]["params"][0].requires_grad:
            return param_groups[idx]["params"][0]
        else:
            return None


def _get_shape(param_groups, idx):
    return param_groups[idx]["param_size"], param_groups[idx]["nelement"]


def get_data(param_groups, param_names, is_get_grad=True):
    data, shapes = [], []

    for idx, param_name in param_names:
        _data = _get_data(param_groups, idx, is_get_grad)

        if _data is not None:
            data.append(_data)
            shapes.append(_get_shape(param_groups, idx))
    return data, shapes


def flatten(tensors, shapes=None, use_cuda=True):
    # init and recover the shapes vec.
    pointers = [0]
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

    # flattening.
    current_device = tensors[0].device
    target_device = tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu"
    vec = torch.empty(pointers[-1], device=target_device)

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = (
            tensor.data.view(-1).to(device=target_device)
            if current_device != target_device
            else tensor.data.view(-1)
        )
    return vec


def unflatten(tensors, synced_tensors, shapes):
    pointer = 0

    for tensor, shape in zip(tensors, shapes):
        param_size, nelement = shape
        tensor.data[:] = synced_tensors[pointer : pointer + nelement].view(param_size)
        pointer += nelement


"""auxiliary."""


def recover_device(data, device=None):
    if device is not None:
        return data.to(device)
    else:
        return data


"""main aggregators."""


class Aggregation(object):
    """Aggregate udpates / models from different processes."""

    def _agg(self, data, op):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        raise NotImplementedError

    def agg_model(self, model, op):
        """Aggregate models by model weight.
        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            param.data = self._agg(param.data, op=op)

    def agg_grad(self, model, op):
        """Aggregate models gradients.
        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            grad = self._agg(param.grad.data, op=op)
            param.grad.data = grad


class CentralizedAggregation(Aggregation):
    """Aggregate udpates / models from different processes."""

    def __init__(self, rank, world, neighbors_info):
        # init
        self.rank = rank

        # define the dist group.
        neighbor_ranks = list(neighbors_info.keys())

        if len(neighbor_ranks) == 0:
            self.group = None
        else:
            self.group = dist.new_group(neighbor_ranks)

        # get the world size from the view of the current rank.
        self.world_size = float(len(neighbor_ranks))

    def _agg(
        self,
        data,
        op=None,
        distributed=True,
        communication_scheme="all_reduce",
        async_op=False,
        **kargs,
    ):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        if not distributed:
            return data

        # do the real sync.
        if communication_scheme == "all_reduce":
            if op == "avg":
                req = dist.all_reduce(
                    data, op=dist.ReduceOp.SUM, group=self.group, async_op=async_op
                )
            elif op == "sum":
                req = dist.all_reduce(
                    data, op=dist.ReduceOp.SUM, group=self.group, async_op=async_op
                )
            else:
                raise NotImplementedError

            if async_op:
                # it would be dangerous to use `avg` operation with async.
                return data, req
            else:
                if op == "avg":
                    return data / self.world_size
                else:
                    return data
        elif communication_scheme == "reduce":
            if op == "sum":
                req = dist.reduce(
                    data,
                    dst=kargs["dst_rank"],
                    op=dist.ReduceOp.SUM,
                    group=self.group,
                    async_op=async_op,
                )
            else:
                raise NotImplementedError

            if async_op:
                return data, req
            else:
                if op == "sum":
                    return data
                else:
                    raise NotImplementedError
        elif communication_scheme == "all_gather":
            gathered_list = [
                torch.empty_like(data) for _ in range(int(self.world_size))
            ]
            req = dist.all_gather(
                gathered_list, data, group=self.group, async_op=async_op
            )
            if async_op:
                return gathered_list, req
            else:
                return gathered_list
        else:
            raise NotImplementedError

    def complete_wait(self, req):
        req.wait()


class DecentralizedAggregation(Aggregation):
    """Aggregate updates in a decentralized manner."""

    def __init__(self, rank, world, graph):
        # init
        self.rank = rank
        self.world = world
        self.world_size = float(len(world))
        self.graph = graph

        # get neighbors information in terms of
        self._update_counter = 0
        self.update_neighbors_info()

    @property
    def update_counter(self):
        return self._update_counter

    def update_neighbors_info(self):
        self._update_counter += 1

        # update the neighbors for in_peers.
        self.in_neighbors_info = self.graph.in_neighborhood
        self.in_neighbor_ranks = [
            neighbor_rank
            for neighbor_rank in self.graph.in_neighborhood.keys()
            if neighbor_rank != self.rank
        ]

        # update the neighbors for out_peers.
        self.out_neighbors_info = self.graph.out_neighborhood
        self.out_neighbor_ranks = [
            neighbor_rank
            for neighbor_rank in self.graph.out_neighborhood.keys()
            if neighbor_rank != self.rank
        ]

    def _agg(self, data, op, force_wait=True):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, `weighted`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        # Create some tensors to host the values from neighborhood.
        in_local_data = {i: torch.empty_like(data) for i in self.in_neighbor_ranks}
        data_clone = data.clone()
        in_local_data[self.rank] = data_clone

        # async send data.
        reqs = []
        for node_rank in self.in_neighbor_ranks:
            reqs.append(dist.irecv(tensor=in_local_data[node_rank], src=node_rank))
        for node_rank in self.out_neighbor_ranks:
            reqs.append(dist.isend(tensor=data_clone, dst=node_rank))

        # wait until finish.
        if force_wait:
            self.complete_wait(reqs)

            # Aggregate local_data
            if op == "avg":
                return sum(in_local_data.values()) / len(in_local_data)
            elif op == "weighted":
                return sum(
                    [
                        tensor * self.in_neighbors_info[rank]
                        for rank, tensor in in_local_data.items()
                    ]
                )
            elif op == "get_raw_sync_data":
                return in_local_data
            elif op == "get_raw_sync_data_and_weighted_data":
                return (
                    in_local_data,
                    sum(
                        [
                            tensor * self.in_neighbors_info[rank]
                            for rank, tensor in in_local_data.items()
                        ]
                    ),
                )
            else:
                raise NotImplementedError("op {} is not supported yet.".format(op))
        else:
            if op == "get_raw_sync_data":
                return reqs, in_local_data
            else:
                raise NotImplementedError("op {} is not supported yet.".format(op))

    def complete_wait(self, reqs):
        for req in reqs:
            req.wait()


def get_aggregators(aggregator_type, graph):
    if "centralized" == aggregator_type:
        return CentralizedAggregation(
            rank=graph.rank,
            world=graph.ranks,
            neighbors_info=dict((rank, 1.0 / graph.n_nodes) for rank in graph.ranks),
        )
    elif "decentralized" == aggregator_type:
        return DecentralizedAggregation(rank=graph.rank, world=graph.ranks, graph=graph)
    else:
        raise NotImplementedError
