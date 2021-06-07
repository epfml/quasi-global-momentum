# -*- coding: utf-8 -*-
import os
import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from parameters import get_args

import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.create_scheduler as create_scheduler

import pcode.utils.topology as topology
import pcode.utils.checkpoint as checkpoint
import pcode.utils.op_paths as op_paths
import pcode.utils.stat_tracker as stat_tracker
import pcode.utils.logging as logging
from pcode.utils.timer import Timer


def init_distributed_world(conf, backend):
    if backend == "mpi":
        dist.init_process_group("mpi")
    elif backend == "nccl" or backend == "gloo":
        # init the process group.
        _tmp_path = os.path.join(conf.checkpoint, "tmp", conf.timestamp)
        op_paths.build_dirs(_tmp_path)

        dist_init_file = os.path.join(_tmp_path, "dist_init")

        torch.distributed.init_process_group(
            backend=backend,
            init_method="file://" + os.path.abspath(dist_init_file),
            timeout=datetime.timedelta(seconds=120),
            world_size=conf.n_mpi_process,
            rank=conf.local_rank,
        )
    else:
        raise NotImplementedError


def main(conf):
    try:
        init_distributed_world(conf, backend=conf.backend)
        conf.distributed = True and conf.n_mpi_process > 1
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    # init the config.
    init_config(conf)

    # define the timer for different operations.
    # if we choose the `train_fast` mode, then we will not track the time.
    conf.timer = Timer(
        verbosity_level=1 if conf.track_time else 0,
        log_fn=conf.logger.log_metric,
        on_cuda=conf.on_cuda,
    )

    # create dataset.
    data_loader = create_dataset.define_dataset(conf, force_shuffle=True)

    # create model
    model = create_model.define_model(conf, data_loader=data_loader)

    # define the optimizer.
    optimizer = create_optimizer.define_optimizer(conf, model)

    # define the lr scheduler.
    scheduler = create_scheduler.Scheduler(conf, optimizer)

    # add model with data-parallel wrapper.
    if conf.graph.on_cuda:
        if conf.n_sub_process > 1:
            model = torch.nn.DataParallel(model, device_ids=conf.graph.device)

    # (optional) reload checkpoint
    try:
        checkpoint.maybe_resume_from_checkpoint(conf, model, optimizer, scheduler)
    except RuntimeError as e:
        conf.logger.log(f"Resume Error: {e}")
        conf.resumed = False

    if "bert" in conf.arch:
        criterion = nn.CrossEntropyLoss(reduction="mean")
        criterion = criterion.cuda() if conf.graph.on_cuda else criterion
        metrics = create_metrics.Metrics(model, task="bert_downstream_tasks")

        # define the best_perf tracker, either empty or from the checkpoint.
        best_tracker = stat_tracker.BestPerf(
            best_perf=None if "best_perf" not in conf else conf.best_perf,
            larger_is_better=True,
        )

        from pcode.distributed_running_bert_finetuning import Trainer
    else:
        # define the criterion and metrics based on arch.
        criterion = nn.CrossEntropyLoss(reduction="mean")
        criterion = criterion.cuda() if conf.graph.on_cuda else criterion
        metrics = create_metrics.Metrics(
            model.module if "DataParallel" == model.__class__.__name__ else model,
            task="classification",
        )

        # define the best_perf tracker, either empty or from the checkpoint.
        best_tracker = stat_tracker.BestPerf(
            best_perf=None if "best_perf" not in conf else conf.best_perf,
            larger_is_better=True,
        )

        # get Trainer class.
        from pcode.distributed_running_cv import Trainer

    # set the tracker in scheduler.
    scheduler.set_best_tracker(best_tracker)

    # save arguments to disk.
    if conf.graph.rank == 0:
        conf.status = "started"
        checkpoint.save_arguments(conf)

    # start training.
    trainer = Trainer(
        conf,
        model=model,
        criterion=criterion,
        scheduler=scheduler,
        optimizer=optimizer,
        metrics=metrics,
        data_loader=data_loader,
    )
    trainer.train_and_validate()
    conf.status = "finished"

    # update the training status.
    if conf.graph.rank == 0:
        checkpoint.save_arguments(conf)
        os.system(f"echo {conf.checkpoint_root} >> {conf.job_id}")


def init_config(conf):
    # init related to randomness on cpu.
    cur_rank = dist.get_rank() if conf.distributed else 0
    conf._manual_seed = (
        conf.manual_seed
        if conf.same_seed_process
        else 1000 * conf.manual_seed + cur_rank
    )
    torch.manual_seed(conf._manual_seed)
    conf.random_state = np.random.RandomState(conf._manual_seed)

    # define the graph for the computation.
    conf.graph = topology.define_graph_topology(
        graph_topology=conf.graph_topology,
        world=conf.world,
        n_mpi_process=conf.n_mpi_process,  # the # of total main processes.
        n_sub_process=conf.n_sub_process,  # the # of subprocess for each main process.
        comm_device=conf.comm_device,
        on_cuda=conf.on_cuda,
        rank=cur_rank,
        rng_state=conf.random_state,
        topology_conf=conf.topology_conf,
    )
    conf.is_centralized = conf.graph_topology == "complete"

    # re-configure batch_size if sub_process > 1.
    if conf.n_sub_process > 1:
        conf.batch_size = conf.batch_size * conf.n_sub_process

    # configure cuda related.
    if conf.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(conf._manual_seed)
        torch.cuda.set_device(conf.graph.device[0])
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # define checkpoint for logging.
    checkpoint.init_checkpoint(conf)

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_dir)

    # display the arguments' info.
    logging.display_args(conf)

    # save extra info to conf.
    for key, value in conf.graph.topology_conf.items():
        setattr(conf, f"tc_{key}", value)


if __name__ == "__main__":
    # parse the arguments.
    conf = get_args()
    main(conf)
