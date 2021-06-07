# -*- coding: utf-8 -*-
import gc
import json
import shutil
import time
from os.path import join, isfile

import torch

import pcode.utils.logging as logging
from pcode.utils.op_paths import build_dirs
from pcode.utils.op_files import is_jsonable


def get_checkpoint_folder_name(conf):
    conf.pretrained_weight_path = join(
        "/mlodata1/tlin/dataset/bert", "pretrained_weights"
    )

    # get communication info.
    if conf.comm_op is None:
        comm_info = ""
    elif "compress" in conf.comm_op:
        comm_info = "{}-{}_".format(conf.comm_op, conf.compress_ratio)
        comm_info += "warmup_epochs-{}".format(conf.compress_warmup_epochs)
        comm_info += "_mask_momentum" if conf.mask_momentum else ""
        comm_info += (
            "_clip_grad-{}".format(conf.clip_grad_val) if conf.clip_grad else ""
        )
    elif conf.comm_op == "quantize_qsgd":
        comm_info = "{}-{}_".format(conf.comm_op, conf.quantize_level)
    elif conf.comm_op == "sign":
        comm_info = "{}_".format(conf.comm_op)
    else:
        comm_info = ""

    # get optimizer info.
    if "choco" in conf.optimizer:
        optim_info = "{}-stepsize-{}".format(conf.optimizer, conf.consensus_stepsize)
    else:
        optim_info = "{}".format(conf.optimizer)

    # concat them together.
    return "{}_seed-{}_l2-{}_lr-{}_factor-{}_epochs-{}_batchsize-{}_basebatchsize-{}_num_mpi_process_{}_n_sub_process-{}_topology-{}_optim-{}_comm_info-{}".format(
        conf.timestamp if conf.timestamp is not None else str(int(time.time())),
        conf.manual_seed,
        conf.weight_decay,
        conf.lr,
        conf.lr_scaleup_factor,
        conf.num_epochs,
        conf.batch_size,
        conf.base_batch_size,
        conf.n_mpi_process,
        conf.n_sub_process,
        conf.graph_topology,
        optim_info,
        comm_info,
    )


def init_checkpoint(conf):
    # init checkpoint dir.
    conf.checkpoint_root = join(
        conf.checkpoint,
        conf.data,
        conf.arch,
        conf.experiment if conf.experiment is not None else "",
        get_checkpoint_folder_name(conf),
    )
    conf.checkpoint_dir = join(conf.checkpoint_root, str(conf.graph.rank))
    if conf.save_some_models is not None:
        conf.save_some_models = conf.save_some_models.split(",")

    # if the directory does not exists, create them.
    build_dirs(conf.checkpoint_dir)


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def save_arguments(conf):
    # save the configure file to the checkpoint.
    if conf.graph.rank == 0:
        with open(join(conf.checkpoint_root, "arguments.json"), "w") as fp:
            json.dump(
                dict(
                    [
                        (k, v)
                        for k, v in conf.__dict__.items()
                        if is_jsonable(v) and type(v) is not torch.Tensor
                    ]
                ),
                fp,
                indent=" ",
            )


def save_to_checkpoint(conf, state, is_best, dirname, filename, save_all=False):
    # save full state.
    checkpoint_path = _save_to_checkpoint(state, dirname, filename)
    best_model_path = join(dirname, "model_best.pth.tar")
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)
    if save_all:
        shutil.copyfile(
            checkpoint_path,
            join(dirname, "checkpoint_epoch_%s.pth.tar" % state["current_epoch"]),
        )
    elif conf.save_some_models is not None:
        if str(state["current_epoch"]) in conf.save_some_models:
            shutil.copyfile(
                checkpoint_path,
                join(dirname, "checkpoint_epoch_%s.pth.tar" % state["current_epoch"]),
            )


def maybe_resume_from_checkpoint(conf, model, optimizer, scheduler):
    if conf.resume:
        if conf.checkpoint_index is not None:
            # reload model from a specific checkpoint index.
            checkpoint_index = "_epoch_" + conf.checkpoint_index
        else:
            # reload model from the latest checkpoint.
            checkpoint_index = ""
        checkpoint_path = join(
            conf.resume,
            str(conf.graph.rank),
            "checkpoint{}.pth.tar".format(checkpoint_index),
        )
        print("try to load previous model from the path:{}".format(checkpoint_path))

        if isfile(checkpoint_path):
            print(
                "=> loading checkpoint {} for {}".format(conf.resume, conf.graph.rank)
            )

            # get checkpoint.
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # reset path for log.
            # try:
            #     remove_folder(conf.checkpoint_root)
            # except RuntimeError as e:
            #     print(f"ignore the error={e}")
            # conf.checkpoint_root = conf.resume
            # conf.checkpoint_dir = join(conf.resume, str(conf.graph.rank))
            # restore model.
            model.load_state_dict(checkpoint["state_dict"])
            # restore optimizer.
            optimizer.load_state_dict(checkpoint["optimizer"])
            # restore some run-time info.
            scheduler.update_from_checkpoint(checkpoint)
            # logging.
            print(
                "=> loaded model from path '{}' checkpointed at (epoch {})".format(
                    conf.resume, checkpoint["current_epoch"]
                )
            )
            # configure logger.
            conf.logger = logging.Logger(conf.checkpoint_dir)

            # try to solve memory issue.
            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()
            return
        else:
            print("=> no checkpoint found at '{}'".format(conf.resume))
