# -*- coding: utf-8 -*-
"""define all global parameters here."""
from os.path import join
import argparse

import pcode.models as models


def get_args():
    ROOT_DIRECTORY = "./"
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, "data/")
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, "checkpoint")

    model_names = sorted(
        name for name in models.__dict__ if name.islower() and not name.startswith("__")
    )

    # feed them to the parser.
    parser = argparse.ArgumentParser(description="PyTorch Training for ConvNet")

    # add arguments.
    parser.add_argument("--work_dir", default=None, type=str)
    parser.add_argument("--remote_exec", default=False, type=str2bool)

    # dataset.
    parser.add_argument("--data", default="cifar10", help="a specific dataset name")
    parser.add_argument(
        "--data_dir", default=RAW_DATA_DIRECTORY, help="path to dataset"
    )
    parser.add_argument(
        "--use_lmdb_data",
        default=False,
        type=str2bool,
        help="use sequential lmdb dataset for better loading.",
    )
    parser.add_argument("--partition_data", default=None, type=str)
    parser.add_argument("--non_iid_alpha", default=0, type=float)
    parser.add_argument("--pin_memory", default=True, type=str2bool)

    # model
    parser.add_argument(
        "--arch",
        "-a",
        default="resnet20",
        help="model architecture: " + " | ".join(model_names) + " (default: resnet20)",
    )
    parser.add_argument("--group_norm_num_groups", default=None, type=int)
    parser.add_argument("--resnet_scaling", type=int, default=1)
    parser.add_argument("--vgg_scaling", default=None, type=int)
    parser.add_argument("--freeze_bn", default=False, type=str2bool)
    parser.add_argument("--freeze_bn_affine", default=False, type=str2bool)
    parser.add_argument("--evonorm_version", default=None, type=str)
    parser.add_argument("--bert_conf", default=None, type=str)

    # training and learning scheme
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--train_fast", type=str2bool, default=False)
    parser.add_argument("--stop_criteria", type=str, default="epoch")
    parser.add_argument("--num_epochs", type=int, default=90)
    parser.add_argument("--num_iterations", type=int, default=32000)

    parser.add_argument("--reshuffle_per_epoch", default=False, type=str2bool)
    parser.add_argument(
        "--batch_size",
        "-b",
        default=256,
        type=int,
        help="mini-batch size (default: 256)",
    )
    parser.add_argument("--base_batch_size", default=None, type=int)

    # learning rate scheme
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="MultiStepLR",
        choices=["MultiStepLR", "ExponentialLR", "ReduceLROnPlateau"],
    )
    parser.add_argument("--lr_milestones", type=str, default=None)
    parser.add_argument("--lr_milestone_ratios", type=str, default=None)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--lr_scaleup", type=str2bool, default=False)
    parser.add_argument("--lr_scaleup_init_lr", type=float, default=None)
    parser.add_argument("--lr_scaleup_factor", type=str, default=None)
    parser.add_argument("--lr_warmup", type=str2bool, default=False)
    parser.add_argument("--lr_warmup_epochs", type=int, default=None)
    parser.add_argument("--lr_warmup_epochs_upper_bound", type=int, default=150)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd")

    parser.add_argument("--adam_beta_1", default=0.9, type=float)
    parser.add_argument("--adam_beta_2", default=0.999, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)

    parser.add_argument("--label_smoothing", default=0.1, type=float)

    # the topology of the decentralized network.
    parser.add_argument("--graph_topology", default="complete", type=str)
    parser.add_argument("--local_step_conf", default=None, type=str)
    parser.add_argument("--slowmo_step_conf", default=None, type=str)
    parser.add_argument("--slow_buffer_conf", default=None, type=str)
    parser.add_argument("--topology_conf", default=None, type=str)

    # communication/compression scheme.
    parser.add_argument("--sync_bn_weight", default=True, type=str2bool)
    parser.add_argument("--comm_algo", default=None, type=str)
    parser.add_argument(
        "--comm_op",
        default=None,
        type=str,
        choices=["compress_top_k", "compress_random_k", "quantize_qsgd", "sign"],
    )
    parser.add_argument("--compress_ratio", default=None, type=float)
    parser.add_argument(
        "--compress_warmup_values", default="0.75,0.9375,0.984375,0.996,0.999", type=str
    )
    parser.add_argument("--compress_warmup_epochs", default=0, type=int)
    parser.add_argument("--quantize_level", default=None, type=int)
    parser.add_argument("--is_biased", default=False, type=str2bool)
    parser.add_argument("--majority_vote", default=False, type=str2bool)

    # momentum scheme
    parser.add_argument("--momentum_factor", default=0.9, type=float)
    parser.add_argument("--use_nesterov", default=False, type=str2bool)
    parser.add_argument("--avg_momentum", default=False, type=str2bool)

    # regularization
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="weight decay (default: 1e-4)"
    )
    parser.add_argument("--drop_rate", default=0.0, type=float)

    # configuration for different models.
    parser.add_argument("--densenet_growth_rate", default=12, type=int)
    parser.add_argument("--densenet_bc_mode", default=False, type=str2bool)
    parser.add_argument("--densenet_compression", default=0.5, type=float)

    parser.add_argument("--wideresnet_widen_factor", default=4, type=int)

    parser.add_argument("--rnn_n_hidden", default=200, type=int)
    parser.add_argument("--rnn_n_layers", default=2, type=int)
    parser.add_argument("--rnn_bptt_len", default=35, type=int)
    parser.add_argument("--rnn_clip", type=float, default=0.25)
    parser.add_argument("--rnn_use_pretrained_emb", type=str2bool, default=True)
    parser.add_argument("--rnn_tie_weights", type=str2bool, default=True)
    parser.add_argument("--rnn_weight_norm", type=str2bool, default=False)

    parser.add_argument("--transformer_n_layers", default=6, type=int)
    parser.add_argument("--transformer_n_head", default=8, type=int)
    parser.add_argument("--transformer_dim_model", default=512, type=int)
    parser.add_argument("--transformer_dim_inner_hidden", default=2048, type=int)
    parser.add_argument("--transformer_n_warmup_steps", default=4000, type=int)
    parser.add_argument("--transformer_share_vocab", default=False, type=str2bool)
    parser.add_argument("--max_sent_len", default=150, type=int)

    # miscs.
    parser.add_argument("--manual_seed", type=int, default=6, help="manual seed")
    parser.add_argument("--same_seed_process", type=str2bool, default=True)
    parser.add_argument(
        "--evaluate",
        "-e",
        dest="evaluate",
        type=str2bool,
        default=False,
        help="evaluate model on validation set",
    )
    parser.add_argument("--evaluate_on_consensus", default=False, type=str2bool)
    parser.add_argument("--summary_freq", default=100, type=int)
    parser.add_argument("--timestamp", default=None, type=str)
    parser.add_argument("--track_time", default=False, type=str2bool)
    parser.add_argument("--track_detailed_time", default=False, type=str2bool)
    parser.add_argument("--display_tracked_time", default=False, type=str2bool)
    parser.add_argument("--track_stat", default=False, type=str2bool)

    # checkpoint.
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=TRAINING_DIRECTORY,
        type=str,
        help="path to save checkpoint (default: checkpoint)",
    )
    parser.add_argument("--checkpoint_index", type=str, default=None)
    parser.add_argument("--save_all_models", type=str2bool, default=False)
    parser.add_argument("--save_some_models", type=str, default=None)

    # device.
    parser.add_argument("--num_cpus", default=None, type=int)
    parser.add_argument("--backend", type=str, default="mpi")
    parser.add_argument("--use_ipc", type=str2bool, default=False)
    parser.add_argument("--hostfile", type=str, default="iccluster/hostfile")
    parser.add_argument("--mpi_path", type=str, default="$HOME/.openmpi")
    parser.add_argument("--mpi_env", type=str, default=None)
    parser.add_argument(
        "--python_path", type=str, default="$HOME/conda/envs/pytorch-py3.8/bin/python"
    )
    parser.add_argument(
        "-j",
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )

    parser.add_argument(
        "--n_mpi_process", default=1, type=int, help="# of the main process."
    )
    parser.add_argument(
        "--n_sub_process",
        default=1,
        type=int,
        help="# of subprocess for each mpi process.",
    )
    parser.add_argument("--world", default=None, type=str)
    parser.add_argument("--on_cuda", type=str2bool, default=True)
    parser.add_argument("--comm_device", type=str, default="cuda")
    parser.add_argument("--local_rank", default=None, type=str)
    parser.add_argument("--clean_python", default=False, type=str2bool)

    # meta info
    parser.add_argument("--display_info", default=True, type=str2bool)
    parser.add_argument("--experiment", type=str, default="debug")
    parser.add_argument("--job_name", type=str, default="default")
    parser.add_argument("--job_id", type=str, default="/tmp/jobrun_logs")
    parser.add_argument("--script_path", default="exp/", type=str)
    parser.add_argument("--script_class_name", default=None, type=str)
    parser.add_argument("--num_jobs_per_node", default=1, type=int)
    parser.add_argument("--wait_in_seconds_per_job", default=30, type=int)

    # parse conf.
    conf = parser.parse_args()
    return conf


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


if __name__ == "__main__":
    args = get_args()
