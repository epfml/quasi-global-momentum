# -*- coding: utf-8 -*-
import os
import torch
import torchtext

from pcode.datasets.partition_data import DataPartitioner
from pcode.datasets.prepare_data import get_dataset
import pcode.models.transformer as transformer
import pcode.utils.auxiliary as auxiliary


def load_data_batch(conf, _input, _target):
    """Load a mini-batch and record the loading time."""
    if conf.graph.on_cuda:
        _input, _target = _input.cuda(), _target.cuda()
    return _input, _target


def define_dataset(conf, force_shuffle=False):
    if "rnn_lm" in conf.arch:
        return define_nlp_dataset(conf, force_shuffle)
    elif "transformer_nmt" in conf.arch:
        return define_transformer_nmt_dataset(conf, force_shuffle)
    elif "bert" in conf.arch:
        return define_ptl_finetuning_dataset(conf, force_shuffle)
    else:
        return define_cv_dataset(conf, force_shuffle)


"""define loaders for different datasets."""
"""bert fine-tuning related task."""


def define_ptl_finetuning_dataset(conf, force_shuffle):
    import pcode.models.bert.predictors.linear_predictors as linear_predictors
    import pcode.models.bert.task_configs as task_configs

    # initialize and extract necessary information.
    assert (
        conf.bert_conf is not None
        and "model_scheme" in conf.bert_conf
        and "max_seq_len" in conf.bert_conf
        and "eval_every_batch" in conf.bert_conf
    )
    conf.bert_conf_ = auxiliary.dict_parser(conf.bert_conf)
    conf.bert_conf_["ptl"] = conf.arch.split("-")[0]
    conf.bert_conf_["task"] = conf.data

    # create data_iter.
    classes = linear_predictors.ptl2classes[conf.bert_conf_["ptl"]]
    tokenizer = classes.tokenizer.from_pretrained(conf.arch)
    data_iter = task_configs.task2dataiter[conf.bert_conf_["task"]](
        conf.bert_conf_["task"],
        conf.arch,
        tokenizer,
        int(conf.bert_conf_["max_seq_len"]),
    )

    # formulate train/val/test.
    print("partitioning the dataset.")
    train_loader, train_partitioner = _define_cv_dataset(
        conf,
        partition_type=conf.partition_data,
        dataset=data_iter.trn_dl,
        dataset_type="train",
        force_shuffle=True,
        task=conf.bert_conf_["task"],
    )
    print("get train_loader")
    val_loader, val_partitioner = _define_cv_dataset(
        conf, partition_type=None, dataset=data_iter.val_dl, dataset_type="test"
    )

    _get_cv_data_stat(conf, train_loader, val_loader)
    return {
        "data_iter": data_iter,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_partitioner": train_partitioner,
        "val_partitioner": val_partitioner,
    }


"""transformer nmt related task."""


def define_transformer_nmt_dataset(conf, force_shuffle):
    conf.logger.log("create {} dataset for rank {}".format(conf.data, conf.graph.rank))

    # create dataset.
    (SRC_TEXT, TGT_TEXT), train, valid, test = get_dataset(
        conf, conf.data, conf.data_dir
    )

    # build vocb.
    MIN_FREQ = 5
    SRC_TEXT.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT_TEXT.build_vocab(train.trg, min_freq=MIN_FREQ)

    if conf.transformer_share_vocab:
        SRC_TEXT.vocab.extend(TGT_TEXT.vocab)
        TGT_TEXT.vocab = SRC_TEXT.vocab

    conf.src_vocab_size = len(SRC_TEXT.vocab)
    conf.tgt_vocab_size = len(TGT_TEXT.vocab)
    conf.logger.log(
        "shared_vocab={}, src_vocab_size={}, tgt_vocab_size={}".format(
            conf.transformer_share_vocab, conf.src_vocab_size, conf.tgt_vocab_size
        )
    )

    # partition training data.
    train_loader, _, _ = torchtext.data.BucketIterator.splits(
        (train, valid, test),
        batch_size=conf.batch_size * conf.graph.n_nodes,
        sort_key=lambda x: -len(x.src),
        sort=True,
        repeat=True,
        device="cuda" if conf.graph.on_cuda else None,
        shuffle=force_shuffle or conf.reshuffle_per_epoch,
    )
    _, val_loader, test_loader = torchtext.data.BucketIterator.splits(
        (train, valid, test),
        batch_size=conf.batch_size,
        sort_key=lambda x: -len(x.src),
        sort=True,
        repeat=False,
        device="cuda" if conf.graph.on_cuda else None,
        shuffle=False,
    )

    # get some stat.
    _get_nlp_data_stat(conf, train, valid, train_loader, val_loader)

    return {
        "SRC_TEXT": SRC_TEXT,
        "TGT_TEXT": TGT_TEXT,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "pad_idx": TGT_TEXT.vocab.stoi[transformer.PAD_WORD],
        "unk_idx": TGT_TEXT.vocab.stoi[transformer.UNK_WORD],
        "bos_idx": TGT_TEXT.vocab.stoi[transformer.BOS_WORD],
        "eos_idx": TGT_TEXT.vocab.stoi[transformer.EOS_WORD],
    }


"""nlp related task."""


def define_nlp_dataset(conf, force_shuffle):
    print("create {} dataset for rank {}".format(conf.data, conf.graph.rank))
    # create dataset.
    TEXT, train, valid, _ = get_dataset(conf, conf.data, conf.data_dir)

    # Build vocb.
    # we can use some precomputed word embeddings,
    # e.g., GloVe vectors with 100, 200, and 300.
    if conf.rnn_use_pretrained_emb:
        try:
            vectors = "glove.6B.{}d".format(conf.rnn_n_hidden)
            vectors_cache = os.path.join(conf.data_dir, ".vector_cache")
        except:
            vectors, vectors_cache = None, None
    else:
        vectors, vectors_cache = None, None
    TEXT.build_vocab(train, vectors=vectors, vectors_cache=vectors_cache)

    # Partition training data.
    train_loader, _ = torchtext.data.BPTTIterator.splits(
        (train, valid),
        batch_size=conf.batch_size * conf.graph.n_nodes,
        bptt_len=conf.rnn_bptt_len,
        device="cuda:{}".format(conf.graph.device[0]) if conf.graph.on_cuda else None,
        repeat=True,
        shuffle=force_shuffle or conf.reshuffle_per_epoch,
    )
    _, val_loader = torchtext.data.BPTTIterator.splits(
        (train, valid),
        batch_size=conf.batch_size,
        bptt_len=conf.rnn_bptt_len,
        device="cuda:{}".format(conf.graph.device[0]) if conf.graph.on_cuda else None,
        shuffle=False,
    )

    # get some stat.
    _get_nlp_data_stat(conf, train, valid, train_loader, val_loader)
    return {"TEXT": TEXT, "train_loader": train_loader, "val_loader": val_loader}


def _get_nlp_data_stat(conf, train, valid, train_loader, val_loader):
    # configure the workload for each worker.
    # Note that: the training will access to the same # of samples (w/ or w/o partition).

    conf.num_batches_train_per_device_per_epoch = len(train_loader)
    conf.num_whole_train_batches_per_worker = (
        conf.num_batches_train_per_device_per_epoch * conf.num_epochs
    )
    conf.num_warmup_train_batches_per_worker = (
        conf.num_batches_train_per_device_per_epoch * conf.lr_warmup_epochs
    )

    # when the training is controlled by the num_iterations.
    conf.num_iterations_per_worker = conf.num_iterations // conf.graph.n_nodes

    # get the data statictics (on behalf of each worker) for val.
    conf.num_batches_val_per_device_per_epoch = len(val_loader)

    # define some parameters for training.
    conf.logger.log(
        "Detailed data stat: we have {} epochs, \
         {} mini-batches per device for training. \
         {} mini-batches per device for val. \
         The batch size: {}.".format(
            conf.num_epochs,
            conf.num_batches_train_per_device_per_epoch,
            conf.num_batches_val_per_device_per_epoch,
            conf.batch_size,
        )
    )


"""cv related task."""


def define_cv_dataset(conf, force_shuffle):
    print("Create dataset: {} for rank {}.".format(conf.data, conf.graph.rank))
    train_dataset = get_dataset(conf, conf.data, conf.data_dir, split="train")
    val_dataset = get_dataset(conf, conf.data, conf.data_dir, split="test")

    train_loader, train_partitioner = _define_cv_dataset(
        conf,
        partition_type=conf.partition_data,
        dataset=train_dataset,
        dataset_type="train",
        force_shuffle=True,
    )
    val_loader, val_partitioner = _define_cv_dataset(
        conf, partition_type=None, dataset=val_dataset, dataset_type="test"
    )

    _get_cv_data_stat(conf, train_loader, val_loader)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_partitioner": train_partitioner,
        "val_partitioner": val_partitioner,
    }


def _define_cv_dataset(
    conf,
    partition_type,
    dataset,
    dataset_type,
    force_shuffle=False,
    data_to_load=None,
    task=None,
):
    """ Given a dataset, partition it. """
    batch_size = conf.batch_size
    world_size = conf.graph.n_nodes

    if data_to_load is None:
        # determine the data to load,
        # either the whole dataset, or a subset specified by partition_type.
        if partition_type is not None and conf.distributed:
            partition_sizes = [1.0 / world_size for _ in range(world_size)]
            partitioner = DataPartitioner(
                conf,
                dataset,
                partition_sizes,
                partition_type=partition_type,
                consistent_indices=True
                if not hasattr(conf, "consistent_indices")
                else conf.consistent_indices,
                task=task,
            )
            data_to_load = partitioner.use(conf.graph.rank)
            print("Data partition: partitioned data and use subdata.")
        else:
            partitioner = None
            data_to_load = dataset
            print("Data partition: used whole data.")
    else:
        print("Data partition: use inputed 'data_to_load'.")
        partitioner = None

    # use Dataloader.
    data_loader = torch.utils.data.DataLoader(
        data_to_load,
        batch_size=batch_size,
        shuffle=force_shuffle,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
    )

    print(
        (
            "Data stat: we have {} samples for {}, "
            + "load {} data for process (rank {}). "
            + "The batch size is {}, number of batches is {}."
        ).format(
            len(dataset),
            dataset_type,
            len(data_to_load),
            conf.graph.rank,
            batch_size,
            len(data_loader),
        )
    )
    return data_loader, partitioner


def shuffle_cv_dataset(conf, dataset_dict):
    # shuffle train.
    if dataset_dict["train_partitioner"] is not None:
        # create the partitioner.
        partition_sizes = [1.0 / conf.graph.n_nodes for _ in range(conf.graph.n_nodes)]
        train_partitioner = DataPartitioner(
            conf,
            dataset_dict["train_dataset"],
            partition_sizes,
            partition_type=conf.partition_data,
        )
        data_to_load = train_partitioner.use(conf.graph.rank)

        # update train data loader dict.
        train_data_loader, train_partitioner = _define_cv_dataset(
            conf,
            partition_type=conf.partition_data,
            dataset=dataset_dict["train_dataset"],
            dataset_type="train",
            data_to_load=data_to_load,
        )
        dataset_dict["train_data_loader"] = train_data_loader
        dataset_dict["train_partitioner"] = train_partitioner
    return dataset_dict


def _get_cv_data_stat(conf, train_loader, val_loader):
    # configure the workload for each worker.
    # Note that: the training will access to the same # of samples (w/ or w/o partition).

    # when it is w/ partition, then return the true local loader size.
    # when it is w/o partition, then return the local loader size / world size.
    conf.num_batches_train_per_device_per_epoch = (
        len(train_loader) // conf.graph.n_nodes
        if conf.partition_data is None
        else len(train_loader)
    )
    conf.num_whole_train_batches_per_worker = (
        conf.num_batches_train_per_device_per_epoch * conf.num_epochs
    )
    conf.num_warmup_train_batches_per_worker = (
        conf.num_batches_train_per_device_per_epoch * conf.lr_warmup_epochs
    )

    # when the training is controlled by the num_iterations.
    conf.num_iterations_per_worker = conf.num_iterations // conf.graph.n_nodes

    # get the data statictics (on behalf of each worker) for val.
    conf.num_batches_val_per_device_per_epoch = len(val_loader)

    # define some parameters for training.
    print(
        "\nData Stat: we have {} epochs, \
         {} mini-batches per device for training. \
         {} mini-batches per device for val. \
         The batch size: {}.".format(
            conf.num_epochs,
            conf.num_batches_train_per_device_per_epoch,
            conf.num_batches_val_per_device_per_epoch,
            conf.batch_size,
        )
    )
