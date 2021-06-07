# -*- coding: utf-8 -*-
import torch.distributed as dist

import pcode.models as models


def define_model(conf, **kargs):
    if "rnn_lm" in conf.arch:
        return define_rnn_lm_model(conf, TEXT=kargs["data_loader"]["TEXT"])
    elif "transformer_nmt" in conf.arch:
        return define_transformer_nmt_model(conf, data_loader=kargs["data_loader"])
    elif "bert" in conf.arch:
        return define_ptl_finetuning_model(conf, data_loader=kargs["data_loader"])
    else:
        return define_cv_model(conf)


"""define loaders for different models."""


def define_cv_model(conf):
    if "wideresnet" in conf.arch:
        model = models.__dict__["wideresnet"](conf)
    elif "resnet" in conf.arch and "resnet_evonorm" not in conf.arch:
        model = models.__dict__["resnet"](conf)
    elif "resnet_evonorm" in conf.arch:
        model = models.__dict__["resnet_evonorm"](conf, arch=conf.arch)
    elif "densenet" in conf.arch:
        model = models.__dict__["densenet"](conf)
    elif "vgg" in conf.arch:
        model = models.__dict__["vgg"](conf)
    elif "lenet" in conf.arch:
        model = models.__dict__["lenet"](conf)
    else:
        model = models.__dict__[conf.arch](conf)

    if conf.graph.on_cuda:
        model = model.cuda()

    # get a consistent init model over the world.
    if conf.distributed:
        consistent_model(conf, model)

    # get the model stat info.
    get_model_stat(conf, model)
    return model


def define_rnn_lm_model(conf, TEXT):
    print("=> creating model '{}'".format(conf.arch))

    # get embdding size and num_tokens.
    weight_matrix = TEXT.vocab.vectors

    if weight_matrix is not None:
        conf.n_tokens, emb_size = weight_matrix.size(0), weight_matrix.size(1)
    else:
        conf.n_tokens, emb_size = len(TEXT.vocab), conf.rnn_n_hidden

    # create model.
    model = models.RNNLM(
        ntoken=conf.n_tokens,
        ninp=emb_size,
        nhid=conf.rnn_n_hidden,
        nlayers=conf.rnn_n_layers,
        tie_weights=conf.rnn_tie_weights,
        dropout=conf.drop_rate,
        weight_norm=conf.rnn_weight_norm,
    )

    # init the model.
    if weight_matrix is not None:
        model.encoder.weight.data.copy_(weight_matrix)

    if conf.graph.on_cuda:
        model = model.cuda()

    # consistent the model.
    consistent_model(conf, model)
    get_model_stat(conf, model)
    return model


def define_transformer_nmt_model(conf, data_loader):
    conf.logger.log("=> creating model '{}'".format(conf.arch))

    # create model.
    model = models.transformer.make_model(
        src_vocab_size=conf.src_vocab_size,
        tgt_vocab_size=conf.tgt_vocab_size,
        share_embedding=conf.transformer_share_vocab,
        num_layers=conf.transformer_n_layers,
        d_model=conf.transformer_dim_model,
        heads=conf.transformer_n_head,
        d_ff=conf.transformer_dim_inner_hidden,
        dropout=conf.drop_rate,  # default here is 0.1
    )

    if conf.graph.on_cuda:
        model = model.cuda()
    return model


def _confirm_experiment(conf, model):
    # we will first put the model on the device (to reduce the initialization time).
    model = model.cuda()

    for param_name, param in model.named_parameters():
        if conf.bert_conf_["ptl"] in param_name:
            param.requires_grad = True
        if "classifier" in param_name:
            param.requires_grad = True
    return model


def define_ptl_finetuning_model(conf, data_loader):
    import pcode.models.bert.predictors.linear_predictors as linear_predictors

    conf.logger.log("=> creating model '{}'".format(conf.arch))

    # init model.
    classes = linear_predictors.ptl2classes[conf.bert_conf_["ptl"]]
    if conf.bert_conf_["model_scheme"] == "vector_cls_sentence":
        model = classes.seqcls.from_pretrained(
            conf.arch,
            num_labels=data_loader["data_iter"].num_labels,
            cache_dir=conf.pretrained_weight_path,
        )
    elif conf.bert_conf_["model_scheme"] == "postagging":
        model = classes.postag.from_pretrained(
            conf.arch,
            out_dim=data_loader["data_iter"].num_labels,
            cache_dir=conf.pretrained_weight_path,
        )
    elif conf.bert_conf_["model_scheme"] == "multiplechoice":
        model = classes.multiplechoice.from_pretrained(
            conf.arch, cache_dir=conf.pretrained_weight_path
        )

    # confirm model.
    model = _confirm_experiment(conf, model)

    if conf.graph.on_cuda:
        model = model.cuda()

    # get a consistent init model over the world.
    if conf.distributed:
        consistent_model(conf, model)

    # get the model stat info.
    get_model_stat(conf, model)
    return model


"""some utilities functions."""


def get_model_stat(conf, model):
    print(
        "=> creating model '{}. total params for process {}: {}M".format(
            conf.arch,
            conf.graph.rank,
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        )
    )


def consistent_model(conf, model):
    """it might because of MPI, the model for each process is not the same.

    This function is proposed to fix this issue,
    i.e., use the  model (rank=0) as the global model.
    """
    print("consistent model for process (rank {})".format(conf.graph.rank))
    cur_rank = conf.graph.rank
    for param in model.parameters():
        param.data = param.data if cur_rank == 0 else param.data - param.data
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
