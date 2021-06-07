# -*- coding: utf-8 -*-


def define_optimizer(conf, model):
    # define the param to optimize.
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": conf.weight_decay,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]

    # define the optimizer.
    if conf.optimizer == "centralized_sgd":
        from pcode.optim.centralized_sgd import SGD

        optim_class = SGD
    elif conf.optimizer == "decentralized_sgd":
        from pcode.optim.decentralized_sgd import SGD

        optim_class = SGD
    elif conf.optimizer == "decentralized_qg_sgd":
        from pcode.optim.decentralized_qg_sgd import SGD

        optim_class = SGD
    elif conf.optimizer == "decentralized_sgd_with_gradient_tracking":
        from pcode.optim.decentralized_sgd_with_gradient_tracking import SGD

        optim_class = SGD
    elif conf.optimizer == "centralized_adam":
        from pcode.optim.centralized_adam import Adam

        optim_class = Adam

    elif conf.optimizer == "decentralized_adam":
        from pcode.optim.decentralized_adam import Adam

        optim_class = Adam
    elif conf.optimizer == "decentralized_qg_adam":
        from pcode.optim.decentralized_qg_adam import Adam

        optim_class = Adam
    elif conf.optimizer == "decentralized_d2":
        from pcode.optim.decentralized_d2 import SGD

        optim_class = SGD
    elif conf.optimizer == "decentralized_d2_v1":
        from pcode.optim.decentralized_d2_v1 import SGD

        optim_class = SGD
    else:
        raise NotImplementedError

    optimizer = optim_class(
        params,
        lr=conf.lr,
        momentum=conf.momentum_factor,
        nesterov=conf.use_nesterov,
        weight_decay=conf.weight_decay,
        conf=conf,
    )
    return optimizer
