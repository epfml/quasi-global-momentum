# -*- coding: utf-8 -*-
import os
import types
import numpy as np

import torch
import torch.distributed as dist

from pcode.create_dataset import shuffle_cv_dataset
from pcode.utils.checkpoint import save_to_checkpoint
from pcode.utils.logging import (
    display_training_stat,
    display_general_stat,
    dispaly_best_test_stat,
)
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.utils.error_handler as error_handler
import pcode.utils.checkpoint as checkpoint

# sys.excepthook = error_handler.global_except_hook


class Trainer:
    def __init__(
        self, conf, model, criterion, scheduler, optimizer, metrics, data_loader
    ):
        # init.
        self.conf = conf
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.metrics = metrics
        self.data_loader = data_loader

        # reset for logging.
        self.logger = conf.logger
        self.log_fn_json = conf.logger.log_metric
        self.log_fn = conf.logger.log

        self.task_metrics = data_loader["data_iter"].metrics
        self.batch_to_device = types.MethodType(
            task2batched_fn[data_loader["data_iter"].task], self
        )

        # logging tools.
        self.tracker_tr = RuntimeTracker(
            metrics_to_track=self.task_metrics, on_cuda=self.conf.graph.on_cuda
        )
        self.timer = conf.timer
        self.model_ptl = conf.bert_conf_["ptl"]

    def train_and_validate(self):
        # break until finish expected full epoch training.
        print("=>>>> start training and validation.")
        print("=>>>> enter the training.\n")
        while True:
            dist.barrier()

            for batched in self.data_loader["train_loader"]:
                with self.timer("load_data", epoch=self.conf.epoch_):
                    uids, golds, batched, _ = self.batch_to_device(batched)

                # forward for "pretrained model+classifier".
                with self.timer("forward_pass", epoch=self.conf.epoch_):
                    logits, *_ = self._model_forward(**batched)
                    loss = self.criterion(logits, golds)
                    self.tracker_tr.update_metrics(
                        metric_stat=[loss.item(), -1], n_samples=len(logits)
                    )

                # backward for "pretrained model+classifier".
                with self.timer("backward_pass", epoch=self.conf.epoch_):
                    loss.backward()

                with self.timer("perform_update", epoch=self.conf.epoch_):
                    n_bits_to_transmit = self.optimizer.step(
                        timer=self.timer,
                        scheduler=self.scheduler,
                        criterion=self.criterion,
                        model=self.model,
                    )
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                # display the logging info.
                display_training_stat(
                    self.conf, self.scheduler, self.tracker_tr, n_bits_to_transmit
                )

                with self.timer("validation", epoch=self.conf.epoch):
                    if (
                        self.conf.local_index
                        % int(self.conf.bert_conf_["eval_every_batch"])
                        == 0
                    ):
                        if self.tracker_tr.stat["loss"].avg > 1e3 or np.isnan(
                            self.tracker_tr.stat["loss"].avg
                        ):
                            print("\nThe process diverges!!!!!Early stop it.")
                            self.conf.logger.save_json()
                            self.conf.status = "diverged"
                            checkpoint.save_arguments(self.conf)
                            os.system(
                                f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}"
                            )
                            error_handler.abort()

                        # each worker enters the validation mode.
                        self.do_validate()

                        # refresh the logging cache at the begining of each epoch.
                        self.tracker_tr.reset()

                        # determine if the training is finished.
                        if self.scheduler.is_stop():
                            # save json.
                            self.conf.logger.save_json()
                            # return to the main.py
                            return

                # display tracking time.
                if (
                    self.conf.graph.rank == 0
                    and self.conf.display_tracked_time
                    and self.scheduler.local_index % self.conf.summary_freq == 0
                ):
                    print(self.timer.summary())

            # reshuffle the data.
            if self.conf.reshuffle_per_epoch:
                print("\nReshuffle the dataset.")
                self.data_loader = shuffle_cv_dataset(self.conf, self.data_loader)

    def _model_forward(self, **kwargs):  # accepting *args is removed for safety...
        if (
            self.model_ptl == "roberta" or self.model_ptl == "distilbert"
        ) and "token_type_ids" in kwargs:
            kwargs.pop("token_type_ids")
        return self.model(**kwargs)

    def do_validate(self):
        """Evaluate the model on the test dataset and save to the checkpoint."""
        # wait until the whole group enters this function, and then evaluate.
        dist.barrier()
        self.conf.logger.log("Enter validation phase.")
        averaged_val_perf_of_local_model = self.validate()

        # remember best performance for the local model and display the val info.
        self.scheduler.best_tracker.update(
            averaged_val_perf_of_local_model[self.tracker_tr.primary_metric],
            self.scheduler.epoch_,
        )
        dispaly_best_test_stat(self.conf, self.scheduler)

        # save to the checkpoint.
        if not self.conf.train_fast:
            checkpoint_dict = {
                "arch": self.conf.arch,
                "current_epoch": self.scheduler.epoch,
                "local_index": self.scheduler.local_index,
                "best_perf": self.scheduler.best_tracker.best_perf,
                "optimizer": self.optimizer.state_dict(),
                "state_dict": self.model.state_dict(),
                "lr_scheduler": self.scheduler.lr_scheduler.state_dict(),
                "tracking_dict": getattr(self.conf, "tracking_dict", {}),
            }
            save_to_checkpoint(
                self.conf,
                checkpoint_dict,
                self.scheduler.best_tracker.is_best,
                dirname=self.conf.checkpoint_dir,
                filename="checkpoint.pth.tar",
                save_all=self.conf.save_all_models,
            )
        self.conf.logger.log("Finished all validation.")

    def validate(self):
        self.model.eval()
        tracker_te = RuntimeTracker(
            metrics_to_track=self.task_metrics, on_cuda=self.conf.graph.on_cuda
        )

        # collect all information.
        all_losses, all_golds, all_preds = [], [], []
        with torch.no_grad():
            for batched in self.data_loader["val_loader"]:
                # golds is used for compute loss, _golds used for i2t convertion
                uids, golds, batched, _golds = self.batch_to_device(batched)
                with torch.no_grad():
                    logits, *_ = self._model_forward(**batched)
                    loss = self.criterion(logits, golds).item()
                    preds = torch.argmax(logits, dim=-1, keepdim=False)
                    all_losses.append(loss)
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_golds.extend(golds.detach().cpu().numpy())

        # evaluate the performance.
        eval_res = self.metrics.evaluate(
            loss=all_losses,
            output=all_preds,
            target=all_golds,
            task_metrics=self.task_metrics,
        )
        tracker_te.update_metrics(
            [sum(all_losses)] + eval_res, n_samples=len(all_losses)
        )

        # display the test stat.
        globally_averaged_things = tracker_te.globally_average_things()
        display_general_stat(
            self.conf,
            self.scheduler,
            globally_averaged_things,
            label="local_model",
            split="test",
        )
        return globally_averaged_things


"""functions for batch_to_device."""


def seqcls_batch_to_device(self, batched):
    uids = batched[0]
    input_ids, golds, attention_mask, token_type_ids = map(
        lambda x: x.cuda(), batched[1:]
    )
    return (
        uids,
        golds,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        None,
    )


def tagging_batch_to_device(self, batched):
    uids = batched[0]
    input_ids, attention_mask, _golds, if_tgts = map(lambda x: x.cuda(), batched[1:])

    golds = []
    for b_step in range(_golds.shape[0]):
        gold = _golds[b_step][if_tgts[b_step]]
        golds.append(gold)

    if self.conf.bert_conf_["task"] != "conll2003":
        return (
            uids,
            torch.cat(golds, dim=0),
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "if_tgts": if_tgts,
            },
            None,
        )
    return (
        uids,
        torch.cat(golds, dim=0),
        {"input_ids": input_ids, "attention_mask": attention_mask, "if_tgts": if_tgts},
        _golds,
    )


task2batched_fn = {
    "mrpc": seqcls_batch_to_device,
    "sst2": seqcls_batch_to_device,
    "mnli": seqcls_batch_to_device,
    "qqp": seqcls_batch_to_device,
    "cola": seqcls_batch_to_device,
    "qnli": seqcls_batch_to_device,
    "rte": seqcls_batch_to_device,
    "posptb": tagging_batch_to_device,
    "swag": seqcls_batch_to_device,
    "agnews": seqcls_batch_to_device,
    "trec": seqcls_batch_to_device,
    "dbpedia": seqcls_batch_to_device,
    "yelp2": seqcls_batch_to_device,
    "semeval16": seqcls_batch_to_device,
    "conll2003": tagging_batch_to_device,
}
