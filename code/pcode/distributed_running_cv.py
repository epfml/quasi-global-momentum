# -*- coding: utf-8 -*-
import os
import copy
import time
import numpy as np
import torch
import torch.distributed as dist

from pcode.create_dataset import load_data_batch, shuffle_cv_dataset

from pcode.utils.checkpoint import save_to_checkpoint
import pcode.utils.logging as logging
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.utils.error_handler as error_handler
import pcode.utils.checkpoint as checkpoint

# sys.excepthook = error_handler.global_except_hook


class Trainer:
    def __init__(
        self, conf, model, criterion, scheduler, optimizer, metrics, data_loader
    ):
        self.conf = conf
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.metrics = metrics
        self.data_loader = data_loader

    def train_and_validate(self):
        print("=>>>> start training and validation.")
        # define runtime stat tracker and start the training.
        self.tracker_tr = RuntimeTracker(
            metrics_to_track=self.metrics.metric_names, on_cuda=self.conf.graph.on_cuda
        )

        # get the timer.
        timer = self.conf.timer
        # break until finish expected full epoch training.
        print("=>>>> enter the training.\n")
        while True:
            # dist.barrier()

            # configure local step.
            for _input, _target in self.data_loader["train_loader"]:
                self.model.train()

                # load data
                with timer("load_data", epoch=self.scheduler.epoch_):
                    _input, _target = load_data_batch(self.conf, _input, _target)

                # inference and get current performance.
                with timer("forward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss = self.inference(
                        self.model, _input, _target, tracker=self.tracker_tr
                    )

                with timer("backward_pass", epoch=self.scheduler.epoch_):
                    loss.backward()

                with timer("sync_and_apply_grad", epoch=self.scheduler.epoch_):
                    n_bits_to_transmit = self.optimizer.step(
                        timer=timer,
                        scheduler=self.scheduler,
                        criterion=self.criterion,
                        data_batch={"input": _input, "target": _target},
                        model=self.model,
                    )
                    self.scheduler.step()

                # display the logging info.
                logging.display_training_stat(
                    self.conf, self.scheduler, self.tracker_tr, n_bits_to_transmit
                )

                # finish one epoch training and to decide if we want to val our model.
                if self.scheduler.epoch_ % 1 == 0:
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

                    # each worker finish one epoch training.
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
                    print(timer.summary())

            # reshuffle the data.
            if self.conf.reshuffle_per_epoch:
                print("\nReshuffle the dataset.")
                self.data_loader = shuffle_cv_dataset(self.conf, self.data_loader)

    def inference(self, model, _input, _target, tracker=None):
        """Inference on the given model and get loss and accuracy."""
        output = model(_input)
        loss = self.criterion(output, _target)
        performance = self.metrics.evaluate(loss, output, _target)
        if tracker is not None:
            tracker.update_metrics(
                [loss.item()] + performance, n_samples=_input.size(0)
            )
        return loss

    def do_validate(self):
        """Evaluate the model on the test dataset and save to the checkpoint."""
        # wait until the whole group enters this function, and then evaluate.
        # dist.barrier()
        self.conf.logger.log("Enter validation phase.")
        averaged_val_perf_of_local_model = self.validate()

        # remember best performance for the local model and display the val info.
        self.scheduler.best_tracker.update(
            averaged_val_perf_of_local_model[self.tracker_tr.primary_metric],
            self.scheduler.epoch_,
        )
        logging.dispaly_best_test_stat(self.conf, self.scheduler)

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
        """A function for model evaluation."""

        def _evaluate(_model, data_loader_label, label):
            # dist.barrier()
            if self.conf.graph.rank == 0:
                print(f"Evaluating: {data_loader_label}---{label}.")

            # define stat.
            tracker = RuntimeTracker(
                metrics_to_track=self.metrics.metric_names,
                on_cuda=self.conf.graph.on_cuda,
            )
            # switch to evaluation mode
            _model.eval()

            for _input, _target in self.data_loader[data_loader_label]:
                # load data and check performance.
                _input, _target = load_data_batch(self.conf, _input, _target)
                with torch.no_grad():
                    self.inference(_model, _input, _target, tracker)

            # display the local stats.
            logging.display_general_stat(
                self.conf,
                self.scheduler,
                tracker.get_metrics_performance(),
                label="_local_model",
                split="train" if "train" in data_loader_label else "test",
            )

            # display the global stat.
            globally_averaged_things = tracker.globally_average_things()
            logging.display_general_stat(
                self.conf,
                self.scheduler,
                globally_averaged_things,
                label,
                split="train" if "train" in data_loader_label else "test",
            )
            # get the global (mean) performance of the current metric.
            return globally_averaged_things

        # evaluate the averaged local model on the validation dataset.
        if self.conf.graph_topology != "complete" and self.conf.evaluate_on_consensus:
            # get averaged_local_model.
            copied_model = copy.deepcopy(
                self.model.module
                if "DataParallel" == self.model.__class__.__name__
                else self.model
            )
            self.optimizer.world_aggregator.agg_model(copied_model, op="avg")

            # evaluate on the val_loader.
            _evaluate(
                copied_model,
                data_loader_label="val_loader",
                label="averaged_local_model",
            )
            # evaluate on the train_loader.
            _evaluate(
                copied_model,
                data_loader_label="train_loader",
                label="averaged_local_model",
            )

        # evaluate each local model on the validation dataset.
        return _evaluate(
            self.model, data_loader_label="val_loader", label="local_model"
        )
