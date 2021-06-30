# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
r"""Instantiate a trainer for training models.

Code from:
https://github.com/kdexd/probnmn-clevr/blob/master/probnmn/trainers/_trainer.py
"""
import json
import os
import torch
import logging

from typing import Any, Dict, Generator, List, Optional

from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.checkpointing import CheckpointManager
from losses import _LossFun


class _Trainer(object):
    r"""
    A base class for generic training of models. This class can have multiple
    models interacting with each other, rather than a single model, which is
    suitable to our use-case (example, ``module_training`` phase has two models:
    :class:`~probnmn.models.program_generator.ProgramGenerator` and
    :class:`~probnmn.models.nmn.NeuralModuleNetwork`). It offers full
    flexibility, with sensible defaults which may be changed (or disabled) while
    extending this class.

    Extended Summary
    ----------------
    1. Default :class:`~torch.optim.Adam` Optimizer, updates parameters of all
        models in this trainer. Learning rate and weight decay for this
        optimizer are picked up from the provided config.

    2. Default :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` learning
        rate scheduler. Gamma and patience arguments are picked up from the
        provided config. Observed metric is assumed to be of type "higher is
        better". For 'lower is better" metrics, make sure to reciprocate.

    3. Tensorboard logging of loss curves, metrics etc.

    4. Serialization of models and optimizer as checkpoint (.pth) files after
       every validation. The observed metric for keeping track of best
       checkpoint is of type "higher is better", follow (2) above if the
       observed metric is of type "lower is better".

    Extend this class and override suitable methods as per requirements, some
    important ones are:

    1. :meth:`step`, provides complete customization, this is the method which
        comprises of one full training iteration, and internally calls (in
        order) - :meth:`_before_iteration`, :meth:`_do_iteration` and
        :meth:`_after_iteration`. Most of the times you may not require
        overriding this method, instead one of the mentioned three methods
        called by `:meth:`step`.

    2. :meth:`_do_iteration`, with core training loop - what happens every
        iteration, given a ``batch`` from the dataloader this class holds.

    3. :meth:`_before_iteration` and :meth:`_after_iteration`, for any pre-
        or post-processing steps. Default behaviour:

        * :meth:`_before_iteration` - call ``optimizer.zero_grad()``
        * :meth:`_after_iteration` - call ``optimizer.step()`` and do
        tensorboard logging.

    4. :meth:`after_validation`, to specify any steps after evaluation. Default
        behaviour is to do learning rate scheduling and log validation metrics
        on tensorboard.

    Notes
    -----
    All models are `passed by assignment`, so they could be shared with an
    external evaluator.
    Do not set ``self._models = ...`` anywhere while extending this class.

    Parameters
    ----------
    config: Config
        A :class:`~probnmn.Config` object with all the relevant configuration parameters.
    dataloader: torch.utils.data.DataLoader
        A :class:`~torch.utils.data.DataLoader` which provides batches of training examples. It
        wraps one of :mod:`probnmn.data.datasets` depending on the evaluation phase.
    models: Dict[str, Type[nn.Module]]
        All the models which interact with each other during training. These are one or more from
        :mod:`probnmn.models` depending on the training phase.
    serialization_dir: str
        Path to a directory for tensorboard logging and serializing checkpoints.
    gpu_ids: List[int], optional (default=[0])
        List of GPU IDs to use or evaluation, ``[-1]`` - use CPU.
    """

    def __init__(
            self,
            config: OmegaConf,
            loss_fn: _LossFun,
            dataloader: DataLoader,
            models: Dict[str, nn.Module],
            serialization_dir: str,
            gpu_ids: List[int] = [0],
            write_metrics_file: str = "metrics.txt",
    ):
        self._C = config

        # Make dataloader cyclic for sampling batches perpetually.
        self._dataloader = self._cycle(dataloader)
        self._models = models

        self._loss_fn = loss_fn

        # Set device according to specified GPU ids.
        self._device = torch.device(
            f"cuda:{gpu_ids[0]}" if gpu_ids[0] >= 0 else "cpu")

        self._serialization_dir = serialization_dir

        # Shift models to device, and wrap in DataParallel for Multi-GPU execution (if needed).
        for model_name in self._models:
            self._models[model_name] = self._models[model_name].to(
                self._device)

            if len(gpu_ids) > 1 and -1 not in gpu_ids:
                # Don't wrap to DataParallel if single GPU ID or -1 (CPU) is provided.
                self._models[model_name] = nn.DataParallel(
                    self._models[model_name], gpu_ids)

        # Accumulate parameters of all models to construct Adam Optimizer.
        all_parameters: List[Any] = []
        for model_name in self._models:
            all_parameters.extend(list(self._models[model_name].parameters()))

        self._optimizer = optim.Adam(all_parameters,
                                     lr=self._C.opt.lr,
                                     weight_decay=self._C.opt.weight_decay)

        # Default learning rate scheduler: (lr *= gamma) when observed metric plateaus for
        # "patience" number of validation steps.
        self._lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode=self._loss_fn.min_or_max,
            factor=self._C.opt.lr_gamma,
            patience=self._C.opt.lr_patience,
            threshold=1e-3,
        )

        # Tensorboard summary writer for logging losses and metrics.
        self._tensorboard_writer = SummaryWriter(log_dir=serialization_dir)
        self._checkpoint_manager = CheckpointManager(
            serialization_dir=serialization_dir,
            models=self._models,
            optimizer=self._optimizer,
            mode=self._loss_fn.min_or_max,
        )
        self._write_metrics_file = os.path.join(serialization_dir,
                                                write_metrics_file)

        # Initialize a counter to keep track of the iteration number.
        # This increments everytime ``step`` is called.
        self._iteration: int = -1

    def step(self, iteration: Optional[int] = None):
        r"""
        Perform one iteration of training.

        Parameters
        ----------
        iteration: int, optional (default = None)
            Iteration number (useful to hard set to any number when loading checkpoint).
            If ``None``, use the internal :attr:`self._iteration` counter.
        """
        self._before_iteration()

        batch = next(self._dataloader)
        batch_loss, output_dict = self._do_iteration(batch)
        self._after_iteration(output_dict)

        self._iteration = iteration or self._iteration + 1

        return batch_loss

    def _before_iteration(self):
        r"""
        Steps to do before doing the forward pass of iteration. Default behavior is to simply
        call :meth:`zero_grad` for optimizer. Called inside :meth:`step`.
        """
        self._optimizer.zero_grad()

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Forward and backward passes on models, given a batch sampled from dataloader.

        Parameters
        ----------
        batch: Dict[str, Any]
            A batch of training examples sampled from dataloader. See :func:`step` and
            :meth:`_cycle` on how this batch is sampled.

        Returns
        -------
        Dict[str, Any]
            An output dictionary typically returned by the models. This would be passed to
            :meth:`_after_iteration` for tensorboard logging.
        """
        # What a single iteration usually would look like.
        iteration_output_dict = self._models["model"](batch)
        batch_loss, metrics = self._loss_fn(model_output=iteration_output_dict,
                                            batch=batch)
        if batch_loss.dim() != 1:
            raise ValueError("Must provide be of size batch_size")
        batch_loss = batch_loss.mean()
        batch_loss.backward()
        return {"loss": batch_loss}, metrics

    def _after_iteration(self, output_dict: Dict[str, Any]):
        r"""
        Steps to do after doing the forward pass of iteration. Default behavior is to simply
        do gradient update through ``optimizer.step()``, and log metrics to tensorboard.

        Parameters
        ----------
        output_dict: Dict[str, Any]
            This is exactly the object returned by :meth:_do_iteration`, which would contain all
            the required losses for tensorboard logging.
        """
        self._optimizer.step()

        # keys: {"loss"} + ... {other keys such as "elbo"}
        for key in output_dict:
            if isinstance(output_dict[key], dict):
                # Use ``add_scalars`` for dicts in a nested ``output_dict``.
                self._tensorboard_writer.add_scalars(f"train/{key}",
                                                     output_dict[key],
                                                     self._iteration)
            else:
                # Use ``add_scalar`` for floats / zero-dim tensors in ``output_dict``.
                self._tensorboard_writer.add_scalar(f"train/{key}",
                                                    output_dict[key],
                                                    self._iteration)

    def after_validation(self,
                         val_metrics: Dict[str, Any],
                         iteration: Optional[int] = None):
        r"""
        Steps to do after an external :class:`~probnmn.evaluators._evaluator._Evaluator` performs
        evaluation. This is not called by :meth:`step`, call it from outside at appropriate time.
        Default behavior is to perform learning rate scheduling, serializaing checkpoint and to
        log validation metrics to tensorboard.

        Since this implementation assumes a key ``"metric"`` in ``val_metrics``, it is convenient
        to set this key while overriding this method, when there are multiple models and multiple
        metrics and there is one metric which decides best checkpoint.

        Parameters
        ----------
        val_metrics: Dict[str, Any]
            Validation metrics for all the models. Returned by ``evaluate`` method of
            :class:`~probnmn.evaluators._evaluator._Evaluator` (or its extended class).
        iteration: int, optional (default = None)
            Iteration number. If ``None``, use the internal :attr:`self._iteration` counter.
        """
        if iteration is not None:
            self._iteration = iteration

        # Serialize model and optimizer and keep track of best checkpoint.
        self._checkpoint_manager.step(val_metrics["metric"], self._iteration)

        # Perform learning rate scheduling based on validation perplexity.
        self._lr_scheduler.step(val_metrics["metric"])

        # Log learning rate after scheduling.
        self._tensorboard_writer.add_scalar(
            "train/lr", self._optimizer.param_groups[0]["lr"], self._iteration)

        self.write_metrics(val_metrics, write_raw_metrics=False)

    def write_metrics(self, val_metrics, eval_split_name=None,
                      test_metric_name=None,
                      write_raw_metrics=True, write_metrics_to_txt=True):
        """Helper function to write a bunch of metrics to disk.
       
        Args: 
          val_metrics: A dict with keys "metric" and "model", where dict["model"]
            is another dict with keys being different metrics and values
            being the average metric values. Another key in val_metrics
            is "{model_name}_raw_metric_values" which is a dict with key as
            metric and value as the list of raw metric values for all dataopoints
            in the split being evaluated.
          eval_split_name: str, name of split being evaluated: train val or test
          test_metric_name: str, name of the validation metric based on which
            we chose the checkpoint to evaluate on test
          write_raw_metrics: bool, whether to write the raw metric values to
            disk
          write_metrics_to_txt: whether to write the aggregate metric values to
            text in addition to writing tensorboard summaries.
        Returns:
          is_repreated_checkpoint: bool, set to True if we were asked to evaluate
            a checkpoint that has already been processed previously. Only
            implemented for validation as for test one might want to save
            results for a new best checkpoint that has just been updated based 
            on validation results.
        """

        # Log all validation metrics to tensorboard (pop the "metric" key, which was only relevant
        # to learning rate scheduling and checkpointing).
        if "metric" in val_metrics.keys():
            val_metrics.pop("metric")
            
        if write_metrics_to_txt == True:
            f = open(self._write_metrics_file, "a")

        for model_name in val_metrics:
            # Only process aggregate metric values.
            if "raw_metric_values" in model_name:
                continue
            for metric_name in val_metrics[model_name]:
                self._tensorboard_writer.add_scalar(
                    f"val/metrics/{model_name}/{metric_name}",
                    val_metrics[model_name][metric_name],
                    self._iteration,
                )
                if write_metrics_to_txt == True:
                    f.write(
                        f"{self._iteration},{metric_name},"
                        f"{val_metrics[model_name][metric_name]}\n"
                    )
            if write_raw_metrics == True:
                if eval_split_name == None:
                    raise ValueError("Must provide an eval split name.")
                
                # File name is like raw_metrics_model_test_modelmap.json for
                # test or raw_metrics_model_val_1000.json for validation.
                if eval_split_name == "test":
                    identifier = test_metric_name 
                else:
                    identifier = self._iteration
                    
                json_path = os.path.join(
                    self._serialization_dir,
                    f"raw_metrics_{model_name}_{eval_split_name}_{identifier}.json"
                )
               
                # If we are testing on validation checkpoints, and we find 
                # a checkpoint that has already been evaluated, return without
                # updating the result file.
                if os.path.exists(json_path) and eval_split_name != "test":
                    is_repeated_checkpoint = True
                    return is_repeated_checkpoint
                
                with open(json_path, 'w') as g:
                    json.dump(val_metrics[f"{model_name}_raw_metric_values"],
                              g)

        if write_metrics_to_txt == True:
            f.close()
        return False


    def load_checkpoint(self,
                        checkpoint_path: str,
                        iteration: Optional[int] = None):
        r"""
        Load a checkpoint to continue training from. The iteration when this checkpoint was
        serialized, is inferred from its name (so do not rename after serialization).

        Parameters
        ----------
        checkpoint_path: str
            Path to a checkpoint containing models and optimizers of the phase which is being
            trained on.

        iteration: int, optional (default = None)
            Iteration number. If ``None``, infer from name of checkpoint file.
        """
        if checkpoint_path is None:
            logging.warn("No checkpoint found to load from.")
            return

        training_checkpoint: Dict[str, Any] = torch.load(checkpoint_path)
        for key in training_checkpoint:
            if key == "optimizer":
                self._optimizer.load_state_dict(training_checkpoint[key])
            else:
                self._models[key].load_state_dict(training_checkpoint[key])

        # Infer iteration number from checkpoint file name, if not specified.
        if "best" not in checkpoint_path or iteration is not None:
            self._iteration = iteration or int(
                checkpoint_path.split("_")[-1][:-4])

        logging.info(f"Loaded from checkpoint {checkpoint_path}")

    def _cycle(self, dataloader: DataLoader
               ) -> Generator[Dict[str, torch.Tensor], None, None]:
        r"""
        A generator which yields a random batch from dataloader perpetually. This generator is
        used in the constructor.

        Extended Summary
        ----------------
        This is done so because we train for a fixed number of iterations, and do not have the
        notion of 'epochs'. Using ``itertools.cycle`` with dataloader is harmful and may cause
        unexpeced memory leaks.
        """
        while True:
            for batch in dataloader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self._device)
                yield batch

    @property
    def iteration(self):
        return self._iteration

    @property
    def models(self):
        return self._models