# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import glob
import numpy as np
import os
import random
import torch

from typing import Any, Dict, Optional, Union, Type
from torch import nn, optim


class CheckpointManager(object):
    r"""
    A :class:`CheckpointManager` periodically serializes models and optimizer as .pth files during
    training, and keeps track of best performing checkpoint based on an observed metric.

    Extended Summary
    ----------------
    It saves state dicts of models and optimizer as ``.pth`` files in a specified directory. This
    class closely follows the API of PyTorch optimizers and learning rate schedulers.

    Notes
    -----
    For :class:`~torch.nn.DataParallel` objects, ``.module.state_dict()`` is called instead of
    ``.state_dict()``.

    Parameters
    ----------
    models: Dict[str, torch.nn.Module]
        Models which need to be serialized as a checkpoint.
    optimizer: torch.optim.Optimizer
        Optimizer which needs to be serialized as a checkpoint.
    serialization_dir: str
        Path to an empty or non-existent directory to save checkpoints.
    mode: str, optional (default="max")
        One of ``min``, ``max``. In ``min`` mode, best checkpoint will be recorded when metric
        hits a lower value; in `max` mode it will be recorded when metric hits a higher value.
    filename_prefix: str, optional (default="checkpoint")
        Prefix of the to-be-saved checkpoint files.

    Examples
    --------
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ckpt_manager = CheckpointManager({"model": model}, optimizer, "/tmp/ckpt", mode="min")
    >>> num_epochs = 20
    >>> for epoch in range(num_epochs):
    ...     train(model)
    ...     val_loss = validate(model)
    ...     ckpt_manager.step(val_loss, epoch)
    """
    def __init__(
        self,
        models: Dict[str, nn.Module],
        optimizer: Type[optim.Optimizer],
        serialization_dir: str,
        mode: str = "max",
        filename_prefix: str = "checkpoint",
    ):
        r"""Initialize checkpoint manager."""
        for key in models:
            if not isinstance(models[key], nn.Module):
                raise TypeError("{} is not a Module".format(type(models).__name__))

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))

        self._models = models
        self._optimizer = optimizer
        self._serialization_dir = serialization_dir

        self._mode = mode
        self._filename_prefix = filename_prefix

        # Initialize members to hold state dict of best checkpoint and its performance.
        self._best_metric: Optional[Union[float, torch.Tensor]] = None
        self._best_ckpt: Dict[str, Any] = {}

    def step(self, metric: Union[float, torch.Tensor], epoch_or_iteration: int):
        r"""Serialize checkpoint and update best checkpoint based on metric and mode."""

        # Update best checkpoint based on metric and metric mode.
        if not self._best_metric:
            self._best_metric = metric

        models_state_dict: Dict[str, Any] = {}
        for key in self._models:
            if isinstance(self._models[key], nn.DataParallel):
                models_state_dict[key] = self._models[key].module.state_dict()
            else:
                models_state_dict[key] = self._models[key].state_dict()

        if (self._mode == "min" and metric < self._best_metric) or (
            self._mode == "max" and metric > self._best_metric
        ):
            self._best_metric = metric
            self._best_ckpt = copy.copy(models_state_dict)

        # Serialize checkpoint corresponding to current epoch (or iteration).
        torch.save(
            {**models_state_dict, "optimizer": self._optimizer.state_dict()},
            os.path.join(
                self._serialization_dir, f"{self._filename_prefix}_{epoch_or_iteration}.pth"
            ),
        )
        # Serialize best performing checkpoint observed so far. By default,
        # the best checkpoint is saved as "_best".
        torch.save(
            self._best_ckpt,
            os.path.join(self._serialization_dir, f"{self._filename_prefix}_best.pth"),
        )

    def best_checkpoint(self, based_on_metric: str = ""):
        r"""Returns the best checkpoint.

        Defaults to returning the best checkpoint stored by step() above, if
        based_on_metric is not provided. If not, then looks for a file like,
        `checkpoint_best_modelmap' for example, if `based_on_metric` is
        `modelmap'.
        """
        # If based_on_metric is like modelmetrics/acc, then read it as
        # modelmetrics_acc
        based_on_metric = based_on_metric.replace("/", "_")
        best_checkpoint_path = os.path.join(
            self._serialization_dir, "%s.pth" % ("_".join(
                [self._filename_prefix, "best", based_on_metric])).rstrip("_"))

        if not os.path.exists(best_checkpoint_path):
            raise ValueError(
                f"Best checkpoint based on {based_on_metric} does not exist.")

        return best_checkpoint_path, None

    @property
    def latest_checkpoint(self):
        all_checkpoint_epochs_or_iterations = glob.glob(
            os.path.join(self._serialization_dir,
                         f"{self._filename_prefix}_*.pth"))

        # TODO(ramav): This is a bit brittle, replace the "best" check with
        # an int check.
        all_checkpoint_epochs_or_iterations = [
            int(x.split("_")[-1].split(".")[0])
            for x in all_checkpoint_epochs_or_iterations if 'best' not in x
        ]

        if len(all_checkpoint_epochs_or_iterations) != 0:
            latest_epoch_or_iteration = np.max(
                all_checkpoint_epochs_or_iterations)
            return os.path.join(
                self._serialization_dir,
                f"{self._filename_prefix}_{latest_epoch_or_iteration}.pth"
            ), latest_epoch_or_iteration
        return None, -1

    def all_checkpoints(self, sort_iterations: bool = False,
                        random_shuffle: bool = False):
        if sort_iterations is True and random is True:
            raise ValueError("Only one of sorted or random can be true.")

        all_checkpoint_epochs_or_iterations = glob.glob(
            os.path.join(self._serialization_dir,
                         f"{self._filename_prefix}_*.pth"))

        # TODO(ramav): This is a bit brittle, replace the "best" check with
        # an int check.
        all_checkpoint_epochs_or_iterations = [
            int(x.split("_")[-1].split(".")[0])
            for x in all_checkpoint_epochs_or_iterations if 'best' not in x
        ]

        # Sort iterations in increasing order, so that when we pop we
        # pick the latest iteration.
        if sort_iterations == True:
            all_checkpoint_epochs_or_iterations = sorted(
                all_checkpoint_epochs_or_iterations)
        elif random_shuffle == True:
            random.shuffle(all_checkpoint_epochs_or_iterations)

        if len(all_checkpoint_epochs_or_iterations) != 0:
            return [(os.path.join(
                self._serialization_dir, f"{self._filename_prefix}_{x}.pth"), x)
                    for x in all_checkpoint_epochs_or_iterations]

        return None, -1
