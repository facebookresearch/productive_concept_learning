# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from typing import Any, Dict, List, Optional, Type

from omegaconf import OmegaConf

from losses import _LossFun
from losses import NegativeLogLikelihoodMultiTask


class _Evaluator(object):
    r"""
    A base class for generic evaluation of models. This class can have multiple models interacting
    with each other, rather than a single model, which is suitable to our use-case (for example,
    ``module_training`` phase has two models:
    :class:`~probnmn.models.program_generator.ProgramGenerator` and
    :class:`~probnmn.models.nmn.NeuralModuleNetwork`). It offers full flexibility, with sensible
    defaults which may be changed (or disabled) while extending this class.

    Extended Summary
    ----------------
    Extend this class and override :meth:`_do_iteration` method, with core evaluation loop - what
    happens every iteration, given a ``batch`` from the dataloader this class holds.

    Notes
    -----
    1. All models are `passed by assignment`, so they could be shared with an external trainer.
       Do not set ``self._models = ...`` anywhere while extending this class.

    2. An instantiation of this class will always be paired in conjunction to a
       :class:`~probnmn.trainers._trainer._Trainer`. Pass the models of trainer class while
       instantiating this class.

    Parameters
    ----------
    config: Config
        A :class:`~probnmn.Config` object with all the relevant configuration parameters.
    dataloader: torch.utils.data.DataLoader
        A :class:`~torch.utils.data.DataLoader` which provides batches of evaluation examples. It
        wraps one of :mod:`probnmn.data.datasets` depending on the evaluation phase.
    models: Dict[str, Type[nn.Module]]
        All the models which interact with each other for evaluation. These are one or more from
        :mod:`probnmn.models` depending on the evaluation phase.
    gpu_ids: List[int], optional (default=[0])
        List of GPU IDs to use or evaluation, ``[-1]`` - use CPU.
    """

    def __init__(
        self,
        config: OmegaConf,
        loss_fn: _LossFun,
        dataloader: DataLoader,
        models: Dict[str, Type[nn.Module]],
        gpu_ids: List[int] = [0],
    ):
        self._C = config
        self._dataloader = dataloader
        self._models = models
        self._loss_fn = loss_fn

        # Set device according to specified GPU ids. This device is only required for batches,
        # models will already be on apropriate device already, if passed from trainer.
        self._device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids[0] >= 0 else "cpu")

    @property
    def models(self):
        return self._models

    def evaluate(self, eval_object: str = "model",
                 num_batches: Optional[int] = None,
                 output_raw_preds=False) -> Dict[str, Any]:
        r"""
        Perform evaluation using first ``num_batches`` of dataloader and return all evaluation
        metrics from the models.

        Parameters
        ----------
        eval_object: str, optional (default=None), can be one of "random",
            "weak_oracle" or "oracle", or "model"
        num_batches: int, optional (default=None)
            Number of batches to use from dataloader. If ``None``, use all batches.

        Returns
        -------
        Dict[str, Any]
            Final evaluation metrics for all the models.
        Dict[str, Any]
            Raw predictions from the model
        """
        # Switch all models to "eval" mode.
        for model_name in self._models:
            self._models[model_name].eval()

        model_outputs = []
        with torch.no_grad():
            for iteration, batch in enumerate(self._dataloader):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self._device)

                model_outputs.append(
                    self._do_iteration(batch, self._loss_fn, eval_object))

                if num_batches is not None and iteration > num_batches:
                    break

        eval_metrics = self._loss_fn.aggregate_metrics()

        if len(self._models) > 1 or list(self._models.keys())[0] != "model":
            raise NotImplementedError("Only supports one model.")

        # Switch all models back to "train" mode.
        for model_name in self._models:
            self._models[model_name].train()

        if output_raw_preds == True:
            return eval_metrics, model_outputs

        return eval_metrics

    def _do_iteration(self, batch: Dict[str, Any], loss_fn: _LossFun,
                      eval_object: str = "model") -> Dict[str, Any]:
        r"""
        Core evaluation logic for one iteration, operates on a batch. This base class has a dummy
        implementation - just forward pass through some "model".

        Parameters
        ----------
        batch: Dict[str, Any]
            A batch of evaluation examples sampled from dataloader. See :func:`evaluate` on how
            this batch is sampled.

        Returns
        -------
        Dict[str, Any]
            An output dictionary typically returned by the models. This may contain predictions
            from models, validation loss etc.
        """
        # Multiple evaluation objects are only supported for the meta
        # learning negative log likelihood metric
        if not isinstance(loss_fn, NegativeLogLikelihoodMultiTask):
            raise NotImplementedError("eval_object support only for NLL loss.")

        if eval_object == "model":
            output_dict = self._models["model"](batch)
        elif "oracle" in eval_object:
            if eval_object == "oracle":
                posterior_dist = batch["posterior_probs_sparse"]
            elif eval_object == "weak_oracle":
                posterior_dist = batch["posterior_probs_train_sparse"]

            # B x N x H
            query_multihot_perdata_labels = batch[
                "query_multihot_perdata_labels"]
            b = query_multihot_perdata_labels.shape[0]

            batch_eval_scores = []
            for it in range(b):
                this_multihot_labels = query_multihot_perdata_labels[it]
                eval_scores = (
                    posterior_dist[it].cpu().detach().unsqueeze(0) *
                    this_multihot_labels.cpu()
                ).sum(-1)
                batch_eval_scores.append(eval_scores)

            batch_eval_scores = torch.stack(batch_eval_scores, dim=0)

        elif eval_object == "random":
            labels = batch["query_labels"]
            batch_eval_scores = torch.rand(labels.shape[0], labels.shape[1])

        if eval_object != "model":
            eval_probs = torch.zeros(batch_eval_scores.shape[0],
                                      batch_eval_scores.shape[1],
                                      2)

            eval_probs[:, :, self._dataloader.dataset.true_class_id] = batch_eval_scores
            eval_probs[:, :, 1 - self._dataloader.dataset.true_class_id] = 1 - batch_eval_scores

            # Add a small quantity for log 0, does not normalize but we dont
            # care here.
            output_dict = {
                "neg_log_p_y": -1 * torch.log(eval_probs + 1e-12), # For log 0
            }
            output_dict["neg_log_p_y"] = output_dict["neg_log_p_y"].to(
                self._device)

        loss_fn(output_dict, batch, metric_prefix=eval_object)
        return output_dict
