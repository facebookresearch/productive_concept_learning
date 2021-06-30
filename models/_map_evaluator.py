# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional, Type

import torch
import time
import logging

from collections import defaultdict

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from omegaconf import OmegaConf

from models._evaluator import _Evaluator
from losses import _LossFun
from losses import MetaLearningMeanAveragePrecision
from models.utils import dict_to_device


class _MapEvaluator(_Evaluator):
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
            loss_fn: MetaLearningMeanAveragePrecision,
            dataloader: DataLoader,
            test_loader: DataLoader,
            models: Dict[str, Type[nn.Module]],
            gpu_ids: List[int] = [0],
    ):
        r"""
        Initialize the MapEvaluator object.

        Args:
          config: An `OmegaConf` object provided by hydra.
          dataloader: A meta learning dataloader with query and support.
          test_loader: A dataloader to access a distinct set of images.
          models: Dict, with key values str, and model.
          gpu_ids: Which gpus to run evaluation on.
        """
        self._C = config
        self._dataloader = dataloader
        self._test_loader = test_loader
        self._models = models
        self._loss_fn = loss_fn

        if isinstance(self._test_loader.sampler, RandomSampler):
            raise ValueError("Expect no shuffling in the test loader.")

        if isinstance(self._dataloader.sampler, RandomSampler):
            raise ValueError("Expect no shuffling in the validation loader.")

        if not isinstance(self._loss_fn, MetaLearningMeanAveragePrecision):
            raise ValueError("Expect meta learning ap object for the loss")

        # Set device according to specified GPU ids. This device is only required for batches,
        # models will already be on apropriate device already, if passed from trainer.
        self._device = torch.device(
            f"cuda:{gpu_ids[0]}" if gpu_ids[0] >= 0 else "cpu")

        self.all_hyp_str_to_idx = {
            v: k
            for k, v in enumerate(self._dataloader.dataset.all_hypotheses_across_splits)
        }
        self.num_total_hypotheses = len(
            self._dataloader.dataset.all_hypotheses_across_splits)

        logging.info("Setting up MAP evaluation.")
        with torch.no_grad():
            self._held_out_image_hypotheses = []
            for _, held_out_batch in enumerate(self._test_loader):
                self._held_out_image_hypotheses.append(
                    held_out_batch["labels"].bool())

            self._held_out_image_hypotheses = torch.cat(
                self._held_out_image_hypotheses, dim=0)
        logging.info("Done setting up map evaluation.")

    def evaluate(self, num_batches: Optional[int] = None,
                 eval_object:str ="model",
                 output_raw_results=False) -> Dict[str, Any]:
        r"""
        Perform evaluation using first ``num_batches`` of dataloader and return all evaluation
        metrics from the models.

        Args:
          num_batches: int, optional (default=None)
            Number of batches to use from dataloader. If ``None``, use all batches.
          eval_object: str, kind of model / approach to evaluate.
          output_raw_results: bool, whether we output raw results or not
        Returns:
          Dict[str, Any]
            Final evaluation metrics for all the models.
        """
        if num_batches is not None:
            logging.info(f"Evaluating on {num_batches} batches.")

        self._held_out_features = []
        self._held_out_image_paths = []

        # Switch all models to "eval" mode.
        for model_name in self._models:
            self._models[model_name].eval()

        model_output = []

        with torch.no_grad():
            if eval_object == "model":
                for _, held_out_batch in enumerate(self._test_loader):
                    held_out_batch = dict_to_device(held_out_batch,
                                                    self._device)
                    feat = self._models["model"].creator.encoder(
                        held_out_batch["datum"])
                    self._held_out_features.append(feat)
                    self._held_out_image_paths.append(held_out_batch["path"])

            cpu_only_tensors = [
                "all_consistent_hypotheses_idx_sparse",
                "posterior_probs_sparse", "posterior_probs_train_sparse"
            ]

            for iteration, batch in enumerate(
                    self._dataloader):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        if key not in cpu_only_tensors:
                            batch[key] = batch[key].to(self._device)

                model_output.append(
                    self._do_iteration(batch, self._loss_fn, eval_object))

                if (iteration + 1) % 50 == 0:
                    logging.info("Finished %d steps of evaluation.", iteration)

                if num_batches is not None and iteration > num_batches:
                    break

        # keys: `self._models.keys()`
        eval_metrics = self._loss_fn.aggregate_metrics()

        # Switch all models back to "train" mode.
        for model_name in self._models:
            self._models[model_name].train()

        if output_raw_results == True:
            return eval_metrics, model_output

        return eval_metrics

    def _do_iteration(self, batch: Dict[str, Any], loss_fn: _LossFun,
                      eval_object: str) -> Dict[str, Any]:
        r"""
        Takes as input a batch of meta learning examples to evaluate, and
        iterates over a large set of test data points (provided by the
        test_loader) to compute metrics.

        Dynamically figures out the labels for the examples in the test set and
        computes the mean average precision loss accordingly.
        """
        if not isinstance(loss_fn, MetaLearningMeanAveragePrecision):
            raise NotImplementedError("eval_object support only for mAP"
                                      " loss.")

        classifier = self._models["model"].creator(batch["support_images"],
                                                   batch["support_labels"])
        labels = batch["all_consistent_hypotheses_idx_sparse"].unsqueeze(
            1) * self._held_out_image_hypotheses.unsqueeze(0)
        if eval_object == "model":
            eval_scores = []
            for test_feat in self._held_out_features:
                log_prob_label = torch.squeeze(
                    -1 * self._models["model"].applier(classifier, test_feat)
                    ["neg_log_p_y"][:, :, self._dataloader.dataset.
                                    true_class_id])  # B x N x L
                eval_scores.append(log_prob_label.cpu().detach())
            eval_scores = torch.cat(eval_scores, dim=1)
        elif eval_object == "oracle":
            eval_scores = (
                batch["posterior_probs_sparse"].detach().unsqueeze(1) *
                self._held_out_image_hypotheses.type(
                    torch.float).unsqueeze(0)).sum(-1)
        elif eval_object == "weak_oracle":
            eval_scores = (
                batch["posterior_probs_train_sparse"].detach().unsqueeze(1) *
                self._held_out_image_hypotheses.type(
                    torch.float).unsqueeze(0)).sum(-1)
        elif eval_object == "random":
            eval_scores = torch.rand(labels.shape[0], labels.shape[1])
        loss_fn({
            "scores": eval_scores,
            "gt_labels": labels,
        },
                batch,
                metric_prefix=eval_object)
        # Filter the gt labels only corresponding to the hypotheses we are
        # working with based on the support set. Above we needed lables for 
        # all hypotheses for the possibility of computing metrics like expected
        # mAP etc.  
        gt_labels = []
        for idx, this_hyp in enumerate(list(batch['hypotheses_idx_dense'])):
            gt_labels.append(labels[idx, :, this_hyp])
        gt_labels = torch.stack(gt_labels, dim=0)
        
        return {"scores": eval_scores, "gt_labels": gt_labels}
