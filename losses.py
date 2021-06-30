# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Define losses for training abstraction models."""

import numpy as np
import torch
import time
import logging

from collections import defaultdict
from sklearn.metrics import average_precision_score
from multiprocessing import Pool

from dataloaders.utils import _numeric_string_array_to_numbers


def compute_map(this_all_hyp_idx_consistent_dense,
                this_posterior_prob,
                this_gt_labels,
                this_batch_model_scores,
                this_true_hyp_idx):
    """Compute mean average precision metrics.

    Currently implements three different kind of mean average precision metrics:
    map, optimistic_map and expected map. MAP labels the ground truth according
    to the true hypothesis corresponding to the concept to be tested, optimistic
    map labels a datapoint as true if it is true for *any* of the concepts
    that are consistent with a support set, while expected MAP computes an
    expectation of the MAP value over the posterior distribution.

    Args:
      this_all_hyp_idx_consistent_dense: A np.array of ints with true hypotheses
      this_posterior_prob: np.array of floats same length as
        this_all_hyp_idx_consistent_dense
      this_gt_labels: A np.array of Bool with ground truth labels. Size: [N x V]
        where V is the total number of hypotheses.
      this_batch_model_scores: An np.array of float with scores. Size: [N]
      this_true_hyp_idx: An np.array of int. Size: [1]
    Returns:
      average_precision: A dict with keys "exepected_map", "map" and
        "optimistic_map" and values scalar float values with map values.
    """
    average_precision = {}

    if this_posterior_prob is not None:
        class_weights = this_posterior_prob
        if len(class_weights) != len(this_all_hyp_idx_consistent_dense):
            raise ValueError("Expect a weight for every class.")
    else:
        class_weights = np.ones(len(this_all_hyp_idx_consistent_dense)) * (
            1 / float(len(this_all_hyp_idx_consistent_dense)))

    average_precision_all_consistent = []

    for this_class_id in this_all_hyp_idx_consistent_dense:
        average_precision_all_consistent.append(
            average_precision_score(
                y_true=(this_gt_labels[:, this_class_id]).numpy(),
                y_score=this_batch_model_scores.numpy()))

    average_precision["expected_map"] = (
        np.sum(np.array(average_precision_all_consistent) * class_weights) /
        np.sum(class_weights))

    true_class_id = [
        x_idx for x_idx, x in enumerate(this_all_hyp_idx_consistent_dense)
        if x == this_true_hyp_idx
    ][0]

    average_precision["map"] = (
        average_precision_all_consistent[true_class_id])

    average_precision["optimistic_map"] = (
        average_precision_score(
            y_true=(torch.sum(this_gt_labels, dim=-1) != 0).numpy(),
            y_score=this_batch_model_scores.numpy()))
    return average_precision


def class_balanced_accuracy(preds: torch.Tensor, gt: torch.Tensor,
                            n_class=2) -> float:
    class_balanced_acc = []

    preds = preds.detach().cpu()
    gt = gt.detach().cpu()

    for cl in range(n_class):
        gt_in_class = torch.eq(gt, cl).float()
        masked_acc_value = torch.eq(preds, gt).float() * gt_in_class
        masked_acc_value = torch.sum(masked_acc_value) / torch.sum(gt_in_class)

        class_balanced_acc.append(masked_acc_value.item())
    return np.mean(class_balanced_acc)


class TotalLoss(object):
    def __init__(self):
        self._losses = []
        self._weights = []
        self._names = []

    def push(
            self,
            this_loss,
            this_name,
            this_weight=1.0,
    ):
        if not isinstance(this_loss, torch.Tensor):
            raise ValueError("Expect `torch.Tensor`")
        if not isinstance(this_weight, float):
            raise ValueError("Expect float")
        if not isinstance(this_name, str):
            raise ValueError("Expect str")
        if this_loss.dim() != 1:
            raise ValueError("Expect a 1D loss tensor.")

        self._losses.append(this_loss)
        self._weights.append(this_weight)
        self._names.append(this_name)

    def aggregate_loss(self):
        if len(self._losses) == 0:
            raise ValueError

        all_losses = self._losses[0] * self._weights[0]
        for idx in range(1, len(self._losses)):
            all_losses = all_losses + self._losses[idx] * self._weights[idx]

        return all_losses

    @property
    def keys(self):
        return self._names

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise ValueError("Expect string key.")
        this_key_idx = [idx for idx, x in enumerate(self._names)
                        if x == key][0]
        return self._losses[this_key_idx]


class _LossFun(object):
    def __init__(self, checkpointing_metric_prefix="model"):
        self._checkpointing_metric_prefix = checkpointing_metric_prefix

        self._metrics_val_list = []
        self._checkpointing_metric_name = (self._checkpointing_metric_prefix
                                           + self.checkpointing_metric[0])
        self._min_or_max = self.checkpointing_metric[1]

    def _compute_loss_and_metrics(self, model_output, batch, metric_prefix):
        raise NotImplementedError("Please implement in derived class.")

    def __call__(self, model_output, batch, metric_prefix=None):
        """Compute the metric of interest on a minibatch.

        The metric_prefix argument is used to provide external information about
        the kind of scores we wish to evaluate. For example, one might want to
        compute the same metrics for random scores or the scores from a model.
        In this case we can provide a different prefix for the two cases.

        Args:
          model_output: A dict with the outputs from the model.
          batch: The input batch fed to the model
          mertric_prefix: Str, the prefix to use for all the metrics. Defaults
            to self._checkpointing_metric_prefix
        Returns:
          loss: A loss `Tensor`
          metrics: A dict with keys metric names and values values of metrics.
        """
        if metric_prefix == None:
            metric_prefix = self._checkpointing_metric_prefix

        loss, metrics = self._compute_loss_and_metrics(model_output, batch,
                                                       metric_prefix)
        if not all([metric_prefix in x for x in metrics.keys()]):
            raise ValueError("Metric keys must incorporate the "
                             "requested metric prefix.")
        self._metrics_val_list.append(metrics)
        return loss, metrics

    def aggregate_metrics(self, report_mean_only=True):
        r"""Aggregates the metrics computed so far.

        Aggregates all the metrics computed so far using the __call__ routine.
        In practice, this is useful as multiple class to the __call__ routine
        are typically, say for different batches in a dataset. Aggregating
        across them, then gives us the mean (or the collated) performance across
        the whole dataset.

        NOTE: Important behavior is that the metrics being tracked are reset
        on every call to this routine as we want to make sure that the metrics
        being computed are not stale.

        Args:
          report_mean_std_only: Bool, if true then report only the mean and
            standard deviation of the metrics, instead of the full values
            computed across the whole batch.

        Returns:
          aggregate_metrics: A dict with keys metric name and value either the
            computed metric across all datapoints in the dataloader or the value
            is the mean metric value across datapoints.
        """
        if len(self._metrics_val_list) == 0:
            raise ValueError("Aggregate after calling loss function.")

        batch_mean_metric_only = True

        aggregate_metrics = defaultdict(list)
        for key in self._metrics_val_list[0]:
            for idx, _ in enumerate(self._metrics_val_list):
                if isinstance(self._metrics_val_list[idx][key], list):
                    batch_mean_metric_only = False
                    aggregate_metrics[key].extend(
                        self._metrics_val_list[idx][key])
                else:
                    aggregate_metrics[key].append(
                        self._metrics_val_list[idx][key])

        mean_aggregate_metrics = {}
        if report_mean_only == True:
            for key in aggregate_metrics:
                mean_aggregate_metrics[key] = float(np.mean(aggregate_metrics[key]))

        metrics_to_return = {}
        if mean_aggregate_metrics.get(self._checkpointing_metric_name) is not None:
            metrics_to_return["metric"] = mean_aggregate_metrics[
                self._checkpointing_metric_name]

        metrics_to_return["model"] = mean_aggregate_metrics
        if batch_mean_metric_only == True:
            metrics_to_return["model_raw_metric_values"] = None
        else:
            metrics_to_return["model_raw_metric_values"] = aggregate_metrics

        self._reset_metrics()

        return metrics_to_return

    def _reset_metrics(self):
        self._metrics_val_list = []

    @property
    def min_or_max(self):
        return self._min_or_max

    @property
    def checkpointing_metric(self):
        """Define the metric to use for checkpointing and if better is lower.

        Returns:
          Name of the metric to use.
          "min" if lower is better, "max" if opposite.
        """
        raise NotImplementedError("Please implement in derived class.")


class MetaLearningMeanAveragePrecision(_LossFun):
    def _compute_loss_and_metrics(self, model_output, batch, metric_prefix):
        """Compute the average precision metrics.

        There are currently two kinds of average precision metrics implemented,
        one is the optimistic average precision metric in which all consistent
        hypotheses with the ground truth are given the credit, and the other
        is the average precision metric where only one is given credit.

        If batch contains a [B x N] `Tensor` with values 1, it means that
        there is only one class that we are considering as positive, and then
        everything with 0 is a negative. However, if [B x N] `Tensor` contains
        values like 1, 2, 3 etc. it means these are all different notions of
        positive classes, and we take an expectation over them. The default
        weights with which we take the expectation is uniform, but can also
        be provided in `batch`.

        Args:
          model_output: Confidence/ scores from the model [B x N x L] `Tensor`
          batch: A dict with keys being the metric name and the value being
            the ground truth for computing mean average precision, which is
            a [B x N] `Tensor`
        Returns:
          average_precision: A dict with key being the metric and value being
            a list with the average precision for each datapoint in the mini
            batch.
        """
        eval_scores = model_output["scores"]
        gt_labels = model_output["gt_labels"]
        all_hyp_idx_consistent_dense = _numeric_string_array_to_numbers(batch[
            "all_consistent_hypotheses_idx_dense"], cast_type="int")
        true_hyp_idx = batch["hypotheses_idx_dense"]
        posterior_probs_dense = _numeric_string_array_to_numbers(
            batch["posterior_probs_dense"])

        average_precision = []
        for idx in range(eval_scores.size(0)):
            average_precision.append(
                compute_map(all_hyp_idx_consistent_dense[idx],
                            posterior_probs_dense[idx],
                            gt_labels[idx].to("cpu"),
                            eval_scores[idx].to("cpu"),
                            true_hyp_idx[idx].to("cpu"))
            )

        metrics = defaultdict(list)
        for m in average_precision:
            for key in m:
                metrics[metric_prefix + key].append(m[key])

        return None, metrics

    @property
    def checkpointing_metric(self):
        return "map", "max"


class NegativeLogLikelihoodMultiTask(_LossFun):
    # TODO(ramav): Add a sentinel like behavior here.
    def __init__(self, num_classes=2, alpha=None, pad_token_idx=None):
        super(NegativeLogLikelihoodMultiTask, self).__init__()
        if alpha == None:
            alpha = 0
        elif pad_token_idx == None:
            raise ValueError("Need to provide pad token index.")

        self._alpha = alpha
        self._pad_token_idx = pad_token_idx
        self._num_classes = num_classes

    def _compute_loss_and_metrics(self, model_output, batch, metric_prefix):
        """Compute the loss that we report for the model.

        Args:
          model_output: A dict with keys "neg_log_p_y" and "logits_for_sequence"
            and values Tensors of [B x N x 2] and [B x T x V] repsectively.
            "logits_for_sequence" can also be None in which case sequence loss
            is not computed.
          batch: A dict with keys and values fed for the minibatch.
          metric_prefix: An str with prefix for metric values

        Returns:
          Aggregate loss across all losses of interest, a `Tensor` of [B]
          metrics: A dict with keys as names of metrics and values scalar values
            of metrics.
        """
        neg_log_p_y = model_output["neg_log_p_y"]
        target_inds = batch["query_labels"]

        all_losses = TotalLoss()
        metrics = {}

        # Optimistic labels where everything that is consistent with the
        # training valid hypotheses are labelled as true.
        optimistic_target_inds = batch["optimistic_query_labels"]
        classification_loss = neg_log_p_y.gather(
            2, target_inds.unsqueeze(-1)).view(-1).reshape_as(target_inds)
        classification_loss = classification_loss.mean(-1)
        all_losses.push(classification_loss, "classification_loss")

        _, y_hat = neg_log_p_y.detach().cpu().min(2)

        # Report Class Balanced Accuracy.
        acc_val = class_balanced_accuracy(preds=y_hat,
                                          gt=target_inds,
                                          n_class=self._num_classes)

        optimistic_acc_val = class_balanced_accuracy(preds=y_hat,
                                                     gt=optimistic_target_inds,
                                                     n_class=self._num_classes)

        if self._alpha != 0 and model_output.get("logits_for_sequence") != None:
            logits_for_sequence = model_output["logits_for_sequence"]
            target_sequence_indices = batch["hypotheses_encoded"]
            sequence_loss = torch.functional.F.cross_entropy(
                logits_for_sequence.transpose(2, 1),
                target_sequence_indices,
                ignore_index=self._pad_token_idx,
                reduction='none')
            sequence_loss = sequence_loss.mean(-1)

            all_losses.push(sequence_loss, "sequence_loss", self._alpha)

        metrics = {}
        metrics[metric_prefix + "metrics/all_losses"] = (
            all_losses.aggregate_loss().mean().detach().item())

        for mtrc in all_losses.keys:
            metrics[f"{metric_prefix}metrics/{mtrc}"] = (
                all_losses[mtrc].mean().detach().item()
            )

        metrics[metric_prefix + "metrics/acc"] = acc_val.item()
        metrics[metric_prefix + "metrics/optimistic_acc"] = (
            optimistic_acc_val.item())

        return all_losses.aggregate_loss(), metrics

    @property
    def checkpointing_metric(self):
        return "metrics/classification_loss", "min"