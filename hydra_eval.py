# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Eval script for running adhoc categorization.
import hydra
import logging
import os

import dataloaders
import losses
import models

from collections import defaultdict
from tqdm import tqdm
from time import time
from time import sleep

from dataloaders.get_dataloader import GetDataloader
from models._trainer import _Trainer
from models._evaluator import _Evaluator
from models._map_evaluator import _MapEvaluator

_PRINT_EVERY = 10
_NUM_EVAL_BATCHES_FOR_SPLIT = defaultdict(lambda: None)
_NUM_EVAL_BATCHES_FOR_SPLIT["train"] = 625


def load_model(cfg, vocabulary):
  get_model_helper = hydra.utils.instantiate(cfg.model)
  return get_model_helper(vocabulary)


class _Workplace(object):
  def __init__(self, cfg):
    self.cfg = cfg
    self._eval_split_name = self.cfg.eval_split_name

    if self._eval_split_name == "qualitative":
        raise ValueError("Use hydra_qualitative.py for qualitative eval.")

    if self._eval_split_name not in self.cfg.splits.split(' & '):
      raise ValueError("Need to load the evaluation split.")

    # Initialize the data loader.
    get_data_loader = hydra.utils.instantiate(self.cfg.data_args)
    dataloaders = get_data_loader(self.cfg, self.cfg._data.map_eval_batch_size)

    # Train, val and test, all vocabularies are the same.
    self._vocabulary = dataloaders[self._eval_split_name].dataset.vocabulary
    logging.warn("Please ensure train, val and test "
                 "loaders give the same vocabulary.")

    # Initialize the model/trainer
    model = load_model(self.cfg, self._vocabulary)
    loss_fn = hydra.utils.instantiate(config=self.cfg.loss,
                                      pad_token_idx=self._vocabulary.pad())

    if self.cfg.model_or_oracle_metrics == "oracle":
      self._best_test_metric_or_oracle = "oracle"
    else:
      self._best_test_metric_or_oracle = self.cfg.eval_cfg.best_test_metric.replace('/', '_')


    # If evaluating on test mention how the validation checkpoint was chosen.
    # replace '/' with '_' if we have a best metric like modelmetrics/acc
    if self._eval_split_name == "test":
        write_metrics_file = (self._best_test_metric_or_oracle + "_" +
                              self.cfg.get(self._eval_split_name).rstrip(".pkl")
                              + "_metrics.txt")
    else:
        write_metrics_file = (self.cfg.get(self._eval_split_name).rstrip(".pkl")
                              + "_metrics.txt")

    # If we are not computing model metrics, then the best test metric used
    # to choose validation points is irrelevant
    trainer = _Trainer(config=self.cfg,
                       dataloader=dataloaders.get("train"),
                       models={"model": model},
                       loss_fn=loss_fn,
                       serialization_dir=os.getcwd(),
                       write_metrics_file=write_metrics_file)

    evaluator = _Evaluator(
      config=self.cfg,
      loss_fn=loss_fn,
      dataloader=dataloaders[self._eval_split_name],
      models={"model": model},
    )

    # Two kinds of evaluators: cheap and costly.
    costly_loss_fn = hydra.utils.instantiate(self.cfg.costly_loss)
    costly_evaluator = _MapEvaluator(
      config=self.cfg,
      loss_fn=costly_loss_fn,
      test_loader=dataloaders["cross_split"],
      dataloader=dataloaders[self._eval_split_name],
      models={"model": model},
    )

    self._trainer = trainer
    self._evaluator = evaluator
    self._costly_evaluator = costly_evaluator

  def run_eval(self):
    # Iterate over all the checkpoints.
    current_iteration = -1
    active_iteration = -1
    num_sleep = 0
    _WAIT=7200

    # Compute metrics that do not depend on a model.
    if self.cfg.model_or_oracle_metrics == "oracle":
      all_oracle_baselines = {}

      # Compute oracle metrics for the query accuracy metric
      logging.info("Computing query accuracy oracles and baselines.")
      for eval_object in ["weak_oracle", "random", "oracle"]:
        all_oracle_baselines["acc_" + eval_object] = self._evaluator.evaluate(
        eval_object=eval_object,
        num_batches=_NUM_EVAL_BATCHES_FOR_SPLIT[self._eval_split_name])

        logging.info(f"Completed {eval_object} evaluation.")

      # Compute oracle metrics for the map metric
      logging.info("Computing mAP oracles and baselines.")
      for eval_object in ["weak_oracle", "random", "oracle"]:
        all_oracle_baselines["map_" + eval_object] = self._costly_evaluator.evaluate(
        eval_object=eval_object,
        num_batches=_NUM_EVAL_BATCHES_FOR_SPLIT[self._eval_split_name])

        logging.info(f"Completed {eval_object} evaluation.")

    if self.cfg.model_or_oracle_metrics == "model" and self.cfg.eval_cfg.evaluate_all == True:
      all_checkpoint_paths_and_idx = (
        self._trainer._checkpoint_manager.all_checkpoints(
          sort_iterations=self.cfg.eval_cfg.sort_validation_checkpoints,
          random_shuffle=not self.cfg.eval_cfg.sort_validation_checkpoints))
      if not isinstance(all_checkpoint_paths_and_idx, list):
        raise ValueError("Not enough checkpoints to evaluate.")

      if self.cfg.eval_cfg.evaluate_once == True:
        raise ValueError("Evaluate once and evaluate all cannot be true at once.")

    while(True):
      if self.cfg.model_or_oracle_metrics == "model":
        if self.cfg.eval_cfg.evaluate_all == True:
          active_checkpoint, active_iteration = all_checkpoint_paths_and_idx.pop()
        else:
          # Test set is always evaluated with the best checkpoint from validation.
          if self._eval_split_name == "test":
            active_checkpoint, active_iteration = (
              self._trainer._checkpoint_manager.best_checkpoint(
                based_on_metric=self._best_test_metric_or_oracle)
            )
          else:
            active_checkpoint, active_iteration = (
              self._trainer._checkpoint_manager.latest_checkpoint)

        # Active iteration is None when we are evaluating the best checkpoint.
        if active_iteration is None and self._eval_split_name != "test":
            raise ValueError("Expect active_iteration to not be None.")
      else:
        active_checkpoint, active_iteration = None, None

      if active_iteration is None or active_iteration != current_iteration:
        all_metrics = {"model": {}, "metric": []}
        if self.cfg.model_or_oracle_metrics == "model":
          logging.info(f"Evaluating checkpoint {active_checkpoint}")
          self._trainer.load_checkpoint(active_checkpoint)
          all_metrics = self._costly_evaluator.evaluate(
            num_batches=_NUM_EVAL_BATCHES_FOR_SPLIT[self._eval_split_name])
          cheap_metrics = self._evaluator.evaluate(
            num_batches=_NUM_EVAL_BATCHES_FOR_SPLIT[self._eval_split_name])
          # Combine all model metrics and write them together.
          all_metrics["model"].update(cheap_metrics["model"])

        if self.cfg.model_or_oracle_metrics == "oracle":
          for _, oracle_or_baseline in all_oracle_baselines.items():
            all_metrics["model"].update(oracle_or_baseline["model"])

        # Serialize metric values for plotting.
        # TODO(ramav): Make changes so that oracle jobs can also write
        # raw metrics. Something that is currently not possible.
        is_repeated_checkpoint = self._trainer.write_metrics(all_metrics,
                                    eval_split_name=self._eval_split_name,
                                    test_metric_name=self._best_test_metric_or_oracle,
                                    write_raw_metrics=
                                    (self._best_test_metric_or_oracle == "model"
                                      and self.cfg.eval_cfg.write_raw_metrics)
        )

        # If we started evaluating checkpoints in descending order and we
        # hit a checkpoint that has already been evaluated, then stop the job.
        if is_repeated_checkpoint == True and self.cfg.eval_cfg.evaluate_all == True and (
          self.cfg.eval_cfg.sort_validation_checkpoints == True
        ):
          logging.info("Reached a checkpoint that has already been "
                       "evaluated, stopping eval now.")
          break


        current_iteration = active_iteration
        num_sleep = 0

      if (self.cfg.model_or_oracle_metrics == "oracle" or
          self.cfg.eval_cfg.evaluate_once == True or
          self._eval_split_name == "test"):
        logging.info("Finished evaluation.")
        break

      if self.cfg.eval_cfg.evaluate_all == True:
        if len(all_checkpoint_paths_and_idx) == 0:
          logging.info("No checkpoints left to evaluate. Finished evaluation.")
          break
      elif self.cfg.eval_cfg.evaluate_all == False:
        logging.info(f"Sleeping for {_WAIT} sec waiting for checkpoint.")
        sleep(_WAIT)
        num_sleep += 1

        if num_sleep == 10:
          logging.info(f"Terminating job after waiting for a new checkpoint.")
          break


@hydra.main(config_path='hydra_cfg/experiment.yaml')
def main(cfg):
  logging.info(cfg.pretty())
  logging.info("Base Directory: %s", os.getcwd())

  if cfg._mode != "eval":
    raise ValueError("Invalid mode %s" % cfg._mode)

  workplace = _Workplace(cfg)
  workplace.run_eval()


if __name__ == "__main__":
  from hypothesis_generation.hypothesis_utils import MetaDatasetExample
  from hypothesis_generation.hypothesis_utils import HypothesisEval
  main()
