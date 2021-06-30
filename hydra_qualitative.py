# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Qualitative examples for running adhoc categorization.
import hydra
import logging
import os
import pickle

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
    if self._eval_split_name != "qualitative":
        raise ValueError("Expect qualitative split.")

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
    write_results_file = (self._best_test_metric_or_oracle + "_" +
                              self.cfg.get(self._eval_split_name).rstrip(".pkl")
                              + "_qualitative.pkl")

    # If we are not computing model metrics, then the best test metric used
    # to choose validation points is irrelevant
    trainer = _Trainer(config=self.cfg,
                       dataloader=dataloaders.get("train"),
                       models={"model": model},
                       loss_fn=loss_fn,
                       serialization_dir=os.getcwd(),
                       write_metrics_file=write_results_file)

    costly_loss_fn = hydra.utils.instantiate(self.cfg.costly_loss)
    costly_evaluator = _MapEvaluator(
      config=self.cfg,
      loss_fn=costly_loss_fn,
      test_loader=dataloaders["cross_split"],
      dataloader=dataloaders[self._eval_split_name],
      models={"model": model},
    )

    self._trainer = trainer
    self._costly_evaluator = costly_evaluator
    self._write_results_file = write_results_file

  def run_eval(self):
    # Iterate over all the checkpoints.
    current_iteration = -1
    active_iteration = -1
    num_sleep = 0
    _WAIT=7200

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
          if self._eval_split_name == "test" or self._eval_split_name == "qualitative":
            active_checkpoint, active_iteration = (
              self._trainer._checkpoint_manager.best_checkpoint(
                based_on_metric=self._best_test_metric_or_oracle)
            )
          else:
            active_checkpoint, active_iteration = (
              self._trainer._checkpoint_manager.latest_checkpoint)

        # Active iteration is None when we are evaluating the best checkpoint.
        if active_iteration is None and self._eval_split_name not in ["test", "qualitative"]:
            raise ValueError("Expect active_iteration to not be None.")
      else:
        active_checkpoint, active_iteration = None, None

      if active_iteration is None or active_iteration != current_iteration:
          logging.info(f"Evaluating checkpoint {active_checkpoint}")
          self._trainer.load_checkpoint(active_checkpoint)
          _, all_qualitative_results = self._costly_evaluator.evaluate(
            num_batches=_NUM_EVAL_BATCHES_FOR_SPLIT[self._eval_split_name],
            output_raw_results=True)
          held_out_image_paths = self._costly_evaluator._held_out_image_paths

      current_iteration = active_iteration
      num_sleep = 0

      if (self.cfg.model_or_oracle_metrics == "oracle" or
          self.cfg.eval_cfg.evaluate_once == True or
          self._eval_split_name == "qualitative"):
        logging.info("Finished evaluation.")
        break

    with open(self._write_results_file, 'wb') as f:
      pickle.dump({"qualitative_results": all_qualitative_results,
                   "held_out_image_paths": held_out_image_paths}, f)


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
