# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Test script for running adhoc categorization.
import hydra
import logging
import os

import dataloaders
import losses
import models

from tqdm import tqdm
from time import time
from time import sleep

from dataloaders.get_dataloader import GetDataloader
from models._trainer import _Trainer
from models._evaluator import _Evaluator
from models._map_evaluator import _MapEvaluator

_PRINT_EVERY = 10


def load_model(cfg, vocabulary):
    get_model_helper = hydra.utils.instantiate(cfg.model)
    return get_model_helper(vocabulary)


class _Workplace(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Initialize the data loader.
        get_data_loader = hydra.utils.instantiate(self.cfg.data_args)
        dataloaders = get_data_loader(self.cfg)

        # Get the vocabulary we are training with.
        self._vocabulary = dataloaders["train"].dataset.vocabulary

        # Initialize the model/trainer
        model = load_model(self.cfg, self._vocabulary)
        total_params_model = sum(p.numel() for p in model.parameters()
                                 if p.requires_grad)
        logging.info(f"Model has {total_params_model} trainable parameters.")

        loss_fn = hydra.utils.instantiate(config=self.cfg.loss,
                                          pad_token_idx=self._vocabulary.pad())

        trainer = _Trainer(config=self.cfg,
                           dataloader=dataloaders["train"],
                           models={"model": model},
                           loss_fn=loss_fn,
                           serialization_dir=os.getcwd())

        evaluator = _Evaluator(
          config=self.cfg,
          loss_fn=loss_fn,
          dataloader=dataloaders["val"],
          models={"model": model},
        )

        # Two kinds of evaluators: cheap and costly.
        costly_loss_fn = hydra.utils.instantiate(self.cfg.costly_loss)

        if dataloaders.get("cross_split") is not None:
            costly_evaluator = _MapEvaluator(
              config=self.cfg,
              loss_fn=costly_loss_fn,
              test_loader=dataloaders["cross_split"],
              dataloader=dataloaders["val"],
              models={"model": model},
            )
        else:
            costly_evaluator = None

        self._trainer = trainer
        self._evaluator = evaluator
        self._costly_evaluator = costly_evaluator

    def run_training(self):
        latest_checkpoint, latest_iteration = self._trainer._checkpoint_manager.latest_checkpoint
        self._trainer.load_checkpoint(latest_checkpoint)

        t = time()
        for step in range(latest_iteration + 1, self.cfg.opt.max_steps):
            loss = self._trainer.step()["loss"]
            if (step + 1) % self.cfg.opt.checkpoint_every == 0:
                # Clear all the metric values accumulated during training
                # TODO(ramav): Fix this more properly, this is a hack for now.
                self._trainer._loss_fn._reset_metrics()
                
                metrics = self._evaluator.evaluate()
                self._trainer.after_validation(metrics, step)

            if (step + 1) % _PRINT_EVERY == 0:
                time_elapsed = (time() - t) / _PRINT_EVERY
                logging.info("%d step]: loss: %f (%f sec per step)",
                            step, loss.detach().cpu().item(), time_elapsed)
                t = time()

    def run_eval(self):
        # Iterate over all the checkpoints.
        current_iteration = -1
        latest_iteration = -1
        num_sleep = 0
        _WAIT=7200

        while(True):
            logging.info(f"Sleeping for {_WAIT} sec waiting for checkpoint.")
            sleep(_WAIT)

            num_sleep += 1
            latest_checkpoint, latest_iteration = (
              self._trainer._checkpoint_manager.latest_checkpoint)

            if latest_iteration != current_iteration:
                self._trainer.load_checkpoint(latest_checkpoint)
                costly_metrics = self._costly_evaluator.evaluate()
                self._trainer.write_metrics(costly_metrics, latest_iteration)

                current_iteration = latest_iteration
                num_sleep = 0

            if num_sleep == 10:
                logging.info(f"Terminating job after waiting for a new checkpoint.")
                break


@hydra.main(config_path='hydra_cfg/experiment.yaml')
def main(cfg):
    logging.info(cfg.pretty())

    logging.info("Base directory: %s", os.getcwd())

    workplace = _Workplace(cfg)

    workplace.run_training()


if __name__ == "__main__":
    from hypothesis_generation.hypothesis_utils import MetaDatasetExample
    from hypothesis_generation.hypothesis_utils import HypothesisEval
    main()
