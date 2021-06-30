# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
r"""Code to pick best checkpoints from the validation runs.

Assumes a nested folder structure for the runs as follows:

sweep_folder/
    job_1/
        0/
            validation_results.txt
        1/
            validation_results.txt
    job_2/
        0/
            validation_results.txt
        1/
            validation_results.txt

The user needs to specify the sweep folder, and the script augments each job
folder job_1/0, job_1/1 etc. with the details of the best checkpoint.
Output of the script is a file like:
    sweep_folder/job_1/0/checkpoint_best_modelmap.pth

which denotes the best validation checkpoint based on the metric of interest,
here, modelmap.
"""
import argparse

import logging
import pandas as pd
import shutil
import os

from glob import glob


def compute_best_valid(job, metric, greater_is_better):
    all_txt_files = glob(job + "/*.txt")
    useful_txt_files = []

    for this_txt_file in all_txt_files:
        # Skip / remove any validation metrics dumped during training.
        if "val" in this_txt_file:
            useful_txt_files.append(this_txt_file)
    del all_txt_files

    if len(useful_txt_files) == 0:
        logging.warning(f"No eval files found for job {job}")
        return

    result_table = None
    for idx, this_txt_file in enumerate(useful_txt_files):
        table = pd.read_csv(this_txt_file)
        table.columns = ["step", "metric", "value"]

        if idx == 0:
            result_table = table
        else:
            result_table = result_table.append(table)

    result_table = result_table[result_table["metric"] == metric]

    if len(result_table) == 0:
        logging.warning(f"{metric} not found in result files for {job}")
        return

    metric_to_print = metric.replace("/", "_")

    best_checkpoint_idx = (result_table.sort_values(
        "value", ascending=greater_is_better)["step"].iloc[-1])
    best_checkpoint = os.path.join(job,
                                   f"checkpoint_{best_checkpoint_idx}.pth")
    target_checkpoint_path = os.path.join(job,
                                          f"checkpoint_best_{metric_to_print}.pth")
    shutil.copy(best_checkpoint, target_checkpoint_path)


def main(args):
    all_jobs = glob(args.sweep_folder + "/*")

    folders = []
    for job in all_jobs:
        folders.extend(glob(job + "/*"))
    del all_jobs

    for job in folders:
        compute_best_valid(job, args.val_metric, args.greater_is_better)
        print(f"Done with {job}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep_folder",
                        type=str,
                        help="Folder which contains all the jobs.")
    parser.add_argument("--val_metric",
                        default="modelmap",
                        type=str,
                        help="Metric to use for picking best checkpoint.")
    parser.add_argument("--greater_is_better", type=int, default=1,
                        help="If greater is better for the chosen val metric,"
                        "1 if true, 0 if false.")

    args = parser.parse_args()

    if args.greater_is_better == 1:
        args.greater_is_better = True
    else:
        args.greater_is_better = False

    main(args)