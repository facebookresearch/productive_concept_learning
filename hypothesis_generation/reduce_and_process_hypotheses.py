# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''Reduce hypotheses and post process them, creating a meta-dataset.

Example Usage:
python hypothesis_generation/reduce_and_process_hypotheses.py \
  --regex_to_result_files '/checkpoint/ramav/adhoc_concept_data/adhoc_images_slurm_v0.2/hypotheses/result_v2_typed_simple_fol_depth_6_trials_2000000_ban_1_*_max_scene_id_200.pkl'
  --add_alternate_hypotheses \
  --train_size 1000 \
  --negative_type 'alternate_hypotheses'
'''
import argparse
import copy
import glob
import json
import logging

import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pickle
import submitit

from functools import partial
from multiprocessing import Pool
from collections import namedtuple
from collections import defaultdict
from collections import OrderedDict
from scipy.special import logsumexp
from tqdm import tqdm
from typing import Dict, List, Tuple

from dataloaders.utils import ImageAccess
from hypothesis_generation.hypothesis_utils import GrammarExpander
from hypothesis_generation.hypothesis_utils import fast_random_negatives
from hypothesis_generation.hypothesis_utils import HypothesisEval
from hypothesis_generation.hypothesis_utils import MetaDatasetExample
from hypothesis_generation.hypothesis_utils import HypothesisSampler
from hypothesis_generation.hypothesis_utils import create_image_index

parser = argparse.ArgumentParser()
logging.basicConfig(level=logging.INFO)

parser.add_argument(
    '--regex_to_result_files',
    default='/checkpoint/ramav/adhoc_concept_data/adhoc_images_slurm_v0.2/'
    'hypotheses/result_v2_typed_simple_fol_depth_6_trials_200000_*.pkl',
    type=str,
    help='Regex path to the directory containing the results files.')

parser.add_argument(
    '--path_to_images',
    default=
    '/checkpoint/ramav/adhoc_concept_data/adhoc_images_slurm_v0.2/images',
    type=str)
parser.add_argument(
    '--output_dir',
    default='/checkpoint/ramav/adhoc_concept_data/adhoc_images_slurm_v0.2/'
    'hypotheses/v2_typed_simple_fol_depth_6_trails_200000',
    type=str,
    help='Output visualization directory.')
parser.add_argument('--num_cpus',
                    type=int,
                    default=1,
                    help='Number of CPUs to use for generating the dataset.')

# Dataset Specific Options.
parser.add_argument('--random_seed',
                    default=42,
                    type=int,
                    help='Random seed for generating the dataset.')
parser.add_argument(
    '--add_alternate_hypotheses',
    default=False,
    action='store_true',
    help='Set to 1 to add the information needed for the posterior_predictive.'
)
parser.add_argument(
    '--positive_threshold',
    default=0.1,
    type=int,
    help='Maximum threshold for how many images a concept fires on to still be '
    'accepted for study.')
parser.add_argument('--min_pos_images_per_episode',
                    default=5,
                    type=int,
                    help='Number of images to display.')
parser.add_argument(
    '--max_neg_images_per_episode',
    default=20,
    type=int,
    help='Minimum number of negative images to use when training. '
    'Based on negative type, one can have more than the minimum.')
parser.add_argument(
    '--negative_type',
    default='random',
    type=str,
    help='Whether to use random negatives, or `alternate_hypotheses` for alternate hypotheses.')
parser.add_argument('--train_size',
                    default=500000,
                    type=int,
                    help='Number of images in training.')
parser.add_argument('--hypothesis_sampler_type',
                    default='log_linear',
                    type=str,
                    help='The kind of sampler to use for concepts: pcfg '
                    'or log_linear')
parser.add_argument(
    '--split_type',
    default='comp',
    type=str,
    help=
    'Regex to the split file path, present in `--output_dir` path, '
    'containing pre-specified splits. Leave empty by default.'
)

MIN_VAL_EXAMPLES = 40
MIN_TEST_EXAMPLES = 100

POS_LABEL_ID = 1

# Data and split related parameters.
_RATIO = (0.8, 0.05, 0.15)
_TRAIN_TEST_RATIO = 0.025
_TRAIN_VAL_RATIO = 0.010
_SPLIT_NAMES = ('train', 'val', 'test')
# TODO(ramav): IID splits should just be implemented via a flag to the code.
_DATA_FILE = '/checkpoint/ramav/global_temp/data.pkl'

# SUBMITIT parameters.
SUBMITIT_TIMEOUT_HOUR = 40
SUBMITIT_PARTITION = 'learnfair'
SUBMITIT_CPUS_PER_TASK = 3


def load_file_with_hypothesis_evaluations(this_file,
                                          num_scenes=500000,
                                          positive_threshold=0.1,
                                          min_pos_images_per_episode=5):
    all_image_evaluations = defaultdict(list)
    with open(this_file, 'rb') as f:
        evaluations = pickle.load(f)['results_per_scene']
        for eval in evaluations:
            if eval is not None:
                if not isinstance(eval[0], int):
                    raise ValueError('Expect evaluation to contain index.')

                if len(
                        eval[1]
                ) / float(num_scenes) < positive_threshold and len(
                        eval[1]
                ) > 2 * min_pos_images_per_episode:  # 2 * episode length for train, test
                    all_image_evaluations[eval[0]] = eval[1]

    return all_image_evaluations


def hypothesis_length(hypo):
    return len(hypo.split(' '))


def load_and_filter_result_files(regex_to_result_files,
                                 min_pos_images_per_episode,
                                 positive_threshold=0.1,
                                 num_processes=100):
    '''Load hypotheses from disk and filter them.

  Loads hypotheses and based on the positive threshold only retains the
  hypotheses for which the firing rate is lesser than the positive
  threshold.

  Args:
    regex_to_result_files: String, with every * being specified as \\*.
    min_pos_images_per_episode: Int, number of positive and negative images
      to put in each meta learning episode.
    positive_threshold: Int, maximum firing rate for a concept (on images).
  Returns:
    all_hypotheses: A list of strings.
    all_image_evaluations: A list of list of Int
    num_scenes: Int, number of total scenes in the dataset.
  '''
    regex_to_result_files = regex_to_result_files.replace('\\*', '*')
    result_files = glob.glob(regex_to_result_files)

    if len(result_files) == 0:
        raise ValueError('Please check files exist that match %s' %
                         (regex_to_result_files))

    with open(result_files[0], 'rb') as f:
        scene_datum = pickle.load(f)
        all_hypotheses = scene_datum['hypotheses']
        all_hypothesis_log_probs = scene_datum['hypotheses_logprob']
        num_scenes = scene_datum['total_scenes']

    logging.info('%d hypotheses found, loading %d result files.' %
                 (len(all_hypotheses), len(result_files)))

    all_image_evaluations = [None] * len(all_hypotheses)
    with Pool(processes=num_processes) as pool:
        logging.info('Started pool with %d processes' % (num_processes))
        all_results = [
            x for x in pool.map(
                partial(load_file_with_hypothesis_evaluations,
                        num_scenes=num_scenes,
                        positive_threshold=positive_threshold,
                        min_pos_images_per_episode=min_pos_images_per_episode),
                result_files)
        ]
        logging.info('Loaded %d files.' % (len(result_files)))
        for this_result in all_results:
            for key, value in this_result.items():
                if all_image_evaluations[key] is not None:
                    raise ValueError(
                        'Dont expect hypothesis overlap betweeen files.')
                all_image_evaluations[key] = value

    filtered_hypothesis_idx = [
        idx for idx, x in enumerate(all_image_evaluations) if x is not None
    ]
    filtered_hypotheses = [
        all_hypotheses[idx] for idx in filtered_hypothesis_idx
    ]
    filtered_evaluations = [
        all_image_evaluations[idx] for idx in filtered_hypothesis_idx
    ]
    filtered_hypotheses_lengths = [
        hypothesis_length(x) for x in filtered_hypotheses
    ]
    filtered_hypotheses_logprobs = [
        all_hypothesis_log_probs[idx] for idx in filtered_hypothesis_idx
    ]

    filtered_hypotheses_evaluations = HypothesisEval(
        filtered_hypotheses, filtered_hypotheses_logprobs,
        filtered_hypotheses_lengths, filtered_evaluations)

    filtered_hypotheses_evaluations = (
        collapse_for_all_and_exists(filtered_hypotheses_evaluations))

    logging.info('Found %d hypotheses' %
                 (len(filtered_hypotheses_evaluations.hypothesis)))

    number_true = [
        len(x) / float(num_scenes)
        for x in filtered_hypotheses_evaluations.image_id_list
    ]

    return all_hypotheses, filtered_hypotheses_evaluations, number_true, num_scenes


def collapse_for_all_and_exists(hypothesis_evaluations):
    current_hypotheses = hypothesis_evaluations.hypothesis
    new_hypotheses = []
    new_hypotheses_logprobs = []
    new_hypotheses_lengths = []
    new_evaluations = []

    for hyp, hyp_logprob, eval in zip(current_hypotheses,
                                      hypothesis_evaluations.logprob,
                                      hypothesis_evaluations.image_id_list):
        hyp_list = hyp.split(' ')
        if 'x' not in hyp_list and 'non-x-S' not in hyp_list:
            # Remove the terminal for-all and exists tokens.
            edited_hyp = ' '.join(hyp_list[:-1])

        else:
            edited_hyp = hyp

        # Checking if the editing led to a duplicate hypothesis, if not add.
        if edited_hyp not in new_hypotheses:
            new_hypotheses.append(edited_hyp)
            new_hypotheses_logprobs.append(hyp_logprob)
            new_hypotheses_lengths.append(hypothesis_length(edited_hyp))
            new_evaluations.append(eval)

    return HypothesisEval(new_hypotheses, new_hypotheses_logprobs,
                          new_hypotheses_lengths, new_evaluations)


def subset_hypotheses(hypothesis_evaluations, idx_in_subset):

    hypotheses_to_return = HypothesisEval([], [], [], [])

    for idx in idx_in_subset:
        for field_idx, _ in enumerate(hypothesis_evaluations._fields):
            hypotheses_to_return[field_idx].append(
                hypothesis_evaluations[field_idx][idx])

    return hypotheses_to_return


def split_train_val_test(hypothesis_evaluations: HypothesisEval,
                         ratio: tuple = _RATIO,
                         split_names: tuple = _SPLIT_NAMES,
                         split_type_or_regex=None):
    '''Split hypotheses into train validation and test.
    
    Aims to partition the hypothesis space into splits given by split names
    based on the specified ratio. Key is that the code ensures that rephrasings
    of the same concept (which we figure out by looking at which concepts)
    entail the same denotation are not distributed across splits.
    
    An optional external split file regex can be provided as filename_%s.json,
    where %s will be substituted with the split names. To load the corresponding
    hypotheses in each split.
    
    Args:
      hypothesis_evaluations: An instance of HypothesisEval containing all the
        hypotheses and their evaluations.
      ratio: A tuple containing the ratio in which we want to split into the
        different splits. Must sum to 1.
      split_names: A tuple containing the names of the splits corresponding
        to the ratios defined in ratio.
      split_type_or_regex: An optional regex with external file that defines
        the splits in train, val and test. 
    Returns:
      A dict with key split_name and value a HypothesisEval object with subset
    Raises:
      ValueError if split_names does not contain a `train` split.
      ValueError if ratios does not sum to 1.
    '''
    split_names_to_idx = defaultdict(list)
    if split_type_or_regex == 'comp':
        if 'train' not in split_names:
            raise ValueError('Must create a train split.')

        if np.sum(ratio) != 1:
            raise ValueError('Provided ratios must sum to 1.')

        if len(ratio) != len(split_names):
            raise ValueError('Length of ratios and split_names must be same.')

        image_id_list_to_hyp_idx_list = defaultdict(list)
        hyp_idx_to_image_id_list = {}

        for hyp_idx, image_id_list in enumerate(
                hypothesis_evaluations.image_id_list):
            image_id_list = tuple(sorted(image_id_list))
            image_id_list_to_hyp_idx_list[image_id_list].append(hyp_idx)
            hyp_idx_to_image_id_list[hyp_idx] = image_id_list

        # Step 1: Iterate over the hypotheses, and put them in splits based on
        # the ratio
        unassigned_hypotheses_idx = list(
            range(len(hypothesis_evaluations.hypothesis)))

        while len(unassigned_hypotheses_idx) > 0:
            # Pick an index at random.
            assign_idx = list(
                np.random.choice(unassigned_hypotheses_idx, replace=False,
                                 size=1))[0]
            # Other concepts with same denotation.
            picked_idx_image_id_list = hyp_idx_to_image_id_list[assign_idx]
            all_assign_hypotheses_idx = image_id_list_to_hyp_idx_list[
                picked_idx_image_id_list]

            # Pick a split for those indices.
            # TODO(ramav): This is what we will generalize in the structured splits.
            # Assignment can be non-random as well.
            picked_split = list(
                np.random.choice(split_names, replace=False, size=1, p=ratio))[0]
            split_names_to_idx[picked_split].extend(all_assign_hypotheses_idx)

            # Remove the all picked index fromjjju unassigned_hypotheses_idx
            for idx in all_assign_hypotheses_idx:
                unassigned_hypotheses_idx.remove(idx)
    else:
        for split_name in split_names:
            fname = split_type_or_regex % (split_name)
            with open(fname, 'r') as f:
                this_hypotheses_in_split = json.load(f)
                this_hypothesis_idx = [hypothesis_evaluations.hypothesis.index(
                    x) for x in this_hypotheses_in_split]
                split_names_to_idx[split_name] = this_hypothesis_idx

    raw_hypothesis_splits = {}

    for this_split, this_split_idx in split_names_to_idx.items():
        raw_hypothesis_splits[this_split] = subset_hypotheses(
            hypothesis_evaluations, this_split_idx)

    return raw_hypothesis_splits, split_names_to_idx


def evaluate_alternate_hypotheses_for_positives(
        positive_image_list: List[int], target_hypothesis: str,
        image_to_hypotheses: Dict[int, List[str]]) -> Tuple[str]:
    r'''Return alternative hypotheses consistent with positive images.
    
    Computes intersection of the hypotheses which are true for a list of images,
    and returns all valid alternative hypotheses for positives.
    
    Args:
      positive_image_list: List of images which are grouped together into one
        concept.
      target_hypothesis: Str, hypothesis for which we grouped positive images
      image_to_hypothesis: Dict, key is image_id and value is a list of strings
    Returns:
      all_consistent_hypotheses: A tuple of str
    '''

    for this_idx, this_image_id in enumerate(positive_image_list):
        if this_idx == 0:
            all_consistent_hypotheses = set(image_to_hypotheses[this_image_id])
        else:
            all_consistent_hypotheses = all_consistent_hypotheses.intersection(
                set(image_to_hypotheses[this_image_id]))

    if target_hypothesis not in all_consistent_hypotheses:
        raise ValueError('Target hypothesis not in all consistent hypothesis.')
    all_consistent_hypotheses.remove(target_hypothesis)

    return tuple(all_consistent_hypotheses)


def get_full_posterior_distribution(
        labels: List[int], alternate_hypotheses: List[str],
        current_hypothesis: str,
        all_hypotheses_across_splits_logprobs: List[float],
        all_hypotheses_across_splits_img_id_list: List[List[int]],
        all_hypotheses_across_splits_str_to_idx: Dict[str, int],
        num_scenes: int,
        random_negative_type: bool):
    '''Get the posterior distribution over hypotheses.
    
    Computes a distribution over valid hypotheses given a set of
    positive and negative examples. Let the count of images for which
    a hypothesis fires be C, and the total number of images (on which)
    the hypothesis is executed be N. Then the likelihood of a positive
    image is 1/C and the likelihood of a negative image is 1/(N-C). We
    compute this likelihood and mix it with the prior probabilities
    to obtain the posterior distribution. 
    
    NOTE: There is no need to normalize the prior distribution since we
    always get normalized posteriors even with unnormalized priors.
    
    Args:
      images: List of int indices of images.
      labels: List of int with values 0 or 1 indicating positive or
        negative labels based on POS_LABEL_ID
      alternate_hypotheses: List of str with alternate hypotheses for
        the positives and negatives.
      current_hypothesis: Str, current hypothesis for which we sampled
        the data.
      all_hypotheses_across_splits_logprobs: List of Float, prior
        logprobs for different hypotheses (where hypotheses are indexed
        across splits like train val or test)
      all_hypotheses_across_splits_img_id_list: List of List of Int, the
        images for which each hypothesis (indexed across all splits)
        fires as True.
      all_hypotheses_across_splits_str_to_idx: Dict str key, where we can
        index into the list with a hypothesis string and retrieve its
        global hypothesis index.
      random_negative_type: Whether the negatives have been sampled at random 
        or from the alternate hypotheses.
    Returns:
      posterior_logprobs: np.Array of float, with posterior log-probs
        for hypotheses.
    '''
    all_hypotheses = alternate_hypotheses + [current_hypothesis]
    prior_logprobs = np.array([
        all_hypotheses_across_splits_logprobs[
            all_hypotheses_across_splits_str_to_idx[hyp_str]]
        for hyp_str in all_hypotheses
    ])

    num_positives = np.sum(labels == POS_LABEL_ID)
    num_negatives = len(labels) - num_positives

    if random_negative_type == True:
        posterior_logprobs = np.array([
            num_positives * -1 * np.log(
                len(all_hypotheses_across_splits_img_id_list[
                    all_hypotheses_across_splits_str_to_idx[hyp_str]])) +
            num_negatives * -1 * np.log(num_scenes)
            for hyp_str in all_hypotheses
        ])
    else:
        posterior_logprobs = np.array([
            num_positives * -1 * np.log(
                len(all_hypotheses_across_splits_img_id_list[
                    all_hypotheses_across_splits_str_to_idx[hyp_str]])) +
            num_negatives * -1 *
            np.log(num_scenes - len(all_hypotheses_across_splits_img_id_list[
                all_hypotheses_across_splits_str_to_idx[hyp_str]]))
            for hyp_str in all_hypotheses
        ])

    posterior_logprobs = prior_logprobs + posterior_logprobs
    posterior_logprobs -= logsumexp(posterior_logprobs)

    return all_hypotheses, posterior_logprobs, prior_logprobs


def negatives_from_alternate_hypotheses(
        alternate_hypotheses_for_positives, positive_image_id_list,
        all_hypotheses_across_splits_img_id_list,
        all_hypotheses_across_splits_str_to_idx, max_neg_images_per_episode,
        num_scenes):
    negative_datum_idx = []
    used_random_negatives = False

    for hyp_str in alternate_hypotheses_for_positives:
        true_for_alternate_only = list(
            set(all_hypotheses_across_splits_img_id_list[
                all_hypotheses_across_splits_str_to_idx[hyp_str]]).difference(
                    set(positive_image_id_list)))

        negative_datum_idx.extend(true_for_alternate_only)

    if len(negative_datum_idx) > 0:
        negative_datum_idx = list(np.unique(negative_datum_idx))
        num_images_to_sample = min(len(negative_datum_idx),
                                   max_neg_images_per_episode)

        negative_datum_idx = list(
            np.random.choice(negative_datum_idx,
                             size=num_images_to_sample,
                             replace=False))

    # Back off to random negatives if there are no alternate hyp.
    if len(negative_datum_idx) != max_neg_images_per_episode:
        used_random_negatives = True

    negative_datum_idx.extend(
        fast_random_negatives(max_neg_images_per_episode -
                              len(negative_datum_idx),
                              num_scenes,
                              positives=positive_image_id_list))

    return negative_datum_idx, used_random_negatives


def generate_meta_example(
        sampled_hypothesis: HypothesisEval,
        min_pos_images_per_episode: int,
        max_neg_images_per_episode: int,
        image_to_hypothesis: Dict[int, str],
        negative_type: str,
        num_scenes: int,
        all_hypotheses_across_splits_img_id_list: List[List[int]],
        all_hypotheses_across_splits_str_to_idx: Dict[str, int],
        all_hypotheses_across_splits_logprobs: List[float],
        this_example_idx: int,
        hypothesis_idx_within_split: int,
        compute_alternate_hypotheses: bool = True) -> Dict[str, List]:
    r'''Generate a single meta example for the dataset.
    
    Generates a meta-example given a hypothesis of interest. A ``meta-example``
    consists of a support set and a query set. Both consist of datapoints and
    labels -- there can be two possible lables (positive and negative).
    
    The query and support sets differ in a subtle way. The set of alternate
    hypotheses that could explain the data is computed on the support --only--.
    This is then copied to the query set. Overall, the code computes valid
    alternate hypotheses, implements random sampling or a smarter sampling
    of negatives based on arguments to the function, and computes the posterior
    predictive distribution (actually an approximation to it) given the sampled
    support examples.
    
    See hypothesis_utils/MetaDatasetExample for a schema that stores the meta
    example generated by this code.
    
    Args:
      sampled_hypothesis: An object of HypothesisEval
      min_pos_images_per_episode: Int, min number of positive images to create
      max_neg_images_per_episode: Int, max number of negative images to create
      image_to_hypothesis: Dict, with key image_id and value list of str 
        hypotheses true for the image
      negative_type: Str, 'alternate_hypothesis' or 'random'
      num_scenes: Int, number of scenes in the dataset.
      all_hypotheses_across_splits_img_id_list: List of List of Int, the
        images for which each hypothesis (indexed across all splits)
        fires as True.
      all_hypotheses_across_splits_str_to_idx: Dict str key, where we can
        index into the list with a hypothesis string and retrieve its
        global hypothesis index
      all_hypotheses_across_splits_logprobs: List of Float, prior
        logprobs for different hypotheses (where hypotheses are indexed
        across splits like train val or test)
      this_example_idx: Int, example index
      hypothesis_idx_within_split: Int, index of hypothesis within split
      compute_alternate_hypotheses: Boolean, whether to compute alternate
        hypotheses or not
    Returns:
      an dict of key meta-split i.e. support or query and value is a list with
      items in the same ordering as hypothesis_utils.MetaDatasetExample
    Raises:
      NotImplementedError: If negative type is not in random or \
          alternate_hypotheses
      RuntimeError: If non-random negatives are positives for target hypothesis
    '''
    if negative_type not in [
            'random', 'alternate_hypotheses'
    ]:
        raise NotImplementedError(f'{negative_type} negatives not implemented')

    get_full_posterior_to_use = partial(
        get_full_posterior_distribution,
        num_scenes=num_scenes,
        all_hypotheses_across_splits_img_id_list=
        all_hypotheses_across_splits_img_id_list,
        all_hypotheses_across_splits_logprobs=
        all_hypotheses_across_splits_logprobs,
        all_hypotheses_across_splits_str_to_idx=
        all_hypotheses_across_splits_str_to_idx,
        random_negative_type=negative_type=='random',
    )

    negatives_from_alternate_hypotheses_to_use = partial(
        negatives_from_alternate_hypotheses,
        all_hypotheses_across_splits_img_id_list=
        all_hypotheses_across_splits_img_id_list,
        all_hypotheses_across_splits_str_to_idx=
        all_hypotheses_across_splits_str_to_idx,
        max_neg_images_per_episode=max_neg_images_per_episode,
        num_scenes=num_scenes,
    )

    positive_datum_idx_train_test = np.random.choice(
        sampled_hypothesis['image_id_list'],
        size=2 * min_pos_images_per_episode,  # For support, query
        replace=False)

    meta_datum = {}
    valid_alternate_hypotheses = []
    for split_idx, meta_split in enumerate(['support', 'query']):
        labels_in_split = []
        optimistic_labels_in_split = []

        ########################### Sample positives ###########################
        positive_datum_idx = positive_datum_idx_train_test[
            split_idx * min_pos_images_per_episode:(split_idx + 1) *
            min_pos_images_per_episode]
        labels_in_split.extend([POS_LABEL_ID] * len(positive_datum_idx))
        optimistic_labels_in_split.extend([POS_LABEL_ID] *
                                          len(positive_datum_idx))

        ########### Evaluate alternate hypotheses for positives ################
        alternate_hypotheses_for_positives = []
        if meta_split == 'support' and compute_alternate_hypotheses == True:
            alternate_hypotheses_for_positives = (
                evaluate_alternate_hypotheses_for_positives(
                    positive_datum_idx, sampled_hypothesis['hypothesis'],
                    image_to_hypothesis))

        ########################### Sample negatives ###########################
        if negative_type == 'random':
            candidate_negative_datum_idx = fast_random_negatives(
                max_neg_images_per_episode,
                num_scenes)
        elif negative_type == 'alternate_hypotheses' and (
                compute_alternate_hypotheses == True):

            candidate_negative_datum_idx, _ = (
                negatives_from_alternate_hypotheses_to_use(
                    alternate_hypotheses_for_positives,
                    sampled_hypothesis['image_id_list'],
                ))

        ########### Identify consistent hypotheses wrt both pos and neg ########
        if meta_split == 'support':
            # Remove the hypotheses for which sampled negatives for the task
            # are actually positives.
            for hyp_str in alternate_hypotheses_for_positives:
                # Positives for a valid alternate hypothesis
                # (overall for positives and negatives) cannot be negatives for
                # the support or query.
                if len(
                        set(all_hypotheses_across_splits_img_id_list[
                            all_hypotheses_across_splits_str_to_idx[hyp_str]]).
                        intersection(set(candidate_negative_datum_idx))) == 0:

                    valid_alternate_hypotheses.append(hyp_str)

        if all_hypotheses_across_splits_img_id_list[
                all_hypotheses_across_splits_str_to_idx[sampled_hypothesis[
                    'hypothesis']]] != sampled_hypothesis['image_id_list']:
            raise ValueError(f'Expected the two lists to be the same.')

        ######################## Label the negative images* ####################
        # *Some of the randomly selected images might be positives for the
        # target hypothesis, in which case they should be labelled as positives.
        # Also in the query set label the images (in the `optimistic`) case
        # if they match the denotation of any of the valid hypotheses.
        for this_neg_idx in candidate_negative_datum_idx:
            if this_neg_idx in set(all_hypotheses_across_splits_img_id_list[
                    all_hypotheses_across_splits_str_to_idx[
                        sampled_hypothesis['hypothesis']]]):

                if negative_type != 'random':
                    raise RuntimeError(f'{negative_type} should not lead '
                                       'to mislabelled negatives.')

                labels_in_split.append(POS_LABEL_ID)
                if meta_split == 'support':
                    optimistic_labels_in_split.append(POS_LABEL_ID)
            else:
                labels_in_split.append(abs(POS_LABEL_ID - 1))
                if meta_split == 'support':
                    optimistic_labels_in_split.append(abs(POS_LABEL_ID - 1))

            # Compute optimistic labels for the query case.
            if meta_split == 'query':
                flag = 0
                for hyp_str in valid_alternate_hypotheses:
                    if this_neg_idx in set(
                            all_hypotheses_across_splits_img_id_list[
                                all_hypotheses_across_splits_str_to_idx[
                                    hyp_str]]):
                        optimistic_labels_in_split.append(POS_LABEL_ID)
                        flag = 1
                        break
                if flag == 0:
                    optimistic_labels_in_split.append(abs(POS_LABEL_ID - 1))

        ######## Compute posterior distribution given pos and neg ##############
        all_valid_hypotheses, posterior_logprobs, prior_logprobs = get_full_posterior_to_use(
            labels_in_split,
            valid_alternate_hypotheses,
            sampled_hypothesis['hypothesis'],
        )

        data_idx = list(positive_datum_idx)
        data_idx.extend(list(candidate_negative_datum_idx))

        if len(optimistic_labels_in_split) != len(labels_in_split):
            raise RuntimeError(f'Optimistic and regular labels must be same'
                               'length.')

        meta_datum[meta_split] = [
            this_example_idx, sampled_hypothesis['hypothesis'],
            hypothesis_idx_within_split, sampled_hypothesis['length'],
            data_idx, labels_in_split, optimistic_labels_in_split,
            all_valid_hypotheses, posterior_logprobs, prior_logprobs,
            alternate_hypotheses_for_positives
        ]

    return meta_datum


def generate_meta_examples_in_range(
        start_example_idx, end_example_idx, max_neg_images_per_episode,
        min_pos_images_per_episode, num_scenes, image_path_access,
        negative_type, compute_alternate_hypotheses, hypothesis_sampler_type,
        MetaDatasetExample, data_serial_path):
    '''Generate meta-dataset examples in a particular range of indices.'''
    with open(data_serial_path, 'rb') as f:
        dataset_metadata = pickle.load(f)

    hypothesis_evaluations_in_split = dataset_metadata[
        'hypothesis_evaluations_in_split']
    image_to_hypothesis = dataset_metadata['image_to_hypothesis']
    all_hypotheses_across_splits_img_id_list = dataset_metadata[
        'all_hypotheses_across_splits_img_id_list']
    all_hypotheses_across_splits_str_to_idx = dataset_metadata[
        'all_hypotheses_across_splits_str_to_idx']
    all_hypotheses_across_splits_logprobs = dataset_metadata[
        'all_hypotheses_across_splits_logprobs']

    hypothesis_sampler = HypothesisSampler(
        hypothesis_evaluations_in_split, sampler_type=hypothesis_sampler_type)

    dataset = []
    for this_example_idx in range(start_example_idx, end_example_idx):
        sampled_hypothesis, hypothesis_idx_within_split = hypothesis_sampler.sample(
        )
        meta_datum = generate_meta_example(
            sampled_hypothesis=sampled_hypothesis,
            min_pos_images_per_episode=min_pos_images_per_episode,
            max_neg_images_per_episode=max_neg_images_per_episode,
            compute_alternate_hypotheses=compute_alternate_hypotheses,
            image_to_hypothesis=image_to_hypothesis,
            num_scenes=num_scenes,
            all_hypotheses_across_splits_img_id_list=
            all_hypotheses_across_splits_img_id_list,
            all_hypotheses_across_splits_str_to_idx=
            all_hypotheses_across_splits_str_to_idx,
            all_hypotheses_across_splits_logprobs=
            all_hypotheses_across_splits_logprobs,
            this_example_idx=this_example_idx,
            negative_type=negative_type,
            hypothesis_idx_within_split=hypothesis_idx_within_split)

        dataset.append(meta_datum)

    return dataset


def create_dataset_for_split(hypothesis_evaluations_in_split,
                             num_examples,
                             min_pos_images_per_episode,
                             max_neg_images_per_episode,
                             negative_type,
                             num_scenes,
                             path_to_images,
                             image_to_hypothesis,
                             all_hypotheses_across_splits,
                             hypothesis_sampler_type,
                             num_cpus=1,
                             compute_alternate_hypotheses=False):

    image_path_access = ImageAccess(root_dir=path_to_images, )
    # Prepare an index of hypothesis to string.
    all_hypotheses_across_splits_str_to_idx = {
        v: k
        for k, v in enumerate(all_hypotheses_across_splits.hypothesis)
    }
    all_hypotheses_across_splits_logprobs = all_hypotheses_across_splits.logprob

    with open(_DATA_FILE, 'wb') as f:
        pickle.dump(
            {
                'hypothesis_evaluations_in_split':
                hypothesis_evaluations_in_split._asdict(),
                'image_to_hypothesis':
                image_to_hypothesis,
                'all_hypotheses_across_splits_img_id_list':
                all_hypotheses_across_splits.image_id_list,
                'all_hypotheses_across_splits_str_to_idx':
                all_hypotheses_across_splits_str_to_idx,
                'all_hypotheses_across_splits_logprobs':
                all_hypotheses_across_splits_logprobs
            }, f)

    generate_meta_examples_in_range_to_use = partial(
        generate_meta_examples_in_range,
        max_neg_images_per_episode=max_neg_images_per_episode,
        min_pos_images_per_episode=min_pos_images_per_episode,
        num_scenes=num_scenes,
        image_path_access=image_path_access,
        negative_type=negative_type,
        compute_alternate_hypotheses=compute_alternate_hypotheses,
        MetaDatasetExample=MetaDatasetExample,
        hypothesis_sampler_type=hypothesis_sampler_type,
        data_serial_path=_DATA_FILE)

    dataset = []

    if num_cpus > 1:
        block_size = math.ceil(num_examples / float(num_cpus))

        start_indices = np.arange(0, num_examples, block_size)
        end_indices = start_indices + block_size
        end_indices[-1] = np.maximum(end_indices[-1], num_examples)

        executor = submitit.AutoExecutor(folder='/'.join(
            _DATA_FILE.split('/')
            [:-1]))  # submission interface (logs are dumped in the folder)
        executor.update_parameters(array_parallelism=num_cpus,
                                   timeout_min=SUBMITIT_TIMEOUT_HOUR * 60,
                                   cpus_per_task=SUBMITIT_CPUS_PER_TASK,
                                   partition=SUBMITIT_PARTITION)
        jobs = executor.map_array(generate_meta_examples_in_range_to_use,
                                  start_indices, end_indices)

        try:
            output = [job.result() for job in jobs]
        except:
            logging.info('Sleeping before collecting all the results.')
            import time
            time.sleep(1000)
            output = [job.result() for job in jobs]

        for out in output:
            dataset.extend(out)

    else:
        dataset = generate_meta_examples_in_range_to_use(0, num_examples)

    final_dataset = []
    for datum in dataset:
        for key in datum:
            datum[key] = MetaDatasetExample(*datum[key])
        final_dataset.append(datum)

    return final_dataset


def create_meta_dataset(raw_hypothesis_splits,
                        splits_to_all_hypothesis_idx,
                        hypothesis_evaluations, min_pos_images_per_episode,
                        max_neg_images_per_episode, num_scenes, metadata,
                        num_examples_per_split, add_alternate_hypotheses,
                        splits_to_output_paths, negative_type, num_cpus,
                        image_to_hypothesis, hypothesis_sampler_type,
                        split_type_or_regex=None):
    '''Create a dataset that will be used for training models.

  Each element of the dataset will contain the following:

  namedtuple:
    index: Int, tells you what the index of this hypothesis is.
    hypothesis_string: Str, tells you the string form of the hypothesis.
    length: Int, ttells you the lengtth of the hypothesis.
    image_ids_positive: List of image id's or json ids
    image_paths_positive: List of paths to image id's.
    image_ids_negative: List of image id's that dont belong to the hypothesis.
    image_paths_negative: List of paths to image id's.

  The dataset overall will contain metadata that will explain other details:
    num_hypotheses
    num_examples
    min_pos_images_per_episode
    grammar_version
    image_version

  Args:
    hypothesis_evaluations: An object of Hypothesis Eval.
    num_examples: Int, number of example episodes to sample.
    min_pos_images_per_episode: Int, number of images to use in each episode.
    num_scenes: Int, total number of scenes in the dataset
    metadata: Dict, metadata about the dataset.
    num_examples_per_split: Dict, key split name and value number of examples.
    add_alternate_hypotheses: Boolean, True means we are adding information
      of alternate hypotheses which could have caused the image grouping we
      are working with.
    splits_to_splits_to_output_pathss: Dict, with key split name and value Str,
      path to the dataset.
    TODO(ramav): Complete this documentation.
  Raises:
    ValueError: If all the images and JSONs in the dataset are not present.
  '''
    for split_name in raw_hypothesis_splits:
        logging.info(f'Creating split {split_name} with '
                     f'{num_examples_per_split[split_name]} examples.')
        meta_dataset_split = create_dataset_for_split(
            hypothesis_evaluations_in_split=raw_hypothesis_splits[split_name],
            num_examples=num_examples_per_split[split_name],
            min_pos_images_per_episode=min_pos_images_per_episode,
            max_neg_images_per_episode=max_neg_images_per_episode,
            negative_type=negative_type,
            num_scenes=num_scenes,
            path_to_images=metadata['path_to_images'],
            image_to_hypothesis=image_to_hypothesis,
            all_hypotheses_across_splits=hypothesis_evaluations,
            hypothesis_sampler_type=hypothesis_sampler_type,
            num_cpus=num_cpus,
            compute_alternate_hypotheses=add_alternate_hypotheses)
        with open(splits_to_output_paths[split_name], 'wb') as f:
            pickle.dump(
                {
                    'meta_dataset': meta_dataset_split,
                    'all_hypotheses_across_splits': hypothesis_evaluations,
                    'split_name_to_all_hypothesis_idx': splits_to_all_hypothesis_idx,
                }, f)


def write_visualization_data(filtered_hypotheses_evaluations,
                             path_to_images,
                             output_dir,
                             min_pos_images_per_episode,
                             num_datapoints=1000):
    vis_data = []
    image_path_access = ImageAccess(root_dir=path_to_images, )
    num_datapoints = min(len(filtered_hypotheses_evaluations), num_datapoints)

    for idx in range(num_datapoints):
        eval_to_display = np.random.choice(
            filtered_hypotheses_evaluations.image_id_list[idx],
            replace=True,
            size=min_pos_images_per_episode)
        eval_paths = [image_path_access(x) for x in eval_to_display]

        vis_data.append({
            'hypothesis':
            filtered_hypotheses_evaluations.hypothesis[idx],
            'image_paths':
            eval_paths
        })

    with open(os.path.join(output_dir, 'vis_hypotheses.json'), 'w') as f:
        json.dump(vis_data, f)


def _get_num_examples_per_split(num_cpus):

    num_examples_per_split = {}
    # If we are using multiple CPUS ensure atleast one example is generated
    # on every CPU
    min_val_examples = max(MIN_VAL_EXAMPLES, num_cpus)
    min_test_examples = max(MIN_TEST_EXAMPLES, num_cpus)

    for this_split in _SPLIT_NAMES:
        if 'train' in this_split:
            num_examples_per_split[this_split] = args.train_size
        elif 'val' in this_split:
            num_examples_per_split[this_split] = int(
                max(args.train_size * _TRAIN_VAL_RATIO, min_val_examples))
        elif 'test' in this_split:
            num_examples_per_split[this_split] = int(
                max(args.train_size * _TRAIN_TEST_RATIO, min_test_examples))
    return num_examples_per_split


def main(args):
    np.random.seed(args.random_seed)
    num_examples_per_split = _get_num_examples_per_split(args.num_cpus)

    ####### This stuff is common regardless of the split of the data###########
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_hypotheses, filtered_hypotheses_evaluations, number_true, num_scenes = (
        load_and_filter_result_files(args.regex_to_result_files,
                                     args.min_pos_images_per_episode,
                                     args.positive_threshold))

    fname = os.path.join(
        args.output_dir, '%d_%0.2f_hypotheses_light.json' %
        (args.min_pos_images_per_episode, args.positive_threshold))

    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            json.dump(
                {
                    'valid_hypotheses':
                    filtered_hypotheses_evaluations.hypothesis,
                    'valid_hypotheses_logprobs':
                    filtered_hypotheses_evaluations.logprob,
                    'valid_evaluations': number_true,
                }, f)
        fname_heavy = fname.replace('light', 'heavy')
        fname_heavy = fname_heavy.replace('.json', '.pkl')
        with open(fname_heavy, 'wb') as f:
            pickle.dump(
                {
                    'hypothesis_and_full_evaluations':
                    filtered_hypotheses_evaluations,
                    'all_hypotheses': all_hypotheses,
                }, f)
        logging.info('Dumped heavy JSON')

    logging.info('Building Image Index...')
    images_to_labels, images_to_hypotheses = create_image_index(
        filtered_hypotheses_evaluations)

    fname = os.path.join(
        args.output_dir, '%d_%0.2f_image_hyp_mapping.json' %
        (args.min_pos_images_per_episode, args.positive_threshold))

    if not os.path.exists(fname):
        with open(fname, 'wb') as f:
            pickle.dump(
                {
                    'images_to_labels': images_to_labels,
                    'images_to_hypotheses': images_to_hypotheses,
                }, f)
        logging.info('Dumped Image to Hypotheses data')

    metadata = {
        'result_files': args.output_dir,
        'threshold': args.positive_threshold,
        'num_pos_images': args.min_pos_images_per_episode,
        'num_neg_images': args.max_neg_images_per_episode,
        'path_to_images': args.path_to_images,
        'negative_type': args.negative_type,
        'random_seed': args.random_seed,
    }
    splits_to_output_paths = {}
    for split_name in num_examples_per_split:
        splits_to_output_paths[split_name] = os.path.join(
            args.output_dir,
            '%s_sampling_%s_%s_threshold_%.2f_pos_im_%d_neg_im_%d_train_examples_%d_neg_type_'
            '%s_alternate_hypo_%d_random_seed_%d.pkl' %
            (args.split_type, args.hypothesis_sampler_type, split_name, args.positive_threshold,
             args.min_pos_images_per_episode, args.max_neg_images_per_episode,
             args.train_size, args.negative_type,
             args.add_alternate_hypotheses, args.random_seed))

    if args.split_type == 'comp':
        split_type_or_regex = args.split_type
    else:
        split_type_or_regex = os.path.join(
            args.output_dir, '%d_%.2f_' %
            (args.min_pos_images_per_episode, args.positive_threshold) +
            args.split_type + '_split_hypothesis_split_%s.json')

    raw_hypothesis_splits, splits_to_all_hypothesis_idx = split_train_val_test(
        filtered_hypotheses_evaluations, split_type_or_regex=split_type_or_regex)

    create_meta_dataset(raw_hypothesis_splits,
                        splits_to_all_hypothesis_idx,
                        filtered_hypotheses_evaluations,
                        args.min_pos_images_per_episode,
                        args.max_neg_images_per_episode,
                        num_scenes,
                        metadata,
                        num_examples_per_split,
                        args.add_alternate_hypotheses,
                        splits_to_output_paths,
                        args.negative_type,
                        args.num_cpus,
                        images_to_hypotheses,
                        args.hypothesis_sampler_type,
                        split_type_or_regex=split_type_or_regex)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)