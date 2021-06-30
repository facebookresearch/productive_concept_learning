# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A set of utilities for hypothesis generation."""
import copy
import unicodedata
import math
import numpy as np
import os

from functools import wraps
from scipy.stats import dirichlet
from collections import defaultdict
from collections import OrderedDict
from typing import NamedTuple
from typing import List, Dict

from tqdm import tqdm

# Emphasis map controls the amount of importance given
# to some expansions over the others. The keys below
# are to be specified in the grammar.
# See ../concept_data/v1_typed_simple_fol.json for an
# example.
EMPHASIS_MAP = {
    "*": 2,
    "-": 1.5,
    "+": 4,
}

MAX_EXPANSION_ATTEMPTS = 10

HypothesisEvalBase = NamedTuple("HypothesisEvalBase",
                                [("hypothesis", List[str]),
                                 ("logprob", List[float]),
                                 ("length", List[int]),
                                 ("image_id_list", List[List[int]])])

MetaDatasetExample = NamedTuple(
    "MetaDatasetExample",
    [
        ("index", int),  # Index of the meta dataset example.
        ("hypothesis", str),  # Hypothesis sampled in this case (str)
        ("hypothesis_idx_within_split", int),  # Hypothesis Index.
        ("hypothesis_length", int),  # Hypothesis length
        ("raw_data_ids", List[int]),  # Image IDs in the example
        ("data_labels", List[int]),  # Data labels for the image IDs
        (
            "optimistic_data_labels", List[int]
        ),  # Optimistic data lables which give credit for matching any valid hypothesis
        (
            "all_valid_hypotheses", List[str]
        ),  # All hypotheses that are consistent with the positives and negatives
        (
            'posterior_logprobs',
            List[float],
        ),
        (
            'prior_logprobs',
            List[float]
        ),
        # Probability of hypotheses under the true posterior distribuiton
        (
            'alternate_hypotheses_for_positives',
            List[str]
            # Alternate hypotheses that explain the positives (irrespective of negatives)
        )
    ])


class HypothesisEval(HypothesisEvalBase):
    __slots__ = ()

    def __len__(self):
        if len(self.hypothesis) != len(self.image_id_list):
            raise ValueError("Both dimensions expected to have same length.")
        return len(self.hypothesis)

    def get(self, idx):
        all_fields = []
        for fidx, _ in enumerate(self._fields):
            all_fields.append(self[fidx][idx])

        return HypothesisEval(*all_fields)


def fast_random_negatives(num_negatives: int,
                          total_datapoints: int = 5000000,
                          discretization: int = 5000,
                          positives=None) -> List[int]:
    """If positives are provided, prune random negatives for positives."""
    negatives = []
    while len(negatives) != num_negatives:
        block_size = int(total_datapoints / discretization)
        block_id = np.random.choice(range(block_size))

        negatives.extend(list([
            block_id * discretization + x
            for x in np.random.choice(range(discretization),
                                      size=num_negatives - len(negatives))
        ]))

        if positives is not None:
            if total_datapoints - len(positives) < num_negatives:
                raise ValueError("Cannot sample non-overlapping negatives "
                                 "without replacement.")
            negatives = list(np.unique(negatives))
            for p in positives:
                if p in negatives:
                    negatives.remove(p)

    return negatives


class HypothesisSampler(object):
    def __init__(self,
                 hypothesis_evaluations: dict,
                 sampler_type: str,
                 temperature: float = 0.2):
        """Initialize the sampler.

    Args:
      hypothesis_evaluations: An instance of HypothesisEval
      sampler_type: 'pcfg' or 'log_linear'
      temperature: float, temperature to use for the log-linear sampler
    """
        self._sampler_type = sampler_type
        self._hypothesis_evaluations = hypothesis_evaluations
        self._temperature = temperature  # Only for log-linear sampler

        if not isinstance(self._hypothesis_evaluations, dict):
            raise ValueError("Expect hypothesis evaluations as a dict.")

        if self._sampler_type not in ['pcfg', 'log_linear']:
            raise ValueError("Invalid argument %s" % self._sampler_type)

    def log_linear_probs(self):
        lengths = np.array(self._hypothesis_evaluations['length'])
        weights = np.exp(-1 * lengths * self._temperature)
        weights /= np.sum(weights)

        return weights

    def pcfg_probs(self):
        weights = np.exp(self._hypothesis_evaluations['logprob'])

        weights /= np.sum(weights)
        return weights

    def sample(self):
        if self._sampler_type == 'pcfg':
            weights = self.pcfg_probs()
        elif self._sampler_type == 'log_linear':
            weights = self.log_linear_probs()

        hypothesis_index = np.random.choice(range(
            len(self._hypothesis_evaluations['hypothesis'])),
                                            p=weights)

        return_hypothesis_eval = OrderedDict()
        for key in self._hypothesis_evaluations:
            return_hypothesis_eval[key] = self._hypothesis_evaluations[key][
                hypothesis_index]

        return return_hypothesis_eval, hypothesis_index


def create_image_index(filtered_hypotheses_evaluations):
    images_to_labels = defaultdict(list)
    images_to_hypotheses = defaultdict(list)

    for this_hyp_str, this_images in tqdm(
            zip(filtered_hypotheses_evaluations.hypothesis,
                filtered_hypotheses_evaluations.image_id_list)):
        this_hyp_idx = [
            idx
            for idx, x in enumerate(filtered_hypotheses_evaluations.hypothesis)
            if x == this_hyp_str
        ]

        if len(this_hyp_idx) != 1:
            raise ValueError("Expect hypotheses to be unique.")

        for im_idx in this_images:
            images_to_labels[im_idx].append(this_hyp_idx[0])
            images_to_hypotheses[im_idx].append(this_hyp_str)

    return images_to_labels, images_to_hypotheses


class GrammarExpander(object):
    """Class to generate expansions from a grammar."""

    def __init__(self,
                 grammar,
                 metadata,
                 max_recursion_depth=5,
                 rejection_sampling=True):
        self._grammar = grammar
        self._grammar_metadata = metadata
        self._rejection_sampling = rejection_sampling
        self._max_recursion_depth = max_recursion_depth

        # All function compleetions.
        all_functions = []
        for f in self._grammar_metadata['functions']:
            if not isinstance(self._grammar[f], list):
                grammar_completions = [self._grammar[f]]
            else:
                grammar_completions = self._grammar[f]
            all_functions.extend(grammar_completions)

        self._all_functions = tuple(all_functions)

        if self._rejection_sampling == False:
            raise NotImplementedError(
                "Rejection sampling scales much better than"
                "alternatives to program syntax changes.")

        self.extract_constant_terminals_from_grammar()

    def extract_constant_terminals_from_grammar(self):
        """Extract terminal nodes, i.e.nodes from which we cannot expand further.

    Iterates over the grammar specification, and checks for the possible
    completions (can either be a list of completions, one completion, or
    a list of lists -- all these possibilities are checked).
    """
        terminal_constants = []
        for literal in self._grammar:
            if literal not in self._grammar_metadata['functions']:
                if isinstance(self._grammar[literal], list):
                    if isinstance(self._grammar[literal][0], list):
                        for completion_literal in self._grammar[literal]:
                            for completion in completion_literal:
                                if "#" not in completion and len(
                                        str(completion).split(" ")) == 1:
                                    terminal_constants.append(completion)
                    else:
                        for completion in self._grammar[literal]:
                            if "#" not in completion and len(
                                    str(completion).split(" ")) == 1:
                                terminal_constants.append(completion)
                else:
                    if "#" not in self._grammar[literal] and len(
                            str(self._grammar[literal]).split(" ")) == 1:
                        terminal_constants.append(self._grammar[literal])

        terminal_constants = [
            x for x in terminal_constants
            if x not in self._grammar_metadata['variables']
        ]
        self._terminal_constants = terminal_constants

    def expand(self, current, tree_depth=0, override_random_selection=False):
        """Expand the current node to generate novel productions."""
        productions_to_use = []
        if current in self._grammar.keys():
            productions_to_use = copy.copy(self._grammar[current])

        if isinstance(productions_to_use, str):
            productions_to_use = [productions_to_use]

        if current in self._grammar.keys():
            if override_random_selection is True and current == 'BOOL':
                string_to_expand = np.random.choice(
                    self._grammar_metadata['best_termination_for_bool'])
            else:
                # If the productions to use are specified as a list of lists,
                # then the second entry is treated as the probability of picking
                # a completion.
                if isinstance(productions_to_use[0], list):
                    flatten_2d_list = lambda l: [t for x in l for t in x]
                    flatten_2d_list_of_list_first = lambda l: [
                        t[0] for x in l for t in x
                    ]

                    probs = []
                    for prod in productions_to_use:
                        if isinstance(prod[0], list):
                            if len(prod[0]) != 2:
                                raise ValueError(
                                    "Expected one emphasis parameter.")
                            emphasis = [EMPHASIS_MAP[x[1]] for x in prod]
                            probs_to_use = dirichlet_mode(alpha=emphasis)
                            probs.append(probs_to_use)
                        else:
                            probs.append([
                                x * 1.0 / len(productions_to_use)
                                for x in [1.0 / len(prod)] * len(prod)
                            ])

                    if isinstance(productions_to_use[0][0], list):
                        completion_strings = flatten_2d_list_of_list_first(
                            productions_to_use)
                    else:
                        completion_strings = flatten_2d_list(
                            productions_to_use)
                    probs = flatten_2d_list(probs)
                else:
                    probs = [1.0 / len(productions_to_use)
                             ] * len(productions_to_use)
                    completion_strings = productions_to_use

                if np.sum(probs) < 1 - 1e-5 and np.sum(probs) > 1 + 1e-5:
                    raise ValueError

                string_to_expand = np.random.choice(completion_strings,
                                                    p=probs)
                expanded_token_idx = np.nonzero(
                    np.array(completion_strings) == string_to_expand)[0][0]
                expanded_logprobs = np.log(probs[expanded_token_idx])
        else:
            string_to_expand = current
            expanded_logprobs = 0.0

        positions_to_expand = []
        all_expansion_logprobs = [expanded_logprobs]
        first_occurence_function = len(string_to_expand.split(' '))
        last_occurence_non_function = 0

        # NOTE: Getting a nice expression tree like representation of what we expand
        # is non trivial and tricky, since we cannot just track what gets epanded
        # vs. what does not get expanded to figure out which is the "operation"
        # and which are the arguments to it, since in a true function sense
        # we expand both functions as well as non-functions in a grammar.
        # So for now we require that the grammar specification be in a postfix
        # expression tree. That is, #OBJECT #F# is correct, since #OBJECT#
        # is an argument to #F#, but the reverse, i.e. #F# #OBJECT# is not
        # permitted.
        for idx, x in enumerate(string_to_expand.split(' ')):
            if needs_expansion(x):
                positions_to_expand.append(idx)
                if strip_hash(x) in self._grammar_metadata[
                        'functions'] and idx < first_occurence_function:
                    first_occurence_function = idx
                elif idx > last_occurence_non_function:
                    last_occurence_non_function = idx

        if first_occurence_function <= last_occurence_non_function:
            raise ValueError(
                "Expect the completions for strings to be written out as #OBJECT# #FUNCTION#"
            )

        # Error check before expanding further.
        has_error = False
        if (tree_depth == self._max_recursion_depth
                and positions_to_expand is not None
                and override_random_selection == False):
            has_error = True
            return None, has_error, None

        op_string = []
        for x in string_to_expand.split(' '):
            if not needs_expansion(x):
                op_string.append(x)
        op_string = ' '.join(op_string)
        expanded_adjacency_list = Node(op_string, current)

        for idx, x in enumerate(string_to_expand.split(' ')):
            if idx in positions_to_expand:
                candidate_node, has_error, log_probs = self.expand(
                    strip_hash(x),
                    tree_depth + 1,
                    override_random_selection=override_random_selection)

                if has_error == True and self._rejection_sampling == True:
                    num_trials = 0
                    while has_error:
                        # If we have tried enough, but cant find a completion without error,
                        # try at a higher tree depth.
                        if num_trials > MAX_EXPANSION_ATTEMPTS:
                            return None, has_error, None

                        candidate_node, has_error, log_probs = self.expand(
                            strip_hash(x),
                            tree_depth + 1,
                            override_random_selection=False)
                        num_trials += 1
                all_expansion_logprobs.append(log_probs)
                expanded_adjacency_list.add_child(candidate_node)

        if any([x is None for x in all_expansion_logprobs]):
            raise ValueError("Do not expect None.")

        all_expansion_logprobs = [
            x for x in all_expansion_logprobs if x is not None
        ]

        # Error check post expansion.
        has_error = self._check_for_error(expanded_adjacency_list)
        return expanded_adjacency_list, has_error, np.sum(
            all_expansion_logprobs)

    def _check_for_error(self, node):
        return self._check_const_only_op(node)

    def _check_const_only_op(self, node):
        num_constant_ops = 0
        for child in node.children():
            if child.op in self._terminal_constants:
                num_constant_ops += 1

        if len(node) == num_constant_ops and num_constant_ops != 0:
            return True
        return False

    @property
    def terminal_constants(self):
        return self._terminal_constants


def list_comprehension(function):
    @wraps(function)
    def wrapper(x):
        if isinstance(x, list):
            return [function(t) for t in x]
        else:
            return function(x)

    return wrapper


def class_list_comprehension(class_fn):
    @wraps(class_fn)
    def wrapper(self, x):
        if isinstance(x, list):
            return [class_fn(self, t) for t in x]
        else:
            return class_fn(self, x)

    return wrapper


def strip_hash(x):
    return x.lstrip('#').rstrip('#')


def needs_expansion(x):
    if x[0] == "#" and x[-1] == "#":
        return True
    return False


def equal_to(obj1, obj2) -> bool:
    if not type(obj1) == type(obj2):
        raise ValueError(
            "Cannot check equality for objects of different types")

    return obj1 == obj2


def greater_than(obj1, obj2) -> bool:
    if isinstance(obj1, str) or isinstance(obj2, str):
        raise ValueError("Cannot compare strings")

    if not type(obj1) == type(obj2):
        raise ValueError(
            "Cannot check equality for objects of different types")

    return obj1 > obj2


def logical_and(obj1: bool, obj2: bool) -> bool:
    if not (isinstance(obj1, bool) and isinstance(obj2, bool)):
        raise ValueError("Expected Bool.")
    return obj1 & obj2


def logical_or(obj1: bool, obj2: bool) -> bool:
    if not (isinstance(obj1, bool) and isinstance(obj2, bool)):
        raise ValueError("Expected Bool.")
    return obj1 | obj2


@list_comprehension
def logical_not(obj1: bool) -> bool:
    if not isinstance(obj1, bool):
        raise ValueError("Expected Bool.")
    return not obj1


def for_all(obj1: list, obj2) -> bool:
    if len(obj1) == 0:
        return False
    if not type(obj2) == type(obj1[0]):
        raise ValueError
    return all([x == obj2 for x in obj1])


def exists(obj1: list, obj2) -> bool:
    if len(obj1) == 0:
        return False
    if not type(obj2) == type(obj1[0]):
        raise ValueError
    return any([x == obj2 for x in obj1])


def count(obj1: list, obj2) -> int:
    if len(obj1) == 0:
        return 0
    if not type(obj2) == type(obj1[0]):
        raise ValueError
    return int(np.sum([x == obj2 for x in obj1]))


@list_comprehension
def get_color(obj) -> str:
    return obj["color"]


@list_comprehension
def get_shape(obj) -> str:
    return obj["shape"]


class get_size_function_for_dataset(object):
    """Return function that maps size to floats.

  The original CLEVR scenes have "large" etc. as strings, convert that
  to float.

  Args:
    size_mapping: A dict, with keys as the discrete size values
      and values as the continuous value associated with it.
  """

    def __init__(self, size_mapping):
        self.size_mapping = size_mapping

    @class_list_comprehension
    def __call__(self, obj) -> float:
        if not isinstance(obj["size"], str):
            raise ValueError("Expect size to be a string.")

        size_to_return = self.size_mapping[obj['size']]

        if not isinstance(size_to_return, float):
            raise ValueError("Expect size to return to be a float")
        return size_to_return


@list_comprehension
def get_material(obj) -> str:
    return obj["material"]


class get_location_function_for_dataset(object):
    def __init__(self, metadata, dim="x"):
        self.metadata = metadata
        self.dim = dim
        self.dimensions_to_idx = metadata['dimensions_to_idx']
        self.dim_bins = metadata["location_bins"][dim]

    @class_list_comprehension
    def __call__(self, obj) -> int:
        if obj["pixel_coords"][self.dimensions_to_idx[
                self.dim]] > self.metadata["image_size"][self.dim]:
            raise ValueError
        return int(
            math.floor(obj["pixel_coords"][self.dimensions_to_idx[self.dim]] /
                       (self.metadata["image_size"][self.dim] *
                        (1.0 / self.dim_bins))))


def dirichlet_mode(alpha=None):
    alpha = np.array(alpha)
    if any(alpha) < 1.0:
        raise ValueError("Alphas can only be greater than 1.")

    # A reeparameterized dirichlet with alpha - 1's mean is the
    # same as the original dirichlet's mode
    return dirichlet.mean(alpha - 1)


def partition_job_id(main_fn):
    def main_wrapper(args, num_examples, max_job_id=0, job_id=None):
        if job_id > max_job_id:
            raise ValueError("Job ID cannot be greater than max_job_id")
        if max_job_id != 0 and job_id is not None:
            partitions = np.arange(max_job_id + 2).astype(np.float32)
            partitions /= partitions[-1]
            partitions *= (num_examples + 1)
            partitions = partitions.astype(np.int32)
            partitions = list(zip(partitions[:-1], partitions[1:]))
            partition_to_pick = partitions[job_id]
        else:
            partition_to_pick = (0, num_examples)

        return main_fn(args, partition_to_pick)

    return main_wrapper


def is_number(s):
    """Check if string corresponds to a number or not"""
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


class Node(object):
    def __init__(self, name, expansion_of):
        self._op = name
        self._children = []
        self._expansion_of = expansion_of

    def add_child(self, child):
        assert (isinstance(child, Node))
        self._children.append(child)

    def children(self):
        for ch in self._children:
            yield ch

    def __len__(self):
        return len(self._children)

    @property
    def op(self):
        return self._op

    @property
    def expansion_of(self):
        return self._expansion_of


def postfix_serialize_program(program, use_brackets=False):
    serialized = []

    if program.op != "" and len(program) != 0 and use_brackets is True:
        serialized.append("{\n")
    for child in program.children():
        serialized.append(postfix_serialize_program(child))

    serialized.append(program.op)
    if program.op != "" and len(program) != 0 and use_brackets is True:
        serialized.append("}\n")

    serialized = [x for x in serialized if x != '']

    return ' '.join(serialized)


def bind_properties_to_grammar(seed_grammar, properties):
    """Convert an abstract grammar specification to something with terminals.

  The seed grammar is an abstract specification of the expansions that
  are possible on BOOL, and the Functions that can be used,
  but the exact values they can take on depends on the dataset.
  These details are specified by the property file corresponding to the
  dataset. Example property file:

    {
    "shapes": {
      "cube": "SmoothCube_v2",
      "sphere": "Sphere",
    },
    "colors": {
      "gray": [87, 87, 87],
      "red": [173, 35, 35],
    },
    "materials": {
      "rubber": "Rubber",
      "metal": "MyMetal"
    },
    "sizes": {
      "large": 0.7,
      "small": 0.35
    }

  Example grammar file:
  {
    "grammar": {
        "START": "lambda x. #BOOL#",
        "BOOL": [
            "and #BOOL# #BOOL#",
            "or #BOOL# #BOOL#",
            "not #BOOL#",
            "#F# #OBJECT#"
        ],
        "OBJECT": "x",
   }
    "metadata": {
       "best_termination_for_bool": [
           "#F# #OBJECT#"
       ]
    }
  }

  Args:
    seed_grammar: A dict with format similar to that in `simple_boolean.json`.
    properties: A dict with format as explained above.

  Returns:
    full_grammar: A dict which incorporates specified properties into the grammar.
  """
    raise NotImplementedError
