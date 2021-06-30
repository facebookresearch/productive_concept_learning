# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Reader class for images from URLs"""
import logging
import os
import torch
import math
import numpy as np
import pickle
import json
import torch.utils.data as data

from collections import namedtuple
from collections import defaultdict
from collections import Counter
from torchvision.datasets.folder import DatasetFolder
from torchvision.datasets.folder import default_loader as default_image_loader
from torchvision.transforms.functional import to_tensor
from typing import Any, Dict, List, Optional, Type, Callable

from hypothesis_generation.hypothesis_utils import MetaDatasetExample
from hypothesis_generation.hypothesis_utils import HypothesisEval
from dataloaders.utils import ImageAccess
from hypothesis_generation.hypothesis_utils import fast_random_negatives
from hypothesis_generation.reduce_and_process_hypotheses import POS_LABEL_ID

from dataloaders.vocabulary import Vocabulary
from dataloaders.vocabulary import ClevrJsonToTensor
from dataloaders.build_sound_scene import ClevrJsonToSoundTensor
from dataloaders.utils import VisionDataset
from dataloaders.utils import DatasetFolderPathIndexing
from dataloaders.utils import tokenizer_programs
from dataloaders.utils import clevr_json_loader
from dataloaders.utils import sound_loader
from dataloaders.utils import to_tensor_sound

_SPLITS_TO_SHUFFLE = {
    "train": True,
    "val": False,
    "test": False,
    "cross_split": False,
}
_TRADITIONAL_DATASET_BS_RATIO = 100
_SPLITS_TO_NUM_HYPOTHESES = {"cross_split": 14929}


def get_loader_for_modality(image_path, json_path, modality,
                            modality_to_transform_fn: dict):
    if modality == "image":
        loader = DatasetFolderPathIndexing(
            image_path,
            default_image_loader,
            extensions=".png",
            transform=modality_to_transform_fn[modality])
    elif modality == "json":
        loader = DatasetFolderPathIndexing(
            json_path,
            clevr_json_loader,
            extensions=".json",
            transform=modality_to_transform_fn[modality])
    return loader


class TraditionalDataset(data.Dataset):
    def __init__(self,
                 path_to_dataset,
                 image_hypothesis_mapping_file,
                 image_path,
                 json_path,
                 modality_to_transform_fn,
                 modality="image",
                 split_name=None,
                 random_seed=42,
                 num_images_per_concept=5):
        """Initialize a batched dataset of images and labels.

        Args:
          path_to_dataset: Raw path to the hypothesis file which contains the list
            of hypotheses and their raw denotations.
          image_path: Path to images.
          json_path: Path to JSON files
          modality: Str, which modality to use. One of "image", "json", "sound"
          split_name: Str with value 'cross_split' contains 5 fixed images from every hypothesis.
          random_seed: Int, random seed.
          num_images_per_concept: Int
        """
        np.random.seed(random_seed)
        self._split_name = split_name
        self._modality = modality
        self._modality_to_transform_fn = modality_to_transform_fn

        with open(path_to_dataset, 'rb') as f:
            all_data = pickle.load(f)
            dataset = all_data["hypothesis_and_full_evaluations"]

        root_dir = '/'.join(path_to_dataset.split('/')[:-1])
        with open(os.path.join(root_dir, image_hypothesis_mapping_file),
                  'rb') as f:
            datum_hypothesis_data = pickle.load(f)
            all_data_to_hypotheses = datum_hypothesis_data[
                "images_to_hypotheses"]
            all_data_to_labels = datum_hypothesis_data["images_to_labels"]

            if len(dataset.hypothesis
                   ) != _SPLITS_TO_NUM_HYPOTHESES[split_name]:
                raise ValueError("Expected %d hypotheses",
                                 _SPLITS_TO_NUM_HYPOTHESES[split_name])

        self._all_data_to_labels = all_data_to_labels
        self._all_data_to_hypotheses = all_data_to_hypotheses

        num_labels = len(dataset.hypothesis)
        if self._split_name is not None:
            this_dataset_datapoint_subset = []

            for hyp_idx, _ in enumerate(dataset.hypothesis):
                this_images = np.random.choice(dataset.image_id_list[hyp_idx],
                                               size=num_images_per_concept,
                                               replace=True)
                this_dataset_datapoint_subset.extend(this_images)

            self.this_dataset_datapoint_subset = this_dataset_datapoint_subset
        self.num_labels = num_labels

        self._loader = get_loader_for_modality(
            image_path,
            json_path,
            self._modality,
            modality_to_transform_fn=self._modality_to_transform_fn)

    def __getitem__(self, index):
        if len(self.this_dataset_datapoint_subset) == 0:
            raise ValueError(
                "Dataset can only provide values when num_images_per_concept is specified."
            )
        if self._split_name == None:
            raise ValueError("Can only call when split name is not None.")

        datum_with_path = self._loader[self.this_dataset_datapoint_subset[index]]

        datum = datum_with_path["datum"]
        path = datum_with_path["path"]

        labels = self._all_data_to_labels[
            self.this_dataset_datapoint_subset[index]]

        label_multi_hot = torch.zeros(self.num_labels).long()
        hypothesis_string_list = self._all_data_to_hypotheses[
            self.this_dataset_datapoint_subset[index]]

        hypothesis_string = ",".join(hypothesis_string_list)
        for this_label in labels:
            label_multi_hot[this_label] = 1

        return {
            "datum": datum,
            "path": path,
            "labels": label_multi_hot,
            "hypotheses_string": hypothesis_string,
        }

    def __len__(self):
        return len(self.this_dataset_datapoint_subset)

    @property
    def all_images_to_hypotheses(self):
        return self._all_data_to_hypotheses

    @property
    def all_images_to_labels(self):
        return self._all_data_to_labels


class MetaDataset(data.Dataset):
    def __init__(self,
                 path_to_meta_dataset: str,
                 image_hypothesis_mapping_file: str,
                 image_path: str,
                 json_path: str,
                 modality_to_transform_fn: Dict[str, Callable],
                 modality: str = "image",
                 split_name: Optional[str] = None,
                 true_class_id: int = POS_LABEL_ID,
                 load_multihot_labels_per_datapoint: bool = False,
                 tokenizer: Callable = tokenizer_programs):
        """Init Meta dataset."""

        if modality not in ["image", "sound", "json"]:
            raise ValueError(f"Unknown modality {modality}")

        with open(path_to_meta_dataset, 'rb') as f:
            meta_dataset_and_all_hypotheses = pickle.load(f)
            meta_dataset = meta_dataset_and_all_hypotheses['meta_dataset']
            all_hypotheses_across_splits = (
                meta_dataset_and_all_hypotheses['all_hypotheses_across_splits']
                .hypothesis)

        if load_multihot_labels_per_datapoint == True:
            root_dir = '/'.join(path_to_meta_dataset.split('/')[:-1])
            with open(os.path.join(root_dir, image_hypothesis_mapping_file),
                      'rb') as f:
                datum_hypothesis_data = pickle.load(f)
                all_data_to_labels = datum_hypothesis_data["images_to_labels"]

        train_hypothesis_idx = meta_dataset_and_all_hypotheses[
            "split_name_to_all_hypothesis_idx"]["train"]

        train_hypothesis_mask = torch.zeros(len(all_hypotheses_across_splits))
        train_hypothesis_mask[train_hypothesis_idx] = 1.0

        if load_multihot_labels_per_datapoint == True:
            self._all_data_to_labels = all_data_to_labels
            self._num_labels = len(all_hypotheses_across_splits)

        self._load_multihot_labels_per_datapoint = load_multihot_labels_per_datapoint
        self._train_hypothesis_mask = train_hypothesis_mask

        vocabulary = Vocabulary(all_hypotheses_across_splits, tokenizer)
        self._meta_dataset = meta_dataset
        self._modality = modality
        self._modality_to_transform_fn = modality_to_transform_fn
        self._split_name = split_name

        self._loader = get_loader_for_modality(
            image_path,
            json_path,
            self._modality,
            modality_to_transform_fn=self._modality_to_transform_fn)
        self._vocabulary = vocabulary
        self._true_class_id = true_class_id
        hypotheses_in_split = set(
            [x['support'].hypothesis for x in self._meta_dataset])
        self._hypotheses_in_split = hypotheses_in_split
        self._all_hypotheses_across_splits = all_hypotheses_across_splits

        self._all_hyp_str_to_idx = {
            v: k
            for k, v in enumerate(self._all_hypotheses_across_splits)
        }

    def __getitem__(self, idx):
        # NOTE: Train here refers to query/ support, not meta-train/ meta-test.
        hypothesis_string = self._meta_dataset[idx]["support"].hypothesis
        hypothesis_encoded = self._vocabulary.encode_string(
            hypothesis_string).long()

        hypothesis_idx_dense = self._all_hyp_str_to_idx[hypothesis_string]

        splits_with_images = {}
        splits_with_labels = {}

        for episode_split in ["support", "query"]:
            data_list = self._meta_dataset[idx][episode_split].raw_data_ids
            labels = np.array(
                self._meta_dataset[idx][episode_split].data_labels)

            data_in_split = [
                x["datum"] for x in self._loader.get_item_list(data_list)
            ]

            if isinstance(data_in_split[0], torch.Tensor):
                data_in_split = torch.stack(data_in_split)

            splits_with_labels[episode_split] = labels
            splits_with_images[episode_split] = data_in_split

            if episode_split == "support":
                all_consistent_hypotheses = ",".join(
                    self._meta_dataset[idx][episode_split].all_valid_hypotheses)

                all_consistent_hypotheses_idx_sparse = torch.zeros(
                    len(self._all_hypotheses_across_splits)).type(torch.bool)
                posterior_probs_sparse = torch.zeros(
                    len(self._all_hypotheses_across_splits))
                all_consistent_hypotheses_idx_dense = []
                for valid_hyp, valid_hyp_log_prob in zip(
                        self._meta_dataset[idx]
                    [episode_split].all_valid_hypotheses,
                        self._meta_dataset[idx]
                    [episode_split].posterior_logprobs):

                    all_consistent_hypotheses_idx_sparse[
                        self._all_hyp_str_to_idx[valid_hyp]] = True

                    posterior_probs_sparse[
                        self._all_hyp_str_to_idx[valid_hyp]] = np.exp(
                            valid_hyp_log_prob)

                    all_consistent_hypotheses_idx_dense.append(
                        "%d" % self._all_hyp_str_to_idx[valid_hyp])

                all_consistent_hypotheses_idx_dense = ",".join(
                    all_consistent_hypotheses_idx_dense)

                posterior_probs_dense = ",".join([
                    "%f" % x for x in np.exp(
                        np.array(self._meta_dataset[idx]
                                 [episode_split].posterior_logprobs))
                ])

                # Get a version of the posterior over only the training
                # hypotheses
                posterior_probs_train_sparse = (
                    posterior_probs_sparse * self._train_hypothesis_mask)
                posterior_probs_train_sparse = posterior_probs_train_sparse / (
                    torch.sum(posterior_probs_train_sparse) + 1e-12)

            if episode_split == "query":
                optimistic_query_labels = np.array(
                    self._meta_dataset[idx]
                    [episode_split].optimistic_data_labels)

                query_multihot_perdata_labels = []
                if self._load_multihot_labels_per_datapoint == True:
                    for datum in data_list:
                        label_multi_hot = torch.zeros(self._num_labels).long()
                        label_multi_hot[self._all_data_to_labels[datum]] = 1
                        query_multihot_perdata_labels.append(label_multi_hot)

                    query_multihot_perdata_labels = torch.stack(
                        query_multihot_perdata_labels, dim=0)

        return {
            "support_images": splits_with_images["support"],
            "support_labels": splits_with_labels["support"],
            "query_images": splits_with_images["query"],
            "query_labels": splits_with_labels["query"],
            "query_multihot_perdata_labels": query_multihot_perdata_labels,
            "optimistic_query_labels": optimistic_query_labels,
            "hypotheses_encoded": hypothesis_encoded,
            "hypotheses_string": hypothesis_string,
            "hypotheses_idx_dense": hypothesis_idx_dense,
            "all_consistent_hypotheses": all_consistent_hypotheses,
            "all_consistent_hypotheses_idx_sparse": all_consistent_hypotheses_idx_sparse,
            "all_consistent_hypotheses_idx_dense": all_consistent_hypotheses_idx_dense,
            "posterior_probs_dense": posterior_probs_dense, # need to send as float %f
            "posterior_probs_sparse": posterior_probs_sparse,
            "posterior_probs_train_sparse": posterior_probs_train_sparse,
        }

    def __len__(self):
        return len(self._meta_dataset)

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def true_class_id(self):
        return self._true_class_id

    @property
    def hypotheses_in_split(self):
        return self._hypotheses_in_split

    @property
    def split_name(self):
        return self._split_name

    @property
    def all_hypotheses_across_splits(self):
        return self._all_hypotheses_across_splits


def get_adhoc_loader(cfg, batch_size, splits):
    HYPOTHESIS_DATASET = cfg.data.path
    data_loader_for_split = {}

    json_transform = ClevrJsonToTensor(cfg.raw_data.properties_file_path)

    modality_to_transform_fn = {
        "image": to_tensor,
        "json": json_transform,
        "sound": to_tensor_sound,
    }

    for this_split in splits:
        logging.info("Loading dataset %s." % (this_split))
        path_to_hypothesis_dataset = os.path.join(HYPOTHESIS_DATASET,
                                                  cfg.get(this_split))

        if this_split != "cross_split":
            this_dataset = MetaDataset(
                path_to_hypothesis_dataset,
                image_hypothesis_mapping_file=cfg.get(
                    "cross_split_hypothesis_image_mapping"),
                image_path=cfg.raw_data.image_path,
                json_path=cfg.raw_data.json_path,
                modality=cfg._data.modality,
                modality_to_transform_fn=modality_to_transform_fn,
                load_multihot_labels_per_datapoint=cfg._mode!="train",
                split_name=this_split)

            use_batch_size = batch_size
        else:
            this_dataset = TraditionalDataset(
                path_to_hypothesis_dataset,
                image_hypothesis_mapping_file=cfg.get(
                    "cross_split_hypothesis_image_mapping"),
                split_name=this_split,
                image_path=cfg.raw_data.image_path,
                json_path=cfg.raw_data.json_path,
                modality=cfg._data.modality,
                modality_to_transform_fn=modality_to_transform_fn,
                num_images_per_concept=cfg.data.map_eval_num_images_per_concept,
            )
            # Traditional dataset has a higher batch size.
            use_batch_size = batch_size * _TRADITIONAL_DATASET_BS_RATIO

        if cfg._mode == "train":
            shuffle = _SPLITS_TO_SHUFFLE[this_split]
        elif cfg._mode == "eval":
            # No shuffling in evaluation mode.
            shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset=this_dataset,
            batch_size=use_batch_size,
            shuffle=shuffle,
            num_workers=cfg.opt.num_workers,
        )
        data_loader_for_split[this_split] = data_loader
    return data_loader_for_split