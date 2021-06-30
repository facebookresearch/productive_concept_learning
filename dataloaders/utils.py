# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A set of utilities for dataloaders."""
import os
import numpy as np
import tempfile
import torch
import torch.utils.data as data
import json
import pickle

from typing import Union, Tuple
from torchvision.datasets.folder import has_file_allowed_extension


def has_allowed_extension(f, extension):
    if len(f) <= len(extension):
        return False
    if f[len(f)-len(extension):] == extension:
        return True
    return False


def tokenizer_programs(prog_string):
    """A tokenizer function for programs."""
    prog_string = prog_string.replace("lambda S.", "lambdaS.")
    return prog_string.split(' ')


def clevr_json_loader(file):
    with open(file, 'r') as f:
        json_scene_data = json.load(f)

    return json_scene_data['objects']

def to_tensor_sound(x):
    return torch.Tensor(x)

def sound_loader(file):
    with open(file, 'rb') as f:
        sound_data = pickle.load(f)
    return sound_data

def _numeric_string_array_to_numbers(numeric_string_array, cast_type="float"):
    if cast_type=="float":
        cast_fn = float
    elif cast_type=="int":
        cast_fn = int

    numeric_array = []
    for t in numeric_string_array:
        numeric_array.append(
                np.array([cast_fn(x) for x in t.split(",")]))
    return numeric_array

class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self,
                 root,
                 transforms=None,
                 transform=None,
                 target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can "
                "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class ImageAccess(object):
    def __init__(self,
                 root_dir,
                 image_string="ADHOC_train_%.8d.png",
                 file_pattern=".png",
                 debug=False):
        all_data = make_adhoc_dataset_with_buffer(
            root_dir, extensions=tuple([file_pattern]))
        self._all_data = all_data

    def __call__(self, image_id):
        return self._all_data[image_id]


def get_validity_fn(extensions: Union[Tuple[str], str]):
    """Return a function that assesses validity of files based on extensions.

    Args:
      extensions: Tuple of str, or str, where each element is a possible extension
    Returns:
      A function that checks if a filepath is valid
    Raises:
      ValueError if extensions is not a list
    """
    def is_valid_file(x):
        return has_file_allowed_extension(x, extensions)
        #return has_allowed_extension(x, extensions)

    return is_valid_file


def make_adhoc_dataset_with_buffer(dir,
                                   extensions=None,
                                   is_valid_file=None,
                                   buffer_threshold=100000):
    """Implements a buffered way to create a folderdataset.

  A modification of `torchvision.datasets.folder.make_dataset` that uses a
  buffer mechanism for faster performance when the number of files in the
  dataset can potentially be very very large.

  Args:
    dir: Str, Directory where the dataset exists
    extensions: List of Str
    is_valid_file: Function
    buffer_threshold: Int, number of entries to flush the buffer with
  Returns:
    all_data: list of tuple of path to object and class index.
  Raises:
    ValueError: If both extensions and is_valid_file are None or if dataset
      files are not in the format x_y_z.extension or if we repeat an index that
      has already been processed
    RuntimeError: If the system command pfind fails
  """
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = get_validity_fn(extensions)

    all_data = {}
    buffer = {}

    _, tfile = tempfile.mkstemp()
    ret = os.system('pfind %s > %s' % (dir, tfile))
    if ret == 0:
        with open(tfile, 'r') as f:
            file_list = [x.rstrip() for x in f.readlines()]
        os.system('rm %s' % (tfile))
    else:
        raise RuntimeError("System command pfind failed.")

    for path in sorted(file_list):
        if is_valid_file(path):
            fname = path.split('/')[-1]
            idx = fname.split('_')

            if len(idx) != 3:
                raise ValueError("Unexpected file format.")

            idx = int(fname.split('_')[-1].split('.')[0])
            if idx in buffer.keys():
                raise ValueError("Index already processed.")
            buffer[idx] = path

            if len(buffer) > buffer_threshold:
                all_data = {**all_data, **buffer}
                del buffer
                buffer = {}

    all_data = {**all_data, **buffer}

    return all_data


class DatasetFolderPathIndexing(VisionDataset):
    def __init__(self,
                 root,
                 loader,
                 extensions=None,
                 transform=None,
                 target_transform=None,
                 is_valid_file=None):
        super(DatasetFolderPathIndexing,
              self).__init__(root,
                             transform=transform,
                             target_transform=target_transform)
        samples = make_adhoc_dataset_with_buffer(self.root, extensions,
                                                 is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " +
                                self.root + "\n"
                                "Supported extensions are: " +
                                ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.samples = samples

    def __getitem__(self, index):
        """
    Args:
        index (int): Index

    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return {"datum": sample, "path": path}

    def get_item_list(self, index_list):
        return [self.__getitem__(x) for x in index_list]

    def __len__(self):
       return len(self.samples)
