# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A vocabulary class definition.

This code largely builds on the vocabulary class definition in FairSeq.
"""
import json
import logging
import numpy as np
import torch

from collections import Counter

class Vocabulary(object):
    def __init__(
            self,
            all_sentences: list,
            tokenizer,
            pad='<pad>',
            eos='</s>',
            unk='<unk>',
            bos='<s>',
    ):
        """Initialize the vocabulary"""
        tokenized_sentences = [tokenizer(x) for x in all_sentences]
        vocabulary = []

        vocabulary.append(pad)
        vocabulary.append(eos)
        vocabulary.append(unk)
        vocabulary.append(bos)

        # Closed world assumption, cannot see a sentence longer than what has been
        # shown at training time.
        max_length_dataset = np.max([len(x) for x in tokenized_sentences])
        vocabulary.extend(
            sorted(
                Counter([x for y in tokenized_sentences for x in y]).keys()))

        logging.info("%d tokens found in the dataset." % (len(vocabulary)))

        self._items = vocabulary
        self._tokenizer = tokenizer
        # Add +2 below for bos and eos tokens.
        self._max_length_dataset = max_length_dataset + 2
        self._item_to_idx = {v: k for k, v in enumerate(vocabulary)}
        self._idx_to_item = {k: v for k, v in enumerate(vocabulary)}
        self._pad_string = pad
        self._eos_string = eos
        self._unk_string = unk
        self._bos_string = bos

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self._items:
            return self._item_to_idx[sym]
        return self.unk()

    def encode_string(self,
                      input_string,
                      append_eos=True,
                      add_start_token=True):
        tokenized_string = self._tokenizer(input_string)
        n_words = len(tokenized_string)

        additional_tokens = 0
        if add_start_token is True:
            additional_tokens += 1

        if append_eos is True:
            additional_tokens += 1

        if n_words + additional_tokens > self._max_length_dataset:
            raise ValueError("String is too long to encode.")
        ids = torch.IntTensor(self._max_length_dataset)
        if add_start_token is True:
            ids[0] = self.bos()

        for i, w in enumerate(tokenized_string):
            ids[i + additional_tokens - 1] = self.index(w)

        if append_eos == True:
            ids[len(tokenized_string) + additional_tokens - 1] = self.eos()

        for i in range(
                len(tokenized_string) + additional_tokens,
                self._max_length_dataset):
            ids[i] = self.pad()

        return ids

    def decode_string(self, tensor, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.

    Can cfgionally remove BPE symbols or escape <unk> words.
    """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t, escape_unk) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self._idx_to_item[int(i)]

        sent = ' '.join(token_string(i) for i in tensor)
        return sent

    def __len__(self):
        return len(self._item_to_idx)

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self._item_to_idx[self._bos_string]

    def pad(self):
        """Helper to get index of pad symbol"""
        return self._item_to_idx[self._pad_string]

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self._item_to_idx[self._eos_string]

    def unk(self):
        """Helper to get index of unk symbol"""
        return self._item_to_idx[self._unk_string]

    def bos_string(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self._bos_string

    def pad_string(self):
        """Helper to get index of pad symbol"""
        return self._pad_string

    def eos_string(self):
        """Helper to get index of end-of-sentence symbol"""
        return self._eos_string

    def unk_string(self, escape_unk=False):
        """Helper to get index of unk symbol"""
        if escape_unk is True:
            return ''
        return self._unk_string

    @property
    def items(self):
        return self._items


class ClevrJsonToTensor(object):
    def __init__(self, properties_file_path):
        with open(properties_file_path, "r") as f:
            properties_json = json.load(f)
            metadata = properties_json["metadata"]
            properties_vocabulary = []
            
            # NOTE: The notion of categorical properties here is based on the
            # CLEVR JSON format, and and not based on the language of thought.
            # This is in many ways a deliberate choice. What is discrete vs not
            # in the language of thought is something that learning should
            # handle, ideally.
            cateogrical_properties = []

            for key in properties_json["properties"]:
                properties_vocabulary.extend(
                    list(properties_json["properties"][key].keys()))
                # Properties file has plural forms of the properties listed.
                # Make it singular.
                cateogrical_properties.append(key.rstrip("s"))

            self._dimension_to_axis_name = {v: k for k, v in properties_json[
                "metadata"]["dimensions_to_idx"].items()}
            self._max_objects_in_scene = properties_json["metadata"]["max_objects"]

        self._vocabulary = properties_vocabulary
        self._categorical_properties = cateogrical_properties
        self._metadata = metadata
        self._idx_to_value = {}
        self._word_to_idx = {v: k for k, v in enumerate(self._vocabulary)}
        self._BAN_FROM_ENCODING="3d_coords"

        for idx in range(len(self._vocabulary)):
            self._idx_to_value[idx] = 1

        self._word_to_idx["pixel_coords_x"] = len(self._word_to_idx)
        self._idx_to_value[len(self._word_to_idx)-1] = lambda x: x/float(
            metadata["image_size"]["x"])

        self._word_to_idx["pixel_coords_y"] = len(self._word_to_idx)
        self._idx_to_value[len(self._word_to_idx)-1] = lambda x: x/float(
            metadata["image_size"]["y"]
        )

        self._word_to_idx["pixel_coords_z"] = len(self._word_to_idx)
        self._idx_to_value[len(self._word_to_idx)-1] = lambda x: x/float(
            metadata["image_size"]["z"])

        self._word_to_idx["rotation"] = len(self._word_to_idx)
        self._idx_to_value[len(self._word_to_idx)-1] = lambda x: x/float(
            metadata["max_rotation"]
        )


    def _encode(self, obj):

        flat_obj = {}
        for prop, value in obj.items():
            if prop == "pixel_coords":
                for loc_idx in range(len(value)):
                    word_string = (prop + "_" +
                                   self._dimension_to_axis_name[loc_idx])
                    flat_obj[word_string] = value[loc_idx]
            elif prop not in self._BAN_FROM_ENCODING:
                flat_obj[prop] = value
        sorted_keys = sorted(list(flat_obj.keys()))
        overall_vec = []
        for prop in sorted_keys:
            value = flat_obj[prop]
            vec = torch.zeros(len(self._word_to_idx))
            if prop in self._categorical_properties:
                vec[self._word_to_idx[value]] = self._idx_to_value[
                    self._word_to_idx[value]]
            else:
                vec[self._word_to_idx[prop]] = self._idx_to_value[
                    self._word_to_idx[prop]](value)
            overall_vec.append(vec)

        return torch.stack(overall_vec, dim=0).sum(0)
    
    def __call__(self, scene):
        x = []
        if len(scene) > self.max_objects_in_scene:
            raise ValueError(
                "Number of objects in scene greater than max objects in scene.")
        
        for obj in scene:
            x.append(self._encode(obj))
        
        for _ in range(len(scene), self.max_objects_in_scene):
            x.append(-1 * torch.ones_like(x[0]))
            
        return torch.stack(x)
   
    @property 
    def max_objects_in_scene(self):
        return self._max_objects_in_scene