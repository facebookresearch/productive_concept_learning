# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import dataloaders

def load(cfg, batch_size, splits):
    if cfg.data.dataset == 'adhoc_concepts':
        ds = (
          dataloaders.adhoc_data_loader.get_adhoc_loader(
            cfg, batch_size, splits)
        )
    else:
        raise ValueError("Unknown dataset: {:s}".format(cfg.data.dataset))

    return ds

class GetDataloader(object):
    def __init__(self, splits):
        if "&" in splits:
            self._splits = splits.split(" & ")
        elif "," in splits:
            self._splits = splits.split(",")
        else:
            self._splits = [splits]
        
    def __call__(self, cfg, batch_size=None):
        if batch_size == None:
            batch_size = cfg._data.batch_size

        data = load(cfg, batch_size, self._splits)

        if cfg.data.split_type not in ["comp", "iid", "color_count",
                                       "color_location", "color_material",
                                        "color", "shape", "color_boolean",
                                        "length_threshold_10"]:
            raise ValueError(f"Unknown split {cfg.data.split_type}")

        train_loader = data.get('train')
        eval_loader = data.get(cfg.eval_split_name)

        if train_loader is not None:
            if len(
                    train_loader.dataset.hypotheses_in_split.intersection(
                        eval_loader.dataset.hypotheses_in_split)
            ) != 0 and (cfg.data.split_type != "iid"):
                if cfg.eval_split_name != "train":
                    raise ValueError(
                        "Expect no overlap in concepts between train and eval"
                        "splits.")

            if (train_loader.dataset.vocabulary.items !=
                eval_loader.dataset.vocabulary.items):
                raise ValueError(
                    "Expect identical vocabularies in train and val.")

        return data
