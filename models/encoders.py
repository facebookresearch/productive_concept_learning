# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Encoders for prototypical networks.

Provides different encoders for prototypical networks corresponding to
modalities like image, json and sound.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

from frozendict import frozendict
from itertools import product

from models.utils import SumPool
from models.utils import build_resnet_base
from models.utils import UnsqueezeRepeatTensor

_MODEL_STAGE_RESNET = 3
_RESNET_FEATURE_DIM_FOR_STAGE = frozendict({3: 256})
_SOUND_SPECTROGRAM_SIZE_FOR_SOUND = frozendict({120000: 201})
_IMAGE_ENCODER_POST_RESNET_STRIDE = (1, 2)
_JSON_EMBED_TO_INPUT_RATIO = 3


class PermuteTensor(nn.Module):
    def __init__(self, perm_order):
        super(PermuteTensor, self).__init__()
        self._perm_order = perm_order

    def forward(self, x):
        return x.permute(self._perm_order)


class FlattenTensor(nn.Module):
    def __init__(self, dim):
        super(FlattenTensor, self).__init__()
        self._dim = dim

    def forward(self, x):
        return x.flatten(self._dim)


#class OuterProductFlattenTensor(nn.Module):
#    def __init__(self, dim):
#        super(OuterProductFlattenTensor, self).__init__()
#        self_dim = dim
#
#    def forward(self, x):
#        """Take an ensemble of differences and use it to create feature."""
#        # x is [B x 5 x 19]
#        x = x.permute(0, )
#        y = x.unsqueeze(-1).repeat(1, 1, 1, x.shape(1)).view(x.shape(0), x.shape(1) * x.shape(1), -1)
#
#        x = x.sum(dim=)

class OptionalPositionEncoding(nn.Module):
    """An nn.module class for position encoding."""
    def __init__(self, num_objects_list, position_encoding=True):
        super(OptionalPositionEncoding, self).__init__()
        self.num_objects_list = num_objects_list
        self.position_encoding = position_encoding

        if position_encoding == True:
            position_embeds = torch.zeros(np.prod(self.num_objects_list),
                                          np.sum(self.num_objects_list))
            locations_offset = ([0] +
                                list(np.cumsum(self.num_objects_list)))[:-1]

            locations_iteator = [
                range(offset, x + offset)
                for x, offset in zip(self.num_objects_list, locations_offset)
            ]

            for prod_idx, locations in enumerate(product(*locations_iteator)):
                for loc in locations:
                    position_embeds[prod_idx][loc] = 1.0
            position_embeds = position_embeds.unsqueeze(0).permute(0, 2, 1)
            position_embeds = position_embeds.reshape(
                position_embeds.size(0), position_embeds.size(1), *self.num_objects_list)

            self.register_buffer('position_embeds', position_embeds)

    def forward(self, x):
        if self.position_encoding == True:
            position_embeds = self.position_embeds.repeat(
                x.size(0), *([1] * (1 + len(self.num_objects_list))))
            return torch.cat([x, position_embeds], dim=1)
        return x

def build_concat_pooling(embed_dim, object_dims, feature_dim):
    concat_pooling = nn.Sequential(
        FlattenTensor(dim=2), # Flatten all the objects. # B x F x {O}
        PermuteTensor((0, 2, 1)),  # B x {O} x F
        FlattenTensor(dim=1),  # B x F x {O}
        nn.Linear(embed_dim * np.prod(object_dims), 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 512, bias=True),  # Just replace with 1x1 conv.
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256, bias=True),  # Just replace with 1x1 conv.
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, feature_dim, bias=True),
    )
    return concat_pooling


class TransformerPooling(nn.Module):
    """Inspired by the model in the following PGM paper:

        Wang, Duo, Mateja Jamnik, and Pietro Lio. 2019.
        Abstract Diagrammatic Reasoning with Multiplex Graph Networks.
        https://openreview.net/pdf?id=ByxQB1BKwH.
    """
    def __init__(self, obj_fdim, num_objects_list, output_dim_f, 
                 n_head=2, num_layers=4, dim_feedforward=512):
        super(TransformerPooling, self).__init__()

        if obj_fdim % n_head != 0:
            raise ValueError("Object dim must be divisible by num heads.")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=obj_fdim, nhead=n_head, dim_feedforward=dim_feedforward)
        transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                    num_layers=num_layers)
        self._trnsf = transformer_encoder

        # 4 here is because input is concat(max(), min(), sum(), mean())
        self._embedder = nn.Linear(4 * obj_fdim, output_dim_f)
        self._num_objects_list = num_objects_list

    def forward(self, x):
        n_objects = np.prod(self._num_objects_list)

        # Reshape to [B x C x n_objects]
        x = x.view(x.size(0), x.size(1), n_objects)
        # Transpose to [n_objects x B x C]
        x = x.permute(2, 0, 1)
        feat_x = self._trnsf(x)

        # Transpose to [B x C x n_objects]
        feat_x = feat_x.permute(1, 2, 0)

        # Transpose to [B x 4*C]
        feat_x = torch.cat([feat_x.max(-1).values,
                            feat_x.min(-1).values,
                            feat_x.sum(-1),
                            feat_x.mean(-1)], dim=-1)

        return  self._embedder(feat_x)


def build_json_object_encoder(feature_dim, pretrained_object_encoder, input_feature_dim):
    if pretrained_object_encoder == True:
        raise ValueError("Cannot use a pretrained encoder for _JSONs")

    input_true_feat_dim = int(input_feature_dim.split(",")[-1])
    if len(input_feature_dim.split(",")) != 2:
        raise ValueError
    json_embed_dim = _JSON_EMBED_TO_INPUT_RATIO * input_true_feat_dim

    object_feature_object_encoder = nn.Sequential(
        PermuteTensor((0, 2, 1)),  # B x 19 x 5
        nn.Conv1d(input_true_feat_dim, json_embed_dim, 1),  # B x 19*3 x 5
        nn.ReLU(),
        nn.Conv1d(json_embed_dim, feature_dim, 1),  # B x 19*3*2 x 5
    )
    return object_feature_object_encoder


def build_image_object_encoder(feature_dim, pretrained_object_encoder,
                               input_feature_dim,
                               image_encoder_stride=_IMAGE_ENCODER_POST_RESNET_STRIDE):
    encoder = build_resnet_base(pretrained=pretrained_object_encoder)
    return nn.Sequential(
        encoder,
        nn.Conv2d(_RESNET_FEATURE_DIM_FOR_STAGE[_MODEL_STAGE_RESNET],
                  feature_dim * 2,
                  1,
                  stride=image_encoder_stride),
        nn.ReLU(),
        nn.Conv2d(feature_dim * 2,
                  feature_dim,
                  1,
                  stride=1))



def infer_output_dim(encoder, input_feature_dim):
    """Infer the output dimensions of the feature that the encoder produces.

    This is useful for initializing a relation network.
    """
    input_feature_dim = [int(x) for x in input_feature_dim.split(',')]

    with torch.no_grad():
        # Needs a batch size of atleast 2 for batchnorm in forward pass.
        dummy_input = torch.zeros([2] + input_feature_dim)
        encoder_out = encoder.forward(dummy_input)

    # Report the last few dimensions.
    if encoder_out.dim() <= 2:
        raise ValueError("Encoder output should have more than 2 dims.")

    return encoder_out.shape[1], encoder_out.shape[
        2:]  # First dimension is batch, second is channels