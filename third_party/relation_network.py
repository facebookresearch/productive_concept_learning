# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Relation Netowrk code for few-shot learning experiments.

Code from https://github.com/AndreaCossu/Relation-Network-PyTorch/
"""
import torch
import torch.nn as nn
import numpy as np
from third_party.mlp import MLP
from itertools import product

REDUCTION_FACTOR = 2


def get_relative_spatial_feature(w, h, spatial_grid_feat_zero_center=True):
    h = int(h)
    w = int(w)
    d_sq = h * w
    d_four = d_sq*d_sq
    spatial_grid_feat = torch.zeros(1, 2, d_four)
    ctr = 0
    for o in range(d_sq):
        for i in range(d_sq):
            o_h = o // w
            o_w = o % w
            i_h = i // w
            i_w = i % w
            spatial_grid_feat[:, 0, ctr] = o_h - i_h
            spatial_grid_feat[:, 1, ctr] = o_w - i_w
            ctr += 1
    if spatial_grid_feat_zero_center:
        spatial_grid_feat -= (spatial_grid_feat.mean())

    return spatial_grid_feat

def get_relative_1d_feature(w, spatial_grid_feat_zero_center=True):
    w = int(w)
    spatial_grid_feat = torch.zeros(1, 1, w*w)
   
    ctr = 0 
    for o in range(w):
        for i in range(w):
            spatial_grid_feat[:, 0, ctr] = o - i
            
            ctr += 1
            
    if spatial_grid_feat_zero_center:
        spatial_grid_feat -= spatial_grid_feat.mean()
        
    return spatial_grid_feat
            


class RelationNetwork(nn.Module):
    def __init__(self, obj_fdim, num_objects_list,
                 hidden_dims_g, output_dim_g, hidden_dims_f,
                 output_dim_f,
                 tanh, use_batch_norm_f=False, relative_position_encoding=True,
                 image_concat_reduce=False):

        super(RelationNetwork, self).__init__()

        self.relative_position_encoding = relative_position_encoding
        self.obj_fdim = obj_fdim
        self.image_concat_reduce = image_concat_reduce
        self.num_objects_list = num_objects_list
        self.input_dim_g = 2 * self.obj_fdim  # g analyzes pairs of objects
        self.hidden_dims_g = hidden_dims_g
        self.output_dim_g = output_dim_g
        self.input_dim_f = self.output_dim_g

        self.hidden_dims_f = hidden_dims_f
        self.output_dim_f = output_dim_f

        self.reduction_factor = 1
        if self.image_concat_reduce == True:
            self.reduction_factor = REDUCTION_FACTOR
            
        self.input_dim_g = self.input_dim_g * (self.reduction_factor)**2
            
        if self.relative_position_encoding == True:
            self.input_dim_g = self.input_dim_g + len(self.num_objects_list)

        self.g = MLP(self.input_dim_g,
                     self.hidden_dims_g,
                     self.output_dim_g,
                     tanh=tanh,
                     nonlinear=True,
                     batch_norm=False)

        # Different from the original paper, we replace / drop dropout layers
        # since models never seem to overfit in our setting.
        self.f = MLP(self.input_dim_f,
                     self.hidden_dims_f,
                     self.output_dim_f,
                     tanh=tanh,
                     batch_norm=use_batch_norm_f)

        if self.relative_position_encoding == True:
            if len(self.num_objects_list) == 2:
                self.register_buffer(
                    'relative_grid_feat',
                    get_relative_spatial_feature(
                        self.num_objects_list[0] / self.reduction_factor,
                        self.num_objects_list[1] / self.reduction_factor))
            elif len(self.num_objects_list) == 1:
                self.register_buffer(
                    'relative_grid_feat',
                    get_relative_1d_feature(
                        self.num_objects_list[0] / self.reduction_factor))
            else:
                raise ValueError("Num object dimensions must be 2 or 1.")
                

    def forward(self, x, q=None):
        """Forward the relation network model.
        
        Args:
          x: A `Tensor` of [B x C x H x W] or [B x C x L]; C is the object
          q: A `Tensor` of [B x C x H x W] or [B x C x L]; C is the object
        Returns:
          A `Tensor` of [B x D]
        """
        n_objects = np.prod(self.num_objects_list)

        # Reshape to [B x C x n_objects]
        x = x.view(x.size(0), x.size(1), n_objects)

        if self.image_concat_reduce == True:
            if len(self.num_objects_list) == 2:
                x = x.view(x.size(0), x.size(1), self.num_objects_list[0],
                          self.num_objects_list[1])
                # TODO(ramav): Remove this hardcoding.
                x = x.permute(0, 2, 3, 1)  # [B x H x W x C]
                x = x.contiguous()
                if x.size(1) % self.reduction_factor != 0 or x.size(2) % self.reduction_factor != 0:
                    raise ValueError("Expect feature width and height to "
                                     "be divisible by 2.")
                x = x.view(x.size(0), int(x.size(1)/self.reduction_factor), int(x.size(2)/self.reduction_factor
                                                                          ), -1)
                n_objects = int(n_objects / self.reduction_factor**2)
                x = x.permute(0, 3, 1, 2)
                x = x.contiguous()
                x = x.view(x.size(0), x.size(1), n_objects)
            else:
                raise NotImplementedError("Concat reduce is only implemented "
                                          "for image models.")
        xi = x.repeat(1, 1, n_objects)  # [B x C x n_objects * n_objects]
        xj = x.unsqueeze(3)  # [B x C x n_objects x 1]
        xj = xj.repeat(1, 1, 1, n_objects)  # [B x C x n_objects x n_objects]
        xj = xj.view(x.size(0), x.size(1),
                     -1)  # [B x C x n_objects * n_objects]
        if q is not None:
            raise NotImplementedError

        pair_concat = torch.cat((xi, xj),
                                dim=1)  # (B, 2*C, n_objects * n_objects)

        if self.relative_position_encoding == True:
            pair_concat = torch.cat([
                pair_concat,
                self.relative_grid_feat.repeat(
                    pair_concat.size(0), 1, 1)
            ], dim=1)

        # MLP will take as input [B , n_objects * n_objects, 2*C]
        pair_concat = pair_concat.permute(0, 2, 1)
        relations = self.g(pair_concat.reshape(
            -1, pair_concat.size(2)))  # (n_objects*n_objects, hidden_dim_g)
        relations = relations.view(pair_concat.size(0), pair_concat.size(1),
                                   self.output_dim_g)

        embedding = torch.sum(relations, dim=1)  # (B x hidden_dim_g)

        out = self.f(embedding)  # (B x hidden_dim_f)

        return out
