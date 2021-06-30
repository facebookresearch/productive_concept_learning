# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 tanh=False,
                 nonlinear=False,
                 batch_norm=False):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.nonlinear = nonlinear
        self.use_batch_norm = batch_norm

        self.linears = nn.ModuleList(
            [nn.Linear(self.input_dim, self.hidden_dims[0])])
        if self.use_batch_norm:
            self.batchnorms = nn.ModuleList([
                nn.BatchNorm1d(self.hidden_dims[i])
                for i in range(len(self.hidden_dims))
            ])

        for i in range(1, len(self.hidden_dims)):
            self.linears.append(
                nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
        self.linears.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

        if tanh:
            self.activation = torch.tanh
        else:
            self.activation = torch.relu

    def forward(self, x):

        x = self.linears[0](x)
        x = self.activation(x)
        if self.use_batch_norm:
            x = self.batchnorms[0](x)

        for i in range(1, len(self.hidden_dims)):
            x = self.linears[i](x)
            x = self.activation(x)
            if self.use_batch_norm and ((i == len(self.hidden_dims) - 1) or
                                        (i == len(self.hidden_dims) - 2)):
                x = self.batchnorms[i](x)

        out = self.linears[-1](x)
        if self.nonlinear:
            out = self.activation(out)

        return out