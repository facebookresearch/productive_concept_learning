# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torchvision
import models.audio_resnet as audio_resnet


class EncoderSequential(nn.Sequential):
    def __init__(self, *args, modality="image", feature_dim=None):
        super(EncoderSequential, self).__init__(*args)
        self._modality = modality
        self._feature_dim = feature_dim

    @property
    def modality(self):
        return self._modality

    @property
    def feature_dim(self):
        return self._feature_dim


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class UnsqueezeRepeatTensor(nn.Module):
    def __init__(self, dim, repeat=3):
        super(UnsqueezeRepeatTensor, self).__init__()
        self._dim = dim
        self._repeat = repeat

    def forward(self, x):
        unsqueeze_pick = [1] * (x.dim() + 1)
        unsqueeze_pick[self._dim] = self._repeat
        return x.unsqueeze(self._dim).repeat(unsqueeze_pick)
    

class ReshapeTensor(nn.Module):
    def __init__(self, new_shape):
        super(ReshapeTensor, self).__init__()
        self._new_shape = new_shape
    
    def forward(self, x):
        return x.reshape(self._new_shape)

class SumPool(nn.Module):
    def __init__(self, dim):
        super(SumPool, self).__init__()
        self._dim = dim

    def forward(self, x):
        return x.sum(dim=self._dim)


def build_resnet_base(model_name='resnet18', model_stage=3, pretrained=False,
                      one_dim_resnet=False, audio_input_channels=201):

    if not hasattr(torchvision.models, model_name):
        raise ValueError('Invalid model "%s"' % model_name)
    if not 'resnet' in model_name:
        raise ValueError('Feature extraction only supports ResNets')
    if one_dim_resnet == True:
        cnn = getattr(audio_resnet, model_name)(pretrained=pretrained,
                                                input_channels=audio_input_channels)
    else:
        cnn = getattr(torchvision.models, model_name)(pretrained=pretrained)
    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(model_stage):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))
    model = nn.Sequential(*layers)
    return model


def euclidean_dist(x, y):
    # x: B x N x D
    # y: B x M x D
    b = x.size(0)
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    assert d == y.size(2)
    assert b == y.size(0)

    x = x.unsqueeze(2).expand(b, n, m, d)
    y = y.unsqueeze(1).expand(b, n, m, d)

    return torch.pow(x - y, 2).sum(3)


def dict_to_device(batch, device):
    """Move a batch from to a target device."""
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    return batch
