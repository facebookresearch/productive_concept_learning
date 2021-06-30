# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implements a prototypical networks model from Snell. et.al.

Snell, Jake, Kevin Swersky, and Richard Zemel. 2017.
''Prototypical Networks for Few-Shot Learning.''
In Advances in Neural Information Processing Systems 30

Original codebase for prototypical networks which has bene modified to include
minibatches of meta-learning examples:
https://www.google.com/search?q=prototypical+networks+code
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.nn.init as init

from models.simple_lstm_decoder import SimpleLSTMDecoder
from models.utils import euclidean_dist
from models.utils import ReshapeTensor
from models.utils import EncoderSequential
from models.utils import Squeeze
from models.utils import SumPool
from models.encoders import build_image_object_encoder
from models.encoders import build_json_object_encoder
from models.encoders import infer_output_dim
from models.encoders import build_concat_pooling
from models.encoders import OptionalPositionEncoding
from models.encoders import TransformerPooling
from third_party.relation_network import RelationNetwork

def weights_init_xavier(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d)\
           or isinstance(m, nn.Linear)\
           or isinstance(m, nn.Embedding):
            print('init xavier', m)
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            init.xavier_uniform(m.weight, gain=1)
            if hasattr(m, 'bias'):
                init.constant(m.bias, 0.)
        # elif isinstance(m, nn.BatchNorm2d):
        #     print('init batchnorm')
        #     m.weight.data.fill_(1)
        #     m.bias.data.zero_()

def weights_init_l2c(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def init_bert_params(model):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)


class GenericMetaClassifier(nn.Module):
    """A general module to define a meta-learning classification model."""

    def __init__(self, creator, applier, program_decoder=None):
        """Initialize a generic meta classifier model.

    Takes an input a creator model and an applier model, and optionally a
    program_decoder model. The creator creates a classifier, and the applier
    applies the classifier on query images, computing a risk function that
    most meta-learning models train with. In addition, also computes a decoding
    of the representation we get into language using an optional program
    decoder.

    Args:
      creator: An instance of nn.Module
      applier: An instance of nn.Module
      program_decoder: An instance of nn.ModuleList
    """
        super(GenericMetaClassifier, self).__init__()
        self.creator = creator
        self.applier = applier
        self.program_decoder = program_decoder

    def forward(self, sample):
        target_sequence_indices = sample["hypotheses_encoded"]
        support_images = sample["support_images"]
        support_labels = sample["support_labels"]

        query_images = sample["query_images"]

        classifier = self.creator(support_images, support_labels)
        model_out = self.applier(classifier, query_images)

        if self.program_decoder is not None:
            classifier_out_rep = self.program_decoder[0](classifier)
            logits_for_sequence, _ = self.program_decoder[1](
                output_tokens=target_sequence_indices, encoder_out=classifier_out_rep)
            model_out["logits_for_sequence"] = logits_for_sequence

        return model_out


class ProtoNetCreator(nn.Module):

    def __init__(self, encoder):
        """Initialize a ProtoNet classification creator."""
        super(ProtoNetCreator, self).__init__()
        self.encoder = encoder

        if not isinstance(self.encoder, EncoderSequential):
            raise ValueError("Expect instance of EncoderSequential")

    def forward(self, support_images_or_features, support_labels):
        """Forward support to generate a classifier.

    Args:
      support_images_or_features: A `torch.Tensor` of size [B x N x 3 x H x W]
        (for images), [B x N x O x P x D] (for JSON), [B x N x T] for sound
        or a `torch.Tensor` of size [B x N x D] where D is an extracted
        feature from either of the modalities.
      support_labels: A `torch.Tensor` of size [B x N] containing
        dense labels for each datapoint.
    """
        batch_size = support_images_or_features.size(0)
        total_images_in_support = support_images_or_features.size(1)

        num_classes_in_support = torch.unique(
          torch.max(support_labels, dim=1)[0]) + 1

        # Assumes labels are 0-indexed, and each episode has both positive
        # and negative examples.
        if torch.min(support_labels) != 0:
            raise ValueError("Expect lables to be 0-indexed.")

        if num_classes_in_support.size(0) != 1:
            raise ValueError("Expect same number of classes for "
                             "every point in the batch.")

        if self.encoder.modality == "image" and (
          support_images_or_features.dim() not in [3, 5]):
            raise ValueError("For images input must be in [3, 5] dims.")

        if self.encoder.modality == "json" and (
          support_images_or_features.dim() not in [3, 4]):
            raise ValueError("For json input must be in [3, 4] dims.")

        if self.encoder.modality == "sound" and (
          support_images_or_features.dim() not in [3]):
            raise ValueError("For sound input must be in 3 dims.")

        if support_images_or_features.size(-1) == self.encoder.feature_dim:
            z = support_images_or_features
            if z.dim() != 3:
                raise ValueError("Expect to see extracted features.")
        else:
            x = support_images_or_features.view(
              batch_size *  total_images_in_support,
                             *support_images_or_features.size()[2:])
            z = self.encoder.forward(x)
            z = z.view(batch_size, total_images_in_support, -1)
        z_proto = []
        for class_idx in range(int(num_classes_in_support)):
            class_mask = (support_labels == class_idx).float().unsqueeze(-1)
            z_proto.append(
              (torch.sum(z * class_mask, dim=1)/torch.sum(class_mask, dim=1)).unsqueeze(1)
            )

        z_proto = torch.cat(z_proto, dim=1)

        return z_proto


class ProtoNetApplier(nn.Module):

    def __init__(self,
                 encoder):
        """Protonet Applier model, which runs inference given classifier.

    Args:
      encoder: An instance of `nn.Module`
    """
        super(ProtoNetApplier, self).__init__()
        self.encoder = encoder

        if not isinstance(self.encoder, EncoderSequential):
            raise ValueError("Expect instance of Sequential with Modality")

    def forward(self, z_proto, query):
        """Forward / apply classifier on datapoints.

    Args:
      z_proto: A [B x L x D] `Tensor`
      query: A [B x N x 3 x H x W] or [B x N x D] `Tensor`
    Returns:
      A dict with keys "neg_log_p_y" (and optionally) "logits_for_sequence"
    """
        batch_size = z_proto.size(0)

        meta_learning_minibatch = query.dim() in [3, 4, 5]
        precomputed_features = query.size(-1) == self.encoder.feature_dim

        # Okay these need to be more specific.
        if query.dim() not in [5, 4, 3, 2]:
            raise ValueError("Expect either Minibatches of meta-learning dataset,"
                             "or images from regular dataset or features from"
                             "a regular dataset. Regular dataset is one which"
                             "has images of size [batch_size, features]")

        if meta_learning_minibatch:
            total_images_in_query = query.size(1)
            if precomputed_features == False:
                x = query.view(batch_size * total_images_in_query,
                             *query.size()[2:])
                zq = self.encoder(x)

                zq = zq.reshape(batch_size, total_images_in_query,
                                self.encoder.feature_dim)
            else:
                zq = query
        elif meta_learning_minibatch == False and precomputed_features == True:
            # This is the non-meta learning case where we just forward the model
            # and make predictions for all batch_size x num_tasks tasks implied
            # by z_proto. This is useful for inference.
            zq = query
            total_images_in_query = query.size(0)
            num_support_meta_learning_batches = z_proto.size(0)
            zq = zq.unsqueeze(0).expand(num_support_meta_learning_batches,
                                        total_images_in_query,
                                        self.encoder.feature_dim)
        else:
            raise ValueError("Precomputed features cannot be false for "
                             "non-meta learing case.")

        distances = euclidean_dist(zq, z_proto)
        # B x N x L `Tensor`
        neg_log_p_y = -1 * F.log_softmax(-distances, dim=2)

        return {"neg_log_p_y": neg_log_p_y}


def get_language_head(feature_dim, vocabulary, n_class):
    proto_feature_model = nn.Sequential(
      ReshapeTensor((-1, n_class * feature_dim))
    )
    # NOTE: the choice of n_class * feature_dim depends on the protonet encoding
    # in the above line.
    lstm_model = SimpleLSTMDecoder(
      dictionary=vocabulary, encoder_hidden_dim=feature_dim * n_class)
    return nn.ModuleList([proto_feature_model, lstm_model])


class GetProtoNetModel(object):
    def __init__(self, modality, pretrained_encoder, feature_dim,
                 obj_fdim, pooling,
                 language_alpha, num_classes, input_dim, init_to_use_pooling=None,
                 use_batch_norm_rel_net=False,
                 absolute_position_encoding_for_modality=True,
                 absolute_position_encoding_for_pooling=True,
                 im_fg=False, pairwise_position_encoding=False):
        self._protonet_feature_dim = feature_dim
        self._obj_fdim = obj_fdim
        self._pooling = pooling
        self._language_alpha = language_alpha
        self._modality = modality
        self._pretrained_encoder = pretrained_encoder
        self._num_classes = num_classes
        self._input_dim = input_dim
        self._init_to_use_pooling = init_to_use_pooling
        self._abs_position_encoding = absolute_position_encoding_for_modality and absolute_position_encoding_for_pooling
        self._pairwise_position_encoding = pairwise_position_encoding
        self._use_batch_norm_rel_net = use_batch_norm_rel_net
        self._im_fg = im_fg

        if self._im_fg == True:
            self._image_encoder_stride = (1, 2)
        elif self._im_fg == False:
            self._image_encoder_stride = 2

        if self._pooling not in ["gap", "rel_net", "concat", "trnsf"]:
            raise ValueError(f"{self._pooling} not implemented.")

        if self._pooling != "rel_net" and self._pairwise_position_encoding == True:
            raise ValueError(f"{self._pooling} cannot do pairwise position encoding.")

        if self._pairwise_position_encoding == True and self._modality == "json":
            raise ValueError("Cannot do pairwise position encoding for JSON.")

    def __call__(self, vocabulary):
        if self._modality == "image":
            object_encoder = build_image_object_encoder(
                self._obj_fdim,
                self._pretrained_encoder,
                self._input_dim,
                image_encoder_stride=self._image_encoder_stride)
        elif self._modality == "json":
            object_encoder = build_json_object_encoder(
                self._obj_fdim, self._pretrained_encoder,
                self._input_dim)

        # Get an initial object encoding, and then attach a position encoding
        # based on the flags.
        _, object_dims = infer_output_dim(object_encoder, self._input_dim)
        object_position_encoder = OptionalPositionEncoding(
            object_dims, position_encoding=self._abs_position_encoding)
        object_encoder = nn.Sequential(object_encoder, object_position_encoder)
        obj_fdim, _ = infer_output_dim(object_encoder, self._input_dim)

        if self._pooling == 'gap':
            gap_mlp = nn.Sequential(
                nn.Linear(obj_fdim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 512, bias=True),  # Just replace with 1x1 conv.
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 384, bias=True),  # Just replace with 1x1 conv.
                nn.BatchNorm1d(384),
                nn.ReLU(),
                nn.Linear(384, self._protonet_feature_dim, bias=True),
            )
            if self._modality == "image":
                pool_module = nn.Sequential(
                  nn.AdaptiveAvgPool2d((1, 1)),
                  Squeeze(),
                  gap_mlp
                )
            elif self._modality == "sound" or self._modality == "json":
                pool_module = nn.Sequential(
                  nn.AdaptiveAvgPool1d(1),
                  Squeeze(),
                  gap_mlp
                )
        elif self._pooling == "rel_net":
            pool_module = RelationNetwork(
                obj_fdim=obj_fdim,
                num_objects_list=object_dims,
                hidden_dims_g=[256, 256, 256],
                output_dim_g=256,
                hidden_dims_f=[256, 256],
                output_dim_f=self._protonet_feature_dim,
                tanh=False,
                use_batch_norm_f=self._use_batch_norm_rel_net,
                relative_position_encoding=self._pairwise_position_encoding,
                image_concat_reduce=False)
        elif self._pooling == "concat":
            pool_module = build_concat_pooling(obj_fdim,
                                               object_dims,
                                               self._protonet_feature_dim)
        elif self._pooling == "trnsf":
            pool_module = TransformerPooling(
                obj_fdim=obj_fdim,
                num_objects_list=object_dims,
                output_dim_f=self._protonet_feature_dim)

        encoder = EncoderSequential(object_encoder,
                                    pool_module,
                                    modality=self._modality,
                                    feature_dim=self._protonet_feature_dim)

        if self._init_to_use_pooling == "xavier":
            weights_init_xavier(pool_module)
        elif self._init_to_use_pooling == "l2c":
            weights_init_l2c(pool_module)
        elif self._init_to_use_pooling == "bert":
            init_bert_params(pool_module)

        creator = ProtoNetCreator(encoder)
        applier = ProtoNetApplier(encoder)
        program_decoder = None

        if self._language_alpha > 0:
            program_decoder = get_language_head(self._protonet_feature_dim,
                                                vocabulary, self._num_classes)

        protonet_model = GenericMetaClassifier(creator, applier, program_decoder)

        return protonet_model
