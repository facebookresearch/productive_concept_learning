# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Visualization routine for few-shot meta learning model."""
import torch
import numpy as np

from protonets.third_party.image_utils import plot_images


def save_visuals_for_fewshot_model(samples, outputs, num_samples=4):
  """Save visualizations for a few-shot prototypical network.
  
  Args:
    samples: A dict provided during forward pass to the model.
    outputs: A list of items returned from the model.
      outputs[0]: is negative log-probs.
      outputs[1]: 
  """
  query_images = samples["query_images"].detach()[:num_samples]
  num_images_row = query_images.shape[1] 

  query_images_concat = torch.reshape(
      query_images, (-1, query_images.shape[-3], query_images.shape[-2],
                     query_images.shape[-1])).numpy()
  # Make images N x H x W x C from N x C x H x W
  query_images_concat = np.swapaxes(query_images_concat, 1, 3)
  query_images_concat = np.swapaxes(query_images_concat, 1, 2)
  pred_labels = np.argmin(
      torch.reshape(outputs["neg_log_p_y"][:num_samples].detach(),
                    (-1, outputs["neg_log_p_y"].shape[-1])).numpy(),
      axis=-1)

  targets = torch.reshape(outputs[1][:num_samples].detach(), [-1]).numpy()
 
  annotations = [] 
  for idx in range(num_samples):
    for row_idx in range(num_images_row):
      annotations.append(
        [
          {"label": samples["hypothesis_string"][idx], "color": "#000801"},
        ]
      )

  viz_images = plot_images(
      query_images_concat,
      n=num_images_row,
      gt_labels=targets,
      annotations=annotations,
      orig_labels=pred_labels)
  
  return viz_images.astype(np.uint8)