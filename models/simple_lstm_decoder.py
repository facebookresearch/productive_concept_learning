# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""LSTM decoder model."""
import torch
import torch.nn as nn
from fairseq.models import FairseqDecoder
from fairseq.data import Dictionary


class SimpleLSTMDecoder(FairseqDecoder):

  def __init__(
      self,
      dictionary,
      encoder_hidden_dim=128,
      embed_dim=128,
      hidden_dim=128,
      dropout=0.1,
  ):
    super().__init__(dictionary)

    self.encoder_hidden_dim = encoder_hidden_dim
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.dropout = dropout

    # Our decoder will embed the inputs before feeding them to the LSTM.
    self.embed_tokens = nn.Embedding(
        num_embeddings=len(dictionary),
        embedding_dim=embed_dim,
        padding_idx=dictionary.pad(),
    )
    self.dropout = nn.Dropout(p=dropout)

    # We'll use a single-layer, unidirectional LSTM for simplicity.
    self.lstm = nn.LSTM(
        # For the first layer we'll concatenate the Encoder's final hidden
        # state with the embedded target tokens.
        input_size=encoder_hidden_dim + embed_dim,
        hidden_size=hidden_dim,
        num_layers=1,
        bidirectional=False,
    )

    self.encoder_projection = nn.Linear(encoder_hidden_dim, hidden_dim)

    # Define the output projection.
    self.output_projection = nn.Linear(hidden_dim, len(dictionary))

    self.register_buffer(
        'sequence_prepend', (torch.ones(1, 1) * self.dictionary.eos()).long())

  # During training Decoders are expected to take the entire target sequence
  # (shifted right by one position) and produce logits over the vocabulary.
  # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
  # ``dictionary.eos()``, followed by the target sequence.
  def forward(self, output_tokens, encoder_out):
    """
        Args:
            output_tokens (LongTensor): outputs of shape
                `(batch, tgt_len)`, for teacher forcing, without any shifting.
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
    """
    batch_size = output_tokens.size()[0]
    sequence_prepend = self.sequence_prepend.expand(batch_size, 1)
    prev_output_tokens = torch.cat(
        [sequence_prepend, output_tokens],
        dim=-1)

    bsz, tgt_len = prev_output_tokens.size()

    # Extract the final hidden state from the Encoder.
    final_encoder_hidden = encoder_out

    # Embed the target sequence, which has been shifted right by one
    # position and now starts with the end-of-sentence symbol.
    x = self.embed_tokens(prev_output_tokens)

    # Apply dropout.
    x = self.dropout(x)

    # Concatenate the Encoder's final hidden state to *every* embedded
    # target token.
    x = torch.cat(
        [x, final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)],
        dim=2,
    )

    # Using PackedSequence objects in the Decoder is harder than in the
    # Encoder, since the targets are not sorted in descending length order,
    # which is a requirement of ``pack_padded_sequence()``. Instead we'll
    # feed nn.LSTM directly.
    initial_hidden_state = self.encoder_projection(
        final_encoder_hidden).unsqueeze(0)
    initial_state = (
        initial_hidden_state,  # hidden
        torch.zeros_like(initial_hidden_state),  # cell
    )
    output, _ = self.lstm(
        x.transpose(0, 1),  # convert to shape `(tgt_len, bsz, dim)`
        initial_state,
    )
    x = output.transpose(0, 1)  # convert to shape `(bsz, tgt_len, hidden)`

    # Project the outputs to the size of the vocabulary, and retain the first
    # target_inds number of dimensions in the time dimension.
    x = self.output_projection(x)[:, :-1, :]

    # Return the logits and ``None`` for the attention weights
    return x, None