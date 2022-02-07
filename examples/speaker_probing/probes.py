# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F


class ConstantBaseline(torch.nn.Module):
    def __init__(self, total_speakers):
        super().__init__()
        self.logits = torch.nn.parameter.Parameter(torch.zeros(total_speakers).float())

    def forward(self, batch):
        bsz = batch["units"].size(0)
        return (
            F.log_softmax(self.logits, dim=-1)
            .unsqueeze(0)
            .expand(bsz, self.logits.size(0))
        )


class DiscreteClassifier(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        n_heads,
        hidden_size,
        n_layers,
        dropout,
        pad_value,
        total_speakers,
    ):
        super().__init__()
        self.pad_value = pad_value.item() if torch.is_tensor(pad_value) else pad_value

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size
        torch.nn.init.normal_(
            self.embedding.weight, mean=0, std=self.embedding_size ** -0.5
        )

        self.encoder_classifier = Classifier(
            embedding_size, n_heads, hidden_size, n_layers, dropout, total_speakers
        )

    def forward(self, batch):
        src = batch["units"]
        padding_mask = src == self.pad_value

        src = src.transpose(1, 0)
        x = self.embedding(src) * math.sqrt(self.embedding_size)
        return self.encoder_classifier(x, padding_mask)


class ContinuousClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size,
        embedding_size,
        n_heads,
        hidden_size,
        n_layers,
        dropout,
        pad_value,
        total_speakers,
    ):
        super().__init__()

        self.pad_value = pad_value.item() if torch.is_tensor(pad_value) else pad_value
        self.embedding = torch.nn.Linear(input_size, embedding_size)

        self.encoder_classifier = Classifier(
            embedding_size, n_heads, hidden_size, n_layers, dropout, total_speakers
        )

    def forward(self, batch):
        src = batch["dense"]
        padding_mask = batch["units"] == self.pad_value

        src = src.transpose(1, 0)
        x = self.embedding(src)  # * math.sqrt(self.embedding_size)
        return self.encoder_classifier(x, padding_mask)


class Classifier(torch.nn.Module):
    def __init__(
        self, embedding_size, n_heads, hidden_size, n_layers, dropout, total_speakers
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_size, dropout=0.0)
        encoder_layers = TransformerEncoderLayer(
            embedding_size, n_heads, hidden_size, dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.embedding_size = embedding_size
        self.classifier = torch.nn.Linear(embedding_size, total_speakers)

    def forward(self, x, padding_mask):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        bos_embedding = x[0, :]
        logits = self.classifier(bos_embedding)
        return F.log_softmax(logits, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        assert x.size(0) < self.pe.size(0), f"{x.size()=} {self.pe.size()=}"

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
