"""Yield-prediction adaptation of the Informer encoder.

The base Informer (Zhou et al. 2021) is encoder-decoder for forecasting.
For Liu et al-style crop-yield regression we only need an encoder + a
sequence-pool + a linear regression head:

  input  : (B, 12, 8)  -- 12 Kharif 16-day windows × 8 environmental features
  marks  : (B, 12, 1)  -- normalized DOY of each window in [0, 1]
  output : (B, 1)      -- predicted yield (in standardized units; caller un-z's)

The encoder uses ProbSparse attention with optional distilling. For our very
short sequence (12 tokens) we DISABLE distillation by default since halving
the sequence is excessive at this scale.
"""

import sys
import os

# Make the cloned Informer2020 importable
INFORMER_ROOT = "/media/sam/writable/Sam Rice Yield Pred/Informer2020"
if INFORMER_ROOT not in sys.path:
    sys.path.insert(0, INFORMER_ROOT)

import torch
import torch.nn as nn

from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.attn import ProbAttention, FullAttention, AttentionLayer
from models.embed import DataEmbedding


class YieldInformer(nn.Module):
    def __init__(
        self,
        n_features=8,
        seq_len=12,
        n_time_marks=1,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        dropout=0.1,
        factor=5,
        attn="prob",          # 'prob' or 'full'
        distil=False,          # short seq -- distil rarely helps
        activation="gelu",
        embed_type="timeF",
        freq="m",              # 1 time mark per step (we use normalized DOY)
        pool="mean",           # 'mean' or 'last'
    ):
        super().__init__()
        # Patch the Informer DataEmbedding so the time-mark Linear matches our n_time_marks
        self.enc_embedding = DataEmbedding(
            c_in=n_features, d_model=d_model, embed_type=embed_type, freq=freq, dropout=dropout
        )
        # The TimeFeatureEmbedding inside DataEmbedding uses freq_map; for freq='m' that's d_inp=1.
        # If you ever want >1 time mark, override the inner Linear here.
        if n_time_marks != 1:
            self.enc_embedding.temporal_embedding.embed = nn.Linear(n_time_marks, d_model)

        AttnCls = ProbAttention if attn == "prob" else FullAttention
        attn_layers = [
            EncoderLayer(
                AttentionLayer(
                    AttnCls(False, factor, attention_dropout=dropout, output_attention=False),
                    d_model, n_heads, mix=False,
                ),
                d_model,
                d_ff,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(e_layers)
        ]
        conv_layers = [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None
        self.encoder = Encoder(
            attn_layers,
            conv_layers,
            norm_layer=nn.LayerNorm(d_model),
        )

        self.pool = pool
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, x_mark):
        # x: (B, T, F), x_mark: (B, T, M)
        z = self.enc_embedding(x, x_mark)         # (B, T, d_model)
        z, _ = self.encoder(z, attn_mask=None)    # (B, T', d_model)
        if self.pool == "mean":
            z = z.mean(dim=1)
        elif self.pool == "last":
            z = z[:, -1, :]
        else:
            raise ValueError("pool must be 'mean' or 'last'")
        return self.head(z).squeeze(-1)           # (B,)
