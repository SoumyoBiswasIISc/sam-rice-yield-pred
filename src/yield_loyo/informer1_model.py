"""Informer1: encoder-decoder forecasting model that maps 9 pre-Kharif + k
observed Kharif feature windows -> the remaining 12-k Kharif feature windows.

We use the full cloned Informer (encoder + decoder) since this IS a
forecasting problem (sequence-to-sequence), unlike Informer2 (yield regression
= sequence-to-scalar).

Input shapes (per batch):
  x_enc      : (B, 9+k, n_features)   -- encoder values
  x_mark_enc : (B, 9+k, n_time_marks)  -- encoder time marks (normalized DOY)
  x_dec      : (B, label_len + pred_len, n_features)
                  -- decoder values: warm-up tokens (copy of last `label_len`
                     of encoder input) followed by `pred_len` zero placeholders
  x_mark_dec : (B, label_len + pred_len, n_time_marks)  -- decoder time marks
Output:
  forecast   : (B, pred_len, n_features)   -- predicted feature windows
"""

import sys
import os
INFORMER_ROOT = "/media/sam/writable/Sam Rice Yield Pred/Informer2020"
if INFORMER_ROOT not in sys.path:
    sys.path.insert(0, INFORMER_ROOT)

import torch
import torch.nn as nn
from models.model import Informer as _BaseInformer


class Informer1(nn.Module):
    """Wraps the cloned Informer with our defaults & a simple forward signature.

    We forecast all n_features channels (so the decoder can be a drop-in
    replacement of the remaining Kharif windows for Informer2).
    """

    def __init__(
        self,
        n_features=8,
        seq_len=15,       # 9 pre-Kharif + k observed Kharif (k=6 default)
        label_len=6,      # k
        pred_len=6,       # 12 - k
        n_time_marks=1,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        factor=5,
        dropout=0.05,
        attn="prob",
        activation="gelu",
        distil=False,     # disable distillation for our short sequences
        mix=True,
        device=None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_len = label_len
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.model = _BaseInformer(
            enc_in=n_features, dec_in=n_features, c_out=n_features,
            seq_len=seq_len, label_len=label_len, out_len=pred_len,
            factor=factor, d_model=d_model, n_heads=n_heads,
            e_layers=e_layers, d_layers=d_layers, d_ff=d_ff,
            dropout=dropout, attn=attn, embed="timeF", freq="m",
            activation=activation, output_attention=False,
            distil=distil, mix=mix, device=device,
        )
        # If we ever want >1 time mark per step the inner Linear must be resized
        if n_time_marks != 1:
            self.model.enc_embedding.temporal_embedding.embed = nn.Linear(n_time_marks, d_model)
            self.model.dec_embedding.temporal_embedding.embed = nn.Linear(n_time_marks, d_model)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # The base Informer returns (B, pred_len, c_out) by indexing in its own forward
        return out


def build_dec_input(x_enc, label_len, pred_len):
    """Construct the decoder input tensor from x_enc.

    Decoder input = [x_enc[:, -label_len:, :], zeros(B, pred_len, F)]
    """
    B, T, F = x_enc.shape
    label_tokens = x_enc[:, -label_len:, :]
    zero_tokens = torch.zeros(B, pred_len, F, device=x_enc.device, dtype=x_enc.dtype)
    return torch.cat([label_tokens, zero_tokens], dim=1)


def build_dec_time_marks(x_mark_enc, future_marks, label_len):
    """Construct decoder time marks: last `label_len` of enc marks + future marks."""
    label_marks = x_mark_enc[:, -label_len:, :]
    return torch.cat([label_marks, future_marks], dim=1)
