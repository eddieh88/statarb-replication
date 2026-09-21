"""
Cross-sectional extension of Guijarro-Ordonez / Pelger / Zanotti (2106.04028).

Their CNNTransformer processes each stock's 30-day residual path INDEPENDENTLY.
The only cross-sectional interaction in the whole model is the final L1
normalisation.  A stat-arb book is fundamentally relative, so this adds:

  1. sinusoidal positional encoding    (theirs has none at all -- the temporal
                                        transformer sees an unordered set, with
                                        ordering supplied only by the causal CNN)
  2. an ISAB cross-sectional block     (Set Transformer, Lee et al. 2019):
                                        pool the day's cross-section into M
                                        learned inducing points, then broadcast
                                        back.  O(N*M) not O(N^2), permutation-
                                        equivariant, handles a variable number
                                        of active names per day via masking.

Everything else -- CNN block, temporal encoder, Sharpe objective, rolling
retrain -- is kept identical so the comparison is clean.
"""
import math
import torch
import torch.nn as nn


class CNN_Block(nn.Module):
    """Verbatim from their models/CNNTransformer.py."""
    def __init__(self, in_filters=1, out_filters=8, normalization=True, filter_size=2):
        super().__init__()
        self.in_filters, self.out_filters = in_filters, out_filters
        self.conv1 = nn.Conv1d(in_filters, out_filters, filter_size)
        self.conv2 = nn.Conv1d(out_filters, out_filters, filter_size)
        self.relu = nn.ReLU(inplace=True)
        self.pad = nn.ConstantPad1d((filter_size-1, 0), 0)      # causal
        self.n1, self.n2 = nn.InstanceNorm1d(in_filters), nn.InstanceNorm1d(out_filters)
        self.normalization = normalization

    def forward(self, x):
        if self.normalization: x = self.n1(x)
        out = self.relu(self.conv1(self.pad(x)))
        if self.normalization: out = self.n2(out)
        out = self.relu(self.conv2(self.pad(out)))
        return out + x.repeat(1, self.out_filters // self.in_filters, 1)


class ISAB(nn.Module):
    """Induced Set Attention Block: pool the cross-section through M inducing
    points, then broadcast back.  Cost O(N*M) instead of O(N^2)."""
    def __init__(self, dim, heads=4, m=16, dropout=0.1):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, dim) * dim**-0.5)
        self.pool = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cast = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2*dim), nn.ReLU(), nn.Linear(2*dim, dim))

    def forward(self, E, pad_mask):
        """E (D, N, C) -- D days, N stocks, C features.  pad_mask (D, N) True = inactive."""
        D = E.shape[0]
        H, _ = self.pool(self.I.expand(D, -1, -1), E, E, key_padding_mask=pad_mask)
        A, _ = self.cast(E, H, H)
        E = self.n1(E + A)
        return self.n2(E + self.ff(E))


PE_BASE = 100.0   # NOT the usual 10000: that schedule is designed for sequences
                  # of ~1000+ steps.  Over T=30, base 10000 leaves 3 of 8 dims
                  # with std < 0.05 (frequencies too slow to move).  Measured
                  # min pairwise distance between positions / dead dims:
                  #   base 10000 -> 0.658 / 3 of 8
                  #   base  1000 -> 0.976 / 2 of 8
                  #   base   100 -> 1.015 / 0 of 8   <- best separability, no waste

def positional_encoding(T, C, device, base=PE_BASE):
    pe = torch.zeros(T, C, device=device)
    pos = torch.arange(T, device=device, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, C, 2, device=device).float() * (-math.log(base)/C))
    pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div[:pe[:, 1::2].shape[1]])
    return pe


class XSectionNet(nn.Module):
    def __init__(self, lookback=30, filters=8, heads=4, hidden_factor=2,
                 dropout=0.25, filter_size=2, inducing=16, use_pos=True,
                 use_xsection=True, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lookback, self.C = lookback, filters
        self.use_pos, self.use_xsection = use_pos, use_xsection
        self.cnn = CNN_Block(1, filters, True, filter_size)
        self.temporal = nn.TransformerEncoderLayer(
            d_model=filters, nhead=heads, dim_feedforward=hidden_factor*filters,
            dropout=dropout, batch_first=False)
        self.pe_scale = nn.Parameter(torch.ones(1)) if use_pos else None
        # CNN features measure |h| ~ 0.80 while the encoding has amplitude 1.0,
        # so a plain add is an arbitrary 1.3x weighting.  Let the model choose.
        # Init 1.0 => identical to standard PE at initialisation.
        self.pe_scale = nn.Parameter(torch.ones(1)) if use_pos else None
        self.xsec = ISAB(filters, heads, inducing, dropout) if use_xsection else None
        self.out = nn.Linear(filters, 1)

    MAX_ROWS = 32768   # their attention kernel caps batch at 65535 rows

    def _embed_chunk(self, x):
        n, T = x.shape
        h = self.cnn(x.reshape(n, 1, T))            # (n, C, T)
        h = h.permute(2, 0, 1)                      # (T, n, C)
        if self.use_pos:
            h = h + self.pe_scale * positional_encoding(T, self.C, x.device).unsqueeze(1)
        h = self.temporal(h)
        return h[-1]                                # (n, C)

    def embed(self, x):
        """x (n, T) flat (day,stock) pairs -> (n, C).  Chunked; gradients flow
        through the cat, so this is identical to one pass."""
        if x.shape[0] <= self.MAX_ROWS:
            return self._embed_chunk(x)
        return torch.cat([self._embed_chunk(x[i:i+self.MAX_ROWS])
                          for i in range(0, x.shape[0], self.MAX_ROWS)])

    def forward(self, x, sub=None):
        """x (n, T) flat pairs.  sub (D, N) bool placing each pair in its day.
        Without sub, behaves like their per-stock model."""
        e = self.embed(x)
        if self.xsec is None or sub is None:
            return self.out(e).squeeze(-1)
        D, N = sub.shape
        E = torch.zeros(D, N, self.C, device=x.device, dtype=e.dtype)
        E[sub] = e
        E = self.xsec(E, pad_mask=~sub)             # stocks see each other
        return self.out(E[sub]).squeeze(-1)
