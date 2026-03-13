"""
model.py
────────
AdaBiD: Adaptive Graph Construction + Bidirectional Diffusion Convolution
for traffic flow forecasting.

Paper sections implemented:
  §4.1  KL-Divergence Adaptive Adjacency Matrix
  §4.2  Adaptive Bidirectional Diffusion Convolution (ABiDC)
  §4.3  MLP Encoder-Decoder

Input  shape: (B, T_in,  N, F)
Output shape: (B, T_out, N, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# 1.  KL-Divergence Adaptive Adjacency Matrix
# ═══════════════════════════════════════════════════════════════════

def build_kl_adjacency(x: torch.Tensor, delta: float = 0.5, eps: float = 1e-10) -> torch.Tensor:
    """
    Construct the adaptive binary adjacency matrix from §4.1.

    Parameters
    ----------
    x     : (B, T, N, F)  – raw (normalised) traffic observations
    delta : sparsity threshold; edge exists when D_KL(i‖j) < delta
    eps   : numerical stability constant

    Returns
    -------
    A_KL  : (B, N, N)  – binary adjacency (1 where similar, 0 otherwise)
    """
    B, T, N, n_feat = x.shape   # avoid shadowing 'F' = torch.nn.functional

    # Use only the first feature (traffic flow) for similarity
    x_flow = x[..., 0]                             # (B, T, N)
    x_flow = x_flow.permute(0, 2, 1)               # (B, N, T)

    # Softmax normalises each node's time-series into a distribution
    p = F.softmax(x_flow, dim=-1)                   # (B, N, T)  Eq.(3)

    # KL divergence  D_KL(p_i ‖ p_j)
    #   = sum_t  p_i_t * (log(p_i_t + eps) - log(p_j_t + eps))
    # Vectorised over all pairs using broadcasting:
    #   p  : (B, N, 1, T)
    #   q  : (B, 1, N, T)
    p_i = p.unsqueeze(2)                            # (B, N, 1, T)
    p_j = p.unsqueeze(1)                            # (B, 1, N, T)

    kl = (p_i * (torch.log(p_i + eps) - torch.log(p_j + eps))).sum(-1)   # (B, N, N)

    # Binary adjacency  Eq.(5)
    A_KL = (kl < delta).float()                     # (B, N, N)
    return A_KL


# ═══════════════════════════════════════════════════════════════════
# 2.  Adaptive Bidirectional Diffusion Convolution (ABiDC)
# ═══════════════════════════════════════════════════════════════════

class AdaptiveBiDiffConv(nn.Module):
    """
    Implements Eq.(6):
        X^p = X W_1  +  α (D^{-1} A_KL) X W_2
                     + (1-α)(D^{-1} A_KL^T) X W_2

    Applied K times (diffusion steps).

    Parameters
    ----------
    in_dim   : input feature dimension
    out_dim  : output feature dimension
    K        : number of diffusion hops
    alpha    : forward / backward mixing coefficient
    """

    def __init__(self, in_dim: int, out_dim: int, K: int = 3, alpha: float = 0.3):
        super().__init__()
        self.K     = K
        self.alpha = alpha

        # Residual / identity projection  (W_1)
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)

        # Diffusion projection  (W_2)  – shared for forward & backward
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _row_norm(A: torch.Tensor) -> torch.Tensor:
        """Row-normalise A → D^{-1} A.  A: (B, N, N)"""
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)   # (B, N, 1)
        return A / deg

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, A_KL: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, T, N, in_dim)
        A_KL : (B, N, N)

        Returns
        -------
        x_p  : (B, T, N, out_dim)
        """
        B, T, N, _ = x.shape

        A_fwd = self._row_norm(A_KL)                # D^{-1} A_KL       (B,N,N)
        A_bwd = self._row_norm(A_KL.transpose(-1,-2))  # D^{-1} A_KL^T  (B,N,N)

        # Residual branch  X W_1  – unchanged across K hops
        x_res = self.W1(x)                           # (B, T, N, out_dim)

        # Diffusion branch  – accumulate over K hops
        x_diff = torch.zeros_like(x_res)
        x_k    = x                                   # running signal

        # Pre-expand adjacency matrices once (avoids repeated alloc inside loop)
        BT = B * T
        A_fwd_e = A_fwd.unsqueeze(1).expand(-1, T, -1, -1).reshape(BT, N, N)
        A_bwd_e = A_bwd.unsqueeze(1).expand(-1, T, -1, -1).reshape(BT, N, N)

        for _ in range(self.K):
            # xw: (B, T, N, out_dim) → (BT, N, out_dim)  for batched matmul
            xw      = self.W2(x_k)                           # (B, T, N, out_dim)
            xw_flat = xw.reshape(BT, N, -1)                  # (BT, N, C)

            fwd_out = torch.bmm(A_fwd_e, xw_flat)            # (BT, N, C)
            bwd_out = torch.bmm(A_bwd_e, xw_flat)            # (BT, N, C)

            x_diff += (  self.alpha       * fwd_out
                       + (1 - self.alpha) * bwd_out
                      ).reshape(B, T, N, -1)

            # Keep x_k stationary (non-recursive 1-hop diffusion per iter)
            x_k = x_k

        x_p = x_res + x_diff                         # residual connection
        return x_p                                   # (B, T, N, out_dim)


# ═══════════════════════════════════════════════════════════════════
# 3.  MLP Block (shared by Encoder & Decoder)
# ═══════════════════════════════════════════════════════════════════

class MLPBlock(nn.Module):
    """
    A 3-layer MLP with BatchNorm, ReLU, Dropout and a residual
    add (projected if dims differ) – matching Fig.1 in the paper.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1, num_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:           # no BN/ReLU on last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.bn  = nn.BatchNorm1d(out_dim)

        # Residual projection when in/out dims differ
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D)  →  (..., out_dim)"""
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])            # (*, D)

        out  = self.net(flat)                           # (*, out_dim)
        res  = self.proj(flat)                          # (*, out_dim)

        out  = self.bn(out + res)                       # BatchNorm over last dim
        return out.reshape(*orig_shape[:-1], -1)


# ═══════════════════════════════════════════════════════════════════
# 4.  AdaBiD  (full model)
# ═══════════════════════════════════════════════════════════════════

class AdaBiD(nn.Module):
    """
    AdaBiD Traffic Forecasting Model.

    Architecture  (§4, Fig.1):
      Input  → KL Adaptive Matrix
             → ABiDC  (spatial)
             → ReLU
             → MLP Encoder  (temporal compression)
             → Dropout + BatchNorm + residual
             → MLP Decoder  (temporal expansion)
             → Dropout + BatchNorm + residual
      Output

    Parameters
    ----------
    num_nodes   : N
    in_steps    : T_in  (historical time steps)
    out_steps   : T_out (prediction horizon)
    in_features : F  (number of input features per node per step)
    hidden_dim  : latent dimension  (paper: 64)
    K           : diffusion hops    (paper: 3)
    alpha       : diffusion mixing  (paper: 0.3)
    delta       : KL sparsity thr.  (paper: 0.5)
    mlp_layers  : depth of MLP blocks (paper: 3)
    dropout     : dropout rate
    """

    def __init__(
        self,
        num_nodes:   int   = 170,
        in_steps:    int   = 12,
        out_steps:   int   = 12,
        in_features: int   = 1,
        hidden_dim:  int   = 64,
        K:           int   = 3,
        alpha:       float = 0.3,
        delta:       float = 0.5,
        mlp_layers:  int   = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.in_steps  = in_steps
        self.out_steps = out_steps
        self.delta     = delta
        self.hidden    = hidden_dim

        # ── Spatial: ABiDC ──────────────────────────────────────────
        self.abidc = AdaptiveBiDiffConv(
            in_dim=in_features,
            out_dim=hidden_dim,
            K=K,
            alpha=alpha,
        )

        # ── Temporal Encoder ────────────────────────────────────────
        # Flattens T × hidden_dim per node, then compresses
        enc_in  = in_steps  * hidden_dim
        enc_out = hidden_dim                    # compact latent per node

        self.encoder = MLPBlock(enc_in, hidden_dim * 2, enc_out,
                                dropout=dropout, num_layers=mlp_layers)

        # ── Temporal Decoder ────────────────────────────────────────
        dec_in  = hidden_dim
        dec_out = out_steps                     # predict T_out steps directly

        self.decoder = MLPBlock(dec_in, hidden_dim * 2, dec_out,
                                dropout=dropout, num_layers=mlp_layers)

        # ── Output projection ────────────────────────────────────────
        # Map to (B, T_out, N, 1)
        self.out_proj = nn.Linear(1, 1)         # trivial; keeps shape consistent

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, T_in, N, F)

        Returns
        -------
        out : (B, T_out, N, 1)
        """
        B, T, N, n_feat = x.shape   # avoid shadowing 'F' = torch.nn.functional

        # Paper §3: only traffic-flow (feature 0) is predicted (F=1).
        # Multi-feature datasets (PEMS08 has speed/flow/occupancy) use only flow.
        x_flow = x[..., :1]                               # (B, T, N, 1)

        # ── 1. Adaptive adjacency (built from flow feature) ──────────
        A_KL = build_kl_adjacency(x_flow, delta=self.delta)  # (B, N, N)

        # ── 2. Bidirectional diffusion convolution ──────────────────
        x_p = self.abidc(x_flow, A_KL)                    # (B, T, N, hidden)
        x_p = F.relu(x_p)

        # ── 3. Encoder (temporal compression) ───────────────────────
        # Reshape: treat each node independently  →  (B, N, T*hidden)
        x_flat = x_p.permute(0, 2, 1, 3).reshape(B, N, T * self.hidden)
        x_e    = self.encoder(x_flat)                    # (B, N, hidden)

        # ── 4. Decoder (temporal expansion) ─────────────────────────
        x_d = self.decoder(x_e)                          # (B, N, T_out)

        # ── 5. Reshape to (B, T_out, N, 1) ──────────────────────────
        out = x_d.permute(0, 2, 1).unsqueeze(-1)         # (B, T_out, N, 1)
        return out


# ─────────────────────────────────────────────
# Quick shape smoke-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    B, T_in, N, F = 2, 12, 170, 1
    model = AdaBiD(num_nodes=N, in_steps=T_in, out_steps=12,
                   in_features=F, hidden_dim=64)
    dummy = torch.randn(B, T_in, N, F)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")   # expect (2, 12, 170, 1)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total:,}")
