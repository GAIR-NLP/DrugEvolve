import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


# ======== Utility sparse activations ========
def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation (Martins & Astudillo, 2016).
    Returns a sparse probability distribution with support determined by logits.
    """
    # Shift by max for numerical stability
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    # Sort in descending order
    zs = torch.sort(shifted, dim=dim, descending=True).values
    # Compute k(z)
    dims = shifted.size(dim)
    range_vals = torch.arange(1, dims + 1, device=logits.device, dtype=logits.dtype).view(
        *([1] * (shifted.dim() - 1)), dims
    )
    # cumsum of sorted scores
    zs_cum = torch.cumsum(zs, dim=dim)
    # Determine support
    cond = 1 + range_vals * zs > zs_cum
    k = cond.sum(dim=dim, keepdim=True)
    # Compute tau
    tau = (zs_cum.gather(dim=dim, index=(k - 1).clamp(min=0)) - 1) / k.clamp(min=1)
    # Compute output
    output = torch.clamp(shifted - tau, min=0)
    # Normalize (sum=1) to be safe
    zsum = output.sum(dim=dim, keepdim=True).clamp(min=1e-12)
    return output / zsum


def _entmax15_bisect(logits: torch.Tensor, dim: int = -1, n_iter: int = 25) -> torch.Tensor:
    """Entmax with alpha=1.5 via bisection (Peters et al., 2019).
    Produces sparse probability distributions with differentiability almost everywhere.
    """
    # Shift for numerical stability
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    alpha = 1.5
    # Lower and upper bounds for tau
    tau_lo = shifted.min(dim=dim, keepdim=True).values - 1.0
    tau_hi = shifted.max(dim=dim, keepdim=True).values

    def _proj(tau):
        # p_i = relu(((alpha-1)*z_i - tau))^{1/(alpha-1)}; for alpha=1.5 -> power=2
        p = torch.clamp((alpha - 1) * shifted - tau, min=0) ** (1 / (alpha - 1))
        return p

    for _ in range(n_iter):
        tau_m = (tau_lo + tau_hi) / 2.0
        p_m = _proj(tau_m)
        s = p_m.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(s > 1, tau_m, tau_lo)
        tau_hi = torch.where(s <= 1, tau_m, tau_hi)
    p = _proj(tau_hi)
    zsum = p.sum(dim=dim, keepdim=True).clamp(min=1e-12)
    return p / zsum


class FCNet(nn.Module):
    """Simple class for non-linear fully connected network with weight normalization.
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act: str = 'ReLU', dropout: float = 0.0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if act and act != '':
                layers.append(getattr(nn, act)())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if act and act != '':
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class BANLayer(nn.Module):
    """
    Bilinear Attention Network (BAN) layer with robust sparsity, calibrated temperatures,
    separable normalization axes, competitive head gating, masked Sinkhorn with true marginal
    constraints, and gradient-correct regularization.

    Major upgrades vs prior version:
    - Exact marginal targets for 'both' axis via masked Sinkhorn-IPFP: row sums -> 1/n_rows, col sums -> 1/n_cols
      on valid entries (rectangular-consistent DS); removes the inconsistent final global renorm.
    - Axis-aware pre-Sinkhorn pruning in 'both' path: optional per-row and per-col Top-K keep-masks applied to logits
      before Sinkhorn; configurable alternations to combine row/col sparsification.
    - Post-dropout DS repair: after stochastic dropout, run masked Sinkhorn again (on log-probs) to restore marginals.
    - Diagnostics for DS residuals and mask coverage, plus retained live regularizers (attention entropy & gate L1).

    Compatibility guarantees:
    - Class name remains BANLayer
    - __init__(**config) supported
    - forward(v, q, softmax=False) preserved and returns (logits, att_maps)
    """

    def __init__(
        self,
        v_dim: int,
        q_dim: int,
        h_dim: int,
        h_out: int,
        act: str = 'ReLU',
        dropout: float = 0.2,
        k: int = 3,
        # attention options
        use_softmax: bool = True,
        attn_temperature: float = 1.0,
        attn_activation: str = 'softmax',  # 'softmax' | 'sparsemax' | 'entmax15'
        attn_dropout: float = 0.1,
        attn_topk_ratio: float = 0.0,      # fraction of kept entries per head (flat or axis-wise)
        attn_axis: str = 'vq',             # {'vq','row','col','both'}
        # temperature and normalization
        per_head_temperature: bool = True,
        t_min: float = 0.05,
        t_max: float = 5.0,
        # normalization on fused features
        norm: str = 'layernorm',           # {'layernorm','batchnorm'}
        # gating options
        gate_heads: bool = True,
        gate_activation: str = 'softmax',  # {'softmax','sigmoid'}
        gate_bias_init: float = -2.0,      # negative bias init -> sparse gates
        gate_hidden: int = 128,
        gate_dropout: float = 0.1,
        gate_temperature: float = 1.0,     # softmax gating temperature
        gate_l1: float = 0.0,              # L1 regularization weight on gate activations
        # low-rank attention rank
        rank: int = 4,
        eps: float = 1e-6,
        # regularization weights (reported via compute_regularization)
        attn_entropy_weight: float = 0.0,
        temperature_l2_weight: float = 0.0,
        alpha_l2_weight: float = 0.0,
        # Sinkhorn options for 'both'
        sinkhorn_iters: int = 2,
        post_sinkhorn_iters: int = 2,
        ds_use_marginals: bool = True,     # enforce row/col targets rather than global renorm
        # Pre-Sinkhorn axis-aware pruning for 'both'
        both_pre_topk: bool = True,
        both_pre_topk_ratio: Optional[float] = None,  # if None -> use attn_topk_ratio
        both_topk_alternations: int = 1,   # number of row/col alternations before Sinkhorn
        **kwargs,
    ):
        super(BANLayer, self).__init__()

        # Allow kwargs override (pipeline compatibility)
        v_dim = kwargs.get('v_dim', v_dim)
        q_dim = kwargs.get('q_dim', q_dim)
        h_dim = kwargs.get('h_dim', h_dim)
        h_out = kwargs.get('h_out', h_out)
        act = kwargs.get('act', act)
        dropout = kwargs.get('dropout', dropout)
        k = kwargs.get('k', k)
        use_softmax = kwargs.get('use_softmax', use_softmax)
        attn_temperature = kwargs.get('attn_temperature', attn_temperature)
        attn_activation = kwargs.get('attn_activation', attn_activation)
        attn_dropout = kwargs.get('attn_dropout', attn_dropout)
        attn_topk_ratio = kwargs.get('attn_topk_ratio', attn_topk_ratio)
        attn_axis = kwargs.get('attn_axis', attn_axis)
        per_head_temperature = kwargs.get('per_head_temperature', per_head_temperature)
        t_min = kwargs.get('t_min', t_min)
        t_max = kwargs.get('t_max', t_max)
        norm = kwargs.get('norm', norm)
        gate_heads = kwargs.get('gate_heads', gate_heads)
        gate_activation = kwargs.get('gate_activation', gate_activation)
        gate_bias_init = kwargs.get('gate_bias_init', gate_bias_init)
        gate_hidden = kwargs.get('gate_hidden', gate_hidden)
        gate_dropout = kwargs.get('gate_dropout', gate_dropout)
        gate_temperature = kwargs.get('gate_temperature', gate_temperature)
        gate_l1 = kwargs.get('gate_l1', gate_l1)
        rank = int(kwargs.get('rank', rank))
        eps = kwargs.get('eps', eps)
        attn_entropy_weight = kwargs.get('attn_entropy_weight', attn_entropy_weight)
        temperature_l2_weight = kwargs.get('temperature_l2_weight', temperature_l2_weight)
        alpha_l2_weight = kwargs.get('alpha_l2_weight', alpha_l2_weight)
        sinkhorn_iters = int(kwargs.get('sinkhorn_iters', sinkhorn_iters))
        post_sinkhorn_iters = int(kwargs.get('post_sinkhorn_iters', post_sinkhorn_iters))
        ds_use_marginals = bool(kwargs.get('ds_use_marginals', ds_use_marginals))
        both_pre_topk = bool(kwargs.get('both_pre_topk', both_pre_topk))
        both_pre_topk_ratio = kwargs.get('both_pre_topk_ratio', both_pre_topk_ratio)
        both_topk_alternations = int(kwargs.get('both_topk_alternations', both_topk_alternations))

        # Constants and shapes
        self.c = 32
        self.k = int(k)
        self.v_dim = int(v_dim)
        self.q_dim = int(q_dim)
        self.h_dim = int(h_dim)
        self.h_out = int(h_out)
        self.use_softmax = bool(use_softmax)
        self.attn_activation = str(attn_activation).lower()
        self.attn_dropout_p = float(attn_dropout) if attn_dropout is not None else 0.0
        self.attn_topk_ratio = max(0.0, min(float(attn_topk_ratio), 1.0))
        self.attn_axis = str(attn_axis).lower()
        self.per_head_temperature = bool(per_head_temperature)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.eps = float(eps)
        self.rank = max(1, int(rank))
        self.attn_entropy_weight = float(attn_entropy_weight)
        self.temperature_l2_weight = float(temperature_l2_weight)
        self.alpha_l2_weight = float(alpha_l2_weight)
        self.sinkhorn_iters = max(0, int(sinkhorn_iters))
        self.post_sinkhorn_iters = max(0, int(post_sinkhorn_iters))
        self.ds_use_marginals = ds_use_marginals
        self.both_pre_topk = both_pre_topk
        # None -> fallback to attn_topk_ratio, else clamp to [0,1]
        self.both_pre_topk_ratio = None if (both_pre_topk_ratio is None) else max(0.0, min(float(both_pre_topk_ratio), 1.0))
        self.both_topk_alternations = max(0, int(both_topk_alternations))

        # Project inputs to bilinear space (MFB with k factors)
        D = self.h_dim * self.k
        self.v_net = FCNet([self.v_dim, D], act=act, dropout=dropout)
        self.q_net = FCNet([self.q_dim, D], act=act, dropout=dropout)
        self.p_net = nn.AvgPool1d(self.k, stride=self.k) if self.k > 1 else None

        # Bilinear attention parameterization (low-rank per head)
        init_scale = 1.0 / math.sqrt(D)
        # Per-head rank factor projections: W_v[h, r, D], W_q[h, r, D]
        self.W_v = nn.Parameter(torch.randn(self.h_out, self.rank, D) * init_scale)
        self.W_q = nn.Parameter(torch.randn(self.h_out, self.rank, D) * init_scale)
        # Optional diagonal term parameter (alias maintained for compatibility)
        self.h_mat = nn.Parameter(torch.randn(1, self.h_out, 1, D) * 0.02)
        self.h_bias = nn.Parameter(torch.zeros(1, self.h_out, 1, 1))
        # Mixture coefficient between diagonal and low-rank terms per head
        self.alpha_logit = nn.Parameter(torch.full((self.h_out,), -2.0))  # start close to diagonal (alpha~0.12)

        # Per-head temperature(s)
        if self.per_head_temperature:
            # softplus parameterization ensures positivity: T = softplus(x)
            inv_softplus = math.log(math.exp(attn_temperature) - 1.0 + 1e-8)
            self.temp_param = nn.Parameter(torch.full((self.h_out,), inv_softplus))
        else:
            self.register_parameter('temp_param', None)
        self.global_temperature = float(attn_temperature)
        # Temperature clamp range
        self.register_buffer('_t_min_tensor', torch.tensor(self.t_min), persistent=False)
        self.register_buffer('_t_max_tensor', torch.tensor(self.t_max), persistent=False)

        # Attention dropout
        self.attn_drop = nn.Dropout(self.attn_dropout_p) if self.attn_dropout_p > 0 else nn.Identity()

        # Learnable, input-conditioned head gating
        self.gate_heads = bool(gate_heads)
        self.gate_activation = str(gate_activation).lower()
        self.gate_temperature = float(gate_temperature)
        self.gate_l1 = float(gate_l1)
        if self.gate_heads:
            # Conditional MLP from pooled v/q contexts
            self.gate_mlp = nn.Sequential(
                weight_norm(nn.Linear(2 * D, gate_hidden), dim=None),
                nn.GELU(),
                weight_norm(nn.Linear(gate_hidden, self.h_out), dim=None),
            )
            self.gate_bias = nn.Parameter(torch.full((self.h_out,), float(gate_bias_init)))
            self.gate_drop = nn.Dropout(gate_dropout) if gate_dropout and gate_dropout > 0 else nn.Identity()
        else:
            self.gate_mlp = None
            self.gate_bias = None
            self.gate_drop = nn.Identity()

        # Fused feature normalization
        if str(norm).lower() == 'layernorm':
            self.norm = nn.LayerNorm(self.h_dim)
        elif str(norm).lower() == 'batchnorm':
            self.norm = nn.BatchNorm1d(self.h_dim)
        else:
            raise ValueError("norm must be one of {'layernorm','batchnorm'}")

        # Diagnostics buffers
        self.register_buffer('last_head_gates', torch.zeros(1, self.h_out), persistent=False)
        self.register_buffer('last_att_entropy', torch.zeros(1), persistent=False)
        self.register_buffer('last_topk_k', torch.zeros(1), persistent=False)
        self.register_buffer('last_gate_sum', torch.zeros(1), persistent=False)
        self.register_buffer('last_sink_row_resid', torch.zeros(1), persistent=False)
        self.register_buffer('last_sink_col_resid', torch.zeros(1), persistent=False)
        self._last_att_maps = None
        self._last_logits = None

        # Optional external attention mask (B,V,Q) set via setter
        self._attn_mask: Optional[torch.Tensor] = None

        # Live tensors for regularization (kept with gradients)
        self._att_maps_for_reg: Optional[torch.Tensor] = None
        self._gates_for_reg: Optional[torch.Tensor] = None

    # ======== Utility hooks ========
    def set_topk_ratio(self, ratio: float):
        self.attn_topk_ratio = max(0.0, min(float(ratio), 1.0))

    def anneal_topk_ratio(self, step: int, total_steps: int, start: float = 0.0, end: float = 0.1):
        step = max(0, min(step, total_steps))
        frac = 0.0 if total_steps <= 0 else float(step) / float(total_steps)
        target = (1.0 - frac) * float(start) + frac * float(end)
        self.set_topk_ratio(target)

    def set_attention_mask(self, mask: Optional[torch.Tensor]):
        """Set external attention mask with shape (B,V,Q), values in {0,1}.
        This mask will be broadcast across heads internally.
        """
        self._attn_mask = mask

    def _get_temperature(self) -> torch.Tensor:
        if self.per_head_temperature and self.temp_param is not None:
            temp = F.softplus(self.temp_param)  # (H,)
        else:
            temp = torch.full((self.h_out,), float(max(self.global_temperature, 1e-6)), device=self._t_min_tensor.device)
        # Clamp for stability
        tmin = self._t_min_tensor.to(temp.device)
        tmax = self._t_max_tensor.to(temp.device)
        temp = temp.clamp(min=float(tmin.item()), max=float(tmax.item()))
        return temp

    # ======== Attention computation ========
    def _compute_attention_logits(self, v_feat: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        """Compute raw attention logits per head via low-rank factorization plus diagonal term.
        v_feat: (B, V, D), q_feat: (B, Q, D)
        returns att_logits: (B, H, V, Q)
        """
        B, V, D = v_feat.shape
        Q = q_feat.size(1)
        # Low-rank projections per head and rank
        # s_v: (B, H, R, V); s_q: (B, H, R, Q)
        s_v = torch.einsum('bvd,hrd->bhrv', v_feat, self.W_v)
        s_q = torch.einsum('bqd,hrd->bhrq', q_feat, self.W_q)
        lowrank = (s_v.unsqueeze(-1) * s_q.unsqueeze(-2)).sum(dim=2)  # (B,H,V,Q)
        # Diagonal term from h_mat
        diag = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_feat, q_feat)) + self.h_bias
        # Mixture per head
        alpha = torch.sigmoid(self.alpha_logit).view(1, self.h_out, 1, 1)
        att_logits = (1.0 - alpha) * diag + alpha * lowrank
        return att_logits

    def _expand_mask_bool(self, B: int, H: int, V: int, Q: int) -> Optional[torch.Tensor]:
        if self._attn_mask is None:
            return None
        m = self._attn_mask
        if m.dim() == 3:
            m = m.unsqueeze(1).expand(B, H, V, Q)
        elif m.dim() == 4:
            if m.size(1) == 1:
                m = m.expand(B, H, V, Q)
        return (m > 0) if m.dtype != torch.bool else m

    def _apply_mask(self, logits: torch.Tensor) -> torch.Tensor:
        mask_bool = self._expand_mask_bool(*logits.shape[:4])
        if mask_bool is None:
            return logits
        very_neg = torch.finfo(logits.dtype).min / 4
        logits = logits.masked_fill(~mask_bool, very_neg)
        return logits

    def _apply_topk_flat(self, flat_logits: torch.Tensor, ratio: Optional[float] = None) -> torch.Tensor:
        """Apply top-k sparsification per sample/head across flattened V*Q logits.
        Positions not in top-k get -inf so that downstream softmax assigns ~0.
        flat_logits: (B, H, N)
        """
        r = self.attn_topk_ratio if ratio is None else float(ratio)
        if r <= 0.0:
            return flat_logits
        B, H, N = flat_logits.shape
        k = max(1, int(r * N))
        topk_vals, topk_idx = torch.topk(flat_logits, k=k, dim=2)
        mask = torch.full_like(flat_logits, float('-inf'))
        mask.scatter_(2, topk_idx, topk_vals)
        # diagnostics
        with torch.no_grad():
            self.last_topk_k.fill_(float(k))
        return mask

    def _build_axis_keepmask(self, logits: torch.Tensor, axis: str, ratio: float, base_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Build boolean keep-mask along a specific axis based on logits Top-K.
        axis 'row': top-k along Q per (B,H,V,:); axis 'col': top-k along V per (B,H,:,Q).
        Returns boolean mask of shape (B,H,V,Q) with True at kept positions.
        base_mask: optional boolean mask of valid entries to respect.
        """
        B, H, V, Q = logits.shape
        keep = torch.zeros_like(logits, dtype=torch.bool)
        if ratio <= 0.0:
            if base_mask is None:
                return torch.ones_like(keep)
            return base_mask.clone()
        if axis == 'row':
            k = max(1, int(ratio * Q))
            x = logits.view(B * H * V, Q)
            if base_mask is not None:
                mb = base_mask.view(B * H * V, Q)
                x = torch.where(mb, x, x.new_full(x.shape, float('-inf')))
            vals, idx = torch.topk(x, k=k, dim=1)
            out = torch.zeros_like(x, dtype=torch.bool)
            out.scatter_(1, idx, True)
            keep = out.view(B, H, V, Q)
        elif axis == 'col':
            k = max(1, int(ratio * V))
            x = logits.permute(0, 1, 3, 2).contiguous().view(B * H * Q, V)
            if base_mask is not None:
                mb = base_mask.permute(0, 1, 3, 2).contiguous().view(B * H * Q, V)
                x = torch.where(mb, x, x.new_full(x.shape, float('-inf')))
            vals, idx = torch.topk(x, k=k, dim=1)
            out = torch.zeros_like(x, dtype=torch.bool)
            out.scatter_(1, idx, True)
            keep = out.view(B, H, Q, V).permute(0, 1, 3, 2).contiguous()
        else:
            keep = torch.ones_like(keep, dtype=torch.bool)
        return keep

    def _compute_targets(self, mask_bool: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Compute row and column target marginals r (B,H,V,1) and c (B,H,1,Q) so that sums are 1.
        For rows/cols with no valid entries, targets are zero.
        """
        B, H, V, Q = mask_bool.shape
        rows_valid = mask_bool.any(dim=3, keepdim=True)  # (B,H,V,1)
        cols_valid = mask_bool.any(dim=2, keepdim=True)  # (B,H,1,Q)
        n_rows = rows_valid.float().sum(dim=2, keepdim=True).clamp(min=1.0)  # (B,H,1,1)
        n_cols = cols_valid.float().sum(dim=3, keepdim=True).clamp(min=1.0)  # (B,H,1,1)
        r = rows_valid.float() / n_rows  # (B,H,V,1), sums to 1 across rows
        c = cols_valid.float() / n_cols  # (B,H,1,Q), sums to 1 across cols
        return r, c

    def _sinkhorn_marginals(self, logits_or_logp: torch.Tensor, mask_bool: Optional[torch.Tensor], n_iters: int) -> torch.Tensor:
        """Masked Sinkhorn-IPFP to match row/col marginals r,c (sum to 1 each) on valid entries.
        Works in exp-domain with stabilization. Returns probs with global sum ~1 and respecting mask.
        """
        B, H, V, Q = logits_or_logp.shape
        if mask_bool is None:
            mask_bool = torch.ones(B, H, V, Q, dtype=torch.bool, device=logits_or_logp.device)
        # valid entries set
        m = mask_bool
        # stabilize
        L = logits_or_logp.clone()
        L = torch.where(m, L, torch.full_like(L, float('-inf')))
        L = L - torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0).amax(dim=(2, 3), keepdim=True)
        S = torch.exp(L)
        S = torch.where(m, S, torch.zeros_like(S))
        # targets
        r, c = self._compute_targets(m)
        eps = self.eps
        iters = max(1, int(n_iters))
        for _ in range(iters):
            # Normalize rows to r (over Q)
            row_sum = (S).sum(dim=3, keepdim=True).clamp(min=eps)
            S = S * (r / row_sum)
            S = torch.where(m, S, torch.zeros_like(S))
            # Normalize cols to c (over V)
            col_sum = (S).sum(dim=2, keepdim=True).clamp(min=eps)
            S = S * (c / col_sum)
            S = torch.where(m, S, torch.zeros_like(S))
        # Guard: if any head has zero mass (fully masked), skip renorm
        total = S.sum(dim=(2, 3), keepdim=True)
        bad = (total <= eps)
        if bad.any():
            # Assign uniform over valid entries for those heads
            uniform = torch.where(m, torch.ones_like(S), torch.zeros_like(S))
            z = uniform.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
            uniform = uniform / z
            S = torch.where(bad, uniform, S)
        # Final ensure total mass 1
        denom = S.sum(dim=(2, 3), keepdim=True).clamp(min=eps)
        S = S / denom
        return S

    def _normalize_attention(self, att_logits: torch.Tensor, apply_softmax: bool) -> torch.Tensor:
        """Normalize attention per head using chosen activation and axis.
        att_logits: (B, H, V, Q)
        returns probs: (B, H, V, Q)
        """
        if not apply_softmax and self.attn_activation == 'none':
            return att_logits
        B, H, V, Q = att_logits.shape
        temps = self._get_temperature().to(att_logits.device).view(1, H, 1, 1)
        scaled = att_logits / temps
        # apply external mask as additive -inf before normalization
        scaled = self._apply_mask(scaled)
        mask_bool = self._expand_mask_bool(B, H, V, Q)
        # Choose axis-wise normalization
        if self.attn_axis == 'vq':
            flat = scaled.view(B, H, V * Q)
            # numerical stability
            flat = flat - flat.max(dim=2, keepdim=True).values
            # sparsify via top-k before activation
            flat = self._apply_topk_flat(flat)
            if self.attn_activation == 'sparsemax':
                probs = _sparsemax(flat, dim=2)
            elif self.attn_activation == 'entmax15':
                probs = _entmax15_bisect(flat, dim=2)
            else:
                probs = torch.softmax(flat, dim=2)
            probs = probs.view(B, H, V, Q)
            probs = self.attn_drop(probs)
            zsum = probs.view(B, H, -1).sum(dim=2, keepdim=True).clamp(min=1e-12)
            probs = (probs.view(B, H, -1) / zsum).view(B, H, V, Q)
        elif self.attn_axis == 'row':
            # optional per-row top-k on logits prior to normalization
            if self.attn_topk_ratio > 0.0:
                scaled = self._apply_topk_flat(scaled.view(B, H, V * Q), ratio=None).view(B, H, V, Q)
            if self.attn_activation == 'sparsemax':
                probs = _sparsemax(scaled, dim=3)
            elif self.attn_activation == 'entmax15':
                probs = _entmax15_bisect(scaled, dim=3)
            else:
                scaled = scaled - scaled.max(dim=3, keepdim=True).values
                probs = torch.softmax(scaled, dim=3)
            probs = self.attn_drop(probs)
            denom = probs.sum(dim=3, keepdim=True).clamp(min=1e-12)
            probs = probs / denom
        elif self.attn_axis == 'col':
            if self.attn_topk_ratio > 0.0:
                # top-k over V for each (B,H,Q)
                B_, H_, V_, Q_ = scaled.shape
                x = scaled.permute(0, 1, 3, 2).contiguous().view(B_ * H_ * Q_, V_)
                k = max(1, int(self.attn_topk_ratio * V_))
                vals, idx = torch.topk(x, k=k, dim=1)
                out = x.new_full(x.shape, float('-inf'))
                out.scatter_(1, idx, x.gather(1, idx))
                scaled = out.view(B_, H_, Q_, V_).permute(0, 1, 3, 2).contiguous()
                with torch.no_grad():
                    self.last_topk_k.fill_(float(k))
            if self.attn_activation == 'sparsemax':
                probs = _sparsemax(scaled, dim=2)
            elif self.attn_activation == 'entmax15':
                probs = _entmax15_bisect(scaled, dim=2)
            else:
                scaled = scaled - scaled.max(dim=2, keepdim=True).values
                probs = torch.softmax(scaled, dim=2)
            probs = self.attn_drop(probs)
            denom = probs.sum(dim=2, keepdim=True).clamp(min=1e-12)
            probs = probs / denom
        elif self.attn_axis == 'both':
            # Optional pre-Sinkhorn axis-aware pruning
            pre_ratio = self.both_pre_topk_ratio if (self.both_pre_topk_ratio is not None) else self.attn_topk_ratio
            combined_mask = mask_bool.clone() if mask_bool is not None else torch.ones_like(scaled, dtype=torch.bool)
            if self.both_pre_topk and pre_ratio > 0.0:
                cur_mask = combined_mask
                # alternate row/col keepmask
                for i in range(max(1, self.both_topk_alternations)):
                    keep_row = self._build_axis_keepmask(scaled, axis='row', ratio=pre_ratio, base_mask=cur_mask)
                    keep_col = self._build_axis_keepmask(scaled, axis='col', ratio=pre_ratio, base_mask=cur_mask)
                    cur_mask = keep_row & keep_col
                combined_mask = cur_mask & combined_mask
                # diagnostics: estimate k used (flat)
                with torch.no_grad():
                    kept = combined_mask.view(B, H, -1).float().sum(dim=2)
                    avg_k = kept.mean().clamp(min=1.0)
                    self.last_topk_k.fill_(float(avg_k.item()))
            # Apply mask to logits
            very_neg = torch.finfo(scaled.dtype).min / 4
            scaled_masked = torch.where(combined_mask, scaled, torch.full_like(scaled, very_neg))
            # First Sinkhorn projection
            if self.sinkhorn_iters > 0 and self.ds_use_marginals:
                probs = self._sinkhorn_marginals(scaled_masked, combined_mask, self.sinkhorn_iters)
            else:
                # fallback: sequential row then col softmax with mask
                tmp = torch.where(combined_mask, scaled_masked, torch.full_like(scaled_masked, very_neg))
                tmp = torch.softmax(tmp, dim=2)
                tmp = torch.softmax(tmp, dim=3)
                probs = torch.where(combined_mask, tmp, torch.zeros_like(tmp))
                zsum = probs.view(B, H, -1).sum(dim=2, keepdim=True).clamp(min=1e-12)
                probs = (probs.view(B, H, -1) / zsum).view(B, H, V, Q)
            # Dropout and DS repair
            probs = self.attn_drop(probs)
            if self.post_sinkhorn_iters > 0 and self.ds_use_marginals:
                probs = self._sinkhorn_marginals(torch.log(probs.clamp_min(self.eps)), combined_mask, self.post_sinkhorn_iters)
            # Diagnostics for DS residuals (mean absolute deviation from targets)
            with torch.no_grad():
                r, c = self._compute_targets(combined_mask)
                row_sum = probs.sum(dim=3, keepdim=True)
                col_sum = probs.sum(dim=2, keepdim=True)
                row_resid = (row_sum - r).abs().sum() / r.numel()
                col_resid = (col_sum - c).abs().sum() / c.numel()
                self.last_sink_row_resid = row_resid.detach().unsqueeze(0)
                self.last_sink_col_resid = col_resid.detach().unsqueeze(0)
        else:
            # default to flat
            flat = scaled.view(B, H, V * Q)
            flat = flat - flat.max(dim=2, keepdim=True).values
            if self.attn_activation == 'sparsemax':
                probs = _sparsemax(flat, dim=2).view(B, H, V, Q)
            elif self.attn_activation == 'entmax15':
                probs = _entmax15_bisect(flat, dim=2).view(B, H, V, Q)
            else:
                probs = torch.softmax(flat, dim=2).view(B, H, V, Q)
            probs = self.attn_drop(probs)
            zsum = probs.view(B, H, -1).sum(dim=2, keepdim=True).clamp(min=1e-12)
            probs = (probs.view(B, H, -1) / zsum).view(B, H, V, Q)
        # Apply mask safety: zero out invalid and renormalize globally
        if mask_bool is not None:
            probs = probs * mask_bool.to(dtype=probs.dtype)
            zsum = probs.view(B, H, -1).sum(dim=2, keepdim=True).clamp(min=1e-12)
            probs = (probs.view(B, H, -1) / zsum).view(B, H, V, Q)
        # diagnostics: attention entropy
        with torch.no_grad():
            flat_p = probs.view(B, H, -1).clamp_min(1e-12)
            ent = -(flat_p * flat_p.log()).sum(dim=2).mean()
            self.last_att_entropy = ent.detach().unsqueeze(0)
            self._last_att_maps = probs.detach()
        return probs

    # ======== Pooling and gating ========
    def _compute_head_gates(self, v_feat: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        if not self.gate_heads:
            gates = torch.ones(v_feat.size(0), self.h_out, device=v_feat.device, dtype=v_feat.dtype)
            with torch.no_grad():
                self.last_head_gates = gates.detach().mean(dim=0, keepdim=True)
                self.last_gate_sum = torch.tensor(float(self.h_out), device=v_feat.device)
            return gates
        # pooled contexts
        v_pool = v_feat.mean(dim=1)  # (B, D)
        q_pool = q_feat.mean(dim=1)  # (B, D)
        ctx = torch.cat([v_pool, q_pool], dim=-1)  # (B, 2D)
        logits = self.gate_mlp(ctx) + self.gate_bias.view(1, -1)  # (B, H)
        if self.gate_activation == 'softmax':
            gates = F.softmax(logits / max(self.gate_temperature, 1e-6), dim=1)
            # dropout then renormalize to preserve convex mixture
            if isinstance(self.gate_drop, nn.Dropout) and self.training:
                gates = self.gate_drop(gates)
                s = gates.sum(dim=1, keepdim=True).clamp(min=1e-12)
                gates = gates / s
        else:
            gates = torch.sigmoid(logits)
            gates = self.gate_drop(gates)
        with torch.no_grad():
            self.last_head_gates = gates.detach().mean(dim=0, keepdim=True)
            self.last_gate_sum = gates.detach().sum(dim=1).mean().unsqueeze(0)
        return gates

    def attention_pooling_all(self, v_feat: torch.Tensor, q_feat: torch.Tensor, att_maps: torch.Tensor) -> torch.Tensor:
        """Vectorized attention pooling across heads. Returns (B, H, h_dim)."""
        fusion = torch.einsum('bvk,bhvq,bqk->bhk', (v_feat, att_maps, q_feat))  # (B,H,D)
        if self.p_net is not None:
            B, H, D = fusion.shape
            fusion = fusion.reshape(B * H, 1, D)
            fusion = self.p_net(fusion).squeeze(1) * self.k  # sum-pooling
            fusion = fusion.reshape(B, H, self.h_dim)
        else:
            fusion = fusion.reshape(fusion.size(0), fusion.size(1), self.h_dim)
        return fusion

    # ======== Forward ========
    def forward(self, v: torch.Tensor, q: torch.Tensor, softmax: Optional[bool] = False):
        """
        v: (B, V, v_dim)
        q: (B, Q, q_dim)
        softmax: legacy flag; if True forces normalized attention (softmax). If False, uses self.use_softmax.
        returns: logits (B, h_dim), att_maps (B, h_out, V, Q)
        """
        # Project to bilinear space
        v_feat = self.v_net(v)  # (B, V, D)
        q_feat = self.q_net(q)  # (B, Q, D)
        # Attention logits
        att_logits = self._compute_attention_logits(v_feat, q_feat)  # (B,H,V,Q)
        # Normalize according to config/override
        apply_softmax = bool(softmax) or self.use_softmax
        att_maps = self._normalize_attention(att_logits, apply_softmax)
        # Pooling per head (vectorized)
        head_logits = self.attention_pooling_all(v_feat, q_feat, att_maps)  # (B,H,h_dim)
        # Head gates and aggregation
        gates = self._compute_head_gates(v_feat, q_feat)  # (B,H)
        logits = torch.einsum('bhk,bh->bk', head_logits, gates)
        # Normalize fused representation
        logits = self.norm(logits)
        # cache for regularization (live tensors for gradient flow)
        self._att_maps_for_reg = att_maps
        self._gates_for_reg = gates
        self._last_logits = logits.detach()
        return logits, att_maps

    # ======== Optional regularization hooks ========
    def compute_regularization(self) -> Dict[str, torch.Tensor]:
        """Return a dict of optional regularization terms based on latest forward pass.
        Includes:
        - gate_l1: L1 on live gate activations (scaled by gate_l1)
        - attn_entropy: mean entropy of attention maps (scaled by attn_entropy_weight)
        - temp_l2: L2 on temperature parameters deviation from init (scaled)
        - alpha_l2: L2 on alpha_logit to prevent overuse of either path (scaled)
        """
        regs: Dict[str, torch.Tensor] = {}
        device = self.h_mat.device
        total = torch.tensor(0.0, device=device)
        # Gate L1 on live gates (use live tensor if available)
        gate_l1_term = torch.tensor(0.0, device=device)
        if self.gate_heads:
            if self._gates_for_reg is not None:
                gate_l1_term = self._gates_for_reg.abs().mean()
            else:
                gate_l1_term = self.last_head_gates.abs().mean()
        regs['gate_l1'] = self.gate_l1 * gate_l1_term
        total = total + regs['gate_l1']
        # Attention entropy (use live att maps if available)
        attn_entropy_term = torch.tensor(0.0, device=device)
        if self._att_maps_for_reg is not None:
            B = self._att_maps_for_reg.size(0)
            flat_p = self._att_maps_for_reg.view(B, self.h_out, -1).clamp_min(1e-12)
            ent = -(flat_p * flat_p.log()).sum(dim=2).mean()
            attn_entropy_term = ent
        elif self._last_att_maps is not None:
            B = self._last_att_maps.size(0)
            flat_p = self._last_att_maps.view(B, self.h_out, -1).clamp_min(1e-12)
            ent = -(flat_p * flat_p.log()).sum(dim=2).mean()
            attn_entropy_term = ent
        regs['attn_entropy'] = self.attn_entropy_weight * attn_entropy_term
        total = total + regs['attn_entropy']
        # Temperature L2 toward init
        temp_l2_term = torch.tensor(0.0, device=device)
        if self.per_head_temperature and (self.temp_param is not None) and self.temperature_l2_weight > 0:
            temp = F.softplus(self.temp_param)
            init = torch.full_like(temp, float(self.global_temperature))
            temp_l2_term = (temp - init).pow(2).mean()
        regs['temp_l2'] = self.temperature_l2_weight * temp_l2_term
        total = total + regs['temp_l2']
        # Alpha L2 regularizer
        alpha_l2_term = torch.tensor(0.0, device=device)
        if self.alpha_l2_weight > 0:
            alpha_l2_term = (self.alpha_logit.pow(2).mean())
        regs['alpha_l2'] = self.alpha_l2_weight * alpha_l2_term
        total = total + regs['alpha_l2']
        regs['total'] = total
        return regs

    def get_diagnostics(self) -> Dict[str, float]:
        """Return lightweight diagnostics from the last forward pass."""
        out = {
            'att_entropy': float(self.last_att_entropy.item()) if self.last_att_entropy is not None else 0.0,
            'gate_mean': float(self.last_head_gates.mean().item()) if self.last_head_gates is not None else 0.0,
            'gate_sum_mean': float(self.last_gate_sum.item()) if self.last_gate_sum is not None else 0.0,
            'topk_k': float(self.last_topk_k.item()) if self.last_topk_k is not None else 0.0,
            'sink_row_resid': float(self.last_sink_row_resid.item()) if self.last_sink_row_resid is not None else 0.0,
            'sink_col_resid': float(self.last_sink_col_resid.item()) if self.last_sink_col_resid is not None else 0.0,
        }
        return out
