"""Attention, embeddings and the DiT block from the main model."""

from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F


def attention(q, k, v, k_lens=None, causal=False):
    if k_lens is not None:
        raise ValueError("Core attention expects fixed-length, unpadded batches.")
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=causal,
    ).transpose(1, 2)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.to(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000,
            -torch.arange(half, device=position.device, dtype=torch.float64) / half,
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(torch.float32)


class TimeEmbed(nn.Module):

    def __init__(self, time_dim: int, model_dim: int):
        super().__init__()
        self.freq_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.to_mod = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 6))

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        te = sinusoidal_embedding_1d(self.freq_dim, t.view(-1))
        e = self.mlp(te)
        m = self.to_mod(e).view(e.shape[0], 6, -1)
        return (e, m)


class GlobalOneHotEmbed(nn.Module):

    def __init__(self, num_classes: int, model_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.to_mod = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 6))

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.mlp(g.float())
        m = self.to_mod(e).view(e.shape[0], 6, -1)
        return (e, m)


class PersonalityEmbed(nn.Module):

    def __init__(self, num_classes: int, model_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.to_mod = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 6))

    def forward(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.mlp(p.float())
        m = self.to_mod(e).view(e.shape[0], 6, -1)
        return (e, m)


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class RoPE1D(nn.Module):
    """1D Rotary Positional Embedding applied per-head to q/k.
    Works for shapes [B, L, H, D]. D must be even.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "head_dim must be even for RoPE"
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_sin_cos(self, L: int, device, dtype):
        pos = torch.arange(L, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("l,d->ld", pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        sin = emb.sin().to(dtype)
        cos = emb.cos().to(dtype)
        return (sin, cos)

    def apply(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (B, L, H, D) = q.shape
        (sin, cos) = self._build_sin_cos(L, q.device, q.dtype)
        sin = sin.view(1, L, 1, D)
        cos = cos.view(1, L, 1, D)
        (q1, q2) = (q[..., ::2], q[..., 1::2])
        (k1, k2) = (k[..., ::2], k[..., 1::2])
        q_rot = torch.stack([-q2, q1], dim=-1).reshape_as(q)
        k_rot = torch.stack([-k2, k1], dim=-1).reshape_as(k)
        q_out = q * cos + q_rot * sin
        k_out = k * cos + k_rot * sin
        return (q_out, k_out)


class MHA(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int,
        qk_norm: bool = True,
        eps: float = 1e-06,
        use_rope: bool = True,
    ):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.rope = RoPE1D(self.dh) if use_rope else None

    def forward(
        self, x: torch.Tensor, k_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        (B, L, C) = x.shape
        (H, D) = (self.heads, self.dh)
        q = self.norm_q(self.q(x)).view(B, L, H, D)
        k = self.norm_k(self.k(x)).view(B, L, H, D)
        v = self.v(x).view(B, L, H, D)
        if self.rope is not None:
            (q, k) = self.rope.apply(q, k)
        y = attention(q, k, v, k_lens=k_lens)
        return self.o(y.reshape(B, L, C))


class CrossMHA(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int,
        qk_norm: bool = True,
        eps: float = 1e-06,
        use_rope: bool = True,
    ):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.rope = RoPE1D(self.dh) if use_rope else None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (B, Lq, C) = x.shape
        (H, D) = (self.heads, self.dh)
        q = self.norm_q(self.q(x)).view(B, Lq, H, D)
        k = self.norm_k(self.k(context)).view(B, -1, H, D)
        v = self.v(context).view(B, -1, H, D)
        if self.rope is not None:
            if Lq != context.shape[1]:
                k_rope = k[:, :Lq]
                (q, k_rope) = self.rope.apply(q, k_rope)
                k = torch.cat([k_rope, k[:, Lq:]], dim=1)
            else:
                (q, k) = self.rope.apply(q, k)
        y = attention(q, k, v, k_lens=context_lens)
        return self.o(y.reshape(B, Lq, C))


class DiTBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        heads: int,
        qk_norm: bool = True,
        eps: float = 1e-06,
        cross_attn: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        self.cross_attn = cross_attn
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.sa = MHA(dim, heads, qk_norm=qk_norm, eps=eps, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.ff = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        if cross_attn:
            self.normc = nn.LayerNorm(dim, eps=eps)
            self.ca = CrossMHA(dim, heads, qk_norm=qk_norm, eps=eps, use_rope=use_rope)
        self.mod = nn.Parameter(torch.zeros(1, 6, dim))

    def forward(
        self,
        x: torch.Tensor,
        tmods: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e = (self.mod + tmods).chunk(6, dim=1)
        y = self.sa(self.norm1(x) * (1 + e[1]) + e[0])
        x = x + y * e[2]
        if self.cross_attn and context is not None:
            x = x + self.ca(self.normc(x), context, context_lens)
        y = self.ff(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x
