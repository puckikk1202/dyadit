import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .motion_generation.motion_gen_utils_dev import LIAMemoryBank, TM2TVQEncoderV5, TANGOWrapedWav2Vec, zero_adaptive_instance_normalization
except ImportError:
    from motion_generation.motion_gen_utils_dev import LIAMemoryBank, TM2TVQEncoderV5, TANGOWrapedWav2Vec, zero_adaptive_instance_normalization
from transformers import Wav2Vec2Processor
# If you keep your flash_attention functional from `attention.py`, import it here.
# Otherwise we'll transparently fall back to SDPA.
try:
    from attention import flash_attention as _flash_attention
    _HAS_FA = True
except Exception:
    _HAS_FA = False

def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    """Fallback to PyTorch scaled dot-product attention.
    Expects shapes [B, L, H, D]. Returns [B, L, H, D].
    """
    B, Lq, H, D = q.shape
    # PyTorch SDPA expects [B, H, L, D]
    q_ = q.transpose(1, 2)
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask, is_causal=False)
    return out.transpose(1, 2)


def _can_use_fa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if not _HAS_FA:
        return False
    # FlashAttention varlen requirements (typical): CUDA, half/bfloat16, head_dim <= 256
    if q.device.type != 'cuda':
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.size(-1) > 256 or k.size(-1) > 256 or v.size(-1) > 256:
        return False
    return True


def attention(q, k, v, k_lens: Optional[torch.Tensor] = None, causal: bool = False):
    """Unified attention wrapper. Inputs [B, L, H, D] -> [B, L, H, D].
    If FlashAttention varlen is available & compatible, we use it; else SDPA.
    """
    if _can_use_fa(q, k, v):
        return _flash_attention(q, k, v, k_lens=k_lens, causal=causal)
    return _sdpa(q, k, v, attn_mask=None)



# -----------------------------
# Positional / time embeddings
# -----------------------------

def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.to(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half, device=position.device, dtype=torch.float64) / half))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(torch.float32)


class TimeEmbed(nn.Module):
    def __init__(self, time_dim: int, model_dim: int):
        super().__init__()
        self.freq_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        # 6-way FiLM like in WAN/DiT: shift/scale pairs for SA, FFN; 3 pairs -> 6 vectors
        self.to_mod = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 6))

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # t: [B] or [B,]
        te = sinusoidal_embedding_1d(self.freq_dim, t.view(-1))  # [B, time_dim]
        e = self.mlp(te)                                         # [B, C]
        m = self.to_mod(e).view(e.shape[0], 6, -1)               # [B, 6, C]
        return e, m

# -----------------------------
# Relationship Projection
# -----------------------------

class GlobalOneHotEmbed(nn.Module):
    def __init__(self, num_classes: int, model_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.to_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, model_dim * 6),
        )

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.mlp(g.float())                         # [B, C]
        m = self.to_mod(e).view(e.shape[0], 6, -1)      # [B, 6, C]
        return e, m


# -----------------------------
# Norms
# -----------------------------

class PersonalityEmbed(nn.Module):
    def __init__(self, num_classes: int, model_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.to_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, model_dim * 6),
        )

    def forward(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.mlp(p.float())                         # [B, C]
        m = self.to_mod(e).view(e.shape[0], 6, -1)      # [B, 6, C]
        return e, m

class MotionOtherEmbed(nn.Module):
    def __init__(self, pose_dim: int, model_dim: int):
        super().__init__()
        self.pose_dim = pose_dim
        self.model_dim = model_dim
        self.encoder = TM2TVQEncoderV5(input_size=pose_dim, n_down=2, channels=[model_dim, model_dim, model_dim])
        self.nonlinear = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = self.encoder(x)  # [B, C]
        e = self.nonlinear(e)
        e = self.norm(e)
        return e

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., C]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


# -----------------------------
# Rotary Positional Embedding (1D) + Attention blocks
# -----------------------------

class RoPE1D(nn.Module):
    """1D Rotary Positional Embedding applied per-head to q/k.
    Works for shapes [B, L, H, D]. D must be even.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "head_dim must be even for RoPE"
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_sin_cos(self, L: int, device, dtype):
        pos = torch.arange(L, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("l,d->ld", pos, self.inv_freq)  # [L, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)               # [L, D]
        sin = emb.sin().to(dtype)
        cos = emb.cos().to(dtype)
        return sin, cos

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q,k: [B, L, H, D]
        B, L, H, D = q.shape
        sin, cos = self._build_sin_cos(L, q.device, q.dtype)  # [L, D]
        # reshape to broadcast: [1, L, 1, D]
        sin = sin.view(1, L, 1, D)
        cos = cos.view(1, L, 1, D)
        # rotate halfs
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]
        q_rot = torch.stack([-q2, q1], dim=-1).reshape_as(q)
        k_rot = torch.stack([-k2, k1], dim=-1).reshape_as(k)
        q_out = q * cos + q_rot * sin
        k_out = k * cos + k_rot * sin
        return q_out, k_out


class MHA(nn.Module):
    def __init__(self, dim: int, heads: int, qk_norm: bool = True, eps: float = 1e-6, use_rope: bool = True):
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

    def forward(self, x: torch.Tensor, k_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        H, D = self.heads, self.dh
        q = self.norm_q(self.q(x)).view(B, L, H, D)
        k = self.norm_k(self.k(x)).view(B, L, H, D)
        v = self.v(x).view(B, L, H, D)
        if self.rope is not None:
            q, k = self.rope.apply(q, k)
        y = attention(q, k, v, k_lens=k_lens)  # [B, L, H, D]
        return self.o(y.reshape(B, L, C))


class CrossMHA(nn.Module):
    def __init__(self, dim: int, heads: int, qk_norm: bool = True, eps: float = 1e-6, use_rope: bool = True):
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
        # For cross, we only apply RoPE along the query time axis; keys come from context with same time grid after align.
        self.rope = RoPE1D(self.dh) if use_rope else None

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, Lq, C], context: [B, Lk, C]
        B, Lq, C = x.shape
        H, D = self.heads, self.dh
        q = self.norm_q(self.q(x)).view(B, Lq, H, D)
        k = self.norm_k(self.k(context)).view(B, -1, H, D)
        v = self.v(context).view(B, -1, H, D)
        if self.rope is not None:
            # If Lk != Lq (unaligned), RoPE on queries only still provides relative signal w.r.t. their own timeline.
            if Lq != context.shape[1]:
                k_rope = k[:, :Lq]
                q, k_rope = self.rope.apply(q, k_rope)
                k = torch.cat([k_rope, k[:, Lq:]], dim=1)   
            else:
                q, k = self.rope.apply(q, k)
        y = attention(q, k, v, k_lens=context_lens)  # [B, Lq, H, D]
        return self.o(y.reshape(B, Lq, C))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int, qk_norm: bool = True, eps: float = 1e-6, cross_attn: bool = True, use_rope: bool = True):
        super().__init__()
        self.cross_attn = cross_attn
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.sa = MHA(dim, heads, qk_norm=qk_norm, eps=eps, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.ff = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim)
        )
        if cross_attn:
            self.normc = nn.LayerNorm(dim, eps=eps)
            self.ca = CrossMHA(dim, heads, qk_norm=qk_norm, eps=eps, use_rope=use_rope)
        # FiLM modulation buffer (learnable shift for stability like WAN)
        self.mod = nn.Parameter(torch.zeros(1, 6, dim))

    def forward(self, x: torch.Tensor, tmods: torch.Tensor, context: Optional[torch.Tensor] = None, context_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, C]; tmods: [B, 6, C]
        e = (self.mod + tmods).chunk(6, dim=1)
        # self-attn
        y = self.sa(self.norm1(x) * (1 + e[1]) + e[0])
        x = x + y * e[2]
        # cross-attn
        if self.cross_attn and context is not None:
            x = x + self.ca(self.normc(x), context, context_lens)
        # ffn
        y = self.ff(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x


# -----------------------------
# Audio→Pose DiT (Diffusion transformer backbone)
# -----------------------------

class Audio2PoseDiTDual_VQVAE(nn.Module):
    """Audio → 129D Pose Diffusion Transformer"""
    def __init__(
        self,
        pose_dim: int = 129,
        audio_channels: int = 1,
        model_dim: int = 512,
        ffn_dim: int = 2048,
        heads: int = 4,
        layers: int = 4,
        time_dim: int = 512,
        use_cross_attn: bool = True,
        use_rope: bool = True,
        fusion_drop: float = 0.0,
        style_drop: float = 0.0,
        wav2vec_layer: int = 1,
        motion_bank_size: int = 64,
        drop_audio_prob: float = 0.0,
        drop_style_prob: float = 0.0,
        drop_memory_bank_prob: float = 0.0,
        relat_drop: float = 0.0,
        person_drop: float = 0.0,
        motion_other_drop: float = 0.0,
        orthogonal_fusion: bool = False,
        downsample_factor: int = 4,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.in_proj = nn.Linear(pose_dim, model_dim)
        self.mask_proj = nn.Linear(1, model_dim)
        self.out_proj = nn.Linear(model_dim, pose_dim)

        self.time = TimeEmbed(time_dim, model_dim)
        self.relat_proj = GlobalOneHotEmbed(num_classes=4, model_dim=model_dim)
        self.relat_alpha = nn.Parameter(torch.ones(1))
        self.relat_norm = nn.LayerNorm(model_dim)
        self.person_proj = PersonalityEmbed(num_classes=5, model_dim=model_dim)
        self.person_alpha = nn.Parameter(torch.ones(1))
        self.person_norm = nn.LayerNorm(model_dim)
        
        self.blocks = nn.ModuleList([
            DiTBlock(model_dim, ffn_dim, heads, qk_norm=True, eps=1e-6, cross_attn=use_cross_attn, use_rope=use_rope)
            for _ in range(layers)
        ])
        self.norm_out = nn.LayerNorm(model_dim)

        # audio encoder used as context for cross-attn
        self.audio_encoder = TANGOWrapedWav2Vec(layers=wav2vec_layer)
        self.audio_encoder_other = TANGOWrapedWav2Vec(layers=wav2vec_layer)
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_proj = nn.Linear(768, model_dim)
        self.audio_proj_other = nn.Linear(768, model_dim)
        self.audio_context_norm = nn.LayerNorm(model_dim)
        self.audio_context_norm_other = nn.LayerNorm(model_dim)
        self.audio_attn = CrossMHA(model_dim, heads, qk_norm=True, eps=1e-6, use_rope=use_rope)
        self.audio_attn_other = CrossMHA(model_dim, heads, qk_norm=True, eps=1e-6, use_rope=use_rope)
        self.audio_mem_attn = CrossMHA(model_dim, heads, qk_norm=True, eps=1e-6, use_rope=False)

        self.audio_ff = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(approximate='tanh'), nn.Linear(model_dim, model_dim)
        )
        self.audio_gate = nn.Linear(model_dim*2, 1)

        self.memory_bank = LIAMemoryBank(model_dim, motion_bank_size)
        self.style_encoder = TM2TVQEncoderV5(input_size=pose_dim, n_down=2, channels=[model_dim, model_dim, model_dim])
        self.style_nonlinear = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim))
        
        self.motion_other_encoder = MotionOtherEmbed(pose_dim, model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
      
        self.fusion_drop = fusion_drop
        self.style_drop = style_drop
        self.relat_drop = relat_drop
        self.person_drop = person_drop
        self.motion_other_drop = motion_other_drop
        self.orthogonal_fusion = orthogonal_fusion
        self.downsample_factor = downsample_factor
        # lightweight init
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_proj.weight)

    def load_from_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"Model loaded from: {checkpoint_path}")

    @torch.no_grad()
    def infer_step_shapes(self, B: int, T: int) -> Tuple[int, int]:
        return B, T

    def forward(
        self,
        noisy_pose: torch.Tensor,   # [B, T, 129] — pose at time t (latent in diffusion)
        timesteps: torch.Tensor,    # [B]
        audio: torch.Tensor,        # [B, 1, T_a]
        audio_other: torch.Tensor,  # [B, 1, T_a]
        cond: Optional[torch.Tensor] = None,         # [B, T, 129]
        style_motion: Optional[torch.Tensor] = None,
        relat_onehot: Optional[torch.Tensor] = None,
        personality_feat: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        motion_other: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = noisy_pose.shape

        if motion_other is None:
            motion_other = torch.zeros(B, T, self.pose_dim, device=audio.device)

        # (1) time embeddings / FiLM mods
        e_vec, e_mod = self.time(timesteps)  # [B, C], [B, 6, C]
        relat_feat, relat_mod = self.relat_proj(relat_onehot)  # [B, C], [B, 6, C]
        if self.training and self.relat_drop > 0.0 and torch.rand(1).item() < self.relat_drop:
            relat_feat = torch.zeros_like(relat_feat)
            relat_mod = torch.zeros_like(relat_mod)
        e_mod = e_mod + self.relat_alpha * relat_mod 
        person_feat, person_mod = self.person_proj(personality_feat)  # [B, C], [B, 6, C]
        if self.training and self.person_drop > 0.0 and torch.rand(1).item() < self.person_drop:
            person_feat = torch.zeros_like(person_feat)
            person_mod = torch.zeros_like(person_mod)
        e_mod = e_mod + self.person_alpha * person_mod
        motion_other_feat = self.motion_other_encoder(motion_other).unsqueeze(1)  # [B, 1, C]
        if self.training and self.motion_other_drop > 0.0 and torch.rand(1).item() < self.motion_other_drop:
            motion_other_feat = torch.zeros_like(motion_other_feat)
        

        # (2) input projection
        x = self.in_proj(noisy_pose)  # [B, T, C]
        if mask is not None:
            # mask: [B, T] or [B, T, 1]
            m = mask.unsqueeze(-1).float() if mask.dim()==2 else mask.float()  # [B,T,1]
            x = x + self.mask_proj(m)  #[B, T, C]
        # (3) audio context aligned to T frames
        # Ensure audio is 1D for each sample (remove batch dimension and channel dimension)
        audio_list = [a.squeeze().cpu().numpy() for a in audio]
        audio_list_other = [a.squeeze().cpu().numpy() for a in audio_other]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        inputs_other = self.audio_processor(audio_list_other, sampling_rate=16000, return_tensors="pt", padding=True).to(audio_other.device)
        a_feat = self.audio_encoder(inputs.input_values)["high_level"]  # [B, L50, 768]
        a_feat_other = self.audio_encoder_other(inputs_other.input_values)["high_level"]  # [B, L50, 768]

        a_feat = F.interpolate(a_feat.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2).contiguous()
        a_feat_other = F.interpolate(a_feat_other.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2).contiguous()

        a_feat = self.audio_proj(a_feat)  # [B, T, C]
        a_feat = self.audio_context_norm(a_feat.float())

        a_feat_other = self.audio_proj_other(a_feat_other)  # [B, T, C]
        a_feat_other = self.audio_context_norm_other(a_feat_other.float())

        # style branch
        if style_motion is None:
            style_motion = torch.zeros(B, T, self.style_encoder.input_size, device=audio.device)
        style_motion = style_motion[:, :T]
        s_feat = self.style_encoder(style_motion)  # [B, C]
        s_feat = self.style_nonlinear(s_feat)
        if self.style_drop > 0.0 and self.training and torch.rand(1).item() < self.style_drop:
            s_feat = torch.zeros_like(s_feat)

        # audio memory attention
        mem = self.memory_bank.weight.transpose(1, 0).unsqueeze(0).expand(B, -1, -1)  # [B, K, C]
        mem = zero_adaptive_instance_normalization(mem, s_feat)           # [B, K, C]
        a_feat = a_feat + self.audio_mem_attn(a_feat, mem)

        if self.orthogonal_fusion:
            den  = (a_feat.pow(2).sum(-1, keepdim=True) + 1e-6)
            proj = ((a_feat_other * a_feat).sum(-1, keepdim=True) / den) * a_feat
            a_feat_other = a_feat_other - proj
        

        a_feat = a_feat + self.audio_attn(a_feat, a_feat_other)
        a_feat_other = a_feat_other + self.audio_attn_other(a_feat_other, a_feat)
        gate = torch.sigmoid(self.audio_gate(torch.cat([a_feat, a_feat_other], dim=-1)))
        ctx_audio = a_feat * gate + a_feat_other * (1 - gate)
        ctx_audio = self.audio_ff(ctx_audio)

        relat_feat = self.relat_norm(relat_feat.unsqueeze(1))
        person_feat = self.person_norm(person_feat.unsqueeze(1))
        ctx = torch.cat([ctx_audio, mem, motion_other_feat, relat_feat, person_feat], dim=1)
        ctx = self.context_norm(ctx.float())
      
        # fusion dropout for classifier-free guidance
        if self.fusion_drop > 0.0 and self.training:
            if torch.rand(1).item() < self.fusion_drop:
                ctx = torch.zeros_like(ctx)

        # transformer blocks
        for blk in self.blocks:
            x = blk(x, e_mod, context=ctx, context_lens=None)

        # output projection
        x = self.out_proj(self.norm_out(x))  # [B, T, 129]
        return x
