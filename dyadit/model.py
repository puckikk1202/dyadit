"""The main dual-audio DyaDiT; parameter names match existing checkpoints."""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2Processor
from .conditioning import (
    LIAMemoryBank,
    TM2TVQEncoderV5,
    TANGOWrapedWav2Vec,
    zero_adaptive_instance_normalization,
)
from .layers import TimeEmbed, GlobalOneHotEmbed, PersonalityEmbed, DiTBlock, CrossMHA


class MotionOtherEmbed(nn.Module):

    def __init__(self, pose_dim: int, model_dim: int):
        super().__init__()
        self.pose_dim = pose_dim
        self.model_dim = model_dim
        self.encoder = TM2TVQEncoderV5(
            input_size=pose_dim, n_down=2, channels=[model_dim, model_dim, model_dim]
        )
        self.nonlinear = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder(x)
        e = self.nonlinear(e)
        e = self.norm(e)
        return e


class Audio2PoseDiTDual_VQVAE(nn.Module):
    """Dual-audio diffusion transformer operating on continuous codec latents."""

    def __init__(
        self,
        pose_dim: int = 64,
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
        motion_bank_size: int = 1000,
        relat_drop: float = 0.0,
        person_drop: float = 0.0,
        motion_other_drop: float = 0.0,
        orthogonal_fusion: bool = False,
        downsample_factor: int = 4,
        audio_model_name="facebook/wav2vec2-base-960h",
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
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    model_dim,
                    ffn_dim,
                    heads,
                    qk_norm=True,
                    eps=1e-06,
                    cross_attn=use_cross_attn,
                    use_rope=use_rope,
                )
                for _ in range(layers)
            ]
        )
        self.norm_out = nn.LayerNorm(model_dim)
        self.audio_encoder = TANGOWrapedWav2Vec(
            layers=wav2vec_layer, audio_model_name=audio_model_name
        )
        self.audio_encoder_other = TANGOWrapedWav2Vec(
            layers=wav2vec_layer, audio_model_name=audio_model_name
        )
        self.audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_name)
        self.audio_proj = nn.Linear(768, model_dim)
        self.audio_proj_other = nn.Linear(768, model_dim)
        self.audio_context_norm = nn.LayerNorm(model_dim)
        self.audio_context_norm_other = nn.LayerNorm(model_dim)
        self.audio_attn = CrossMHA(
            model_dim, heads, qk_norm=True, eps=1e-06, use_rope=use_rope
        )
        self.audio_attn_other = CrossMHA(
            model_dim, heads, qk_norm=True, eps=1e-06, use_rope=use_rope
        )
        self.audio_mem_attn = CrossMHA(
            model_dim, heads, qk_norm=True, eps=1e-06, use_rope=False
        )
        self.audio_ff = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(model_dim, model_dim),
        )
        self.audio_gate = nn.Linear(model_dim * 2, 1)
        self.memory_bank = LIAMemoryBank(model_dim, motion_bank_size)
        self.style_encoder = TM2TVQEncoderV5(
            input_size=pose_dim, n_down=2, channels=[model_dim, model_dim, model_dim]
        )
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
        self.reset_parameters()

    def reset_parameters(self):
        # Both speech encoders have already loaded pretrained weights. Only
        # initialize the new motion/conditioning layers; traversing every
        # Linear would overwrite Wav2Vec's projection and transformer weights.
        pretrained = {
            id(module)
            for encoder in (self.audio_encoder, self.audio_encoder_other)
            for module in encoder.modules()
        }
        for m in self.modules():
            if id(m) not in pretrained and isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        noisy_pose: torch.Tensor,
        timesteps: torch.Tensor,
        audio: torch.Tensor,
        audio_other: torch.Tensor,
        style_motion: Optional[torch.Tensor] = None,
        relat_onehot: Optional[torch.Tensor] = None,
        personality_feat: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        motion_other: Optional[torch.Tensor] = None,
        drop_context: bool = False,
    ) -> torch.Tensor:
        (B, T, _) = noisy_pose.shape
        if relat_onehot is None:
            relat_onehot = noisy_pose.new_zeros(B, 4)
        if personality_feat is None:
            personality_feat = noisy_pose.new_zeros(B, 5)
        if motion_other is None:
            motion_other = torch.zeros(B, T, self.pose_dim, device=audio.device)
        (e_vec, e_mod) = self.time(timesteps)
        (relat_feat, relat_mod) = self.relat_proj(relat_onehot)
        if (
            self.training
            and self.relat_drop > 0.0
            and (torch.rand(1).item() < self.relat_drop)
        ):
            relat_feat = torch.zeros_like(relat_feat)
            relat_mod = torch.zeros_like(relat_mod)
        e_mod = e_mod + self.relat_alpha * relat_mod
        (person_feat, person_mod) = self.person_proj(personality_feat)
        if (
            self.training
            and self.person_drop > 0.0
            and (torch.rand(1).item() < self.person_drop)
        ):
            person_feat = torch.zeros_like(person_feat)
            person_mod = torch.zeros_like(person_mod)
        e_mod = e_mod + self.person_alpha * person_mod
        motion_other_feat = self.motion_other_encoder(motion_other)
        if (
            self.training
            and self.motion_other_drop > 0.0
            and (torch.rand(1).item() < self.motion_other_drop)
        ):
            motion_other_feat = torch.zeros_like(motion_other_feat)
        x = self.in_proj(noisy_pose)
        if mask is not None:
            m = mask.unsqueeze(-1).float() if mask.dim() == 2 else mask.float()
            x = x + self.mask_proj(m)
        audio_list = [a.squeeze().cpu().numpy() for a in audio]
        audio_list_other = [a.squeeze().cpu().numpy() for a in audio_other]
        inputs = self.audio_processor(
            audio_list, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(audio.device)
        inputs_other = self.audio_processor(
            audio_list_other, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(audio_other.device)
        a_feat = self.audio_encoder(inputs.input_values)["high_level"]
        a_feat_other = self.audio_encoder_other(inputs_other.input_values)["high_level"]
        a_feat = (
            F.interpolate(
                a_feat.transpose(1, 2), size=T, mode="linear", align_corners=False
            )
            .transpose(1, 2)
            .contiguous()
        )
        a_feat_other = (
            F.interpolate(
                a_feat_other.transpose(1, 2), size=T, mode="linear", align_corners=False
            )
            .transpose(1, 2)
            .contiguous()
        )
        a_feat = self.audio_proj(a_feat)
        a_feat = self.audio_context_norm(a_feat.float())
        a_feat_other = self.audio_proj_other(a_feat_other)
        a_feat_other = self.audio_context_norm_other(a_feat_other.float())
        if style_motion is None:
            style_motion = torch.zeros(
                B, T, self.style_encoder.input_size, device=audio.device
            )
        style_motion = style_motion[:, :T]
        s_feat = self.style_encoder(style_motion)
        s_feat = self.style_nonlinear(s_feat)
        if (
            self.style_drop > 0.0
            and self.training
            and (torch.rand(1).item() < self.style_drop)
        ):
            s_feat = torch.zeros_like(s_feat)
        mem = self.memory_bank.weight.transpose(1, 0).unsqueeze(0).expand(B, -1, -1)
        mem = zero_adaptive_instance_normalization(mem, s_feat)
        a_feat = a_feat + self.audio_mem_attn(a_feat, mem)
        if self.orthogonal_fusion:
            den = a_feat.pow(2).sum(-1, keepdim=True) + 1e-06
            proj = (a_feat_other * a_feat).sum(-1, keepdim=True) / den * a_feat
            a_feat_other = a_feat_other - proj
        a_feat = a_feat + self.audio_attn(a_feat, a_feat_other)
        a_feat_other = a_feat_other + self.audio_attn_other(a_feat_other, a_feat)
        gate = torch.sigmoid(self.audio_gate(torch.cat([a_feat, a_feat_other], dim=-1)))
        ctx_audio = a_feat * gate + a_feat_other * (1 - gate)
        ctx_audio = self.audio_ff(ctx_audio)
        relat_feat = self.relat_norm(relat_feat.unsqueeze(1))
        person_feat = self.person_norm(person_feat.unsqueeze(1))
        ctx = torch.cat(
            [ctx_audio, mem, motion_other_feat, relat_feat, person_feat], dim=1
        )
        ctx = self.context_norm(ctx.float())
        if drop_context or (
            self.training
            and self.fusion_drop > 0.0
            and torch.rand(1).item() < self.fusion_drop
        ):
            ctx = torch.zeros_like(ctx)
        for blk in self.blocks:
            x = blk(x, e_mod, context=ctx, context_lens=None)
        x = self.out_proj(self.norm_out(x))
        return x
