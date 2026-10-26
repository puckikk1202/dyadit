"""Only the audio, motion and memory helpers used by the main model."""

from typing import Optional
import torch
from torch import nn
from transformers import Wav2Vec2Model


class LIAMemoryBank(nn.Module):
    """Learnable memory tokens; only the weight is used by DyaDiT."""

    def __init__(self, latent_dim, num_direction):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(latent_dim, num_direction))


def zero_adaptive_instance_normalization(content, style, eps=1e-05):
    assert len(content.shape) == 3
    mean_content = content.mean(dim=1, keepdim=True)
    std_content = content.std(dim=1, keepdim=True)
    mean_style = style.mean(dim=1, keepdim=True)
    std_style = style.std(dim=1, keepdim=True)
    normalized_content = (content - mean_content) / (std_content + eps)
    return normalized_content * (1 + std_style) + mean_style


class TANGOWrapedWav2Vec(nn.Module):
    """
    Modified from:
    TANGO: Co-Speech Gesture Video Reenactment with Hierarchical Audio-Motion Embedding and Diffusion Interpolation
    https://github.com/CyberAgentAILab/TANGO
    """

    def __init__(self, layers: int = 1, audio_model_name="facebook/wav2vec2-base-960h"):
        super(TANGOWrapedWav2Vec, self).__init__()
        base = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.feature_extractor = base.feature_extractor
        self.feature_projection = base.feature_projection
        self.encoder = base.encoder
        self.encoder.layers = self.encoder.layers[:layers]

    def forward(
        self,
        inputs,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        finetune_audio_low = self.feature_extractor(inputs).transpose(1, 2)
        (hidden_states, _) = self.feature_projection(finetune_audio_low.detach())
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        return {"low_level": finetune_audio_low, "high_level": hidden_states}


class ResBlock(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(channel, channel, 3, 1, 1),
        )

    def forward(self, x):
        return self.model(x) + x


def init_weight(m):
    if (
        isinstance(m, nn.Conv1d)
        or isinstance(m, nn.Linear)
        or isinstance(m, nn.ConvTranspose1d)
    ):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class TM2TVQEncoderV5(nn.Module):
    """
    Modified from:
    TM2T: Stochastical and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts
    https://github.com/EricGuo5513/TM2T
    """

    def __init__(self, input_size=512, n_down=2, channels=[768, 768, 768]):
        super().__init__()
        self.input_size = input_size
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            ResBlock(channels[0]),
        ]
        for i in range(1, n_down + 1):
            layers += [
                nn.Conv1d(channels[i - 1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs
