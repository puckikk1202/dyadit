"""The retained legacy model must also preserve pretrained speech weights."""

import torch
from torch import nn


def test_legacy_constructor_preserves_both_pretrained_speech_encoders(monkeypatch):
    from model import dyadit as legacy

    snapshots = []

    class PretrainedAudio(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.feature_projection = nn.Linear(4, 768)
            self.encoder = nn.Sequential(nn.Linear(768, 768))
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.fill_(0.125 + len(snapshots) * 0.25)
            snapshots.append(
                {name: value.clone() for name, value in self.state_dict().items()}
            )

    monkeypatch.setattr(legacy, "TANGOWrapedWav2Vec", PretrainedAudio)
    monkeypatch.setattr(
        legacy.Wav2Vec2Processor, "from_pretrained", lambda *args, **kwargs: object()
    )
    model = legacy.Audio2PoseDiTDual_VQVAE(
        model_dim=16, ffn_dim=32, heads=2, layers=1, time_dim=16, motion_bank_size=8
    )
    assert len(snapshots) == 2
    for encoder, expected in zip(
        (model.audio_encoder, model.audio_encoder_other), snapshots
    ):
        for name, value in encoder.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert torch.count_nonzero(model.in_proj.weight) > 0
    assert torch.count_nonzero(model.out_proj.weight) == 0
