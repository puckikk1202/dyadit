"""Read current state dictionaries and legacy Lightning checkpoint metadata."""

import pickle
from types import SimpleNamespace

import torch


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Older Lightning files also pickle the codec and DiffusionConfig inside
        # hyper_parameters. Resolve these names locally without importing old scripts.
        if module == "dyadic_dit_vq.model.vqvae_downsampled":
            from . import codec

            return getattr(codec, name)
        if module == "dyadic_dit_vq.model.vqvae":
            from . import quantizer

            return getattr(quantizer, name)
        if name == "DiffusionConfig" and module in (
            "__main__",
            "train_wan_ours",
            "_dyadit_training_implementation",
        ):
            from .diffusion import DiffusionConfig

            return DiffusionConfig
        return super().find_class(module, name)


_legacy_pickle = SimpleNamespace(
    __name__="dyadit_legacy_pickle",
    Unpickler=LegacyUnpickler,
    load=pickle.load,
    loads=pickle.loads,
)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(
        path,
        map_location=map_location,
        weights_only=False,
        pickle_module=_legacy_pickle,
    )


def load_system_weights(system, path):
    """Load the original model.* / vqvae.* Lightning parameter layout strictly."""
    checkpoint = load_checkpoint(path)
    state = checkpoint.get("state_dict", checkpoint)
    system.load_state_dict(state, strict=True)
    return checkpoint
