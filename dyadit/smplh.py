"""SMPL-H assets, including numeric reading of older Chumpy-based model files."""

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np


class _ChArray:
    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def r(self):
        return np.asarray(self.x)


class _ChSelect(_ChArray):
    @property
    def r(self):
        source = self.a.r if isinstance(self.a, _ChArray) else np.asarray(self.a)
        selected = source.reshape(-1)[self.idxs]
        return selected.reshape(self.preferred_shape)


class _ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("chumpy.ch", "Ch"):
            return _ChArray
        if (module, name) == ("chumpy.reordering", "Select"):
            return _ChSelect
        return super().find_class(module, name)


def resolve_model_path(path, gender):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in (".pkl", ".npz"):
            raise ValueError("SMPL-H model must be a .pkl or .npz file")
        return path
    names = [
        f"SMPLH_{value}.{suffix}"
        for value in (gender.upper(), gender.lower())
        for suffix in ("npz", "pkl")
    ]
    candidates = [
        directory / name for directory in (path, path / "smplh") for name in names
    ]
    candidates += [
        directory / "model.npz"
        for directory in (path / gender, path / "smplh" / gender, path)
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No SMPL-H {gender} model found under {path}. Pass a model file directly or its model directory."
    )


@lru_cache(maxsize=4)
def _model_data(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            data = dict(archive)
    else:
        with path.open("rb") as stream:
            data = _ModelUnpickler(stream, encoding="latin1").load()
    return {
        key: value.r if isinstance(value, _ChArray) else value
        for key, value in data.items()
    }


def load_body_model(path, gender="male", device="cpu", flat_hand_mean=True):
    import smplx
    from smplx.utils import Struct

    path = resolve_model_path(path, gender)
    data = dict(_model_data(str(path)))
    # Some full-axis-angle SMPL-H files omit the optional MANO PCA metadata.
    # PCA is disabled here; these identity components are never applied.
    for name in ("hands_componentsl", "hands_componentsr"):
        data.setdefault(name, np.eye(45, dtype=np.float32))
    for name in ("hands_meanl", "hands_meanr"):
        if name not in data and not flat_hand_mean:
            raise ValueError(
                f"{path} omits MANO mean poses. Use --hand-mean flat with this model."
            )
        data.setdefault(name, np.zeros(45, dtype=np.float32))
    return (
        smplx.SMPLH(
            str(path),
            data_struct=Struct(**data),
            gender=gender,
            use_pca=False,
            flat_hand_mean=flat_hand_mean,
            num_betas=10,
        )
        .to(device)
        .eval()
    )
