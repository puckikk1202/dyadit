"""DyaDiT training entry point.

This wrapper reuses the repository-level training implementation while keeping
``dyadit_code`` first on ``sys.path``.  Consequently, imports such as ``model``,
``datasets`` and ``dyadic_dit_vq`` resolve to the implementations shipped in
this directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_DYADIT_DIR = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _DYADIT_DIR.parent
_TRAINING_IMPLEMENTATION = _REPOSITORY_ROOT / "train_wan_ours.py"

if str(_DYADIT_DIR) in sys.path:
    sys.path.remove(str(_DYADIT_DIR))
sys.path.insert(0, str(_DYADIT_DIR))

_spec = importlib.util.spec_from_file_location(
    "_dyadit_training_implementation",
    _TRAINING_IMPLEMENTATION,
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load training implementation: {_TRAINING_IMPLEMENTATION}")

_implementation = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _implementation
_spec.loader.exec_module(_implementation)

# Preserve the existing import interface used by inference_dyadic.py and other
# callers (for example LitAudio2PoseDiT and DiffusionConfig).
for _name, _value in vars(_implementation).items():
    if not _name.startswith("_"):
        globals()[_name] = _value


if __name__ == "__main__":
    _implementation.main()
