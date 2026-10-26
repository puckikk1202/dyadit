"""A relocated checkout must launch its own code and independent ablation run."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("override_directory", [False, True])
def test_no_orca_launcher_uses_submitted_checkout(tmp_path, override_directory):
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    record = tmp_path / "record.json"
    recorder = tmp_path / "record.py"
    recorder.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "keys = ['DYADIT_OUTPUT_DIR', 'DYADIT_RUN_NAME', 'WANDB_RUN_ID']\n"
        f"Path({str(record)!r}).write_text(json.dumps({{'env': {{key: os.environ[key] for key in keys}}, 'args': sys.argv[1:]}}))\n"
    )
    (checkout / "qsub_4gpu.sh").write_text(
        f'#!/bin/bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(recorder))} "$@"\n'
    )
    environment = os.environ.copy()
    environment.update(
        JOB_ID="1234567",
        SGE_O_WORKDIR=str(checkout),
        WANDB_RUN_ID="active-main-run",
        DYADIT_OUTPUT_DIR="active-main-output",
    )
    if override_directory:
        environment["DYADIT_CORE_DIR"] = str(checkout)
        environment["SGE_O_WORKDIR"] = str(tmp_path / "different directory")
    else:
        environment.pop("DYADIT_CORE_DIR", None)
    script = Path(__file__).resolve().parents[1] / "qsub_no_orca_4gpu.sh"
    subprocess.run(
        ["bash", str(script), "--set", "fid.num_samples=8"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(record.read_text())
    assert result["args"] == [
        "--set",
        "fid.num_samples=8",
        "--set",
        "model.orthogonal_fusion=false",
    ]
    assert result["env"] == {
        "DYADIT_OUTPUT_DIR": str(checkout / "outputs/cluster/no_orca_job_1234567"),
        "DYADIT_RUN_NAME": "dyadit_no_orca_4gpu_1234567",
        "WANDB_RUN_ID": "dyadit-no-orca-1234567",
    }
