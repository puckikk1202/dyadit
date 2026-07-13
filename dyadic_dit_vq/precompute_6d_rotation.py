#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precompute 6D rotation representation (aligned with get_motion_segment)
- Input : SMPL-H NPZ (axis-angle) with keys:
          smplh:body_pose (F,21,3), smplh:left_hand_pose (F,15,3), smplh:right_hand_pose (F,15,3)
- Pack   : [upper_body(13 joints) → left_hand(15) → right_hand(15)] = (F,129)
           * Upper body = all body joints EXCEPT lower body [0,1,3,4,6,7,9,10]
- Output : NPY (F,258) 6D rotation
"""

import os
import sys
import glob
import numpy as np
import torch
from tqdm import tqdm

# Add project root (where utils.motion_utils lives)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.motion_utils import convert_axis_angle_to_rotation_6d

# -------- Config: joint sets (must match your loader exactly) --------
LOWER_BODY_JOINTS = [0, 1, 3, 4, 6, 7, 9, 10]
UPPER_BODY_JOINTS = [j for j in range(21) if j not in LOWER_BODY_JOINTS]  # -> [2,5,8,11,12,13,14,15,16,17,18,19,20]


def pack_upper_hands_axis_angle(npz_data: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    Pack axis-angle features to (F,129) with the EXACT same order as the loader:
      [upper_body(13*3) → left_hand(15*3) → right_hand(15*3)]
    """
    feats = []

    if 'smplh:body_pose' in npz_data:
        body_pose = np.asarray(npz_data['smplh:body_pose'], dtype=np.float32)  # (F,21,3)
        assert body_pose.ndim == 3 and body_pose.shape[1:] == (21, 3), f"bad body_pose shape {body_pose.shape}"
        upper = body_pose[:, UPPER_BODY_JOINTS, :]                               # (F,13,3)
        feats.append(upper.reshape(upper.shape[0], -1))                          # (F,39)
    if 'smplh:left_hand_pose' in npz_data:
        lh = np.asarray(npz_data['smplh:left_hand_pose'], dtype=np.float32)      # (F,15,3)
        assert lh.ndim == 3 and lh.shape[1:] == (15, 3), f"bad left_hand shape {lh.shape}"
        feats.append(lh.reshape(lh.shape[0], -1))                                 # (F,45)
    if 'smplh:right_hand_pose' in npz_data:
        rh = np.asarray(npz_data['smplh:right_hand_pose'], dtype=np.float32)     # (F,15,3)
        assert rh.ndim == 3 and rh.shape[1:] == (15, 3), f"bad right_hand shape {rh.shape}"
        feats.append(rh.reshape(rh.shape[0], -1))                                 # (F,45)

    if not feats:
        raise ValueError("No required motion features (body/hands) found in NPZ")

    x = np.concatenate(feats, axis=1).astype(np.float32)  # (F,129)
    assert x.ndim == 2 and x.shape[1] == 129, f"packed feature shape {x.shape} != (F,129)"
    return x


@torch.no_grad()
def aa129_to_6d258(x_aa: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    (F,129) axis-angle → (F,258) 6D on given device.
    """
    assert x_aa.ndim == 2 and x_aa.shape[1] == 129, f"expected (F,129), got {x_aa.shape}"
    tensor = torch.from_numpy(x_aa).to(device=device, dtype=torch.float32)  # (F,129)
    batch = tensor.unsqueeze(0).contiguous()                                 # (1,F,129)
    x6d = convert_axis_angle_to_rotation_6d(batch)                           # (1,F,258)
    x6d = x6d.squeeze(0).contiguous().cpu().numpy().astype(np.float32)      # (F,258)
    assert x6d.ndim == 2 and x6d.shape[1] == 258, f"6D shape {x6d.shape} != (F,258)"
    return x6d


def precompute_6d_rotation(
    data_dir: str = 'data/motion',
    output_dir: str = 'data/6d_rotation/',
    pattern: str = '*.npz',
    recursive: bool = False,
    device: str = None,
    overwrite: bool = False
):
    """
    Convert ALL frames in each NPZ to 6D and save as .npy

    Args:
        data_dir : directory of NPZ files
        output_dir : where to save NPY 6D files
        pattern : glob pattern (default: '*.npz')
        recursive : whether to search subdirectories
        device : 'cuda' or 'cpu' (default: 'cuda' if available)
        overwrite : overwrite existing outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    glob_pattern = os.path.join(data_dir, '**', pattern) if recursive else os.path.join(data_dir, pattern)
    motion_files = sorted(glob.glob(glob_pattern, recursive=recursive))
    print(f"Found {len(motion_files)} motion files (recursive={recursive})")

    for motion_path in tqdm(motion_files, desc="Converting to 6D rotation"):
        try:
            base = os.path.basename(motion_path)
            out_name = base.replace('.npz', '_6d.npy')
            out_path = os.path.join(output_dir, out_name)

            if (not overwrite) and os.path.exists(out_path):
                # quick shape check
                try:
                    arr = np.load(out_path, mmap_mode='r')
                    if arr.ndim == 2 and arr.shape[1] == 258:
                        continue
                except Exception:
                    pass  # fall through to recompute

            # Load NPZ (readonly)
            data = np.load(motion_path, allow_pickle=False, mmap_mode='r')

            # 1) Pack to (F,129) with EXACT same joints & order as loader
            x_aa = pack_upper_hands_axis_angle(data)  # (F,129)

            # 2) All frames -> 6D (F,258)
            x_6d = aa129_to_6d258(x_aa, device=device)

            # 3) Save as .npy
            np.save(out_path, x_6d)
        except Exception as e:
            print(f"\n[!] Error processing {motion_path}: {e}")
            import traceback; traceback.print_exc()
            continue

    print(f"\nPrecomputation complete! Files saved to: {output_dir}")
    print("   Original (axis-angle): (F,129)  [upper13 + Lhand15 + Rhand15]")
    print("   Converted (6D rot)   : (F,258)  [same order × 2]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute 6D rotation representation (aligned with loader)")
    parser.add_argument("--data_dir", type=str, default="/mnt/nas2/Projects/meta_data/naturalistic/test/motion", help="Input directory containing .npz")
    parser.add_argument("--output_dir", type=str, default="data/test_motion_6d_rotation/", help="Output directory for .npy")
    parser.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern for input files")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    parser.add_argument("--device", type=str, default="cpu", choices=[None, "cpu", "cuda"], help="Force device")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    precompute_6d_rotation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        device=args.device,
        overwrite=args.overwrite,
    )
