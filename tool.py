#!/usr/bin/env python3
"""
Tool functions for motion denormalization and audio conversion
"""

import numpy as np
import torch
import soundfile as sf
import os
import matplotlib.pyplot as plt
import sys

# Add the parent directory to sys.path to import motion_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.motion_utils import convert_rotation_6d_to_axis_angle


def denormalize_motion(motion_tensor, motion_mean, motion_std, is_6d_rotation=False):
    """
    Denormalize motion data: motion * std + mean
    
    Args:
        motion_tensor: Normalized motion tensor [batch, frames, dim] or [frames, dim]
        motion_mean: Mean for denormalization (dim,)
        motion_std: Std for denormalization (dim,)
        is_6d_rotation: Whether the motion data is in 6D rotation format (258 dims)
        
    Returns:
        denormalized_motion: Denormalized motion tensor
    """
    if motion_mean is None or motion_std is None:
        print("Warning: Cannot denormalize - mean or std is None")
        return motion_tensor
    
    # Convert numpy arrays to torch tensors if needed
    if isinstance(motion_mean, np.ndarray):
        motion_mean = torch.from_numpy(motion_mean).float()
    if isinstance(motion_std, np.ndarray):
        motion_std = torch.from_numpy(motion_std).float()
    
    # Move to same device as motion_tensor
    motion_mean = motion_mean.to(motion_tensor.device)
    motion_std = motion_std.to(motion_tensor.device)
    
    # Apply denormalization: motion * std + mean
    denormalized_motion = motion_tensor * motion_std + motion_mean
    
    print(f"  Denormalized motion: {motion_tensor.shape} -> {denormalized_motion.shape}")
    print(f"  Mean range: [{motion_mean.min():.6f}, {motion_mean.max():.6f}]")
    print(f"  Std range: [{motion_std.min():.6f}, {motion_std.max():.6f}]")
    
    if is_6d_rotation:
        print(f"  6D rotation data detected (dim={motion_tensor.shape[-1]})")
    
    return denormalized_motion


def reconstruct_motion_npz(motion_data, upper_body_joints=None, is_6d_rotation=False):
    """
    Reconstruct motion data from flattened format back to NPZ structure with all SMPL keys.
    
    Args:
        motion_data: Flattened motion data of shape (frames, 129) or (frames, 258) for 6D rotation
        upper_body_joints: List of upper body joint indices (default: [2,5,8,11,12,13,14,15,16,17,18,19,20])
        is_6d_rotation: Whether the motion data is in 6D rotation format (258 dims)
    
    Returns:
        dict: NPZ-compatible motion data with all SMPL keys:
            - 'smplh:body_pose': (frames, 21, 3) - full body pose
            - 'smplh:left_hand_pose': (frames, 15, 3) - left hand pose  
            - 'smplh:right_hand_pose': (frames, 15, 3) - right hand pose
            - 'smplh:global_orient': (frames, 3) - global orientation
            - 'smplh:translation': (frames, 3) - translation
            - 'smplh:is_valid': (frames,) - validity flag
    """
    if upper_body_joints is None:
        # Default upper body joints (excluding lower body: [0,1,3,4,6,7,9,10])
        # Upper body: [2,5,8,11,12,13,14,15,16,17,18,19,20] (spine, neck, collar, head, shoulder, elbow, wrist)
        upper_body_joints = [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    frames = motion_data.shape[0]
    
    # Convert 6D rotation to axis-angle if needed
    if is_6d_rotation:
        print(f"Converting 6D rotation to axis-angle format...")
        # Convert to torch tensor for conversion
        motion_tensor = torch.from_numpy(motion_data).float()
        motion_batch = motion_tensor.unsqueeze(0)  # (1, frames, 258)
        
        # Convert 6D rotation to axis-angle: (1, frames, 258) -> (1, frames, 129)
        motion_axis_angle = convert_rotation_6d_to_axis_angle(motion_batch)
        motion_axis_angle = motion_axis_angle.squeeze(0)  # (frames, 129)
        motion_data = motion_axis_angle.cpu().detach().numpy()
        
        print(f"Converted 6D rotation to axis-angle: {motion_tensor.shape} -> {motion_axis_angle.shape}")
    
    # Initialize full body pose (21 joints, 3D each) - lower body will remain zero
    full_body_pose = np.zeros((frames, 21, 3))
    
    # Extract upper body pose (first 39 dimensions: 13 joints * 3)
    upper_body_flat = motion_data[:, :39]  # (frames, 39)
    upper_body_pose = upper_body_flat.reshape(frames, 13, 3)  # (frames, 13, 3)
    
    # Place upper body joints in full body pose
    for i, joint_idx in enumerate(upper_body_joints):
        if joint_idx < 21:  # Ensure joint index is valid
            full_body_pose[:, joint_idx, :] = upper_body_pose[:, i, :]
    
    # Extract left hand pose (next 45 dimensions: 15 joints * 3)
    left_hand_flat = motion_data[:, 39:84]  # (frames, 45)
    left_hand_pose = left_hand_flat.reshape(frames, 15, 3)  # (frames, 15, 3)
    
    # Extract right hand pose (last 45 dimensions: 15 joints * 3)
    right_hand_flat = motion_data[:, 84:129]  # (frames, 45)
    right_hand_pose = right_hand_flat.reshape(frames, 15, 3)  # (frames, 15, 3)
    
    # Add missing SMPL keys with default values
    # Global orientation (identity rotation)
    global_orient = np.zeros((frames, 3))
    
    # Translation (zero translation)
    translation = np.zeros((frames, 3))
    
    # Validity flag (all frames are valid)
    is_valid = np.ones((frames,), dtype=bool)
    
    return {
        'smplh:body_pose': full_body_pose,
        'smplh:left_hand_pose': left_hand_pose,
        'smplh:right_hand_pose': right_hand_pose,
        'smplh:global_orient': global_orient,
        'smplh:translation': translation,
        'smplh:is_valid': is_valid
    }


def save_audio_as_wav(audio_tensor, output_path, sample_rate=16000, audio_right=None):
    """
    Save audio tensor as WAV file
    
    Args:
        audio_tensor: Audio tensor [B, T] or [T] or [B, C, T] (left channel if audio_right is provided)
        output_path: Path to save WAV file
        sample_rate: Sample rate for WAV file (default: 16000)
        audio_right: Optional right channel audio tensor [B, T] or [T] or [B, C, T]
                     If provided, saves as stereo WAV with audio_tensor as left channel
    """
    def extract_audio_numpy(audio_tensor):
        """Extract audio numpy array from tensor"""
        if isinstance(audio_tensor, torch.Tensor):
            if audio_tensor.dim() == 3:  # [B, C, T] or [B, T, C]
                # Check if last dimension is channels (typically 2 for stereo)
                if audio_tensor.shape[2] <= 2:  # [B, T, C] where C is 1 or 2
                    audio = audio_tensor[0, :, 0].cpu().detach().numpy()  # Take first sample, first channel
                else:  # [B, C, T]
                    audio = audio_tensor[0, 0].cpu().detach().numpy()  # Take first sample, first channel
            elif audio_tensor.dim() == 2:  # [B, T] or [T, C]
                # Check if second dimension is channels
                if audio_tensor.shape[1] <= 2:  # [T, C] where C is 1 or 2
                    audio = audio_tensor[:, 0].cpu().detach().numpy()  # Take first channel
                else:  # [B, T]
                    audio = audio_tensor[0].cpu().detach().numpy()  # Take first sample
            else:  # [T]
                audio = audio_tensor.cpu().detach().numpy()
        else:
            # Already numpy array
            if audio_tensor.ndim == 3:  # [B, T, C] or [B, C, T]
                # Check if last dimension is channels
                if audio_tensor.shape[2] <= 2:  # [B, T, C] where C is 1 or 2
                    audio = audio_tensor[0, :, 0] if audio_tensor.shape[0] == 1 else audio_tensor[:, :, 0]
                else:  # [B, C, T]
                    audio = audio_tensor[0, 0] if audio_tensor.shape[0] == 1 else audio_tensor[0]
            elif audio_tensor.ndim == 2:  # [B, T] or [T, C]
                # Check if second dimension is channels
                if audio_tensor.shape[1] <= 2:  # [T, C] where C is 1 or 2
                    audio = audio_tensor[:, 0]  # Take first channel
                else:  # [B, T]
                    audio = audio_tensor[0] if audio_tensor.shape[0] == 1 else audio_tensor
            elif audio_tensor.ndim == 1:  # [T]
                audio = audio_tensor
            else:
                audio = audio_tensor
        return audio
    
    # Extract left channel audio
    audio_left = extract_audio_numpy(audio_tensor)
    
    # If audio_right is provided, create stereo audio
    if audio_right is not None:
        audio_right_np = extract_audio_numpy(audio_right)
        
        # Ensure both channels have the same length
        min_len = min(len(audio_left), len(audio_right_np))
        audio_left = audio_left[:min_len]
        audio_right_np = audio_right_np[:min_len]
        
        # Stack to create stereo: [T, 2] where column 0 is left, column 1 is right
        audio_stereo = np.stack([audio_left, audio_right_np], axis=1)
        
        # Ensure audio is in the correct range for WAV format (-1.0 to 1.0)
        audio_stereo = np.clip(audio_stereo, -1.0, 1.0)
        
        # Convert to float32 format (soundfile expects float32 for WAV)
        audio_stereo = audio_stereo.astype(np.float32)
        
        # Save as stereo WAV file
        sf.write(output_path, audio_stereo, sample_rate)
        print(f"Saved stereo audio as WAV: {output_path}")
        print(f"   Audio shape: {audio_stereo.shape} (stereo), Sample rate: {sample_rate}Hz")
        print(f"   Duration: {len(audio_stereo) / sample_rate:.2f}s")
    else:
        # Mono audio
        # Ensure audio is in the correct range for WAV format (-1.0 to 1.0)
        audio_left = np.clip(audio_left, -1.0, 1.0)
        
        # Convert to float32 format (soundfile expects float32 for WAV)
        audio_left = audio_left.astype(np.float32)
        
        # Save as WAV file
        sf.write(output_path, audio_left, sample_rate)
        print(f"Saved audio as WAV: {output_path}")
        print(f"   Audio shape: {audio_left.shape}, Sample rate: {sample_rate}Hz")
        print(f"   Duration: {len(audio_left) / sample_rate:.2f}s")


def save_motion_and_audio_outputs(
    motion_denorm,
    gt_motion_denorm,
    audio,
    batch,
    config,
    is_6d_rotation=False,
    motion_other_denorm=None,
    audio_other=None,
    output_prefix=None,
    output_dir=None,
):
    """
    Save denormalized motion as NPZ files and audio as WAV files
    
    Args:
        motion_denorm: Denormalized generated motion tensor
        gt_motion_denorm: Denormalized target motion tensor  
        audio: Audio tensor
        batch: Data batch containing metadata
        config: Configuration object
        is_6d_rotation: Whether the motion data is in 6D rotation format (258 dims)
        motion_other_denorm: Denormalized other motion tensor (optional)
        audio_other: Other audio tensor (optional)
        output_prefix: Optional subdirectory name suffix for organizing outputs
    """
    # Create output directories
    if output_dir:
        npz_output_dir = output_dir
        audio_output_dir = output_dir
    else:
        npz_output_dir = "motion_outputs"
        audio_output_dir = "audio_outputs"

        if output_prefix:
            npz_output_dir = os.path.join(npz_output_dir, output_prefix)
            audio_output_dir = os.path.join(audio_output_dir, output_prefix)

    os.makedirs(npz_output_dir, exist_ok=True)
    if audio_output_dir != npz_output_dir:
        os.makedirs(audio_output_dir, exist_ok=True)
    
    # Get audio sample rate from config
    audio_sr = config.model.audio_sr if hasattr(config, 'model') and hasattr(config.model, 'audio_sr') else 16000
    
    # Convert to numpy and remove batch dimension
    motion_denorm_np = motion_denorm[0].cpu().detach().numpy()  # [frames, 129] or [frames, 258]
    gt_motion_denorm_np = gt_motion_denorm[0].cpu().detach().numpy()  # [frames, 129] or [frames, 258]
    
    print(f"\nSaving denormalized motion as NPZ files...")
    print(f"Motion data shape: {motion_denorm_np.shape}")
    print(f"6D rotation format: {is_6d_rotation}")
    
    # Save motion as NPZ files
    try:
        generated_npz = reconstruct_motion_npz(motion_denorm_np, is_6d_rotation=is_6d_rotation)
        np.savez_compressed(os.path.join(npz_output_dir, "generated_motion_denorm.npz"), **generated_npz)
        print(f"Saved denormalized generated motion as NPZ: {list(generated_npz.keys())}")
    except Exception as e:
        print(f"Failed to save generated motion as NPZ: {e}")
    
    try:
        target_npz = reconstruct_motion_npz(gt_motion_denorm_np, is_6d_rotation=is_6d_rotation)
        np.savez_compressed(os.path.join(npz_output_dir, "target_motion_denorm.npz"), **target_npz)
        print(f"Saved denormalized target motion as NPZ: {list(target_npz.keys())}")
    except Exception as e:
        print(f"Failed to save target motion as NPZ: {e}")
    
    # Save motion_other_denorm if provided
    if motion_other_denorm is not None:
        print(f"\nSaving denormalized motion_other as NPZ files...")
        motion_other_denorm_np = motion_other_denorm[0].cpu().detach().numpy()  # [frames, 129] or [frames, 258]
        print(f"Motion_other data shape: {motion_other_denorm_np.shape}")
        
        try:
            target_other_npz = reconstruct_motion_npz(motion_other_denorm_np, is_6d_rotation=is_6d_rotation)
            np.savez_compressed(os.path.join(npz_output_dir, "target_motion_other_denorm.npz"), **target_other_npz)
            print(f"Saved denormalized target_motion_other as NPZ: {list(target_other_npz.keys())}")
        except Exception as e:
            print(f"Failed to save target_motion_other as NPZ: {e}")
        
        # Also save as numpy array
        np.save(os.path.join(npz_output_dir, "target_motion_other_denorm.npy"), motion_other_denorm_np)
        print(f"Saved denormalized motion_other as numpy array")
    
    # Save input audio as WAV file
    print(f"\nSaving input audio as WAV file...")
    
    # Save input audio as WAV
    save_audio_as_wav(
        audio, 
        os.path.join(audio_output_dir, "input_audio.wav"),
        sample_rate=audio_sr
    )
    
    # Save audio_other if provided
    if audio_other is not None:
        print(f"\nSaving audio_other as WAV file...")
        save_audio_as_wav(
            audio_other,
            os.path.join(audio_output_dir, "input_audio_other.wav"),
            sample_rate=audio_sr
        )
    
    # Also save the original audio from NPZ if available
    if hasattr(batch, 'audio_path') or (isinstance(batch, dict) and 'audio_path' in batch):
        audio_path = batch['audio_path'] if isinstance(batch, dict) else batch.audio_path
        # Handle case where audio_path might be a list
        if isinstance(audio_path, list):
            audio_path = audio_path[0]  # Take the first path
        print(f"Original audio NPZ path: {audio_path}")
        
        # Load original audio from NPZ and save as WAV
        try:
            audio_data = np.load(audio_path, allow_pickle=True)
            audio_keys = ['audio', 'audio_data', 'signal', 'waveform']
            original_audio = None
            
            for key in audio_keys:
                if key in audio_data:
                    original_audio = audio_data[key]
                    break
            
            if original_audio is None:
                keys = list(audio_data.keys())
                if keys:
                    original_audio = audio_data[keys[0]]
            
            if original_audio is not None:
                # Ensure audio is 1D
                if len(original_audio.shape) > 1:
                    original_audio = original_audio.flatten()
                
                # Save original full audio as WAV
                original_audio_clipped = np.clip(original_audio, -1.0, 1.0)
                original_audio_int16 = (original_audio_clipped * 32767).astype(np.int16)
                sf.write(os.path.join(audio_output_dir, "original_full_audio.wav"), original_audio_int16, audio_sr)
                print(f"Saved original full audio as WAV: {original_audio.shape}")
                print(f"   Duration: {len(original_audio) / audio_sr:.2f}s")
            else:
                print("Could not extract original audio from NPZ")
        except Exception as e:
            print(f"Failed to load original audio from NPZ: {e}")
    else:
        print("No audio_path found in batch, skipping original audio extraction")


def apply_denormalization(motion, gt_motion, dataset):
    """
    Apply denormalization to motion data if normalization was used during training
    
    Args:
        motion: Generated motion tensor
        gt_motion: Ground truth motion tensor
        dataset: Dataset object containing normalization parameters
        
    Returns:
        tuple: (motion_denorm, gt_motion_denorm, is_6d_rotation)
    """
    # Get normalization stats from dataset
    motion_mean = dataset.motion_mean if hasattr(dataset, 'motion_mean') else None
    motion_std = dataset.motion_std if hasattr(dataset, 'motion_std') else None
    normalize_motion = dataset.normalize_motion if hasattr(dataset, 'normalize_motion') else False
    
    # Detect if motion data is in 6D rotation format
    is_6d_rotation = motion.shape[-1] == 258  # 6D rotation has 258 dimensions
    
    print(f"Normalization enabled: {normalize_motion}")
    print(f"6D rotation format: {is_6d_rotation}")
    if normalize_motion and motion_mean is not None and motion_std is not None:
        print(f"Motion mean shape: {motion_mean.shape}")
        print(f"Motion std shape: {motion_std.shape}")
    
    # Apply denormalization if normalization was used during training
    motion_denorm = None
    gt_motion_denorm = None
    
    if normalize_motion and motion_mean is not None and motion_std is not None:
        print(f"\nApplying denormalization to predicted motion...")
        motion_denorm = denormalize_motion(motion, motion_mean, motion_std, is_6d_rotation=is_6d_rotation)
        
        print(f"Applying denormalization to target motion...")
        gt_motion_denorm = denormalize_motion(gt_motion, motion_mean, motion_std, is_6d_rotation=is_6d_rotation)
        print(f"Target motion denormalization completed")
    else:
        print(f"\nNo denormalization applied (normalize_motion={normalize_motion}, mean/std available: {motion_mean is not None and motion_std is not None})")
        motion_denorm = motion
        gt_motion_denorm = gt_motion
    
    return motion_denorm, gt_motion_denorm, is_6d_rotation


def visualize_motion_comparison(motion_pred, motion_gt, audio=None, save_path=None, title="Motion Comparison"):
    """
    Visualize generated motion vs ground truth motion with optional audio waveform background
    
    Args:
        motion_pred: Generated motion tensor [B, T, pose_dim] or [T, pose_dim]
        motion_gt: Ground truth motion tensor [B, T, pose_dim] or [T, pose_dim]
        audio: Audio tensor [B, T] or [B, C, T] for background waveform (optional)
        save_path: Path to save the plot (optional)
        title: Title for the plot
    """
    # Convert to numpy and handle batch dimension
    if motion_pred.dim() == 3:  # [B, T, pose_dim]
        motion_pred = motion_pred[0].cpu().detach().numpy()  # Take first sample
    else:  # [T, pose_dim]
        motion_pred = motion_pred.cpu().detach().numpy()
        
    if motion_gt.dim() == 3:  # [B, T, pose_dim]
        motion_gt = motion_gt[0].cpu().detach().numpy()  # Take first sample
    else:  # [T, pose_dim]
        motion_gt = motion_gt.cpu().detach().numpy()
    
    T, pose_dim = motion_pred.shape
    
    # Process audio for background if provided
    audio_bg = None
    if audio is not None:
        if audio.dim() == 3:  # [B, C, T]
            audio_bg = audio[0, 0].cpu().detach().numpy()  # Take first sample, first channel
        elif audio.dim() == 2:  # [B, T]
            audio_bg = audio[0].cpu().detach().numpy()  # Take first sample
        else:  # [T]
            audio_bg = audio.cpu().detach().numpy()
        
        # Keep original audio range for dual y-axis visualization
        # audio_bg will use its own y-axis scale
        
        # Resample audio to match motion length if needed
        if len(audio_bg) != T:
            audio_bg = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(audio_bg)), audio_bg)
    
    # Create subplots for different pose components
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # Plot different groups of pose dimensions
    # Check if data is in 6D rotation format (258 dims) or axis-angle format (129 dims)
    if pose_dim == 258:
        # 6D rotation format: 43 joints * 6 dimensions = 258
        # Group by joint types (6 dimensions per joint)
        pose_groups = [
            (0, 6, "Root Position"),       # Root position (6D rotation)
            (6, 12, "Root Rotation"),      # Root rotation (6D rotation)
            (12, 18, "Hip Joints"),        # Hip joints (6D rotation)
            (18, 24, "Knee Joints"),       # Knee joints (6D rotation)
            (24, 30, "Ankle Joints"),      # Ankle joints (6D rotation)
            (30, 36, "Spine"),             # Spine (6D rotation)
            (36, 42, "Shoulder"),          # Shoulder (6D rotation)
            (42, 48, "Elbow"),             # Elbow (6D rotation)
            (48, 54, "Wrist")              # Wrist (6D rotation)
        ]
    else:
        # Axis-angle format: 43 joints * 3 dimensions = 129
        pose_groups = [
            (0, 3, "Root Position"),      # Root position (x, y, z)
            (3, 6, "Root Rotation"),      # Root rotation (rx, ry, rz)
            (6, 9, "Hip Joints"),         # Hip joints
            (9, 12, "Knee Joints"),       # Knee joints
            (12, 15, "Ankle Joints"),     # Ankle joints
            (15, 18, "Spine"),            # Spine
            (18, 21, "Shoulder"),         # Shoulder
            (21, 24, "Elbow"),            # Elbow
            (24, 27, "Wrist")             # Wrist
        ]
    
    for i, (start_idx, end_idx, group_name) in enumerate(pose_groups):
        if i >= 9:  # Only plot first 9 groups
            break
            
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Plot audio waveform with dual y-axis for time alignment if available
        if audio_bg is not None:
            # Create secondary y-axis for audio
            ax2 = ax.twinx()
            ax2.plot(audio_bg, color='gray', alpha=0.4, linewidth=0.5, label='Audio Waveform')
            # Set audio y-axis limits to be smaller and at the bottom
            audio_min, audio_max = audio_bg.min(), audio_bg.max()
            audio_range = audio_max - audio_min
            ax2.set_ylim(audio_min - audio_range * 0.1, audio_max + audio_range * 0.1)
            ax2.set_ylabel('Audio', color='gray', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
            # Hide the secondary y-axis spine to reduce clutter
            ax2.spines['right'].set_visible(False)
        
        # Plot each dimension in the group with same color for same dimension
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for j in range(start_idx, min(end_idx, pose_dim)):
            dim_idx = j - start_idx  # Index within the joint group
            color = colors[dim_idx % len(colors)]  # Use same color for same dimension
            
            ax.plot(motion_gt[:, j], color=color, label=f'GT dim{j}', alpha=0.7, linewidth=1)
            ax.plot(motion_pred[:, j], '--', color=color, label=f'Pred dim{j}', alpha=0.7, linewidth=1)
        
        ax.set_title(group_name)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(pose_groups), 9):
        row, col = i // 3, i % 3
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Motion visualization saved to: {save_path}")
    
    plt.show()


def visualize_motion_heatmap(motion_pred, motion_gt, save_path=None, title="Motion Heatmap Comparison"):
    """
    Create heatmap visualization of motion data
    
    Args:
        motion_pred: Generated motion tensor [B, T, pose_dim] or [T, pose_dim]
        motion_gt: Ground truth motion tensor [B, T, pose_dim] or [T, pose_dim]
        save_path: Path to save the plot (optional)
        title: Title for the plot
    """
    # Convert to numpy and handle batch dimension
    if motion_pred.dim() == 3:  # [B, T, pose_dim]
        motion_pred = motion_pred[0].cpu().detach().numpy()  # Take first sample
    else:  # [T, pose_dim]
        motion_pred = motion_pred.cpu().detach().numpy()
        
    if motion_gt.dim() == 3:  # [B, T, pose_dim]
        motion_gt = motion_gt[0].cpu().detach().numpy()  # Take first sample
    else:  # [T, pose_dim]
        motion_gt = motion_gt.cpu().detach().numpy()
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)
    
    # Ground truth heatmap
    im1 = ax1.imshow(motion_gt.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title('Ground Truth Motion')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Pose Dimension')
    plt.colorbar(im1, ax=ax1)
    
    # Predicted motion heatmap
    im2 = ax2.imshow(motion_pred.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_title('Predicted Motion')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Pose Dimension')
    plt.colorbar(im2, ax=ax2)
    
    # Difference heatmap
    diff = motion_pred - motion_gt
    im3 = ax3.imshow(diff.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax3.set_title('Difference (Pred - GT)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Pose Dimension')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Motion heatmap saved to: {save_path}")
    
    plt.show()


def visualize_audio_waveform(audio, save_path=None, title="Audio Waveform"):
    """
    Visualize audio waveform
    
    Args:
        audio: Audio tensor [B, T] or [B, C, T]
        save_path: Path to save the plot (optional)
        title: Title for the plot
    """
    # Convert to numpy and handle batch dimension
    if audio.dim() == 3:  # [B, C, T]
        audio = audio[0, 0].cpu().detach().numpy()  # Take first sample, first channel
    elif audio.dim() == 2:  # [B, T]
        audio = audio[0].cpu().detach().numpy()  # Take first sample
    else:  # [T]
        audio = audio.cpu().detach().numpy()
    
    # Create time axis
    time_axis = np.linspace(0, len(audio) / 16000, len(audio))  # Assuming 16kHz sample rate
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    fig.suptitle(title, fontsize=16)
    
    # Plot waveform
    ax.plot(time_axis, audio, linewidth=0.9, alpha=1)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform')
    ax.grid(True, alpha=1)
    
    # Add some statistics
    ax.text(0.02, 0.95, f'Duration: {time_axis[-1]:.2f}s\nSamples: {len(audio)}\nMax: {audio.max():.3f}\nMin: {audio.min():.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Audio waveform saved to: {save_path}")
    
    plt.show()
