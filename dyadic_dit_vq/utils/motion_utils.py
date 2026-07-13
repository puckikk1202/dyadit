"""
Motion Utilities for Rotation Conversions

This module provides utility functions for converting between different rotation representations
used in motion data processing. These functions should be used to preprocess data before
feeding into the Motion VQVAE model.
"""

import torch
import torch.nn.functional as F


def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def quaternion_to_axis_angle(quaternions):
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles]*angles[small_angles])/48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def rotation_6d_to_axis_angle(rot6d):
    return matrix_to_axis_angle(rotation_6d_to_matrix(rot6d))

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def axis_angle_to_rotation_6d(axis_angle):
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def convert_axis_angle_to_rotation_6d(axis_angle_motion):
    """
    Convert axis-angle motion to rotation 6d format for model input
    
    Args:
        axis_angle_motion: [batch_size, seq_len, motion_dim] - Motion in axis-angle format (165D)
        
    Returns:
        rot6d_motion: [batch_size, seq_len, rot6d_dim] - Motion in rotation 6d format (330D)
    """
    batch_size, seq_len, motion_dim = axis_angle_motion.shape
    n_joints = motion_dim // 3
    
    # Reshape to [batch_size, seq_len, n_joints, 3]
    axis_angle = axis_angle_motion.reshape(batch_size, seq_len, n_joints, 3)
    
    # Convert to rotation 6d: [batch_size, seq_len, n_joints, 6]
    rot6d = axis_angle_to_rotation_6d(axis_angle)
    
    # Reshape back: [batch_size, seq_len, n_joints * 6]
    rot6d_motion = rot6d.reshape(batch_size, seq_len, n_joints * 6)
    
    return rot6d_motion


def convert_rotation_6d_to_axis_angle(rot6d_motion):
    """
    Convert rotation 6d motion back to axis-angle format
    
    Args:
        rot6d_motion: [batch_size, seq_len, rot6d_dim] - Motion in rotation 6d format (330D)
        
    Returns:
        axis_angle_motion: [batch_size, seq_len, motion_dim] - Motion in axis-angle format (165D)
    """
    batch_size, seq_len, rot6d_dim = rot6d_motion.shape
    n_joints = rot6d_dim // 6
    
    # Reshape to [batch_size, seq_len, n_joints, 6]
    rot6d = rot6d_motion.reshape(batch_size, seq_len, n_joints, 6)
    
    # Convert to axis-angle: [batch_size, seq_len, n_joints, 3]
    axis_angle = rotation_6d_to_axis_angle(rot6d)
    
    # Reshape back: [batch_size, seq_len, n_joints * 3]
    axis_angle_motion = axis_angle.reshape(batch_size, seq_len, n_joints * 3)
    
    return axis_angle_motion
