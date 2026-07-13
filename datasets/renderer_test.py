#!/usr/bin/env python3
"""
SMPL-H Motion Renderer

Usage examples:
  # Test mode - render frames 0, 10, 20 from default file
  python renderer_test.py --mode test
  
  # Test mode - render frames 0, 10, 20 from specific file
  python renderer_test.py --mode test --npz_file /path/to/your/file.npz
  
  # Full mode - render first 300 frames (default)
  python renderer_test.py --mode full --npz_file /path/to/your/file.npz
  
  # Full mode - render all frames
  python renderer_test.py --mode full --npz_file /path/to/your/file.npz --frame_num 0 --max_frames 1000 --step 5
  
  # Full mode - render first 500 frames with video creation
  python renderer_test.py --mode full --output_dir my_output --frame_num 500 --create_video --fps 30
  
  # Full mode with video and audio (NEW)
  python renderer_test.py --mode full --npz_file motion.npz --max_frames 500 --create_video --fps 30 --audio output.wav
  
  # Comparison mode - render original vs reconstructed side by side
  python renderer_test.py --mode full --comparison --npz_file original.npz --reconstructed_file reconstructed.npz --create_video
  
  # Comparison mode with audio (NEW)
  python renderer_test.py --mode full --comparison --npz_file original.npz --reconstructed_file reconstructed.npz --output_dir comparison_output --max_frames 200 --step 2 --create_video --fps 24 --audio audio.wav
  
  # Use MANO hands_mean during rendering (loads from npy files)
  python renderer_test.py --mode full --npz_file motion.npz --use_hand_mean --create_video
"""

import os
# Enable headless rendering - use software rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Use software rendering to avoid GPU permission issues
import numpy as np
import torch, smplx, trimesh, pyrender, imageio
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='SMPL/H/X Motion Renderer')
    parser.add_argument('--npz_file', type=str, 
                       default='/home/peng/test_converted.npz',
                       help='Path to NPZ file to render')
    parser.add_argument('--model_type', type=str, choices=['smpl', 'smplh', 'smplx'], 
                       default='smplh',
                       help='Model type: smpl, smplh (with hands), or smplx (with hands and face)')
    parser.add_argument('--mode', type=str, choices=['test', 'full'], 
                       default='test',
                       help='Rendering mode: test (frames 0,10,20) or full (all frames)')
    parser.add_argument('--output_dir', type=str, default='frames',
                       help='Output directory for rendered frames')
    parser.add_argument('--max_frames', type=int, default=2000,
                       help='Maximum frames to render in full mode')
    parser.add_argument('--step', type=int, default=1,
                       help='Frame step size in full mode')
    parser.add_argument('--comparison', action='store_true',
                       help='Enable comparison mode (original vs reconstructed). Only works with --mode full. Requires both original and reconstructed NPZ files.')
    parser.add_argument('--reconstructed_file', type=str, default=None,
                       help='Path to reconstructed NPZ file for comparison mode')
    parser.add_argument('--create_video', action='store_true',
                       help='Create MP4 video from rendered frames')
    parser.add_argument('--fps', type=int, default=30,
                       help='FPS for output video')
    parser.add_argument('--audio', type=str, default=None,
                       help='Path to audio file to merge with video (e.g., output.wav)')
    parser.add_argument('--frame_num', type=int, default=300,
                       help='Number of frames to render (default: 300, 0 means all frames)')
    parser.add_argument('--use_hand_mean', action='store_true',
                       help='Add MANO hands_mean to hand poses during rendering')
    return parser.parse_args()


# Parse command line arguments
args = parse_args()

# Load MANO hands_mean if requested
left_hand_mean = None
right_hand_mean = None
if args.use_hand_mean:
    print("Loading MANO hands_mean from npy files...")
    try:
        # Load from npy files (no chumpy dependency needed)
        left_hand_mean = np.load('./datasets/mano_hands/LEFT_hands_mean.npy')
        right_hand_mean = np.load('./datasets/mano_hands/RIGHT_hands_mean.npy')
        
        print(f"  Left hand mean shape: {left_hand_mean.shape}")
        print(f"  Right hand mean shape: {right_hand_mean.shape}")
        print(f"  Left hand mean range: [{left_hand_mean.min():.4f}, {left_hand_mean.max():.4f}]")
        print(f"  Right hand mean range: [{right_hand_mean.min():.4f}, {right_hand_mean.max():.4f}]")
        print("MANO hands_mean loaded successfully from npy files!")
        
    except FileNotFoundError as e:
        print(f"Warning: Could not find npy files: {e}")
        print("  Expected files:")
        print("    /home/peng/SMPLest-X/human_models/LEFT_hands_mean.npy")
        print("    /home/peng/SMPLest-X/human_models/RIGHT_hands_mean.npy")
        print("  Continuing without hands_mean...")
        left_hand_mean = None
        right_hand_mean = None
    except Exception as e:
        print(f"Warning: Failed to load MANO hands_mean: {e}")
        print("  Continuing without hands_mean...")
        left_hand_mean = None
        right_hand_mean = None

# Load data from specified file
data_file = args.npz_file
print(f"Loading data from: {data_file}")
data = np.load(data_file)

# Check available keys and show file info
print(f"Available keys in NPZ file:")
for key in sorted(data.keys()):
    if hasattr(data[key], 'shape'):
        print(f"   {key}: {data[key].shape}")
    else:
        print(f"   {key}: {type(data[key])}")

# Check required keys based on model type
print(f"\nModel type: {args.model_type.upper()}")
if args.model_type == 'smplx':
    # SMPL-X requires body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, global_orient
    required_keys = [f'{args.model_type}:body_pose', f'{args.model_type}:global_orient']
    optional_but_important = [f'{args.model_type}:left_hand_pose', f'{args.model_type}:right_hand_pose', 
                              f'{args.model_type}:jaw_pose', f'{args.model_type}:expression']
elif args.model_type == 'smplh':
    # SMPL-H requires body_pose, left_hand_pose, right_hand_pose, global_orient
    required_keys = [f'{args.model_type}:body_pose', f'{args.model_type}:left_hand_pose', 
                    f'{args.model_type}:right_hand_pose', f'{args.model_type}:global_orient']
    optional_but_important = []
elif args.model_type == 'smpl':
    # SMPL requires body_pose, global_orient
    required_keys = [f'{args.model_type}:body_pose', f'{args.model_type}:global_orient']
    optional_but_important = []
else:
    required_keys = []

missing_keys = [key for key in required_keys if key not in data]
if missing_keys:
    raise KeyError(f"Missing required keys in NPZ file for {args.model_type}: {missing_keys}")

print(f"All required {args.model_type.upper()} keys found!")

# Warn about missing optional keys
missing_optional = [key for key in optional_but_important if key not in data]
if missing_optional:
    print(f"Warning: Missing optional keys: {missing_optional}")
    print(f"  These will be set to zero.")

# Handle comparison mode validation and data loading
reconstructed_data = None
if args.comparison:
    if args.mode != 'full':
        raise ValueError("Comparison mode (--comparison) only works with full mode (--mode full)")
    if not args.reconstructed_file:
        raise ValueError("Comparison mode requires --reconstructed_file argument")
    if not os.path.exists(args.reconstructed_file):
        raise FileNotFoundError(f"Reconstructed file not found: {args.reconstructed_file}")
    
    print(f"\nCOMPARISON MODE ENABLED")
    print(f"   Original file: {data_file}")
    print(f"   Reconstructed file: {args.reconstructed_file}")
    
    # Load reconstructed data
    reconstructed_data = np.load(args.reconstructed_file)
    print(f"Reconstructed file keys:")
    for key in sorted(reconstructed_data.keys()):
        if hasattr(reconstructed_data[key], 'shape'):
            print(f"   {key}: {reconstructed_data[key].shape}")
    
    # Check that both files have same sequence length
    body_pose_key = f"{args.model_type}:body_pose"
    orig_frames = data[body_pose_key].shape[0]
    recon_frames = reconstructed_data[body_pose_key].shape[0]
    if orig_frames != recon_frames:
        print(f"Warning: Frame count mismatch - Original: {orig_frames}, Reconstructed: {recon_frames}")
        total_frames_comparison = min(orig_frames, recon_frames)
        print(f"   Will render {total_frames_comparison} frames")
    else:
        total_frames_comparison = orig_frames
        print(f"Frame counts match: {total_frames_comparison}")

MODEL_DIR = "/home/peng/project/dyadic_motion/"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define lower body joint indices in body_pose (21 joints for SMPL/H/X)
# Based on previous analysis: [0, 1, 3, 4, 6, 7, 9, 10]
LOWER_BODY_JOINT_INDICES = [0, 1, 3, 4, 6, 7, 9, 10]  # hip, knee, ankle, foot joints
print(f"Lower body joint indices to zero out: {LOWER_BODY_JOINT_INDICES}")

def zero_out_lower_body(body_pose_data):
    """Set lower body joints to zero"""
    modified_data = body_pose_data.copy()
    for joint_idx in LOWER_BODY_JOINT_INDICES:
        modified_data[:, joint_idx, :] = 0.0  # Set all 3 rotation values to 0
    return modified_data

# Modify the body pose data to zero out lower body
body_pose_key = f"{args.model_type}:body_pose"
original_body_pose = data[body_pose_key]
# modified_body_pose = zero_out_lower_body(original_body_pose)
modified_body_pose = original_body_pose

print(f"Original body_pose shape: {original_body_pose.shape}")
print(f"Modified body_pose shape: {modified_body_pose.shape}")
print("Lower body joints have been set to (0,0,0)")


def to_torch(x):  # numpy -> torch float32 (1, D)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(DEVICE)

print(f"Loading {args.model_type.upper()} model...")
model = smplx.create(
    MODEL_DIR, model_type=args.model_type, gender="neutral",
    use_pca=False, num_expression_coeffs=10 if args.model_type == 'smplx' else None
).to(DEVICE)
print(f"{args.model_type.upper()} model loaded successfully!")

# Store model prefix for data access
MODEL_PREFIX = args.model_type

def render_one_frame(i, out_path="frame.png", betas=None, fixed_cam_pose=None, fixed_light_center=None,
                     left_hand_mean=None, right_hand_mean=None):
    print(f"Starting to render frame {i} to {out_path}")
    if betas is None:
        betas = torch.zeros(1, 10, device=DEVICE)

    # Show comparison between original and modified body pose for this frame (only for first few frames)
    if i < 3:  # Only show for first 3 frames to reduce output
        print(f"Frame {i} - Lower body joint comparison:")
        for joint_idx in LOWER_BODY_JOINT_INDICES:
            joint_names = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot']
            joint_name = joint_names[LOWER_BODY_JOINT_INDICES.index(joint_idx)]
            orig_val = original_body_pose[i, joint_idx, :]
            mod_val = modified_body_pose[i, joint_idx, :]
            print(f"  {joint_name} (joint {joint_idx}): {orig_val} -> {mod_val}")

    body_pose = to_torch(modified_body_pose[i].reshape(-1))  # (1, 21*3) - using modified data with lower body zeroed
    
    # Apply hands_mean if provided (for SMPL-H and SMPL-X)
    left_hand_pose = to_torch(np.zeros(45))  # Default: 15 joints * 3D
    right_hand_pose = to_torch(np.zeros(45))
    
    if MODEL_PREFIX in ['smplh', 'smplx'] and f"{MODEL_PREFIX}:left_hand_pose" in data:
        left_hand_data = data[f"{MODEL_PREFIX}:left_hand_pose"][i].reshape(-1).copy()
        right_hand_data = data[f"{MODEL_PREFIX}:right_hand_pose"][i].reshape(-1).copy()
        if left_hand_mean is not None:
            left_hand_data = left_hand_data - left_hand_mean
        if right_hand_mean is not None:
            right_hand_data = right_hand_data - right_hand_mean
        left_hand_pose = to_torch(left_hand_data)
        right_hand_pose = to_torch(right_hand_data)
    
    global_orient = to_torch(np.zeros(3))  # Set global orientation to zeros
    
    # Handle translation - use from file if available, otherwise use zeros
    if f"{MODEL_PREFIX}:translation" in data:
        transl = to_torch(np.zeros_like(data[f"{MODEL_PREFIX}:translation"][i]))
    else:
        transl = to_torch(np.zeros(3))

    # Handle SMPL-X specific parameters
    kwargs = {
        'betas': betas,
        'body_pose': body_pose,
        'global_orient': global_orient,
        'return_verts': True,
    }
    
    if MODEL_PREFIX in ['smplh', 'smplx']:
        kwargs['left_hand_pose'] = left_hand_pose
        kwargs['right_hand_pose'] = right_hand_pose
    
    if MODEL_PREFIX == 'smplx':
        if f"{MODEL_PREFIX}:expression" in data:
            kwargs['expression'] = to_torch(data[f"{MODEL_PREFIX}:expression"][i])
        if f"{MODEL_PREFIX}:jaw_pose" in data:
            kwargs['jaw_pose'] = to_torch(data[f"{MODEL_PREFIX}:jaw_pose"][i])
    
    kwargs['transl'] = transl
    
    out = model(**kwargs)
    verts = out.vertices[0].detach().cpu().numpy()
    faces = model.faces

    if i == 0:  # Only show detailed info for first frame
        print(f"Vertices shape: {verts.shape}")
        print(f"Vertices bounds: min={verts.min(axis=0)}, max={verts.max(axis=0)}")
        print(f"Model center: {verts.mean(axis=0)}")

 
    rot_x = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    rot_y = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    
    # 组合旋转：先X轴翻转，再Y轴转向
    rotation_matrix = rot_y @ rot_x
    
    # 应用旋转到顶点
    verts = verts @ rotation_matrix.T
    
    if i == 0:  # Only show detailed info for first frame
        print(f"After rotation - Vertices bounds: min={verts.min(axis=0)}, max={verts.max(axis=0)}")
        print(f"After rotation - Model center: {verts.mean(axis=0)}")


    mesh = trimesh.Trimesh(verts, faces, process=False)
    

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.3, 0.3, 0.6, 1.0],  
        metallicFactor=0.0,
        roughnessFactor=1.0
    )
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)


    scene = pyrender.Scene(bg_color=[0.3, 0.3, 0.3, 1.0], ambient_light=[0.4, 0.4, 0.4, 1.0])
    scene.add(mesh)


    if fixed_light_center is not None:
        light_center = fixed_light_center
    else:
        light_center = verts.mean(axis=0)  # fallback to current frame center
    

    light1 = pyrender.PointLight(intensity=8.0)
    light_pose1 = np.eye(4)
    light_pose1[:3, 3] = [light_center[0] - 1, light_center[1] + 1.5, light_center[2] - 1]
    scene.add(light1, pose=light_pose1)
    
    light2 = pyrender.PointLight(intensity=6.0)
    light_pose2 = np.eye(4)
    light_pose2[:3, 3] = [light_center[0] + 1.5, light_center[1] + 1, light_center[2] - 1]
    scene.add(light2, pose=light_pose2)
    
    light3 = pyrender.PointLight(intensity=5.0)
    light_pose3 = np.eye(4)
    light_pose3[:3, 3] = [light_center[0], light_center[1] + 2, light_center[2] - 0.5]
    scene.add(light3, pose=light_pose3)
    
    light4 = pyrender.PointLight(intensity=3.0)
    light_pose4 = np.eye(4)
    light_pose4[:3, 3] = [light_center[0], light_center[1] + 0.5, light_center[2] + 1.5]
    scene.add(light4, pose=light_pose4)
    
    if i == 0:  # Only show lighting info for first frame
        print(f"Fixed lighting setup (center: {light_center}):")
        print(f"Key Light (intensity 8.0): {light_pose1[:3, 3]}")
        print(f"Fill Light (intensity 6.0): {light_pose2[:3, 3]}")
        print(f"Top Light (intensity 5.0): {light_pose3[:3, 3]}")
        print(f"Rim Light (intensity 3.0): {light_pose4[:3, 3]}")

    cam = pyrender.PerspectiveCamera(yfov=np.pi/5.0)
    
    if fixed_cam_pose is not None and False:  # Force recalculate camera position
        cam_pose = fixed_cam_pose.copy()
        if i == 0:
            print(f"Using fixed camera position: {cam_pose[:3, 3]}")
    else:
        cam_pose = np.eye(4)
        # Use root joint position (translation) as alignment point
        root_position = transl.detach().cpu().numpy() if hasattr(transl, 'detach') else transl
        print(f"Root position: {root_position}")
        # Camera position relative to root joint with fixed offset
        cam_pose[:3, 3] = root_position + [0.0, 0.1, -2.0]  # Fixed offset relative to root joint
        
        # Use fixed camera orientation to avoid flipping
        up = np.array([0, 1, 0])      # Fixed up direction
        forward = np.array([0, 0, 1]) # Fixed forward direction (towards positive Z-axis)
        right = np.array([1, 0, 0])   # Fixed right direction
        
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up  
        cam_pose[:3, 2] = -forward  
        
        if i == 0:
            print(f"Calculated camera position: {cam_pose[:3, 3]}")
            print(f"Looking at model center: {verts.mean(axis=0)}")
    
    scene.add(cam, pose=cam_pose)

    if i == 0:
        print("Creating offscreen renderer...")
    r = pyrender.OffscreenRenderer(viewport_width=450, viewport_height=800)
    if i == 0:
        print("Rendering scene...")
    color, _ = r.render(scene)
    if i == 0:
        print(f"Rendered image shape: {color.shape}")
        print(f"Saving image to: {out_path}")
    imageio.imwrite(out_path, color)
    if i == 0:
        print(f"Image saved successfully to {out_path}")
    r.delete()
    
    # Return camera pose and light center for first frame to use in subsequent frames
    if i == 0 and fixed_cam_pose is None:
        return cam_pose, light_center
    return None, None


def render_single_motion_frame(i, motion_data, out_path, betas=None, fixed_cam_pose=None, 
                              fixed_light_center=None, apply_lower_body_zero=True,
                              left_hand_mean=None, right_hand_mean=None):
    """Render a single motion frame using the same logic as render_one_frame"""
    if betas is None:
        betas = torch.zeros(1, 10, device=DEVICE)

    # Process body pose data
    body_pose_data = motion_data[f"{MODEL_PREFIX}:body_pose"]
    if apply_lower_body_zero:
        body_pose_data = zero_out_lower_body(body_pose_data)

    if i == 0 and apply_lower_body_zero:
        print(f"Frame {i} - Lower body joint comparison:")
        for joint_idx in LOWER_BODY_JOINT_INDICES:
            joint_names = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot']
            if joint_idx < len(joint_names):
                joint_name = joint_names[joint_idx] if joint_idx < len(LOWER_BODY_JOINT_INDICES) else f"joint_{joint_idx}"
                orig_val = motion_data[f"{MODEL_PREFIX}:body_pose"][i, joint_idx, :]
                mod_val = body_pose_data[i, joint_idx, :]
                print(f"  {joint_name} (joint {joint_idx}): {orig_val} -> {mod_val}")

    body_pose = to_torch(body_pose_data[i].reshape(-1))
    
    # Handle hand poses
    left_hand_pose = to_torch(np.zeros(45))
    right_hand_pose = to_torch(np.zeros(45))
    
    if MODEL_PREFIX in ['smplh', 'smplx'] and f"{MODEL_PREFIX}:left_hand_pose" in motion_data:
        left_hand_data = motion_data[f"{MODEL_PREFIX}:left_hand_pose"][i].reshape(-1).copy()
        right_hand_data = motion_data[f"{MODEL_PREFIX}:right_hand_pose"][i].reshape(-1).copy()
        if left_hand_mean is not None:
            left_hand_data = left_hand_data - left_hand_mean
        if right_hand_mean is not None:
            right_hand_data = right_hand_data - right_hand_mean
        left_hand_pose = to_torch(left_hand_data)
        right_hand_pose = to_torch(right_hand_data)
    
    global_orient = to_torch(np.zeros(3))
    
    # Handle translation
    if f"{MODEL_PREFIX}:translation" in motion_data:
        transl = to_torch(np.zeros_like(motion_data[f"{MODEL_PREFIX}:translation"][i]))
    else:
        transl = to_torch(np.zeros(3))

    # Build kwargs for model
    kwargs = {
        'betas': betas,
        'body_pose': body_pose,
        'global_orient': global_orient,
        'return_verts': True,
    }
    
    if MODEL_PREFIX in ['smplh', 'smplx']:
        kwargs['left_hand_pose'] = left_hand_pose
        kwargs['right_hand_pose'] = right_hand_pose
    
    if MODEL_PREFIX == 'smplx':
        if f"{MODEL_PREFIX}:expression" in motion_data:
            kwargs['expression'] = to_torch(motion_data[f"{MODEL_PREFIX}:expression"][i])
        if f"{MODEL_PREFIX}:jaw_pose" in motion_data:
            kwargs['jaw_pose'] = to_torch(motion_data[f"{MODEL_PREFIX}:jaw_pose"][i])
    
    kwargs['transl'] = transl
    
    out = model(**kwargs)
    verts = out.vertices[0].detach().cpu().numpy()
    faces = model.faces

    if i == 0 and apply_lower_body_zero:  # Only show detailed info for first frame
        print(f"Vertices shape: {verts.shape}")
        print(f"Vertices bounds: min={verts.min(axis=0)}, max={verts.max(axis=0)}")
        print(f"Model center: {verts.mean(axis=0)}")

    # Apply rotation
    rot_x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rot_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    rotation_matrix = rot_y @ rot_x
    verts = verts @ rotation_matrix.T
    
    if i == 0 and apply_lower_body_zero:
        print(f"After rotation - Vertices bounds: min={verts.min(axis=0)}, max={verts.max(axis=0)}")
        print(f"After rotation - Model center: {verts.mean(axis=0)}")

    # Create mesh
    mesh = trimesh.Trimesh(verts, faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.3, 0.3, 0.6, 1.0],
        metallicFactor=0.3, roughnessFactor=0.8
    )
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = pyrender.Scene(bg_color=[0.3, 0.3, 0.3, 1.0], ambient_light=[0.4, 0.4, 0.4, 1.0])
    scene.add(mesh)

    # Use lighting and camera setup
    if fixed_light_center is not None:
        light_center = fixed_light_center
    else:
        light_center = verts.mean(axis=0)
    
    # Add lights (same as render_one_frame)
    light1 = pyrender.PointLight(intensity=8.0)
    light_pose1 = np.eye(4)
    light_pose1[:3, 3] = [light_center[0] - 1, light_center[1] + 1.5, light_center[2] - 1]
    scene.add(light1, pose=light_pose1)
    
    light2 = pyrender.PointLight(intensity=6.0)
    light_pose2 = np.eye(4)
    light_pose2[:3, 3] = [light_center[0] + 1.5, light_center[1] + 1, light_center[2] - 1]
    scene.add(light2, pose=light_pose2)
    
    light3 = pyrender.PointLight(intensity=5.0)
    light_pose3 = np.eye(4)
    light_pose3[:3, 3] = [light_center[0], light_center[1] + 2, light_center[2] - 0.5]
    scene.add(light3, pose=light_pose3)
    
    light4 = pyrender.PointLight(intensity=3.0)
    light_pose4 = np.eye(4)
    light_pose4[:3, 3] = [light_center[0], light_center[1] + 0.5, light_center[2] + 1.5]
    scene.add(light4, pose=light_pose4)

    # Camera setup
    cam = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
    
    if fixed_cam_pose is not None and False:  # Force recalculate camera position
        cam_pose = fixed_cam_pose.copy()
    else:
        cam_pose = np.eye(4)
        # Use root joint position (translation) as alignment point
        root_position = transl.detach().cpu().numpy() if hasattr(transl, 'detach') else transl
        # Camera position relative to root joint with fixed offset
        cam_pose[:3, 3] = root_position + [0.0, 0.0, -2.3]  # Fixed offset relative to root joint
        # print('root_position: ', root_position)
        # Use fixed camera orientation to avoid flipping
        up = np.array([0, 1, 0])      # Fixed up direction
        forward = np.array([0, 0, 1]) # Fixed forward direction (towards positive Z-axis)
        right = np.array([1, 0, 0])   # Fixed right direction
        
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up  
        cam_pose[:3, 2] = -forward
    
    scene.add(cam, pose=cam_pose)

    # Render
    r = pyrender.OffscreenRenderer(viewport_width=450, viewport_height=800)
    color, _ = r.render(scene)
    imageio.imwrite(out_path, color)
    r.delete()
    
    if i == 0 and fixed_cam_pose is None:
        return cam_pose, light_center
    return None, None


def render_comparison_frame(i, original_data, reconstructed_data, out_path="comparison_frame.png", 
                          betas=None, fixed_cam_pose=None, fixed_light_center=None,
                          left_hand_mean=None, right_hand_mean=None):
    """Render comparison frame by separately rendering original and reconstructed, then combining"""
    print(f"Rendering comparison frame {i} to {out_path}")
    
    import os
    
    # Create temporary files for individual renders
    temp_dir = os.path.dirname(out_path)
    temp_original = os.path.join(temp_dir, f"temp_original_{i:06d}.png")
    temp_reconstructed = os.path.join(temp_dir, f"temp_reconstructed_{i:06d}.png")
    
    try:
        # Render original motion (with lower body zeroed out)
        if i == 0:
            print("  Rendering original motion...")
            fixed_cam_pose, fixed_light_center = render_single_motion_frame(
                i, original_data, temp_original, betas, None, None, apply_lower_body_zero=True,
                left_hand_mean=left_hand_mean, right_hand_mean=right_hand_mean
            )
        else:
            render_single_motion_frame(
                i, original_data, temp_original, betas, fixed_cam_pose, fixed_light_center, apply_lower_body_zero=True,
                left_hand_mean=left_hand_mean, right_hand_mean=right_hand_mean
            )
        
        # Render reconstructed motion (without lower body zeroing)
        if i == 0:
            print("  Rendering reconstructed motion...")
        render_single_motion_frame(
            i, reconstructed_data, temp_reconstructed, betas, fixed_cam_pose, fixed_light_center, apply_lower_body_zero=False,
            left_hand_mean=left_hand_mean, right_hand_mean=right_hand_mean
        )
        
        # Combine the two images side by side
        from PIL import Image, ImageDraw, ImageFont
        try:
            # Load the two rendered images
            img_original = Image.open(temp_original)
            img_reconstructed = Image.open(temp_reconstructed)
            
            # Get dimensions
            width, height = img_original.size
            
            # Create combined image (side by side)
            combined_img = Image.new('RGB', (width * 2, height))
            combined_img.paste(img_original, (0, 0))
            combined_img.paste(img_reconstructed, (width, 0))
            
            # Add text labels
            draw = ImageDraw.Draw(combined_img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((50, 30), "Original", fill=(255, 255, 255, 255), font=font)
            draw.text((width + 50, 30), "Generated", fill=(255, 255, 255, 255), font=font)
            
            combined_img.save(out_path)
            
        except ImportError:
            print("PIL not available, using imageio for combining")
            img_original = imageio.imread(temp_original)
            img_reconstructed = imageio.imread(temp_reconstructed)
            combined_img = np.concatenate([img_original, img_reconstructed], axis=1)
            imageio.imwrite(out_path, combined_img)
    
    finally:
        # Clean up temporary files
        for temp_file in [temp_original, temp_reconstructed]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    if i == 0:
        return fixed_cam_pose, fixed_light_center
    return None, None

print(f"Data shapes:")
for key in data.keys():
    if hasattr(data[key], 'shape'):
        print(f"{key}: {data[key].shape}")

total_frames = data[f"{MODEL_PREFIX}:body_pose"].shape[0]
print(f"\nTotal available frames: {total_frames}")

def render_video_frames(max_frames=1000, step=5, output_dir="frames", comparison_mode=False, 
                       original_data=None, reconstructed_data=None, frame_num=0,
                       left_hand_mean=None, right_hand_mean=None):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Handle frame count for comparison mode
    if comparison_mode and original_data is not None and reconstructed_data is not None:
        orig_frames = original_data[f"{MODEL_PREFIX}:body_pose"].shape[0]
        recon_frames = reconstructed_data[f"{MODEL_PREFIX}:body_pose"].shape[0]
        available_frames = min(orig_frames, recon_frames)
        print(f"Comparison mode: using {available_frames} frames")
    else:
        available_frames = total_frames
    
    # Apply frame_num limit if specified (0 means all frames)
    if frame_num > 0:
        available_frames = min(available_frames, frame_num)
        print(f"Frame number limit: will render up to {frame_num} frames")
    
    frames_to_render = min(max_frames, (available_frames // step))
    
    if comparison_mode:
        print(f"Will render {frames_to_render} COMPARISON frames (step={step}) with FIXED camera")
        print("Original (skin color) vs Reconstructed (greenish)")
    else:
        print(f"Will render {frames_to_render} frames (step={step}) with FIXED camera")

    fixed_cam_pose = None
    fixed_light_center = None
    
    for i in range(0, frames_to_render * step, step):
        if i >= available_frames:
            break
            
        if comparison_mode:
            frame_filename = f"{output_dir}/comparison_{i:06d}.png"
        else:
            frame_filename = f"{output_dir}/frame_{i:06d}.png"
        
        if i == 0:
            print(f"\nRendering FIRST frame {i}/{available_frames-1} -> {frame_filename}")
            print("Calculating fixed camera and light positions...")
            
            if comparison_mode:
                fixed_cam_pose, fixed_light_center = render_comparison_frame(
                    i, original_data, reconstructed_data, frame_filename,
                    left_hand_mean=left_hand_mean, right_hand_mean=right_hand_mean
                )
            else:
                fixed_cam_pose, fixed_light_center = render_one_frame(
                    i, frame_filename,
                    left_hand_mean=left_hand_mean, right_hand_mean=right_hand_mean
                )
            
            print("Fixed camera and lighting parameters calculated!")
        else:
            if i % (step * 10) == 0:  
                print(f"Rendering frame {i}/{available_frames-1} (using fixed camera)")
            
            if comparison_mode:
                render_comparison_frame(
                    i, original_data, reconstructed_data, frame_filename,
                    fixed_cam_pose=fixed_cam_pose, fixed_light_center=fixed_light_center,
                    left_hand_mean=left_hand_mean, right_hand_mean=right_hand_mean
                )
            else:
                render_one_frame(
                    i, frame_filename, 
                    fixed_cam_pose=fixed_cam_pose, fixed_light_center=fixed_light_center,
                    left_hand_mean=left_hand_mean, right_hand_mean=right_hand_mean
                )
    
    print(f"\nFinished rendering {frames_to_render} frames with FIXED camera!")
    return frames_to_render

def render_test_frames(output_dir="frames", left_hand_mean=None, right_hand_mean=None):
    """Render test frames: 0, 10, 20"""
    import os
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_frames = [0, 10, 20]
    
    print(f"\nTEST MODE: Rendering frames {test_frames}")
    
    # Get fixed camera and light parameters from first frame
    fixed_cam_pose = None
    fixed_light_center = None
    
    for i, frame_num in enumerate(test_frames):
        if frame_num >= total_frames:
            print(f"Frame {frame_num} exceeds total frames ({total_frames}), skipping...")
            continue
            
        frame_filename = f"{output_dir}/test_frame_{frame_num:06d}.png"
        
        if i == 0:
            print(f"\nRendering FIRST test frame {frame_num} -> {frame_filename}")
            print("Calculating fixed camera and light positions...")
            fixed_cam_pose, fixed_light_center = render_one_frame(
                frame_num, frame_filename,
                left_hand_mean=left_hand_mean, right_hand_mean=right_hand_mean
            )
            print("Fixed camera and lighting parameters calculated!")
        else:
            print(f"Rendering test frame {frame_num} -> {frame_filename}")
            render_one_frame(
                frame_num, frame_filename, 
                fixed_cam_pose=fixed_cam_pose, fixed_light_center=fixed_light_center,
                left_hand_mean=left_hand_mean, right_hand_mean=right_hand_mean
            )
    
    print(f"\nTest rendering complete! Check the generated images in {output_dir}/:")
    for frame_num in test_frames:
        if frame_num < total_frames:
            print(f"  - test_frame_{frame_num:06d}.png")
    
    return len([f for f in test_frames if f < total_frames])


def cleanup_png_files(output_dir, frame_pattern="frame_*.png"):
    """
    Delete PNG files after video creation to save disk space.
    
    Args:
        output_dir: Directory containing frames
        frame_pattern: Pattern to match frame files to delete
    """
    import glob
    import os
    
    print(f"\nCleaning up PNG files...")
    print(f"   Directory: {output_dir}")
    print(f"   Pattern: {frame_pattern}")
    
    # Find all PNG files matching the pattern
    png_files = glob.glob(os.path.join(output_dir, frame_pattern))
    
    if not png_files:
        print(f"   No PNG files found matching pattern: {frame_pattern}")
        return
    
    print(f"   Found {len(png_files)} PNG files to delete")
    
    deleted_count = 0
    for png_file in png_files:
        try:
            os.remove(png_file)
            deleted_count += 1
        except OSError as e:
            print(f"   Warning: Could not delete {png_file}: {e}")
    
    print(f"   Successfully deleted {deleted_count} PNG files")
    print(f"   Disk space saved!")


def create_video_from_frames(output_dir, fps=30, video_name="motion_video.mp4", frame_pattern="frame_*.png", audio_path=None):
    """
    Create MP4 video from rendered frames, optionally with audio.
    
    Args:
        output_dir: Directory containing frames
        fps: Frame rate
        video_name: Output video filename
        frame_pattern: Pattern to match frame files
        audio_path: Optional path to audio file to merge with video
    """
    import glob
    import subprocess
    import os
    
    print(f"\nCreating video from frames...")
    print(f"   Input directory: {output_dir}")
    print(f"   Frame pattern: {frame_pattern}")
    print(f"   FPS: {fps}")
    if audio_path:
        print(f"   Audio: {audio_path}")
    
    # Find all frame files
    frame_files = glob.glob(os.path.join(output_dir, frame_pattern))
    if not frame_files:
        print(f"No frames found matching pattern: {frame_pattern}")
        return False
    
    frame_files.sort()
    print(f"   Found {len(frame_files)} frames")
    
    # Output video path
    video_path = os.path.join(output_dir, video_name)
    
    # Use ffmpeg to create video
    # Create a temporary file list for ffmpeg
    filelist_path = os.path.join(output_dir, "frame_list.txt")
    
    try:
        # Method 1: Try using ffmpeg with image sequence
        if frame_pattern.startswith("comparison_"):
            input_pattern = os.path.join(output_dir, "comparison_%06d.png")
        else:
            input_pattern = os.path.join(output_dir, "frame_%06d.png")
        
        # Try different encoders in order of preference
        encoders_to_try = [
            ('libx264', ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23']),
            ('libx265', ['-c:v', 'libx265', '-pix_fmt', 'yuv420p', '-crf', '23']),
            ('mpeg4', ['-c:v', 'mpeg4', '-q:v', '3']),
            ('libvpx', ['-c:v', 'libvpx', '-crf', '10', '-b:v', '0']),
            ('copy', ['-c:v', 'copy'])  # Fallback - no re-encoding
        ]
        
        video_created = False
        for encoder_name, encoder_args in encoders_to_try:
            # Build ffmpeg command with optional audio
            cmd = [
                '/usr/bin/ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', input_pattern,
            ]
            
            # Add audio input if provided
            if audio_path and os.path.exists(audio_path):
                cmd.extend(['-i', audio_path])
                # Map video and audio streams
                cmd.extend(['-map', '0:v:0', '-map', '1:a:0'])
                # Audio codec
                cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
                # Ensure video is not longer than audio
                cmd.extend(['-shortest'])
            
            # Add video encoder args and output path
            cmd.extend(encoder_args)
            cmd.append(video_path)
            
            print(f"Trying encoder: {encoder_name}")
            print(f"   Command: {' '.join(cmd)}")
            
            # Use system libraries to avoid conda conflicts
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = '/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu'
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                if audio_path and os.path.exists(audio_path):
                    print(f"Video with audio created successfully with {encoder_name}: {video_path}")
                    print(f"   Audio merged from: {audio_path}")
                else:
                    print(f"Video created successfully with {encoder_name}: {video_path}")
                video_created = True
                break
            else:
                print(f"{encoder_name} failed: {result.stderr.strip()}")
                continue
        
        if video_created:
            # Get file size
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            print(f"   Video size: {file_size:.2f} MB")
            return True
        else:
            # Try alternative method with file list
            print("Trying alternative method...")
            with open(filelist_path, 'w') as f:
                for frame_file in frame_files:
                    f.write(f"file '{os.path.basename(frame_file)}'\n")
            
            # Try alternative method with different encoders
            for encoder_name, encoder_args in encoders_to_try:
                cmd_alt = [
                    '/usr/bin/ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', filelist_path,
                    '-r', str(fps),
                ]
                
                # Add audio input if provided
                if audio_path and os.path.exists(audio_path):
                    cmd_alt.extend(['-i', audio_path])
                    cmd_alt.extend(['-map', '0:v:0', '-map', '1:a:0'])
                    cmd_alt.extend(['-c:a', 'aac', '-b:a', '128k'])
                    cmd_alt.extend(['-shortest'])
                
                cmd_alt.extend(encoder_args)
                cmd_alt.append(video_path)
                
                print(f"Alternative method with encoder: {encoder_name}")
                # Use system libraries to avoid conda conflicts
                env = os.environ.copy()
                env['LD_LIBRARY_PATH'] = '/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu'
                result_alt = subprocess.run(cmd_alt, capture_output=True, text=True, cwd=output_dir, env=env)
                
                if result_alt.returncode == 0:
                    if audio_path and os.path.exists(audio_path):
                        print(f"Video with audio created successfully (alternative method with {encoder_name}): {video_path}")
                        print(f"   Audio merged from: {audio_path}")
                    else:
                        print(f"Video created successfully (alternative method with {encoder_name}): {video_path}")
                    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                    print(f"   Video size: {file_size:.2f} MB")
                    return True
                else:
                    print(f"Alternative method with {encoder_name} failed: {result_alt.stderr.strip()}")
                    continue
            
            print(f"All alternative methods also failed")
            return False
                
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to create videos.")
        print("   On Ubuntu: sudo apt install ffmpeg")
        return False
    
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        return False
    
    finally:
        # Clean up temporary file
        if os.path.exists(filelist_path):
            os.remove(filelist_path)

# Main execution based on configuration
print("\n" + "="*60)
print(f"MOTION RENDERER - Mode: {args.mode.upper()}")
print(f"NPZ File: {data_file}")
print(f"Output Directory: {args.output_dir}")
print("="*60)

if args.mode == 'test':
    # Test mode: render frames 0, 10, 20
    if args.comparison:
        print("Warning: Comparison mode is disabled in test mode")
    num_rendered = render_test_frames(
        output_dir=args.output_dir,
        left_hand_mean=left_hand_mean,
        right_hand_mean=right_hand_mean
    )
    video_created = False
    
else:
    # Full mode: render all frames
    if args.comparison:
        # Comparison mode: render original vs reconstructed
        num_rendered = render_video_frames(
            max_frames=args.max_frames, 
            step=args.step, 
            output_dir=args.output_dir,
            comparison_mode=True,
            original_data=data,
            reconstructed_data=reconstructed_data,
            frame_num=args.frame_num,
            left_hand_mean=left_hand_mean,
            right_hand_mean=right_hand_mean
        )
        frame_pattern = "comparison_*.png"
        video_name = "comparison_video.mp4"
    else:
        # Normal mode: render single motion
        num_rendered = render_video_frames(
            max_frames=args.max_frames, 
            step=args.step, 
            output_dir=args.output_dir,
            comparison_mode=False,
            frame_num=args.frame_num,
            left_hand_mean=left_hand_mean,
            right_hand_mean=right_hand_mean
        )
        frame_pattern = "frame_*.png"
        video_name = "motion_video.mp4"
    
    # Create video if requested
    video_created = False
    if args.create_video:
        video_created = create_video_from_frames(
            output_dir=args.output_dir,
            fps=args.fps,
            video_name=video_name,
            frame_pattern=frame_pattern,
            audio_path=args.audio
        )
        
        # Clean up PNG files after successful video creation
        if video_created:
            cleanup_png_files(args.output_dir, frame_pattern)

print(f"\nRendering completed! Total frames rendered: {num_rendered}")
print(f"Output saved to: {args.output_dir}")

if args.mode == 'full':
    if args.comparison:
        print(f"Comparison mode: Original (skin color) vs Reconstructed (greenish)")
    if args.create_video:
        if video_created:
            print(f"Video created successfully!")
        else:
            print(f"Video creation failed")