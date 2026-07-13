import random
import numpy as np
import torch
from torch.utils.data import Dataset
import json 
import librosa
import pandas as pd
import os
import sys
from collections import defaultdict
import re
from tqdm import tqdm

# Add the parent directory to sys.path to import motion_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.motion_utils import convert_axis_angle_to_rotation_6d

class DyadicLoader(Dataset):
    def __init__(self, cfg=None, data_path=None):
        self.cfg = cfg
        # Allow override of data path (useful for validation set with different data)
        self.data_path = data_path if data_path is not None else cfg.paths.data_root
        
        # Optional separate motion path (for emage training)
        # If specified, motion files will be loaded from this path, while audio still from data_path
        self.motion_path = getattr(cfg.data, 'motion_path', None) if cfg and hasattr(cfg, 'data') else None
        
        # Audio and motion parameters
        self.fps = cfg.model.pose_fps if cfg and hasattr(cfg, 'model') else 30
        self.audio_sr = cfg.model.audio_sr if cfg and hasattr(cfg, 'model') else 16000
        self.pose_length = cfg.model.pose_length if cfg and hasattr(cfg, 'model') else 150
        
        # Limit number of files to use (for single file overfitting)
        self.max_files = cfg.data.max_files if cfg and hasattr(cfg, 'data') and hasattr(cfg.data, 'max_files') else None
        
        # Specific files to use (for single file overfitting)
        self.specific_files = cfg.data.specific_files if cfg and hasattr(cfg, 'data') and hasattr(cfg.data, 'specific_files') else None
        
        # Samples per conversation - how many different segments to sample from each file
        samples_per_conversation_config = cfg.data.samples_per_conversation if cfg and hasattr(cfg, 'data') and hasattr(cfg.data, 'samples_per_conversation') else -1
        
        # Rotation representation - 6D rotation or 3D euler angles
        self.use_6d_rotation = cfg.data.use_6d_rotation if cfg and hasattr(cfg, 'data') and hasattr(cfg.data, 'use_6d_rotation') else True
        
        # Flag to track if we're using preprocessed 6D data or converting on-the-fly
        self.use_preprocessed_6d = True
        
        # Load relationship metadata
        self.relationship_map = self._load_relationship_map()
        
        # Load participants metadata for personality traits
        self.participants_map = self._load_participants_map()
        
        # Initialize data_list first (needed for normalization stats computation)
        self.data_list = self._get_data_list()
        print(f"Found {len(self.data_list)} data file pairs")
        
        if len(self.data_list) == 0:
            print("=" * 60)
            print("ERROR: No data files found!")
            print(f"  Motion path: {getattr(self, 'motion_path', 'Not set (using default)')}")
            print(f"  Data path: {self.data_path}")
            print("=" * 60)
            raise ValueError(
                f"No data files found. Please check:\n"
                f"  1. Motion directory exists and contains .npz files\n"
                f"  2. Audio directory exists at {os.path.join(self.data_path, 'audio_npz')}\n"
                f"  3. Motion and audio files have matching filenames"
            )
        
        # Normalization parameters
        self.normalize_motion = cfg.data.normalize_motion if cfg and hasattr(cfg, 'data') and hasattr(cfg.data, 'normalize_motion') else True
        
        if self.use_6d_rotation:
            print(f"Using 6D rotation representation")
            if self.normalize_motion:
                # Load 6D rotation normalization stats from cache
                self.motion_mean, self.motion_std = self._load_6d_normalization_stats()
                print(f"6D rotation normalization enabled: mean/std shapes ({self.motion_mean.shape})")
            else:
                print(f"6D rotation normalization disabled")
        else:
            print(f"Using 3D euler angles representation")
            if self.normalize_motion:
                # Load 3D euler normalization stats from cache
                self.motion_mean, self.motion_std = self._load_3d_euler_normalization_stats()
                print(f"3D euler normalization enabled: mean/std shapes ({self.motion_mean.shape})")
            else:
                print(f"3D euler normalization disabled")
        
        # Get total frames for each motion file
        self.total_frames_per_file = self._get_total_frames_per_file()
        
        # Calculate actual samples_per_conversation
        if samples_per_conversation_config == -1:
            # -1 means use all possible non-overlapping segments from all files
            total_possible_samples = sum(max(1, frames // self.pose_length) for frames in self.total_frames_per_file)
            self.samples_per_conversation = total_possible_samples
            print(f"Total samples across all files: {self.samples_per_conversation}")
        else:
            self.samples_per_conversation = samples_per_conversation_config * len(self.data_list)
            print(f"Samples per file: {samples_per_conversation_config}, Total samples: {self.samples_per_conversation}")

    def _get_data_list(self):
        """Scan data directory for all available motion and audio files, grouping pairs by prefix"""
        # Determine motion directory
        # If motion_path is specified (e.g., for emage training), use it directly
        if self.motion_path is not None:
            motion_dir = self.motion_path
            print(f"Using custom motion path: {motion_dir}")
            # Check if custom motion path exists
            if not os.path.exists(motion_dir):
                print(f"WARNING: Custom motion directory not found: {motion_dir}")
                return []
            # Assume it's already 6D if using 6D rotation
            if self.use_6d_rotation:
                self.use_preprocessed_6d = True
        else:
            # Use motion_6d directory if using 6D rotation, otherwise use motion directory
            if self.use_6d_rotation:
                motion_dir = os.path.join(self.data_path, 'motion_6d')
                print(f"Using 6D motion data from: {motion_dir}")
                
                # Check if motion_6d directory exists, if not fallback to motion directory
                if not os.path.exists(motion_dir):
                    print(f"WARNING: 6D motion directory not found: {motion_dir}")
                    motion_dir = os.path.join(self.data_path, 'motion')
                    print(f"Falling back to motion directory: {motion_dir}")
                    if not os.path.exists(motion_dir):
                        print(f"WARNING:  Motion directory not found: {motion_dir}")
                        return []
                    print(f"Will perform 6D conversion on-the-fly from motion data")
                    self.use_preprocessed_6d = False
            else:
                motion_dir = os.path.join(self.data_path, 'motion')
        
        # Audio directory always from data_path
        audio_dir = os.path.join(self.data_path, 'audio_npz')
        print(f"Audio directory: {audio_dir}")
        
        data_list = []
        
        if not os.path.exists(audio_dir):
            print(f"ERROR: Audio directory not found: {audio_dir}")
            print(f"Please check that data_path is correct: {self.data_path}")
            return []
        
        # Check if audio directory has files
        try:
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.npz')]
            print(f"Found {len(audio_files)} audio files in {audio_dir}")
        except Exception as e:
            print(f"ERROR: Failed to list files in audio directory {audio_dir}: {e}")
            return []
        
        # Determine file extension based on whether using custom motion_path
        # If using custom motion_path, read _6d.npy files; otherwise read .npz files
        if self.motion_path is not None:
            motion_ext = '_6d.npy'
            print(f"Using _6d.npy extension for motion files (custom motion_path)")
        else:
            motion_ext = '.npz'
        
        # Get motion files based on configuration
        if self.specific_files is not None:
            # Use specific files if provided
            motion_files = []
            for filename in self.specific_files:
                # Remove extension if present, then add the correct one
                base_name = filename.replace('_6d.npy', '').replace('.npz', '').replace('.npy', '')
                motion_files.append(base_name + motion_ext)
            print(f"Using specific files: {motion_files}")
        else:
            # Get all motion files with the correct extension
            try:
                motion_files = [f for f in os.listdir(motion_dir) if f.endswith(motion_ext)]
            except Exception as e:
                print(f"ERROR: Failed to list files in motion directory {motion_dir}: {e}")
                return []
            
            # Sort files for consistent ordering
            motion_files.sort()
            print(f"Found {len(motion_files)} motion files ({motion_ext}) in {motion_dir}")
            
            # Limit number of files if specified
            if self.max_files is not None:
                motion_files = motion_files[:self.max_files]
                print(f"Limited to {self.max_files} files for overfitting")
        
        # Group files by prefix (VXX_SXXXX_IXXXXXXXX)
        file_groups = {}
        for motion_file in motion_files:
            # Extract prefix: VXX_SXXXX_IXXXXXXXX
            parts = motion_file.split('_')
            if len(parts) >= 3:
                prefix = '_'.join(parts[:3])  # VXX_SXXXX_IXXXXXXXX
                if prefix not in file_groups:
                    file_groups[prefix] = []
                file_groups[prefix].append(motion_file)
        
        # Process each group to find pairs
        pairs_found = 0
        for prefix, files in file_groups.items():
            if len(files) >= 2:
                # Sort files to ensure consistent pairing
                files.sort()
                file1, file2 = files[0], files[1]
                
                # Motion files use the determined extension (_6d.npy or .npz)
                motion_path1 = os.path.join(motion_dir, file1)
                motion_path2 = os.path.join(motion_dir, file2)
                
                # Audio files always use .npz extension
                # Convert motion filename to audio filename if needed
                # Motion: V00_S0030_I00000125_P0045_6d.npy -> Audio: V00_S0030_I00000125_P0045.npz
                if file1.endswith('_6d.npy'):
                    audio_file1 = file1.replace('_6d.npy', '.npz')
                elif file1.endswith('.npy'):
                    audio_file1 = file1.replace('.npy', '.npz')
                else:
                    audio_file1 = file1
                
                if file2.endswith('_6d.npy'):
                    audio_file2 = file2.replace('_6d.npy', '.npz')
                elif file2.endswith('.npy'):
                    audio_file2 = file2.replace('.npy', '.npz')
                else:
                    audio_file2 = file2
                
                audio_path1 = os.path.join(audio_dir, audio_file1)
                audio_path2 = os.path.join(audio_dir, audio_file2)
                
                # Check if motion files exist
                motion_files_exist = os.path.exists(motion_path1) and os.path.exists(motion_path2)
                
                # Check if audio files exist (NPZ, WAV, or MP3 fallback)
                audio_files_exist = self._check_audio_files_exist(audio_path1, audio_path2)
                
                if motion_files_exist and audio_files_exist:
                    data_list.append([motion_path1, audio_path1, motion_path2, audio_path2])
                    pairs_found += 1
                    if pairs_found <= 5:  # Print first 5 pairs for debugging
                        print(f"Found data pair {pairs_found}: {file1} <-> {file2}")
                else:
                    print(f"WARNING:  Missing files for pair: {file1} <-> {file2}")
                    if not motion_files_exist:
                        print(f"  Motion files missing: {motion_path1} exists={os.path.exists(motion_path1)}, {motion_path2} exists={os.path.exists(motion_path2)}")
                    if not audio_files_exist:
                        print(f"  Audio files missing: {audio_path1} or {audio_path2}")
            else:
                print(f"WARNING:  Only {len(files)} file(s) found for prefix {prefix}, need at least 2 for pairing")
        
        print(f"Total file groups: {len(file_groups)}, Valid pairs found: {pairs_found}")
        return data_list
    
    def _check_audio_files_exist(self, audio_path1, audio_path2):
        """Check if audio files exist, supporting NPZ, WAV, and MP3 fallback"""
        # Check NPZ files first
        npz_exists = os.path.exists(audio_path1) and os.path.exists(audio_path2)
        if npz_exists:
            return True
        
        # Check WAV/MP3 fallback files
        audio_paths1 = self._get_audio_paths_from_npz_path(audio_path1)
        audio_paths2 = self._get_audio_paths_from_npz_path(audio_path2)
        
        # Try to find matching format for both files
        for ext in ['wav', 'mp3']:
            path1 = audio_paths1[ext]
            path2 = audio_paths2[ext]
            if os.path.exists(path1) and os.path.exists(path2):
                print(f"  Using {ext.upper()} fallback files: {os.path.basename(path1)}, {os.path.basename(path2)}")
                return True
        
        return False
    
    def _get_audio_paths_from_npz_path(self, audio_npz_path):
        """Convert NPZ path to audio file paths (WAV and MP3) for fallback"""
        audio_npz_dir = os.path.dirname(audio_npz_path)
        audio_npz_filename = os.path.basename(audio_npz_path)
        
        # Replace 'audio_npz' with 'audio' in the path
        audio_dir = audio_npz_dir.replace('audio_npz', 'audio')
        
        # Generate paths for both WAV and MP3
        base_filename = audio_npz_filename.replace('.npz', '')
        wav_path = os.path.join(audio_dir, base_filename + '.wav')
        mp3_path = os.path.join(audio_dir, base_filename + '.mp3')
        
        return {'wav': wav_path, 'mp3': mp3_path}
    
    def _get_wav_path_from_npz_path(self, audio_npz_path):
        """Convert NPZ path to WAV path for fallback (deprecated, use _get_audio_paths_from_npz_path)"""
        audio_paths = self._get_audio_paths_from_npz_path(audio_npz_path)
        return audio_paths['wav']
    
    def _load_relationship_map(self):
        """Load relationship details from CSV file into a mapping dictionary"""
        # csv_path = '/mnt/nas2/Projects/meta_data/relationships.csv'
        csv_path = os.path.join(self.data_path, 'csv', 'relationships.csv')
        
        if not os.path.exists(csv_path):
            print(f"WARNING: Relationship CSV not found at {csv_path}")
            return {}
        
        relationship_map = {}
        
        try:
            # Read CSV with session_id as string to preserve leading zeros
            df = pd.read_csv(csv_path, dtype={'session_id': str})
            
            for _, row in df.iterrows():
                vendor_id = str(row['vendor_id'])  # Ensure string type
                session_id = str(row['session_id'])  # Ensure string type, preserve leading zeros
                relationship_detail = row['relationship_detail']
                
                key = (vendor_id, session_id)
                
                # Map relationship_detail to one-hot categories
                # friends/coworkers, stranger, family-generic/familiar-generic, dating/spouse/romantic_partner
                if relationship_detail in ['friends', 'coworkers']:
                    category = [1, 0, 0, 0]  # friends/coworkers
                elif relationship_detail == 'stranger':
                    category = [0, 1, 0, 0]  # stranger
                elif relationship_detail in ['family-generic', 'familiar-generic']:
                    category = [0, 0, 1, 0]  # family-generic/familiar-generic
                elif relationship_detail == 'dating/spouse/romantic_partner':
                    category = [0, 0, 0, 1]  # dating/spouse/romantic_partner
                else:
                    # For other categories not in the 4-dim vector, set all zeros
                    category = [0, 0, 0, 0]
                
                relationship_map[key] = np.array(category, dtype=np.float32)
            
            print(f"Loaded {len(relationship_map)} relationship entries from CSV")
            
            # Print sample keys for debugging
            sample_keys = list(relationship_map.keys())[:5]
            print(f"Sample keys (first 5): {sample_keys}")
            
            # Print statistics
            category_counts = {'friends_coworkers': 0, 'stranger': 0, 
                             'family_familiar_generic': 0, 'dating_romantic': 0, 'other': 0}
            
            for category in relationship_map.values():
                if np.array_equal(category, [1, 0, 0, 0]):
                    category_counts['friends_coworkers'] += 1
                elif np.array_equal(category, [0, 1, 0, 0]):
                    category_counts['stranger'] += 1
                elif np.array_equal(category, [0, 0, 1, 0]):
                    category_counts['family_familiar_generic'] += 1
                elif np.array_equal(category, [0, 0, 0, 1]):
                    category_counts['dating_romantic'] += 1
                else:
                    category_counts['other'] += 1
            
            print(f"Relationship distribution: {category_counts}")
            
        except Exception as e:
            print(f"WARNING: Error loading relationship CSV: {e}")
            relationship_map = {}
        
        return relationship_map
    
    def get_relationship_onehot(self, filename):
        """
        Extract VXX_SXXXX from filename and return 4-dim one-hot vector for relationship
        
        Args:
            filename: string like "V00_S0030_I00000125_P0045.npz" or "V00_S0030_I00000125_P0045.npy" or just "V00_S0030_I00000125_P0045"
        
        Returns:
            numpy array of shape (4,) with one-hot encoding:
            [friends/coworkers, stranger, family-generic/familiar-generic, dating/spouse/romantic_partner]
        """
        # Remove .npz, .npy, or _6d.npy extension if present
        if filename.endswith('_6d.npy'):
            filename = filename[:-7]  # Remove '_6d.npy'
        elif filename.endswith('.npz'):
            filename = filename[:-4]  # Remove '.npz'
        elif filename.endswith('.npy'):
            filename = filename[:-4]  # Remove '.npy'
        
        # Extract vendor_id and session_id from filename
        # Format: VXX_SXXXX_IXXXXXXXX_PXXXX
        parts = filename.split('_')
        if len(parts) < 2:
            print(f"WARNING: Cannot parse filename {filename}, returning all zeros")
            return np.array([0, 0, 0, 0], dtype=np.float32)
        
        vendor_id = parts[0]  # V00
        session_id_with_prefix = parts[1]  # S0030
        
        # Remove 'S' prefix from session_id to match CSV format
        if session_id_with_prefix.startswith('S'):
            session_id = session_id_with_prefix[1:]  # Remove 'S' prefix: S0030 -> 0030
        else:
            session_id = session_id_with_prefix  # Keep as is if no prefix
        
        key = (vendor_id, session_id)

        # Debug: print key being searched
        # print(f"DEBUG: Searching relationship for key: {key}, filename was: {filename}")
        if key in self.relationship_map:
            return self.relationship_map[key]
        else:
            # Try alternative key format (without V prefix)
            vendor_id_alt = vendor_id[1:] if vendor_id.startswith('V') else vendor_id
            key_alt = (vendor_id_alt, session_id)
            if key_alt in self.relationship_map:
                return self.relationship_map[key_alt]
            print(f"WARNING: No relationship found for {vendor_id}_{session_id} (tried keys: {key}, {key_alt}), returning all zeros")
            return np.array([0, 0, 0, 0], dtype=np.float32)
    
    def _load_participants_map(self):
        """Load participant personality traits from CSV file into a mapping dictionary"""
        # Try multiple possible paths for participants.csv
        # possible_paths = [
        #     os.path.join(self.data_path, 'csv', 'participants.csv'),
        #     os.path.join(self.cfg.paths.data_root, 'csv', 'participants.csv') if self.cfg and hasattr(self.cfg, 'paths') else None,
        #     '/mnt/nas2/Projects/meta_data/participants.csv',
        # ]
        
        # csv_path = None
        # for path in possible_paths:
        #     if path and os.path.exists(path):
        #         csv_path = path
        #         break
        csv_path = os.path.join(self.data_path, 'csv', 'participants.csv')
        
        # if csv_path is None:
        #     print(f"WARNING: Participants CSV not found at any of the following paths:")
        #     for path in possible_paths:
        #         if path:
        #             print(f"  - {path}")
        #     return {}
        
        participants_map = {}
        
        try:
            # Read CSV with vendor_id and participant_id as strings to preserve leading zeros
            df = pd.read_csv(csv_path, dtype={'vendor_id': str, 'participant_id': str})
            
            required_columns = ['vendor_id', 'participant_id', 'extraversion_raw', 
                              'agreeableness_raw', 'conscientiousness_raw', 
                              'neuroticism_raw', 'openness_raw']
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"WARNING: Missing columns in participants CSV: {missing_columns}")
                return {}
            
            for _, row in df.iterrows():
                vendor_id = str(row['vendor_id'])  # Ensure string type
                participant_id = str(row['participant_id'])  # Ensure string type, preserve leading zeros
                
                # Helper function to safely convert to float, handling "Undisclosed" and NaN
                def safe_float(value):
                    if pd.isna(value):
                        return 0.0
                    if isinstance(value, str) and value.lower() == 'undisclosed':
                        return 0.0
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return 0.0
                
                # Extract personality traits in order: extraversion, agreeableness, conscientiousness, neuroticism, openness
                personality_traits = np.array([
                    safe_float(row['extraversion_raw']),
                    safe_float(row['agreeableness_raw']),
                    safe_float(row['conscientiousness_raw']),
                    safe_float(row['neuroticism_raw']),
                    safe_float(row['openness_raw']),
                ], dtype=np.float32)
                
                key = (vendor_id, participant_id)
                participants_map[key] = personality_traits
            
            print(f"Loaded {len(participants_map)} participant personality entries from CSV")
            
            # Print sample keys for debugging
            sample_keys = list(participants_map.keys())[:5]
            print(f"Sample participant keys (first 5): {sample_keys}")
            
            # Print statistics
            trait_stats = {
                'extraversion': {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')},
                'agreeableness': {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')},
                'conscientiousness': {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')},
                'neuroticism': {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')},
                'openness': {'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')},
            }
            
            trait_names = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
            all_traits = np.array(list(participants_map.values()))
            
            if len(all_traits) > 0:
                for i, trait_name in enumerate(trait_names):
                    trait_values = all_traits[:, i]
                    trait_stats[trait_name]['mean'] = np.mean(trait_values)
                    trait_stats[trait_name]['std'] = np.std(trait_values)
                    trait_stats[trait_name]['min'] = np.min(trait_values)
                    trait_stats[trait_name]['max'] = np.max(trait_values)
                
                print(f"Personality traits statistics:")
                for trait_name in trait_names:
                    stats = trait_stats[trait_name]
                    print(f"  {trait_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                          f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")
            
        except Exception as e:
            print(f"WARNING: Error loading participants CSV: {e}")
            participants_map = {}
        
        return participants_map
    
    def get_personality_traits(self, filename):
        """
        Extract VXX and PXXXX from filename and return 5-dim vector for personality traits
        
        Args:
            filename: string like "V00_S0030_I00000125_P0045.npz" or "V00_S0030_I00000125_P0045.npy" or just "V00_S0030_I00000125_P0045"
        
        Returns:
            numpy array of shape (5,) with normalized personality traits (range: 0 to 1):
            [extraversion_raw, agreeableness_raw, conscientiousness_raw, neuroticism_raw, openness_raw]
            Values are normalized using: personality_feat/5.0
        """
        # Remove .npz, .npy, or _6d.npy extension if present
        if filename.endswith('_6d.npy'):
            filename = filename[:-7]  # Remove '_6d.npy'
        elif filename.endswith('.npz'):
            filename = filename[:-4]  # Remove '.npz'
        elif filename.endswith('.npy'):
            filename = filename[:-4]  # Remove '.npy'
        
        # Extract vendor_id and participant_id from filename
        # Format: VXX_SXXXX_IXXXXXXXX_PXXXX
        parts = filename.split('_')
        if len(parts) < 4:
            print(f"WARNING: Cannot parse filename {filename}, returning normalized zeros (0)")
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        vendor_id = parts[0]  # V00
        participant_id_with_prefix = parts[3]  # P0045
        
        # Remove 'P' prefix from participant_id to match CSV format
        if participant_id_with_prefix.startswith('P'):
            participant_id = participant_id_with_prefix[1:]  # Remove 'P' prefix: P0045 -> 0045
        else:
            participant_id = participant_id_with_prefix  # Keep as is if no prefix
        
        # Remove 'V' prefix from vendor_id if present
        if vendor_id.startswith('V'):
            vendor_id_clean = vendor_id[1:]  # Remove 'V' prefix: V00 -> 00
        else:
            vendor_id_clean = vendor_id
        
        key = (vendor_id_clean, participant_id)
        
        if key in self.participants_map:
            personality_feat = self.participants_map[key]
            # Normalize to (0, 1) range: personality_feat/5.0
            personality_feat_normalized = personality_feat / 5.0
            return personality_feat_normalized
        else:
            # Debug: check if similar keys exist
            similar_keys = [k for k in self.participants_map.keys() if k[0] == vendor_id_clean or k[1] == participant_id]
            if similar_keys:
                print(f"WARNING: No personality traits found for {vendor_id_clean}_{participant_id}, but found similar keys: {similar_keys[:5]}")
            else:
                print(f"WARNING: No personality traits found for {vendor_id_clean}_{participant_id}, returning all zeros")
            # Return normalized zeros (which is still 0.0 after normalization: 0/5.0 = 0)
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def _get_total_frames_per_file(self):
        """Get total number of frames for each motion file pair, filtering out files with 0 frames"""
        total_frames_per_file = []
        valid_indices = []
        
        for idx, (motion_path1, _, motion_path2, _) in enumerate(self.data_list):
            try:
                frames1 = frames2 = None
                
                # Check if files are .npy (direct array) or .npz (dictionary)
                if motion_path1.endswith('.npy') or motion_path1.endswith('_6d.npy'):
                    # .npy or _6d.npy file: load directly as array
                    motion_data1 = np.load(motion_path1, allow_pickle=True)
                    frames1 = motion_data1.shape[0]
                else:
                    # .npz file: load as dictionary
                    motion_dict1 = np.load(motion_path1, allow_pickle=True)
                    # Get frames from body_pose for file1
                    if 'smplh:body_pose' in motion_dict1:
                        frames1 = motion_dict1['smplh:body_pose'].shape[0]
                
                if motion_path2.endswith('.npy') or motion_path2.endswith('_6d.npy'):
                    # .npy or _6d.npy file: load directly as array
                    motion_data2 = np.load(motion_path2, allow_pickle=True)
                    frames2 = motion_data2.shape[0]
                else:
                    # .npz file: load as dictionary
                    motion_dict2 = np.load(motion_path2, allow_pickle=True)
                    # Get frames from body_pose for file2
                    if 'smplh:body_pose' in motion_dict2:
                        frames2 = motion_dict2['smplh:body_pose'].shape[0]
                
                if frames1 is not None and frames2 is not None:
                    # Check if either file has 0 frames
                    if frames1 == 0 or frames2 == 0:
                        print(f"SKIPPING pair with 0 frames: {os.path.basename(motion_path1)} <-> {os.path.basename(motion_path2)} (frames: {frames1} <-> {frames2})")
                        continue  # Skip this pair
                    
                    # Use the minimum frames to ensure both files have enough data
                    total_frames = min(frames1, frames2)
                    
                    # Only add if we have enough frames
                    if total_frames < self.pose_length:
                        print(f"SKIPPING pair with insufficient frames: {os.path.basename(motion_path1)} <-> {os.path.basename(motion_path2)} (only {total_frames} frames, need {self.pose_length})")
                        continue
                    
                    total_frames_per_file.append(total_frames)
                    valid_indices.append(idx)
                    # print(f"Frames in pair {os.path.basename(motion_path1)} <-> {os.path.basename(motion_path2)}: {frames1} <-> {frames2} (using {total_frames})")
                else:
                    print(f"WARNING:  Could not determine frames for pair, skipping")
            except Exception as e:
                print(f"WARNING:  Error getting frames for pair: {e}, skipping")
        
        # Filter data_list to only include valid pairs
        if len(valid_indices) < len(self.data_list):
            self.data_list = [self.data_list[i] for i in valid_indices]
            print(f"Filtered out {len(self.data_list) - len(valid_indices)} invalid pairs, {len(self.data_list)} valid pairs remaining")
        
        return total_frames_per_file

    def _load_6d_normalization_stats(self):
        """Load pre-computed 6D rotation normalization stats."""
        print(f"Loading 6D rotation normalization stats from: {self.data_path}")
        cache_dir = self.cfg.paths.data_root
        cache_path = os.path.join(cache_dir, "motion_6d_normalization.npz")
        
        if os.path.exists(cache_path):
            try:
                cache_data = np.load(cache_path, allow_pickle=True)
                motion_mean = cache_data['motion_mean']
                motion_std = cache_data['motion_std']
                
                # Verify dimensions
                if motion_mean.shape != (258,) or motion_std.shape != (258,):
                    print(f"WARNING:  Invalid normalization stats dimensions: mean {motion_mean.shape}, std {motion_std.shape}")
                    raise ValueError("Invalid normalization stats dimensions")
                
                print(f"Loaded 6D rotation normalization stats from: {cache_path}")
                print(f"Stats info: mean shape {motion_mean.shape}, std shape {motion_std.shape}")
                
                # Show additional info if available
                if 'total_frames' in cache_data:
                    print(f"Total frames used: {cache_data['total_frames']}")
                if 'description' in cache_data:
                    print(f"Description: {cache_data['description']}")
                
                return motion_mean, motion_std
            except Exception as e:
                print(f"WARNING:  Failed to load 6D rotation normalization stats: {e}")
        
        # If no cache exists, compute and save normalization stats
        print(" Computing 6D rotation normalization stats...")
        motion_mean, motion_std = self._compute_6d_normalization_stats()
        
        # Save computed stats
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache_path, 
                    motion_mean=motion_mean, 
                    motion_std=motion_std,
                    motion_dim=258,
                    total_frames=len(self.data_list),  # Approximate
                    description="6D rotation normalization statistics",
                    data_path=self.data_path)
            print(f"Saved 6D rotation normalization stats to: {cache_path}")
            print(f"Stats saved: mean shape {motion_mean.shape}, std shape {motion_std.shape}")
        except Exception as e:
            print(f"WARNING:  Failed to save normalization stats: {e}")
        
        return motion_mean, motion_std
    
    def _load_3d_euler_normalization_stats(self):
        """Load pre-computed 3D euler normalization stats."""
        print(f"Loading 3D euler normalization stats from: {self.data_path}")
        cache_dir = 'data'
        cache_path = os.path.join(cache_dir, "motion_normalization_overall.npz")
        
        if os.path.exists(cache_path):
            try:
                cache_data = np.load(cache_path, allow_pickle=True)
                motion_mean = cache_data['motion_mean']
                motion_std = cache_data['motion_std']
                
                # Verify dimensions for 3D euler (129 dims)
                if motion_mean.shape != (129,) or motion_std.shape != (129,):
                    print(f"WARNING:  Invalid normalization stats dimensions: mean {motion_mean.shape}, std {motion_std.shape}")
                    raise ValueError("Invalid normalization stats dimensions")
                
                print(f"Loaded 3D euler normalization stats from: {cache_path}")
                print(f"Stats info: mean shape {motion_mean.shape}, std shape {motion_std.shape}")
                
                # Show additional info if available
                if 'total_frames' in cache_data:
                    print(f"Total frames used: {cache_data['total_frames']}")
                if 'description' in cache_data:
                    print(f"Description: {cache_data['description']}")
                
                return motion_mean, motion_std
            except Exception as e:
                print(f"WARNING:  Failed to load 3D euler normalization stats: {e}")
        
        # If no cache exists, compute and save normalization stats
        print(" Computing 3D euler normalization stats...")
        motion_mean, motion_std = self._compute_3d_euler_normalization_stats()
        
        # Save computed stats
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache_path, 
                    motion_mean=motion_mean, 
                    motion_std=motion_std,
                    motion_dim=129,
                    total_frames=len(self.data_list),  # Approximate
                    description="3D euler normalization statistics",
                    data_path=self.data_path)
            print(f"Saved 3D euler normalization stats to: {cache_path}")
            print(f"Stats saved: mean shape {motion_mean.shape}, std shape {motion_std.shape}")
        except Exception as e:
            print(f"WARNING:  Failed to save normalization stats: {e}")
        
        return motion_mean, motion_std
    
    def _compute_3d_euler_normalization_stats(self):
        """Compute 3D euler normalization statistics from all data."""
        print(" Computing 3D euler normalization statistics...")
        
        all_motion_data = []
        
        # Sample data from all files to compute statistics
        # Each element in data_list is [motion_path1, audio_path1, motion_path2, audio_path2]
        for motion_path1, _, motion_path2, _ in tqdm(self.data_list, desc="Computing 3D euler stats"):
            # Process both motion files in the pair
            for motion_path in [motion_path1, motion_path2]:
                try:
                    # Load motion data and extract 3D euler features
                    motion_dict = np.load(motion_path, allow_pickle=True)
                    
                    # Extract motion features (same as get_motion_segment but without 6D conversion)
                    motion_features = []
                    lower_body_joints = [0, 1, 3, 4, 6, 7, 9, 10]
                    upper_body_joints = [j for j in range(21) if j not in lower_body_joints]
                    
                    if 'smplh:body_pose' in motion_dict:
                        body_pose = motion_dict['smplh:body_pose']
                        upper_body_pose = body_pose[:, upper_body_joints, :]
                        upper_body_flat = upper_body_pose.reshape(upper_body_pose.shape[0], -1)
                        motion_features.append(upper_body_flat)
                    
                    if 'smplh:left_hand_pose' in motion_dict:
                        left_hand_pose = motion_dict['smplh:left_hand_pose']
                        left_hand_flat = left_hand_pose.reshape(left_hand_pose.shape[0], -1)
                        motion_features.append(left_hand_flat)
                    
                    if 'smplh:right_hand_pose' in motion_dict:
                        right_hand_pose = motion_dict['smplh:right_hand_pose']
                        right_hand_flat = right_hand_pose.reshape(right_hand_pose.shape[0], -1)
                        motion_features.append(right_hand_flat)
                    
                    if motion_features:
                        # Concatenate features: (frames, 129) - keep as 3D euler
                        motion_data = np.concatenate(motion_features, axis=1)
                        all_motion_data.append(motion_data)
                        
                except Exception as e:
                    print(f"WARNING:  Error processing {motion_path}: {e}")
                    continue
        
        if not all_motion_data:
            print("WARNING:  No motion data found, using default normalization")
            return np.zeros(129), np.ones(129)
        
        # Concatenate all motion data
        all_motion = np.concatenate(all_motion_data, axis=0)  # (total_frames, 129)
        
        # Compute mean and std
        motion_mean = np.mean(all_motion, axis=0)  # (129,)
        motion_std = np.std(all_motion, axis=0)   # (129,)
        
        # Avoid division by zero
        motion_std = np.where(motion_std < 1e-6, 1.0, motion_std)
        
        print(f"Computed 3D euler normalization stats:")
        print(f" Mean shape: {motion_mean.shape}")
        print(f" Std shape: {motion_std.shape}")
        print(f" Mean range: [{motion_mean.min():.4f}, {motion_mean.max():.4f}]")
        print(f" Std range: [{motion_std.min():.4f}, {motion_std.max():.4f}]")
        
        return motion_mean, motion_std
    
    def _compute_6d_normalization_stats(self):
        """Compute 6D rotation normalization statistics from all data."""
        print(" Computing 6D rotation normalization statistics...")
        
        all_motion_data = []
        
        # Sample data from all files to compute statistics
        # Each element in data_list is [motion_path1, audio_path1, motion_path2, audio_path2]
        for motion_path1, _, motion_path2, _ in tqdm(self.data_list, desc="Computing 6D rotation stats"):
            # Process both motion files in the pair
            for motion_path in [motion_path1, motion_path2]:
                try:
                    # Load motion data and convert to 6D rotation
                    motion_dict = np.load(motion_path, allow_pickle=True)
                    
                    # Extract motion features (same as get_motion_segment)
                    motion_features = []
                    lower_body_joints = [0, 1, 3, 4, 6, 7, 9, 10]
                    upper_body_joints = [j for j in range(21) if j not in lower_body_joints]
                    
                    if 'smplh:body_pose' in motion_dict:
                        body_pose = motion_dict['smplh:body_pose']
                        upper_body_pose = body_pose[:, upper_body_joints, :]
                        upper_body_flat = upper_body_pose.reshape(upper_body_pose.shape[0], -1)
                        motion_features.append(upper_body_flat)
                    
                    if 'smplh:left_hand_pose' in motion_dict:
                        left_hand_pose = motion_dict['smplh:left_hand_pose']
                        left_hand_flat = left_hand_pose.reshape(left_hand_pose.shape[0], -1)
                        motion_features.append(left_hand_flat)
                    
                    if 'smplh:right_hand_pose' in motion_dict:
                        right_hand_pose = motion_dict['smplh:right_hand_pose']
                        right_hand_flat = right_hand_pose.reshape(right_hand_pose.shape[0], -1)
                        motion_features.append(right_hand_flat)
                    
                    if motion_features:
                        # Concatenate features: (frames, 129)
                        motion_data = np.concatenate(motion_features, axis=1)
                        
                        # Convert to 6D rotation
                        motion_tensor = torch.from_numpy(motion_data).float()
                        motion_batch = motion_tensor.unsqueeze(0)  # (1, frames, 129)
                        motion_6d = convert_axis_angle_to_rotation_6d(motion_batch)
                        motion_6d = motion_6d.squeeze(0)  # (frames, 258)
                        
                        all_motion_data.append(motion_6d.numpy())
                        
                except Exception as e:
                    print(f"WARNING:  Error processing {motion_path}: {e}")
                    continue
        
        if not all_motion_data:
            print("WARNING:  No motion data found, using default normalization")
            return np.zeros(258), np.ones(258)
        
        # Concatenate all motion data
        all_motion = np.concatenate(all_motion_data, axis=0)  # (total_frames, 258)
        
        # Compute mean and std
        motion_mean = np.mean(all_motion, axis=0)  # (258,)
        motion_std = np.std(all_motion, axis=0)   # (258,)
        
        # Avoid division by zero
        motion_std = np.where(motion_std < 1e-6, 1.0, motion_std)
        
        print(f"Computed 6D rotation normalization stats:")
        print(f" Mean shape: {motion_mean.shape}")
        print(f" Std shape: {motion_std.shape}")
        print(f" Mean range: [{motion_mean.min():.4f}, {motion_mean.max():.4f}]")
        print(f" Std range: [{motion_std.min():.4f}, {motion_std.max():.4f}]")
        
        return motion_mean, motion_std
    
    @staticmethod
    def normalize(motion, mean, std):
        """Normalize motion data: (motion - mean) / std"""
        return (motion - mean) / (std + 1e-7)
    
    def get_audio_segment_from_npz(self, audio_npz_path, start_frame=0, length_frames=None):
        """Load audio segment from NPZ file, with fallback to WAV or MP3 file if NPZ fails"""
        try:
            # Load NPZ file containing preprocessed audio
            audio_data = np.load(audio_npz_path, allow_pickle=True)
            
            # Extract audio array from NPZ
            # Common keys for audio in NPZ files
            audio_keys = ['audio', 'audio_data', 'signal', 'waveform']
            audio = None
            
            for key in audio_keys:
                if key in audio_data:
                    audio = audio_data[key]
                    break
            
            if audio is None:
                # Try to get the first array from the NPZ
                keys = list(audio_data.keys())
                if keys:
                    audio = audio_data[keys[0]]
                else:
                    raise ValueError(f"No audio data found in {audio_npz_path}")
            
            # Ensure audio is 1D
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            if length_frames is None:
                length_frames = self.pose_length
            
            # Calculate audio frame count
            audio_frame_count = len(audio) * self.fps // self.audio_sr
            
            # Apply same adjustment logic as get_motion_segment
            end_frame = start_frame + length_frames
            original_start = start_frame
            if end_frame > audio_frame_count:
                end_frame = audio_frame_count
                start_frame = max(0, end_frame - length_frames)
                if start_frame != original_start:
                    print(f"Audio NPZ segment adjusted: start_frame {original_start} -> {start_frame} (file has {audio_frame_count} frames)")
            
            # Convert frame indices to audio samples
            start_sample = int(start_frame * (1 / self.fps) * self.audio_sr)
            end_sample = int(end_frame * (1 / self.fps) * self.audio_sr)
            
            # Extract segment
            audio_segment = audio[start_sample:end_sample]
            
            # Handle padding if needed
            expected_length = int(length_frames * (1 / self.fps) * self.audio_sr)
            if len(audio_segment) < expected_length:
                # Pad with last value
                padding = np.repeat(audio_segment[-1] if len(audio_segment) > 0 else 0, 
                                   expected_length - len(audio_segment))
                audio_segment = np.concatenate([audio_segment, padding])
            elif len(audio_segment) > expected_length:
                # Truncate
                audio_segment = audio_segment[:expected_length]
            
            # Ensure audio segment is not empty and has valid values
            if len(audio_segment) == 0:
                print(f"WARNING:  Empty audio segment, using zeros")
                audio_segment = np.zeros(expected_length, dtype=np.float32)
            elif np.all(audio_segment == 0):
                print(f"WARNING:  All-zero audio segment detected")
                # Add small random noise to avoid wav2vec2 issues
                audio_segment = np.random.normal(0, 1e-6, len(audio_segment)).astype(np.float32)
            elif np.std(audio_segment) < 1e-8:
                print(f"WARNING:  Very low variance audio segment, adding small noise")
                audio_segment = audio_segment + np.random.normal(0, 1e-6, len(audio_segment))
                
            return torch.from_numpy(audio_segment).float()
            
        except Exception as e:
            print(f"Error loading audio from NPZ {audio_npz_path}: {e}")
            print(f"Attempting to load from WAV or MP3 file as fallback...")
            
            # Fallback: try to load from corresponding WAV or MP3 file
            return self._load_audio_from_wav_fallback(audio_npz_path, start_frame, length_frames)

    def _load_audio_from_wav_fallback(self, audio_npz_path, start_frame=0, length_frames=None):
        """Fallback method to load audio from WAV or MP3 file when NPZ fails"""
        # Get paths for both WAV and MP3
        audio_paths = self._get_audio_paths_from_npz_path(audio_npz_path)
        
        # Try WAV first, then MP3
        audio_formats = ['wav', 'mp3']
        last_error = None
        
        for audio_format in audio_formats:
            audio_path = audio_paths[audio_format]
            
            try:
                if not os.path.exists(audio_path):
                    continue
                
                print(f"Attempting to load {audio_format.upper()} file: {audio_path}")
                
                # Load audio file using librosa (supports both WAV and MP3)
                audio, sr = librosa.load(audio_path, sr=self.audio_sr, mono=True)
                
                if length_frames is None:
                    length_frames = self.pose_length
                
                # Calculate audio frame count
                audio_frame_count = len(audio) * self.fps // self.audio_sr
                
                # Apply same adjustment logic as NPZ loading
                end_frame = start_frame + length_frames
                original_start = start_frame
                if end_frame > audio_frame_count:
                    end_frame = audio_frame_count
                    start_frame = max(0, end_frame - length_frames)
                    if start_frame != original_start:
                        print(f"Audio {audio_format.upper()} segment adjusted: start_frame {original_start} -> {start_frame} (file has {audio_frame_count} frames)")
                
                # Convert frame indices to audio samples
                start_sample = int(start_frame * (1 / self.fps) * self.audio_sr)
                end_sample = int(end_frame * (1 / self.fps) * self.audio_sr)
                
                # Extract segment
                audio_segment = audio[start_sample:end_sample]
                
                # Handle padding if needed
                expected_length = int(length_frames * (1 / self.fps) * self.audio_sr)
                if len(audio_segment) < expected_length:
                    # Pad with last value
                    padding = np.repeat(audio_segment[-1] if len(audio_segment) > 0 else 0, 
                                       expected_length - len(audio_segment))
                    audio_segment = np.concatenate([audio_segment, padding])
                elif len(audio_segment) > expected_length:
                    # Truncate
                    audio_segment = audio_segment[:expected_length]
                
                # Ensure audio segment is not empty and has valid values
                if len(audio_segment) == 0:
                    print(f"WARNING:  Empty audio segment from {audio_format.upper()}, using zeros")
                    audio_segment = np.zeros(expected_length, dtype=np.float32)
                elif np.all(audio_segment == 0):
                    print(f"WARNING:  All-zero audio segment from {audio_format.upper()} detected")
                    # Add small random noise to avoid wav2vec2 issues
                    audio_segment = np.random.normal(0, 1e-6, len(audio_segment)).astype(np.float32)
                elif np.std(audio_segment) < 1e-8:
                    print(f"WARNING:  Very low variance audio segment from {audio_format.upper()}, adding small noise")
                    audio_segment = audio_segment + np.random.normal(0, 1e-6, len(audio_segment))
                
                print(f"Successfully loaded audio from {audio_format.upper()} file: {audio_path}")
                return torch.from_numpy(audio_segment).float()
                
            except Exception as e:
                last_error = e
                print(f"Error loading {audio_format.upper()} file {audio_path}: {e}")
                continue
        
        # If all formats failed, return zero audio
        print(f"Error loading audio from fallback files: {last_error if last_error else 'No audio files found'}")
        expected_length = int((self.pose_length) * (self.audio_sr / self.fps))
        return torch.zeros(expected_length, dtype=torch.float32)

    def get_motion_segment(self, motion_path, start_frame=0, length_frames=None):
        """Load motion segment from NPZ or NPY file"""
        try:
            if length_frames is None:
                length_frames = self.pose_length
            
            # Check if file is .npy (direct array) or .npz (dictionary)
            if motion_path.endswith('.npy') or motion_path.endswith('_6d.npy'):
                # Load .npy or _6d.npy file (direct array, assumed to be 6D motion data)
                motion_data = np.load(motion_path, allow_pickle=True)  # (frames, 258)
                
                # Extract segment
                end_frame = start_frame + length_frames
                original_start = start_frame
                if end_frame > motion_data.shape[0]:
                    end_frame = motion_data.shape[0]
                    start_frame = max(0, end_frame - length_frames)
                    if start_frame != original_start:
                        print(f"Motion segment adjusted: start_frame {original_start} -> {start_frame} (file has {motion_data.shape[0]} frames)")
                
                motion_segment = motion_data[start_frame:end_frame]
                motion_output = torch.from_numpy(motion_segment).float()
                
                # Apply normalization if enabled
                if self.normalize_motion:
                    motion_output = self.normalize(motion_output, self.motion_mean, self.motion_std)
                
                return motion_output
            else:
                # Load .npz file (dictionary format)
                motion_dict = np.load(motion_path, allow_pickle=True)
            
            if self.use_6d_rotation and self.use_preprocessed_6d and 'smplh:motion_6d' in motion_dict:
                # Use preprocessed 6D rotation data
                motion_6d_data = motion_dict['smplh:motion_6d']  # (frames, 258)
                
                # Extract segment
                end_frame = start_frame + length_frames
                original_start = start_frame
                if end_frame > motion_6d_data.shape[0]:
                    end_frame = motion_6d_data.shape[0]
                    start_frame = max(0, end_frame - length_frames)
                    if start_frame != original_start:
                        print(f"6D motion segment adjusted: start_frame {original_start} -> {start_frame} (file has {motion_6d_data.shape[0]} frames)")
                
                motion_segment = motion_6d_data[start_frame:end_frame]
                motion_output = torch.from_numpy(motion_segment).float()
                
            else:
                # Extract only upper body + hands features (original logic)
                motion_features = []
                
                # Define upper body joint indices (exclude lower body)
                # Lower body: [0, 1, 3, 4, 6, 7, 9, 10] (hip, knee, ankle, foot)
                # Upper body: [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] (spine, neck, collar, head, shoulder, elbow, wrist)
                lower_body_joints = [0, 1, 3, 4, 6, 7, 9, 10]
                upper_body_joints = [j for j in range(21) if j not in lower_body_joints]
                
                # 1. Extract upper body joints from body_pose
                if 'smplh:body_pose' in motion_dict:
                    body_pose = motion_dict['smplh:body_pose']  # (frames, 21, 3)
                    upper_body_pose = body_pose[:, upper_body_joints, :]  # (frames, 13, 3)
                    upper_body_flat = upper_body_pose.reshape(upper_body_pose.shape[0], -1)  # (frames, 39)
                    motion_features.append(upper_body_flat)
                
                # 2. Extract left hand pose
                if 'smplh:left_hand_pose' in motion_dict:
                    left_hand_pose = motion_dict['smplh:left_hand_pose']  # (frames, 15, 3)
                    left_hand_flat = left_hand_pose.reshape(left_hand_pose.shape[0], -1)  # (frames, 45)
                    motion_features.append(left_hand_flat)
                
                # 3. Extract right hand pose  
                if 'smplh:right_hand_pose' in motion_dict:
                    right_hand_pose = motion_dict['smplh:right_hand_pose']  # (frames, 15, 3)
                    right_hand_flat = right_hand_pose.reshape(right_hand_pose.shape[0], -1)  # (frames, 45)
                    motion_features.append(right_hand_flat)
                
                if not motion_features:
                    raise ValueError(f"No required motion features found in {motion_path}")
                
                # Concatenate all features: upper_body(39) + left_hand(45) + right_hand(45) = 129 dims
                motion_data = np.concatenate(motion_features, axis=1)
                
                # Extract segment BEFORE conversion to avoid converting padding
                end_frame = start_frame + length_frames
                original_start = start_frame
                if end_frame > motion_data.shape[0]:
                    end_frame = motion_data.shape[0]
                    start_frame = max(0, end_frame - length_frames)
                    if start_frame != original_start:
                        print(f"Motion segment adjusted: start_frame {original_start} -> {start_frame} (file has {motion_data.shape[0]} frames)")
                
                motion_segment = motion_data[start_frame:end_frame]
                
                # Convert to torch tensor
                motion_tensor = torch.from_numpy(motion_segment).float()
                
                if self.use_6d_rotation:
                    # Convert axis-angle to 6D rotation: (seq_len, 129) -> (seq_len, 258)
                    # Add batch dimension for conversion function: (1, seq_len, 129)
                    motion_batch = motion_tensor.unsqueeze(0)
                    motion_6d = convert_axis_angle_to_rotation_6d(motion_batch)
                    # Remove batch dimension: (seq_len, 258)
                    motion_output = motion_6d.squeeze(0)
                else:
                    # Keep as 3D euler angles: (seq_len, 129)
                    motion_output = motion_tensor
            
            # Apply normalization if enabled
            if self.normalize_motion:
                motion_output = self.normalize(motion_output, self.motion_mean, self.motion_std)
            
            return motion_output
            
        except Exception as e:
            print(f"Error loading motion from {motion_path}: {e}")
            # Return zero motion with correct dimensions based on rotation representation
            if self.use_6d_rotation:
                return torch.zeros((self.pose_length, 258), dtype=torch.float32)  # 6D rotation
            else:
                return torch.zeros((self.pose_length, 129), dtype=torch.float32)  # 3D euler

    def __len__(self):
        # Each pair produces 2 samples: A-B and B-A
        return self.samples_per_conversation * 2

    def __getitem__(self, index):
        # Determine which file pair and segment to use
        # Each pair produces 2 samples: A-B (index % 2 == 0) and B-A (index % 2 == 1)
        pair_index = index // 2
        is_reversed = (index % 2) == 1
        
        file_index, segment_index = self._get_file_and_segment_index(pair_index)
        motion_path1, audio_path1, motion_path2, audio_path2 = self.data_list[file_index]
        total_frames = self.total_frames_per_file[file_index]
        
        # Calculate start frame based on segment index
        max_start_frame = max(0, total_frames - self.pose_length)
        
        # Calculate start frame for this segment
        if self.samples_per_conversation > len(self.data_list):
            # Multiple segments per file - evenly space them
            samples_per_file = self.samples_per_conversation // len(self.data_list)
            segment_size = max_start_frame / max(1, samples_per_file - 1)
            start_frame = int(segment_index * segment_size)
            start_frame = min(start_frame, max_start_frame)  # Ensure within bounds
        else:
            # One segment per file or use all segments
            if segment_index == 0:
                start_frame = 0
            else:
                # Use non-overlapping consecutive segments
                start_frame = segment_index * self.pose_length
                start_frame = min(start_frame, max_start_frame)
        
        # Determine which file is A and which is B based on is_reversed
        if is_reversed:
            # B-A version: file2 is motion_latent, file1 is motion_latent_other
            motion_latent_path = motion_path2
            motion_latent_other_path = motion_path1
            audio_path = audio_path2
            audio_other_path = audio_path1
            video_id_suffix = "B-A"
        else:
            # A-B version: file1 is motion_latent, file2 is motion_latent_other
            motion_latent_path = motion_path1
            motion_latent_other_path = motion_path2
            audio_path = audio_path1
            audio_other_path = audio_path2
            video_id_suffix = "A-B"
        
        # Load motion and audio from the calculated start frame
        motion_latent = self.get_motion_segment(motion_latent_path, start_frame=start_frame)
        motion_latent_other = self.get_motion_segment(motion_latent_other_path, start_frame=start_frame)
        audio = self.get_audio_segment_from_npz(audio_path, start_frame=start_frame)
        audio_other = self.get_audio_segment_from_npz(audio_other_path, start_frame=start_frame)
        
        # Load style_latent as a random segment from the same file as motion_latent
        max_start_frame = max(0, total_frames - self.pose_length)
        style_start_frame = random.randint(0, max_start_frame) if max_start_frame > 0 else 0
        style_latent = self.get_motion_segment(motion_latent_path, start_frame=style_start_frame)
        
        # Get relationship one-hot vector based on the filename
        motion_filename = os.path.basename(motion_latent_path)
        # Ensure filename is properly handled (remove _6d.npy extension if present)
        # get_relationship_onehot and get_personality_traits will handle the extension removal
        relationship_onehot = torch.from_numpy(self.get_relationship_onehot(motion_filename))
        
        # Get personality traits based on the filename
        personality_traits = torch.from_numpy(self.get_personality_traits(motion_filename))
        
        return {
            # Main data with pair information
            'motion_latent': motion_latent,           # Target motion (A or B)
            'motion_latent_other': motion_latent_other,  # Other motion (B or A)
            'audio': audio,                           # Input audio (A or B)
            'audio_other': audio_other,              # Other audio (B or A)

            'style_latent': style_latent,            # Random segment from same file as motion_latent
            'relationship_onehot': relationship_onehot,  # 4-dim one-hot: [friends/coworkers, stranger, family-generic/familiar-generic, dating/spouse/romantic_partner]
            'personality_traits': personality_traits,  # 5-dim vector (normalized to 0~1): [extraversion_raw, agreeableness_raw, conscientiousness_raw, neuroticism_raw, openness_raw]
            
            # Additional metadata
            'motion_path': motion_latent_path,
            'motion_other_path': motion_latent_other_path,
            'audio_path': audio_path,
            'audio_other_path': audio_other_path,
            'video_id': f"{os.path.basename(motion_latent_path).replace('.npz', '')}_{video_id_suffix}_segment_{segment_index}",
            'start_frame': start_frame,
            'file_index': file_index,
            'segment_index': segment_index,
            'is_reversed': is_reversed,
        }
    
    def _get_file_and_segment_index(self, pair_index):
        """Convert pair index to file index and segment index"""
        if len(self.data_list) == 0:
            return 0, 0
        
        # Calculate samples per file pair
        samples_per_file = self.samples_per_conversation // len(self.data_list)
        
        # Determine file index and segment index
        file_index = pair_index % len(self.data_list)
        segment_index = pair_index // len(self.data_list)
        
        return file_index, segment_index
    
    @classmethod
    def create_single_file_loader(cls, filename, cfg=None, use_6d_rotation=True):
        """Create a loader that only uses one specific file for overfitting"""
        if cfg is None:
            # Create a minimal config if none provided
            class MinimalConfig:
                class Model:
                    pose_fps = 30
                    audio_sr = 16000
                    pose_length = 150
                
                class Data:
                    # max_files = 1
                    # specific_files = [filename]
                    samples_per_conversation = -1
                    normalize_motion = True

                class Paths:
                    data_root = "/mnt/nas2/Projects/meta_data/naturalistic/test"
                
                model = Model()
                data = Data()
            
            cfg = MinimalConfig()
            cfg.data.use_6d_rotation = use_6d_rotation
            cfg.data.samples_per_conversation = -1
            cfg.paths.data_root = "/mnt/nas2/Projects/meta_data/naturalistic/test"
        else:
            # Modify existing config to use only one file
            if not hasattr(cfg, 'data'):
                cfg.data = type('Data', (), {})()
            # cfg.data.max_files = 1
            # cfg.data.specific_files = [filename]
            cfg.data.samples_per_conversation = -1
            cfg.data.use_6d_rotation = use_6d_rotation
        
        return cls(cfg)

if __name__ == '__main__':
    # Test normal loader (all files)
    print("=== Testing normal loader (all files) ===")
    
    # Create a minimal config for testing
    class TestConfig:
        class Paths:
            data_root = '/mnt/nas2/Projects/meta_data/naturalistic/test'  # Adjust path as needed
        
        class Model:
            pose_fps = 30
            audio_sr = 16000
            pose_length = 150
        
        class Data:
            max_files = None
            specific_files = None
            samples_per_conversation = -1
            use_6d_rotation = True
            normalize_motion = True
        
        paths = Paths()
        model = Model()
        data = Data()
    
    cfg = TestConfig()
    # Update to use local data path
    cfg.paths.data_root = '/home/peng/dyadic_motion/dyadic_dit/data'
    dyadic_loader = DyadicLoader(cfg)
    print(f"\nDataset length: {len(dyadic_loader)}")
    print(f"Number of data file pairs: {len(dyadic_loader.data_list)}")
    
    # Test samples from the loader
    if len(dyadic_loader) > 0:
        print("\n=== Testing pair loader samples ===")
        
        # Test A-B version (index 0)
        sample_ab = dyadic_loader[0]
        print(f"\nA-B sample:")
        print(f" motion_latent: {sample_ab['motion_latent'].shape}")
        print(f" motion_latent_other: {sample_ab['motion_latent_other'].shape}")
        print(f" audio: {sample_ab['audio'].shape}")
        print(f" audio_other: {sample_ab['audio_other'].shape}")
        print(f" relationship_onehot: {sample_ab['relationship_onehot']}")
        print(f" personality_traits: {sample_ab['personality_traits']}")
        print(f" video_id: {sample_ab['video_id']}")
        print(f" is_reversed: {sample_ab['is_reversed']}")
        print(f" motion_path: {os.path.basename(sample_ab['motion_path'])}")
        print(f" motion_other_path: {os.path.basename(sample_ab['motion_other_path'])}")
        print(f" audio_path: {os.path.basename(sample_ab['audio_path'])}")
        print(f" audio_other_path: {os.path.basename(sample_ab['audio_other_path'])}")
        
        # Test B-A version (index 1)
        if len(dyadic_loader) > 1:
            sample_ba = dyadic_loader[1]
            print(f"\nB-A sample:")
            print(f" motion_latent: {sample_ba['motion_latent'].shape}")
            print(f" motion_latent_other: {sample_ba['motion_latent_other'].shape}")
            print(f" audio: {sample_ba['audio'].shape}")
            print(f" audio_other: {sample_ba['audio_other'].shape}")
            print(f" relationship_onehot: {sample_ba['relationship_onehot']}")
            print(f" personality_traits: {sample_ba['personality_traits']}")
            print(f" video_id: {sample_ba['video_id']}")
            print(f" is_reversed: {sample_ba['is_reversed']}")
            print(f" motion_path: {os.path.basename(sample_ba['motion_path'])}")
            print(f" motion_other_path: {os.path.basename(sample_ba['motion_other_path'])}")
            print(f" audio_path: {os.path.basename(sample_ba['audio_path'])}")
            print(f" audio_other_path: {os.path.basename(sample_ba['audio_other_path'])}")
            
            # Verify that A-B and B-A are indeed reversed
            print(f"\nVerification:")
            print(f" A-B motion_latent == B-A motion_latent_other: {torch.equal(sample_ab['motion_latent'], sample_ba['motion_latent_other'])}")
            print(f" A-B motion_latent_other == B-A motion_latent: {torch.equal(sample_ab['motion_latent_other'], sample_ba['motion_latent'])}")
            print(f" A-B audio == B-A audio_other: {torch.equal(sample_ab['audio'], sample_ba['audio_other'])}")
            print(f" A-B audio_other == B-A audio: {torch.equal(sample_ab['audio_other'], sample_ba['audio'])}")
            
            # Test style_latent functionality
            print(f"\nStyle latent test:")
            print(f" A-B style_latent shape: {sample_ab['style_latent'].shape}")
            print(f" A-B motion_latent shape: {sample_ab['motion_latent'].shape}")
            print(f" A-B style_latent == motion_latent: {torch.equal(sample_ab['style_latent'], sample_ab['motion_latent'])}")
            print(f" B-A style_latent shape: {sample_ba['style_latent'].shape}")
            print(f" B-A motion_latent shape: {sample_ba['motion_latent'].shape}")
            print(f" B-A style_latent == motion_latent: {torch.equal(sample_ba['style_latent'], sample_ba['motion_latent'])}")
            
            # Test multiple samples to verify randomness
            print(f"\nTesting style_latent randomness:")
            sample1 = dyadic_loader[0]
            sample2 = dyadic_loader[0]  # Same index, should get different style_latent
            print(f" Same index, different style_latent: {not torch.equal(sample1['style_latent'], sample2['style_latent'])}")
            
            # Test relationship_onehot consistency
            print(f"\nTesting relationship_onehot:")
            print(f" A-B relationship_onehot: {sample_ab['relationship_onehot']}")
            print(f" B-A relationship_onehot: {sample_ba['relationship_onehot']}")
            print(f" A-B and B-A have same relationship: {torch.equal(sample_ab['relationship_onehot'], sample_ba['relationship_onehot'])}")
            
            # Test personality_traits consistency
            print(f"\nTesting personality_traits:")
            print(f" A-B personality_traits: {sample_ab['personality_traits']}")
            print(f" B-A personality_traits: {sample_ba['personality_traits']}")
            print(f" A-B personality_traits shape: {sample_ab['personality_traits'].shape}")
            print(f" B-A personality_traits shape: {sample_ba['personality_traits'].shape}")
        
        # Test audio fallback functionality
        print("\n=== Testing audio fallback functionality ===")
        if len(dyadic_loader.data_list) > 0:
            # Test with a non-existent NPZ file to trigger fallback
            test_npz_path = "/fake/path/to/nonexistent.npz"
            print(f"Testing fallback with non-existent NPZ: {test_npz_path}")
            try:
                audio_fallback = dyadic_loader._load_audio_from_wav_fallback(test_npz_path, start_frame=0)
                print(f"Fallback audio shape: {audio_fallback.shape}")
                print("Fallback test completed (expected to fail gracefully)")
            except Exception as e:
                print(f"Fallback test failed as expected: {e}")
        
        # Test with actual NPZ file if available
        if len(dyadic_loader.data_list) > 0:
            motion_path1, audio_path1, motion_path2, audio_path2 = dyadic_loader.data_list[0]
            print(f"\nTesting audio loading from actual file: {os.path.basename(audio_path1)}")
            try:
                audio_segment = dyadic_loader.get_audio_segment_from_npz(audio_path1, start_frame=0)
                print(f"Successfully loaded audio segment: {audio_segment.shape}")
                print(f"Audio stats: mean={audio_segment.mean():.6f}, std={audio_segment.std():.6f}")
            except Exception as e:
                print(f"Error loading audio: {e}")
    else:
        print("No data files found for testing")