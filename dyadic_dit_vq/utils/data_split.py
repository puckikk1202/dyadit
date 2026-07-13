"""
Data splitting utilities for reproducible train/val/test splits
"""

import json
import os
import torch
from torch.utils.data import Subset


def get_fixed_split_indices(dataset_size, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Get fixed split indices for train/val/test splits
    
    Args:
        dataset_size: Total number of samples
        train_ratio: Ratio of training data (default: 0.8)
        val_ratio: Ratio of validation data (default: 0.1)
        test_ratio: Ratio of test data (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        train_indices, val_indices, test_indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    
    # Generate shuffled indices
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    
    # Calculate split sizes
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size  # Remaining goes to test
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"Dataset split (seed={seed}):")
    print(f"   Total: {dataset_size}")
    print(f"   Train: {len(train_indices)} ({len(train_indices)/dataset_size*100:.1f}%)")
    print(f"   Val:   {len(val_indices)} ({len(val_indices)/dataset_size*100:.1f}%)")
    print(f"   Test:  {len(test_indices)} ({len(test_indices)/dataset_size*100:.1f}%)")
    
    return train_indices, val_indices, test_indices


def create_fixed_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Create fixed train/val/test splits using Subset
    
    Args:
        dataset: PyTorch Dataset
        train_ratio: Ratio of training data (default: 0.8)
        val_ratio: Ratio of validation data (default: 0.1)
        test_ratio: Ratio of test data (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        train_dataset, val_dataset, test_dataset (Subset objects)
    """
    train_indices, val_indices, test_indices = get_fixed_split_indices(
        len(dataset), train_ratio, val_ratio, test_ratio, seed
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def get_split_by_name(dataset, split_name='train', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Get a specific split by name
    
    Args:
        dataset: PyTorch Dataset
        split_name: 'train', 'val', or 'test'
        train_ratio: Ratio of training data (default: 0.8)
        val_ratio: Ratio of validation data (default: 0.1)
        test_ratio: Ratio of test data (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Subset for the requested split
    """
    train_indices, val_indices, test_indices = get_fixed_split_indices(
        len(dataset), train_ratio, val_ratio, test_ratio, seed
    )
    
    if split_name == 'train':
        return Subset(dataset, train_indices)
    elif split_name in ['val', 'validation']:
        return Subset(dataset, val_indices)
    elif split_name == 'test':
        return Subset(dataset, test_indices)
    else:
        raise ValueError(f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'")


def create_splits_from_json(dataset, split_json_path, data_root=None):
    """
    Create train/val/test splits based on a JSON file that specifies the split
    
    Args:
        dataset: PyTorch Dataset (OverfitLoader or similar)
        split_json_path: Path to JSON file containing split information
        data_root: Root directory for data files (optional, used to resolve paths)
    
    Returns:
        train_dataset, val_dataset, test_dataset (Subset objects)
        
    JSON format expected:
    {
        "train": {
            "V00_S0030_I00000125": [...],
            "V00_S0030_I00000129": [...],
            ...
        },
        "val": {
            "V00_S0672_I00000543": [...],
            ...
        },
        "test": {
            "V00_S0750_I00000583": [...],
            ...
        }
    }
    """
    print(f"📂 Loading split information from: {split_json_path}")
    
    # Load JSON file
    with open(split_json_path, 'r') as f:
        split_info = json.load(f)
    
    # Extract interaction IDs for each split
    train_ids = set(split_info.get('train', {}).keys())
    val_ids = set(split_info.get('val', {}).keys())
    test_ids = set(split_info.get('test', {}).keys())
    
    print(f"   JSON contains:")
    print(f"   - Train: {len(train_ids)} interaction IDs")
    print(f"   - Val:   {len(val_ids)} interaction IDs")
    print(f"   - Test:  {len(test_ids)} interaction IDs")
    
    # Map dataset indices to splits
    # For OverfitLoader, each dataset index corresponds to a pair sample
    # We need to extract the interaction ID from the file paths
    train_indices = []
    val_indices = []
    test_indices = []
    unmatched_indices = []
    
    # Helper function to extract interaction ID from file path
    def extract_interaction_id(file_path):
        """Extract interaction ID (V00_S0030_I00000125) from file path"""
        basename = os.path.basename(file_path)
        # Remove extension and participant suffix
        # Example: "V00_S0030_I00000125_P0045A.npz" -> "V00_S0030_I00000125"
        parts = basename.replace('.npz', '').split('_')
        if len(parts) >= 3:
            # Extract V00_S0030_I00000125 part
            interaction_id = '_'.join(parts[:3])
            return interaction_id
        return None
    
    # Efficient method: Check if dataset has data_list attribute (like OverfitLoader)
    if hasattr(dataset, 'data_list'):
        print(f"\n🔍 Analyzing {len(dataset.data_list)} data file pairs...")
        
        # Build a mapping from dataset index to interaction ID
        # OverfitLoader returns 2 samples per file pair (A-B and B-A)
        # So we need to map each dataset index to the corresponding file pair
        for pair_idx, data_paths in enumerate(dataset.data_list):
            # data_paths = [motion_path1, audio_path1, motion_path2, audio_path2]
            motion_path1 = data_paths[0]
            
            # Extract interaction ID from the first motion file
            interaction_id = extract_interaction_id(motion_path1)
            
            if interaction_id:
                # OverfitLoader produces 2 samples per pair: A-B (pair_idx * 2) and B-A (pair_idx * 2 + 1)
                dataset_idx_ab = pair_idx * 2
                dataset_idx_ba = pair_idx * 2 + 1
                
                # Assign both A-B and B-A samples to the same split
                if interaction_id in train_ids:
                    train_indices.extend([dataset_idx_ab, dataset_idx_ba])
                elif interaction_id in val_ids:
                    val_indices.extend([dataset_idx_ab, dataset_idx_ba])
                elif interaction_id in test_ids:
                    test_indices.extend([dataset_idx_ab, dataset_idx_ba])
                else:
                    unmatched_indices.extend([dataset_idx_ab, dataset_idx_ba])
            else:
                print(f"Could not extract interaction ID from: {motion_path1}")
                unmatched_indices.extend([pair_idx * 2, pair_idx * 2 + 1])
    
    else:
        # Fallback: Iterate through dataset (slow, loads data)
        print(f"\n🔍 Analyzing dataset (slow method - loading samples)...")
        print("   Consider adding get_sample_id() method to dataset for faster splitting")
        
        for idx in range(len(dataset)):
            # Get sample from dataset (THIS IS SLOW!)
            sample = dataset[idx]
            
            # Extract interaction ID from sample
            if isinstance(sample, dict):
                # Try to get from video_id or motion_path
                if 'video_id' in sample:
                    video_id = sample['video_id']
                    # Extract interaction ID from video_id
                    interaction_id = extract_interaction_id(video_id)
                elif 'motion_path' in sample:
                    interaction_id = extract_interaction_id(sample['motion_path'])
                else:
                    interaction_id = None
            else:
                interaction_id = None
            
            # Assign to appropriate split
            if interaction_id:
                if interaction_id in train_ids:
                    train_indices.append(idx)
                elif interaction_id in val_ids:
                    val_indices.append(idx)
                elif interaction_id in test_ids:
                    test_indices.append(idx)
                else:
                    unmatched_indices.append(idx)
            else:
                unmatched_indices.append(idx)
    
    # Print statistics
    print(f"\nDataset split from JSON:")
    print(f"   Total dataset size: {len(dataset)}")
    print(f"   Train: {len(train_indices)} samples ({len(train_indices)/len(dataset)*100:.1f}%)")
    print(f"   Val:   {len(val_indices)} samples ({len(val_indices)/len(dataset)*100:.1f}%)")
    print(f"   Test:  {len(test_indices)} samples ({len(test_indices)/len(dataset)*100:.1f}%)")
    
    if unmatched_indices:
        print(f"   Unmatched: {len(unmatched_indices)} samples (will be excluded)")
        # Show a few examples of unmatched interaction IDs
        if hasattr(dataset, 'data_list') and len(dataset.data_list) > 0:
            print(f"   Example unmatched files:")
            for i, idx in enumerate(unmatched_indices[:5]):  # Show first 5
                pair_idx = idx // 2
                if pair_idx < len(dataset.data_list):
                    motion_path = dataset.data_list[pair_idx][0]
                    interaction_id = extract_interaction_id(motion_path)
                    print(f"     - {interaction_id} (from {os.path.basename(motion_path)})")
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def print_split_info(train_dataset, val_dataset, test_dataset):
    """Print information about dataset splits"""
    total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    
    print("\n" + "="*60)
    print("Dataset Split Information")
    print("="*60)
    print(f"Train: {len(train_dataset):5d} samples ({len(train_dataset)/total*100:.1f}%)")
    print(f"Val:   {len(val_dataset):5d} samples ({len(val_dataset)/total*100:.1f}%)")
    print(f"Test:  {len(test_dataset):5d} samples ({len(test_dataset)/total*100:.1f}%)")
    print(f"Total: {total:5d} samples")
    print("="*60 + "\n")

