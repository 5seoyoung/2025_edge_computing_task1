"""
Create sample dataset for Colab testing
Selects a subset of videos from the full dataset
"""
import pandas as pd
import shutil
from pathlib import Path
import random

def create_sample_dataset(
    source_data_root: str,
    target_data_root: str,
    num_samples_per_split: dict = {'TRAIN': 100, 'VAL': 20, 'TEST': 20},
    seed: int = 42
):
    """
    Create a sample dataset by copying a subset of videos
    
    Args:
        source_data_root: Path to original dataset
        target_data_root: Path to save sample dataset
        num_samples_per_split: Number of samples per split
        seed: Random seed
    """
    source_path = Path(source_data_root)
    target_path = Path(target_data_root)
    
    # Read original FileList.csv
    filelist_path = source_path / "FileList.csv"
    if not filelist_path.exists():
        raise FileNotFoundError(f"FileList.csv not found at {filelist_path}")
    
    df = pd.read_csv(filelist_path)
    print(f"Original dataset: {len(df)} videos")
    
    # Create target directories
    target_path.mkdir(exist_ok=True, parents=True)
    (target_path / "Videos").mkdir(exist_ok=True, parents=True)
    
    # Sample videos by split
    sample_df_list = []
    random.seed(seed)
    
    for split, num_samples in num_samples_per_split.items():
        split_df = df[df['Split'] == split].copy()
        if len(split_df) < num_samples:
            print(f"Warning: Only {len(split_df)} samples in {split}, using all")
            selected_df = split_df
        else:
            selected_df = split_df.sample(n=num_samples, random_state=seed)
        
        print(f"{split}: Selected {len(selected_df)} samples")
        sample_df_list.append(selected_df)
        
        # Copy video files
        source_video_dir = source_path / "Videos"
        target_video_dir = target_path / "Videos"
        
        for _, row in selected_df.iterrows():
            filename = row['FileName']
            if not filename.endswith('.avi'):
                filename = filename + '.avi'
            
            source_video = source_video_dir / filename
            target_video = target_video_dir / filename
            
            if source_video.exists():
                shutil.copy2(source_video, target_video)
            else:
                print(f"Warning: Video not found: {source_video}")
    
    # Combine and save sample FileList.csv
    sample_df = pd.concat(sample_df_list, ignore_index=True)
    sample_filelist_path = target_path / "FileList.csv"
    sample_df.to_csv(sample_filelist_path, index=False)
    
    print(f"\nSample dataset created:")
    print(f"  Total videos: {len(sample_df)}")
    print(f"  Location: {target_path}")
    print(f"  FileList.csv: {sample_filelist_path}")
    
    # Print split distribution
    print(f"\nSplit distribution:")
    for split in ['TRAIN', 'VAL', 'TEST']:
        count = len(sample_df[sample_df['Split'] == split])
        print(f"  {split}: {count}")

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python create_sample_data.py <source_path> <target_path> [train_samples] [val_samples] [test_samples]")
        print("\nExample:")
        print("  python create_sample_data.py /path/to/echonet_dynamic /path/to/sample_echonet 100 20 20")
        sys.exit(1)
    
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    
    # Optional: specify number of samples per split
    num_samples = {
        'TRAIN': int(sys.argv[3]) if len(sys.argv) > 3 else 100,
        'VAL': int(sys.argv[4]) if len(sys.argv) > 4 else 20,
        'TEST': int(sys.argv[5]) if len(sys.argv) > 5 else 20
    }
    
    create_sample_dataset(source_path, target_path, num_samples)

