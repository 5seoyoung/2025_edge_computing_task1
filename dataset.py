"""
EchoNet-Dynamic Video Dataset Loader
"""
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional


class EchoNetVideoDataset(Dataset):
    """
    Dataset class for loading EchoNet-Dynamic videos
    
    Args:
        video_dir: Directory containing .avi video files
        filelist_path: Path to FileList.csv containing (FileName, EF) pairs
        num_frames: Number of frames to sample uniformly from video
        image_size: Target image size (will be resized to image_size x image_size)
        transform: Optional torchvision transforms
        split: 'train', 'val', or 'test' (for future use)
    """
    
    def __init__(
        self,
        video_dir: Path,
        filelist_path: Path,
        num_frames: int = 32,
        image_size: int = 112,
        transform: Optional[transforms.Compose] = None,
        split: str = "train"
    ):
        self.video_dir = Path(video_dir)
        self.filelist_path = Path(filelist_path)
        self.num_frames = num_frames
        self.image_size = image_size
        
        # Load file list
        if not self.filelist_path.exists():
            raise FileNotFoundError(f"FileList.csv not found at {filelist_path}")
        
        df = pd.read_csv(self.filelist_path)
        
        # Debug: Print CSV info
        print(f"Loaded CSV with {len(df)} rows")
        print(f"CSV columns: {df.columns.tolist()}")
        
        # Check for required columns
        if 'FileName' not in df.columns:
            # Try common alternative column names
            if 'filename' in df.columns:
                df['FileName'] = df['filename']
            elif 'file_name' in df.columns:
                df['FileName'] = df['file_name']
            elif 'Video' in df.columns:
                df['FileName'] = df['Video']
            else:
                raise ValueError(f"CSV must contain 'FileName' column. Available columns: {df.columns.tolist()}")
        
        if 'EF' not in df.columns:
            # Try common alternative column names
            if 'ef' in df.columns:
                df['EF'] = df['ef']
            elif 'EjectionFraction' in df.columns:
                df['EF'] = df['EjectionFraction']
            else:
                raise ValueError(f"CSV must contain 'EF' column. Available columns: {df.columns.tolist()}")
        
        # Filter by split if needed (assuming FileList has a 'Split' column)
        # If not, use all data
        if 'Split' in df.columns:
            # CSV uses uppercase: TRAIN, VAL, TEST
            # Convert split parameter to uppercase for matching
            split_upper = split.upper()
            self.df = df[df['Split'] == split_upper].reset_index(drop=True)
            print(f"After filtering by Split='{split_upper}': {len(self.df)} rows")
        else:
            self.df = df.reset_index(drop=True)
            print(f"Using all data: {len(self.df)} rows")
        
        if len(self.df) == 0:
            raise ValueError(f"Dataset is empty! Check FileList.csv and video directory.")
        
        # Default transform: resize + normalize (ImageNet stats)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Load video and return sampled frames with EF label
        
        Returns:
            video_tensor: (num_frames, 3, H, W)
            ef_label: float
        """
        row = self.df.iloc[idx]
        filename = row['FileName']
        # Convert to float32 to match model's expected dtype
        ef_label = torch.tensor(float(row['EF']), dtype=torch.float32)
        
        # Add .avi extension if not present
        if not filename.endswith('.avi'):
            filename = filename + '.avi'
        
        # Load video file
        video_path = self.video_dir / filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Read video using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Read all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        
        # Uniform sampling
        if len(frames) >= self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            # If video has fewer frames, repeat last frame
            sampled_frames = frames.copy()
            while len(sampled_frames) < self.num_frames:
                sampled_frames.append(frames[-1])
        
        # Apply transforms
        transformed_frames = []
        for frame in sampled_frames:
            transformed_frame = self.transform(frame)
            transformed_frames.append(transformed_frame)
        
        # Stack frames: (num_frames, 3, H, W)
        video_tensor = torch.stack(transformed_frames, dim=0)
        
        return video_tensor, ef_label


def create_data_loaders(
    video_dir: Path,
    filelist_path: Path,
    num_frames: int = 32,
    image_size: int = 112,
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Create train/val/test data loaders
    
    If CSV has 'Split' column, use it. Otherwise, use random split.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader, random_split
    import pandas as pd
    
    # Check if CSV has Split column
    df = pd.read_csv(filelist_path)
    has_split_column = 'Split' in df.columns
    
    if has_split_column:
        # Use predefined splits from CSV
        print(f"Loading dataset from: {filelist_path}")
        print(f"Video directory: {video_dir}")
        print("Using predefined splits from CSV 'Split' column")
        
        # Create separate datasets for each split
        train_dataset = EchoNetVideoDataset(
            video_dir=video_dir,
            filelist_path=filelist_path,
            num_frames=num_frames,
            image_size=image_size,
            split='TRAIN'  # CSV uses 'TRAIN', 'VAL', 'TEST'
        )
        
        val_dataset = EchoNetVideoDataset(
            video_dir=video_dir,
            filelist_path=filelist_path,
            num_frames=num_frames,
            image_size=image_size,
            split='VAL'
        )
        
        test_dataset = EchoNetVideoDataset(
            video_dir=video_dir,
            filelist_path=filelist_path,
            num_frames=num_frames,
            image_size=image_size,
            split='TEST'
        )
        
        print(f"Dataset sizes from CSV Split:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Val: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")
        
    else:
        # Use random split
        print(f"Loading dataset from: {filelist_path}")
        print(f"Video directory: {video_dir}")
        print("Using random split (CSV has no 'Split' column)")
        
        full_dataset = EchoNetVideoDataset(
            video_dir=video_dir,
            filelist_path=filelist_path,
            num_frames=num_frames,
            image_size=image_size
        )
        
        # Split dataset
        total_size = len(full_dataset)
        print(f"Total dataset size: {total_size}")
        
        if total_size == 0:
            raise ValueError(
                f"Dataset is empty! Please check:\n"
                f"  1. FileList.csv exists at: {filelist_path}\n"
                f"  2. FileList.csv contains 'FileName' and 'EF' columns\n"
                f"  3. Video directory exists at: {video_dir}\n"
                f"  4. Videos match the filenames in FileList.csv"
            )
        
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

