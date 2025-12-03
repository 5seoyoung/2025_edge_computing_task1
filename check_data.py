"""
Quick script to check data structure
"""
import pandas as pd
from pathlib import Path
from config import Config

print("="*70)
print("Checking Data Structure")
print("="*70)

# Check paths
print(f"\n1. Checking paths:")
print(f"   DATA_ROOT: {Config.DATA_ROOT}")
print(f"   VIDEO_DIR: {Config.VIDEO_DIR}")
print(f"   FILELIST_PATH: {Config.FILELIST_PATH}")

print(f"\n   VIDEO_DIR exists: {Config.VIDEO_DIR.exists()}")
print(f"   FILELIST_PATH exists: {Config.FILELIST_PATH.exists()}")

# Check FileList.csv
if Config.FILELIST_PATH.exists():
    print(f"\n2. Reading FileList.csv...")
    df = pd.read_csv(Config.FILELIST_PATH)
    
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"\n   First few rows:")
    print(df.head())
    
    # Check for required columns
    print(f"\n3. Checking required columns:")
    has_filename = 'FileName' in df.columns or 'filename' in df.columns or 'file_name' in df.columns or 'Video' in df.columns
    has_ef = 'EF' in df.columns or 'ef' in df.columns or 'EjectionFraction' in df.columns
    
    print(f"   Has filename column: {has_filename}")
    print(f"   Has EF column: {has_ef}")
    
    if has_filename and has_ef:
        # Get actual column names
        filename_col = None
        for col in ['FileName', 'filename', 'file_name', 'Video']:
            if col in df.columns:
                filename_col = col
                break
        
        ef_col = None
        for col in ['EF', 'ef', 'EjectionFraction']:
            if col in df.columns:
                ef_col = col
                break
        
        print(f"\n4. Sample filenames:")
        print(f"   Filename column: {filename_col}")
        print(f"   EF column: {ef_col}")
        print(f"   Sample filenames: {df[filename_col].head().tolist()}")
        
        # Check if videos exist
        if Config.VIDEO_DIR.exists():
            print(f"\n5. Checking video files:")
            video_files = list(Config.VIDEO_DIR.glob("*.avi"))
            print(f"   Number of .avi files in Videos/: {len(video_files)}")
            if len(video_files) > 0:
                print(f"   Sample video files: {[f.name for f in video_files[:5]]}")
            
            # Check if first filename in CSV matches a video
            if len(df) > 0:
                first_filename = df[filename_col].iloc[0]
                first_video_path = Config.VIDEO_DIR / first_filename
                print(f"\n   First filename in CSV: {first_filename}")
                print(f"   Video exists: {first_video_path.exists()}")
else:
    print(f"\n   ERROR: FileList.csv not found at {Config.FILELIST_PATH}")

print("\n" + "="*70)

