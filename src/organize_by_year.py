import os
import shutil
import pandas as pd
from pathlib import Path

# Configuration
CSV_FILE = "job_videos_data.csv"
SOURCE_FOLDER = "Andy Frisella_transcripts"  # Root folder with .txt files
OUTPUT_BASE = "Andy Frisella_transcripts"  # Same folder, will create year subfolders

# Load the CSV with video data
df = pd.read_csv(CSV_FILE)

# Convert published_at to datetime and extract year
df["published_at"] = pd.to_datetime(df["published_at"])
df["year"] = df["published_at"].dt.year

# Create a mapping of video_id to year
video_year_map = dict(zip(df["video_id"], df["year"]))

print(f"Loaded {len(video_year_map)} video mappings")

# Get all .txt files in the source folder (root level only)
txt_files = list(Path(SOURCE_FOLDER).glob("*.txt"))
print(f"Found {len(txt_files)} .txt files in {SOURCE_FOLDER}")

# Organize files by year
moved_count = 0
not_found_count = 0

for txt_file in txt_files:
    # Extract video_id from filename (assuming filename is video_id.txt)
    video_id = txt_file.stem

    # Get the year for this video
    year = video_year_map.get(video_id)

    if year:
        # Create year folder if it doesn't exist
        year_folder = Path(OUTPUT_BASE) / str(int(year))
        year_folder.mkdir(parents=True, exist_ok=True)

        # Move the file
        dest_path = year_folder / txt_file.name
        shutil.move(str(txt_file), str(dest_path))
        moved_count += 1
        print(f"Moved {txt_file.name} -> {year}/")
    else:
        not_found_count += 1
        print(f"⚠️  No year found for {txt_file.name}")

print(f"\n✓ Done! Moved {moved_count} files, {not_found_count} not found in data")
