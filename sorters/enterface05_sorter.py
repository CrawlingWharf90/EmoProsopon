import os
import shutil
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', action='store_true')
parser.add_argument('--video', action='store_true')
args, _ = parser.parse_known_args()

modality = "video" if args.video else "video"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UNPACK_DIR = os.path.join(BASE_DIR, "unpkged_datasets", modality, "eNTERFACE05")
TARGET_DIR = os.path.join(BASE_DIR, "sorted_datasets", modality)

GREEN, YELLOW, RED, RESET = '\033[92m', '\033[93m', '\033[91m', '\033[0m'

EMOTION_MAP = {
    "anger": "Angry", "disgust": "Disgust", "fear": "Fear", 
    "happiness": "Happy", "sadness": "Sad", "surprise": "Surprise"
}

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def sort_enterface05(dry_run=False):
    if not os.path.exists(UNPACK_DIR):
        print(f"Error: {UNPACK_DIR} not found.")
        sys.exit(1)

    video_files = []
    for root, _, files in os.walk(UNPACK_DIR):
        for file in files:
            if file.endswith(".avi"):
                video_files.append(os.path.join(root, file))

    total_files = len(video_files)
    if total_files == 0:
        print(f"\n{RED}No .avi files found in the unpacked eNTERFACE05 folder.{RESET}")
        sys.exit(1)

    print(f"\n{GREEN}Found {total_files} eNTERFACE05 files. Sorting by emotion...{RESET}")

    for i, filepath in enumerate(video_files):
        filename = os.path.basename(filepath)
        
        parent_folder = os.path.basename(os.path.dirname(os.path.dirname(filepath))).lower()
        
        if parent_folder not in EMOTION_MAP:
            parent_folder = os.path.basename(os.path.dirname(filepath)).lower()
        elif parent_folder in EMOTION_MAP:
            emotion_name = EMOTION_MAP[parent_folder]
            
            target_folder = os.path.join(TARGET_DIR, emotion_name)
            os.makedirs(target_folder, exist_ok=True)
            
            unique_filename = f"eNT_{i}_{filename}"
            dst = os.path.join(target_folder, unique_filename)
            
            if dry_run: print(f"[DRY RUN]: {filepath} -> {dst}")
            elif not os.path.exists(dst): shutil.copy2(filepath, dst)
        else:
            print(f"\n{YELLOW}Unrecognized folder '{parent_folder}' for: {filepath}{RESET}")
        
        print_progress(i + 1, total_files, prefix="Sorting")

if __name__ == "__main__":
    sort_enterface05()