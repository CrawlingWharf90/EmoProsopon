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
UNPACK_DIR = os.path.join(BASE_DIR, "unpkged_datasets", modality, "CREMA-D")
TARGET_DIR = os.path.join(BASE_DIR, "sorted_datasets", modality)

#* CREMA-D Emotion Codes
EMOTION_MAP = {
    "ANG": "Angry", "DIS": "Disgust", "FEA": "Fear", 
    "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"
}

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def sort_cremad():
    if not os.path.exists(UNPACK_DIR):
        print(f"Error: {UNPACK_DIR} not found.")
        sys.exit(1)

    video_files = []
    for root, _, files in os.walk(UNPACK_DIR):
        for file in files:
            if file.endswith((".mp4", ".flv", ".avi")):
                video_files.append(os.path.join(root, file))

    total_files = len(video_files)
    if total_files == 0:
        print("No video files found in the unpacked CREMA-D folder.")
        sys.exit(1)

    print(f"Found {total_files} CREMA-D files. Sorting by emotion...")

    for i, filepath in enumerate(video_files):
        filename = os.path.basename(filepath)
        
        parts = filename.split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
            
            if emotion_code in EMOTION_MAP:
                emotion_name = EMOTION_MAP[emotion_code]
                
                target_folder = os.path.join(TARGET_DIR, emotion_name)
                os.makedirs(target_folder, exist_ok=True)
                
                dst = os.path.join(target_folder, f"cremad_{filename}")
                if not os.path.exists(dst): 
                    shutil.copy2(filepath, dst)
        
        print_progress(i + 1, total_files, prefix="Sorting")

if __name__ == "__main__":
    sort_cremad()