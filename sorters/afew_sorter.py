import os
import shutil
import sys

#* AFEW typically uses Capitalized proper names for folders
EMOTION_MAP = {
    "Angry": "Angry", "Disgust": "Disgust", "Fear": "Fear", 
    "Happy": "Happy", "Neutral": "Neutral", "Sad": "Sad", "Surprise": "Surprise"
}

#! THEORETICAL SORTER: Based on EmotiW AFEW challenge hierarchy
#! Awaiting community verification on exact extraction paths.
UNPACK_DIR = os.path.join("unpkged_datasets", "AFEW") 
TARGET_DIR = "sorted_datasets" 

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def sort_afew():
    if not os.path.exists(UNPACK_DIR):
        print(f"Error: {UNPACK_DIR} not found.")
        sys.exit(1)

    video_files = []
    for root, _, files in os.walk(UNPACK_DIR):
        for file in files:
            if file.endswith((".avi", ".mp4")):
                video_files.append(os.path.join(root, file))

    total_files = len(video_files)
    if total_files == 0:
        print("No video files found in the unpacked AFEW folder.")
        sys.exit(1)

    print(f"Found {total_files} AFEW files. Sorting by emotion...")

    for i, filepath in enumerate(video_files):
        filename = os.path.basename(filepath)
        
        #? The emotion is the name of the immediate parent folder (e.g. Train/Angry/video.mp4)
        parent_folder = os.path.basename(os.path.dirname(filepath))
        
        if parent_folder in EMOTION_MAP:
            emotion_name = EMOTION_MAP[parent_folder]
            
            target_folder = os.path.join(TARGET_DIR, emotion_name)
            os.makedirs(target_folder, exist_ok=True)
            
            dst = os.path.join(target_folder, f"AFEW_{filename}")
            if not os.path.exists(dst): 
                shutil.copy2(filepath, dst)
        
        print_progress(i + 1, total_files, prefix="Sorting")

if __name__ == "__main__":
    sort_afew()