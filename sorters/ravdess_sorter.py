import os
import shutil
import sys

#* RAVDESS Emotion Codes (3rd number in the filename)
EMOTION_MAP = {
    "01": "Neutral", "02": "Neutral", #! Merging Calm into Neutral
    "03": "Happy", "04": "Sad", "05": "Angry", 
    "06": "Fear", "07": "Disgust", "08": "Surprise"
}

#! The TUI extracted it here
UNPACK_DIR = os.path.join("unpkged_datasets", "RAVDESS") 
#! Sort it here so extract_dataset.py can find it cleanly separated from raw zips
TARGET_DIR = "sorted_datasets" 

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def sort_ravdess():
    if not os.path.exists(UNPACK_DIR):
        print("Error: RAVDESS unpacked directory not found.")
        sys.exit(1)

    #* Gather all video files (including subdirectories)
    video_files = []
    for root, _, files in os.walk(UNPACK_DIR):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))

    total_files = len(video_files)
    if total_files == 0:
        print("No .mp4 files found in the unpacked RAVDESS folder.")
        sys.exit(1)

    print(f"Found {total_files} RAVDESS files. Sorting by emotion...")

    for i, filepath in enumerate(video_files):
        filename = os.path.basename(filepath)
        
        #* Example filename: 03-01-05-01-01-01-01.mp4 (05 = Angry)
        parts = filename.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            
            if emotion_code in EMOTION_MAP:
                emotion_name = EMOTION_MAP[emotion_code]
                
                #? Create the target emotion folder (e.g., sorted_datasets/Happy)
                target_folder = os.path.join(TARGET_DIR, emotion_name)
                os.makedirs(target_folder, exist_ok=True)
                
                #? Copy the file to the final destination
                dst = os.path.join(target_folder, filename)
                if not os.path.exists(dst): #! Don't overwrite if it's already there
                    shutil.copy2(filepath, dst)
        
        print_progress(i + 1, total_files, prefix="Sorting")

if __name__ == "__main__":
    sort_ravdess()