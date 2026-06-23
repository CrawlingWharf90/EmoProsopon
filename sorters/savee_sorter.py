import os
import shutil
import sys
import re

#* SAVEE Emotion Codes (Letters before the numbers in the filename)
EMOTION_MAP = {
    "a": "Angry", "d": "Disgust", "f": "Fear", 
    "h": "Happy", "n": "Neutral", "sa": "Sad", "su": "Surprise"
}

UNPACK_DIR = os.path.join("unpkged_datasets", "SAVEE") 
TARGET_DIR = "sorted_datasets" 

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def sort_savee():
    if not os.path.exists(UNPACK_DIR):
        print(f"Error: {UNPACK_DIR} not found.")
        sys.exit(1)

    video_files = []
    for root, _, files in os.walk(UNPACK_DIR):
        for file in files:
            if file.endswith((".mp4", ".avi")):
                video_files.append(os.path.join(root, file))

    total_files = len(video_files)
    if total_files == 0:
        print("No video files found in the unpacked SAVEE folder.")
        sys.exit(1)

    print(f"Found {total_files} SAVEE files. Sorting by emotion...")

    for i, filepath in enumerate(video_files):
        filename = os.path.basename(filepath)
        
        #* Example: DC_a01.avi -> Split by '_' to get 'a01.avi'
        parts = filename.split('_')
        if len(parts) == 2:
            code_and_num = parts[1]
            
            #? Use regex to strip the numbers out, leaving only the emotion letters ('a', 'sa', 'su')
            match = re.match(r"([a-zA-Z]+)\d+", code_and_num)
            if match:
                emotion_code = match.group(1)
                
                if emotion_code in EMOTION_MAP:
                    emotion_name = EMOTION_MAP[emotion_code]
                    
                    target_folder = os.path.join(TARGET_DIR, emotion_name)
                    os.makedirs(target_folder, exist_ok=True)

                    dst = os.path.join(target_folder, f"savee_{filename}")
                    if not os.path.exists(dst): 
                        shutil.copy2(filepath, dst)
        
        print_progress(i + 1, total_files, prefix="Sorting")

if __name__ == "__main__":
    sort_savee()