import os
import shutil
import sys

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

#* eNTERFACE05 uses lowercase folder names for the 6 emotions (No Neutral)
EMOTION_MAP = {
    "anger": "Angry", "disgust": "Disgust", "fear": "Fear", 
    "happiness": "Happy", "sadness": "Sad", "surprise": "Surprise"
}

UNPACK_DIR = os.path.join("unpkged_datasets", "eNTERFACE05") 
TARGET_DIR = "sorted_datasets" 

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def sort_enterface05(dry_run=False): #! Dry Run is a debug safeguard to print intended actions without making changes. Its value can be chnaged only via code, not command line args.
    if not os.path.exists(UNPACK_DIR):
        print(f"Error: {UNPACK_DIR} not found.")
        sys.exit(1)

    video_files = []
    #* Walk through all nested directories
    for root, _, files in os.walk(UNPACK_DIR):
        for file in files:
            if file.endswith(".avi"): # eNTERFACE05 natively uses AVI
                video_files.append(os.path.join(root, file))

    total_files = len(video_files)
    if total_files == 0:
        print(f"\n{RED}No .avi files found in the unpacked eNTERFACE05 folder.{RESET}")
        sys.exit(1)

    print(f"\n{GREEN}Found {total_files} eNTERFACE05 files. Sorting by emotion...{RESET}")

    for i, filepath in enumerate(video_files):
        filename = os.path.basename(filepath)
        
        #? Since the emotion is the name of the parent folder, we extract it from the path
        parent_folder = os.path.basename(os.path.dirname(os.path.dirname(filepath))).lower()
        
        if parent_folder not in EMOTION_MAP:
            parent_folder = os.path.basename(os.path.dirname(filepath)).lower()
        elif parent_folder in EMOTION_MAP:
            emotion_name = EMOTION_MAP[parent_folder]
            
            target_folder = os.path.join(TARGET_DIR, emotion_name)
            os.makedirs(target_folder, exist_ok=True)
            
            #* Append a prefix so files named 'video.avi' don't overwrite each other!
            unique_filename = f"eNT_{i}_{filename}"
            dst = os.path.join(target_folder, unique_filename)
            
            if dry_run:
                print(f"[DRY RUN]: {filepath} -> {dst}")
            elif not os.path.exists(dst): 
                shutil.copy2(filepath, dst)
        else:
            print(f"\n{YELLOW}Unrecognized folder '{parent_folder}' for: {filepath}{RESET}")
        
        print_progress(i + 1, total_files, prefix="Sorting")

if __name__ == "__main__":
    sort_enterface05()