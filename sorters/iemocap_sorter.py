import os
import shutil
import sys
import re

#* IEMOCAP Evaluation Classes
EMOTION_MAP = {
    "ang": "Angry", "dis": "Disgust", "fea": "Fear", 
    "hap": "Happy", "exc": "Happy", #! Merging "Excited" into Happy to fit our 7 classes
    "neu": "Neutral", "sad": "Sad", "sur": "Surprise"
}

#! THEORETICAL SORTER: Assumes pre-split AVIs in dialog/avi/ and EmoEvaluation files
UNPACK_DIR = os.path.join("unpkged_datasets", "IEMOCAP") 
TARGET_DIR = "sorted_datasets" 

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def sort_iemocap():
    if not os.path.exists(UNPACK_DIR):
        print(f"Error: {UNPACK_DIR} not found.")
        sys.exit(1)

    #* 1. Find all EmoEvaluation .txt files across the 5 Sessions
    eval_files = []
    for root, _, files in os.walk(UNPACK_DIR):
        if "EmoEvaluation" in root:
            for file in files:
                if file.endswith(".txt") and not file.startswith("."):
                    eval_files.append(os.path.join(root, file))

    if not eval_files:
        print("Error: Could not find EmoEvaluation .txt files.")
        sys.exit(1)

    #* 2. Parse the evaluations: e.g., "[10.00 - 15.00] Ses01F_impro01_F000 neu [2.5, 2.5, 2.5]"
    master_labels = {}
    for eval_path in eval_files:
        with open(eval_path, 'r') as f:
            for line in f:
                #? Regex to grab the utterance ID and the 3-letter emotion code
                match = re.search(r'\[.+\]\s+(Ses\w+)\s+([a-z]{3})\s+\[', line)
                if match:
                    utterance_id = match.group(1)
                    emotion_code = match.group(2)
                    master_labels[utterance_id] = emotion_code

    #* 3. Hunt down the actual video files
    video_files = []
    for root, _, files in os.walk(UNPACK_DIR):
        for file in files:
            if file.endswith((".avi", ".mp4")):
                video_files.append(os.path.join(root, file))

    total_files = len(video_files)
    print(f"Parsed Evaluations. Found {total_files} IEMOCAP video files. Sorting...")

    for i, filepath in enumerate(video_files):
        #? Extract 'Ses01F_impro01_F000' from 'Ses01F_impro01_F000.avi'
        filename = os.path.basename(filepath)
        utterance_id = os.path.splitext(filename)[0]
        
        if utterance_id in master_labels:
            raw_emo = master_labels[utterance_id]
            if raw_emo in EMOTION_MAP:
                emotion_name = EMOTION_MAP[raw_emo]
                
                target_folder = os.path.join(TARGET_DIR, emotion_name)
                os.makedirs(target_folder, exist_ok=True)
                
                dst = os.path.join(target_folder, f"IEMOCAP_{filename}")
                if not os.path.exists(dst): 
                    shutil.copy2(filepath, dst)
                    
        print_progress(i + 1, total_files, prefix="Sorting")

if __name__ == "__main__":
    sort_iemocap()