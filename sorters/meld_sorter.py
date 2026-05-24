import os
import shutil
import sys
import csv

#! MELD maps string emotions in the CSV to standard Title Case
EMOTION_MAP = {
    "neutral": "Neutral", "joy": "Happy", "sadness": "Sad", 
    "anger": "Angry", "fear": "Fear", "surprise": "Surprise", 
    "disgust": "Disgust"
}

UNPACK_DIR = os.path.join("unpkged_datasets", "MELD") 
TARGET_DIR = "sorted_datasets" 

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def sort_meld():
    if not os.path.exists(UNPACK_DIR):
        print(f"Error: {UNPACK_DIR} not found.")
        sys.exit(1)

    #* 1. Find all CSV files in the extracted MELD folder
    csv_files = []
    for root, _, files in os.walk(UNPACK_DIR):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
                
    if not csv_files:
        print("Error: Could not find MELD label CSV files (train_sent_emo.csv, etc.)")
        sys.exit(1)

    #* 2. Build a massive dictionary mapping 'diaX_uttY' -> 'Emotion'
    master_labels = {}
    for csv_path in csv_files:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dia_id = row['Dialogue_ID']
                    utt_id = row['Utterance_ID']
                    emotion = row['Emotion'].lower()
                    
                    video_name = f"dia{dia_id}_utt{utt_id}.mp4"
                    master_labels[video_name] = emotion
                except KeyError:
                    pass # Skip rows that are malformed

    #* 3. Hunt down the actual video files and sort them based on the dictionary
    video_files = []
    for root, _, files in os.walk(UNPACK_DIR):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))

    total_files = len(video_files)
    print(f"Parsed CSVs. Found {total_files} MELD video files. Sorting...")

    for i, filepath in enumerate(video_files):
        filename = os.path.basename(filepath)
        
        if filename in master_labels:
            raw_emo = master_labels[filename]
            if raw_emo in EMOTION_MAP:
                emotion_name = EMOTION_MAP[raw_emo]
                
                target_folder = os.path.join(TARGET_DIR, emotion_name)
                os.makedirs(target_folder, exist_ok=True)
                
                dst = os.path.join(target_folder, f"MELD_{filename}")
                if not os.path.exists(dst): 
                    shutil.copy2(filepath, dst)
                    
        print_progress(i + 1, total_files, prefix="Sorting")

if __name__ == "__main__":
    sort_meld()