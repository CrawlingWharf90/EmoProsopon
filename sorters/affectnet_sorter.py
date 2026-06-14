import os
import sys
import argparse
import shutil
import csv
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--image', action='store_true')
parser.add_argument('--video', action='store_true')
args, _ = parser.parse_known_args()

modality = "image" if args.image else "video"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(BASE_DIR, 'unpkged_datasets', modality, 'AffectNet')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_datasets', modality)

# AffectNet Expression Map: 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger
EMOTION_MAP = {
    '0': 'Neutral', '1': 'Happy', '2': 'Sad', 
    '3': 'Surprise', '4': 'Fear', '5': 'Disgust', '6': 'Angry'
}

def sort():
    for emotion in EMOTION_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    # Find all CSV files (handles training.csv, validation.csv, etc.)
    csv_files = glob.glob(os.path.join(INPUT_DIR, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print("Error: No annotation CSV files found for AffectNet.")
        sys.exit(1)

    print("Parsing AffectNet CSV annotations...")
    for csv_path in csv_files:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # AffectNet CSVs usually use 'subDirectory_filePath' and 'expression'
                if 'subDirectory_filePath' in row and 'expression' in row:
                    img_rel_path = row['subDirectory_filePath']
                    expression_id = row['expression']
                    
                    if expression_id in EMOTION_MAP:
                        emotion = EMOTION_MAP[expression_id]
                        # Fix path separators based on OS
                        clean_rel_path = img_rel_path.replace('/', os.sep).replace('\\', os.sep)
                        src = os.path.join(INPUT_DIR, 'Manually_Annotated_Images', clean_rel_path)
                        
                        if os.path.exists(src):
                            filename = os.path.basename(src)
                            dst = os.path.join(OUTPUT_DIR, emotion, f"affectnet_{filename}")
                            shutil.copy2(src, dst)

if __name__ == "__main__":
    sort()