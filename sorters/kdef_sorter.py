import os
import sys
import argparse
import shutil
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--image', action='store_true')
parser.add_argument('--video', action='store_true')
args, _ = parser.parse_known_args()

modality = "image" if args.image else "video"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(BASE_DIR, 'unpkged_datasets', modality, 'KDEF')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_datasets', modality)

# KDEF encodes the emotion in characters 5 and 6 of the filename (e.g., AF01ANHL.JPG -> AN = Angry)
EMOTION_MAP = {
    'AN': 'Angry', 'DI': 'Disgust', 'AF': 'Fear', 
    'HA': 'Happy', 'SA': 'Sad', 'SU': 'Surprise', 'NE': 'Neutral'
}

def sort():
    for emotion in EMOTION_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    images = glob.glob(os.path.join(INPUT_DIR, '**', '*.[jJ][pP][gG]'), recursive=True)
    
    if not images:
        print("Error: No images found in KDEF unpkged directory.")
        sys.exit(1)

    print("Parsing KDEF filenames...")
    for img_path in images:
        filename = os.path.basename(img_path)
        
        # Standard KDEF names are exactly 8 characters + extension
        if len(filename) >= 8:
            emotion_code = filename[4:6].upper()
            if emotion_code in EMOTION_MAP:
                emotion = EMOTION_MAP[emotion_code]
                dst = os.path.join(OUTPUT_DIR, emotion, f"kdef_{filename}")
                shutil.copy2(img_path, dst)

if __name__ == "__main__":
    sort()