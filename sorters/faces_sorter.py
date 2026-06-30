import os
import sys
import argparse
import shutil
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--image', action='store_true')
parser.add_argument('--video', action='store_true')
args, _ = parser.parse_known_args()

modality = "image" if args.image else "image"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(BASE_DIR, 'unpkged_datasets', modality, 'FACES')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_datasets', modality)

EMOTION_MAP = {
    'a': 'Angry', 'd': 'Disgust', 'f': 'Fear', 
    'h': 'Happy', 'n': 'Neutral', 's': 'Sad'
}

def sort():
    for emotion in EMOTION_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    images = glob.glob(os.path.join(INPUT_DIR, '**', '*.[jJ][pP][gG]'), recursive=True)
    if not images:
        print("Error: No images found in FACES unpkged directory.")
        sys.exit(1)

    print(f"Sorting {len(images)} FACES images...")
    for img_path in images:
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        
        # Format is typically ID_age_gender_emotion_set.jpg
        if len(parts) >= 4:
            emotion_code = parts[3] 
            if emotion_code in EMOTION_MAP:
                emotion = EMOTION_MAP[emotion_code]
                dst = os.path.join(OUTPUT_DIR, emotion, f"faces_{filename}")
                shutil.copy2(img_path, dst)

if __name__ == "__main__":
    sort()