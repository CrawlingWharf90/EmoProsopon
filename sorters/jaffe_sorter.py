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
INPUT_DIR = os.path.join(BASE_DIR, 'unpkged_datasets', modality, 'JAFFE')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_datasets', modality)

EMOTION_MAP = {
    'AN': 'Angry', 'DI': 'Disgust', 'FE': 'Fear', 
    'HA': 'Happy', 'SA': 'Sad', 'SU': 'Surprise', 'NE': 'Neutral'
}

def sort():
    for emotion in EMOTION_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    # Bulletproofed: Case-insensitive extension matching for Unix/Linux systems
    images = glob.glob(os.path.join(INPUT_DIR, '**', '*.[tT][iI][fF][fF]'), recursive=True)
    if not images:
        print("Error: No .tiff images found in JAFFE unpkged directory.")
        sys.exit(1)

    print(f"Sorting {len(images)} JAFFE images...")
    for img_path in images:
        filename = os.path.basename(img_path)
        # JAFFE standard format: Initials.EmotionNum.Identifier.tiff (e.g., KA.AN1.39.tiff)
        parts = filename.split('.')
        if len(parts) >= 2:
            emotion_code = parts[1][:2] # Grab 'AN' from 'AN1'
            if emotion_code in EMOTION_MAP:
                emotion = EMOTION_MAP[emotion_code]
                dst = os.path.join(OUTPUT_DIR, emotion, f"jaffe_{filename}")
                shutil.copy2(img_path, dst)

if __name__ == "__main__":
    sort()