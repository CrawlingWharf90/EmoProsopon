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
INPUT_DIR = os.path.join(BASE_DIR, 'unpkged_datasets', modality, 'AR-Face')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_datasets', modality)

# AR Face Database Expression Conditions
# 1/14: Neutral, 2/15: Smile (Happy), 3/16: Anger, 4/17: Scream (Mapped to Surprise)
EMOTION_MAP = {
    '1': 'Neutral', '14': 'Neutral',
    '2': 'Happy',   '15': 'Happy',
    '3': 'Angry',   '16': 'Angry',
    '4': 'Surprise', '17': 'Surprise' 
}

def sort():
    for emotion in set(EMOTION_MAP.values()):
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    # Grab only bmp images, ignoring the __MACOSX junk folders in the zip
    images = glob.glob(os.path.join(INPUT_DIR, '**', '*.[bB][mM][pP]'), recursive=True)
    images = [img for img in images if '__MACOSX' not in img]

    if not images:
        print("Error: No valid .bmp images found in AR-Face unpkged directory.")
        sys.exit(1)

    print(f"Sorting {len(images)} AR-Face images...")
    for img_path in images:
        filename = os.path.basename(img_path)
        parent_folder = os.path.basename(os.path.dirname(img_path)) # e.g., 's1'
        
        # The filename itself is the condition code (e.g., '1.bmp' -> '1')
        condition = filename.split('.')[0]
        
        if condition in EMOTION_MAP:
            emotion = EMOTION_MAP[condition]
            # Output: arface_s1_1.bmp
            dst = os.path.join(OUTPUT_DIR, emotion, f"arface_{parent_folder}_{filename}")
            shutil.copy2(img_path, dst)

if __name__ == "__main__":
    sort()