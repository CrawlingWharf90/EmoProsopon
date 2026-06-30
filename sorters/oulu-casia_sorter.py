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
INPUT_DIR = os.path.join(BASE_DIR, 'unpkged_datasets', modality, 'Oulu-CASIA')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_datasets', modality)

EMOTION_MAP = {
    'Anger': 'Angry', 'Disgust': 'Disgust', 'Fear': 'Fear', 
    'Happiness': 'Happy', 'Sadness': 'Sad', 'Surprise': 'Surprise'
}

def sort():
    for emotion in EMOTION_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    images = glob.glob(os.path.join(INPUT_DIR, '**', '*.*'), recursive=True)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print("Error: No images found in Oulu-CASIA unpkged directory.")
        sys.exit(1)

    print(f"Sorting {len(images)} Oulu-CASIA images...")
    for img_path in images:
        parent_folder = os.path.basename(os.path.dirname(img_path))
        
        if parent_folder in EMOTION_MAP:
            emotion = EMOTION_MAP[parent_folder]
            filename = os.path.basename(img_path)
            dst = os.path.join(OUTPUT_DIR, emotion, f"oulucasia_{filename}")
            shutil.copy2(img_path, dst)

if __name__ == "__main__":
    sort()