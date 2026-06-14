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
INPUT_DIR = os.path.join(BASE_DIR, 'unpkged_datasets', modality, 'EED')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_datasets', modality)

# EED typically uses numeric classes or specific affective states in its folder structure.
EMOTION_MAP = {
    '0': 'Neutral', '1': 'Happy', '2': 'Sad', 
    '3': 'Surprise', '4': 'Fear', '5': 'Disgust', '6': 'Angry'
}

def sort():
    for emotion in set(EMOTION_MAP.values()):
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    images = glob.glob(os.path.join(INPUT_DIR, '**', '*.*'), recursive=True)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print("Error: No images found in EED unpkged directory.")
        sys.exit(1)

    print(f"Sorting {len(images)} EED images...")
    for img_path in images:
        # Check the name of the folder the image is sitting inside
        parent_folder = os.path.basename(os.path.dirname(img_path))
        
        if parent_folder in EMOTION_MAP:
            emotion = EMOTION_MAP[parent_folder]
            filename = os.path.basename(img_path)
            dst = os.path.join(OUTPUT_DIR, emotion, f"eed_{filename}")
            shutil.copy2(img_path, dst)

if __name__ == "__main__":
    sort()