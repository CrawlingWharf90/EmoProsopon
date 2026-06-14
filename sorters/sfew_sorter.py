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
INPUT_DIR = os.path.join(BASE_DIR, 'unpkged_datasets', modality, 'SFEW')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_datasets', modality)

# SFEW uses the exact emotion names for its subfolders
VALID_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def sort():
    for emotion in VALID_EMOTIONS:
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    # Grab all images regardless of how deep they are nested
    images = glob.glob(os.path.join(INPUT_DIR, '**', '*.*'), recursive=True)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print("Error: No images found in SFEW unpkged directory.")
        sys.exit(1)

    print(f"Sorting {len(images)} SFEW images from nested folders...")
    for img_path in images:
        # Get the name of the immediate parent folder
        parent_folder = os.path.basename(os.path.dirname(img_path))
        
        # Capitalize it just in case the dataset uses 'happy' instead of 'Happy'
        parent_folder = parent_folder.capitalize()

        if parent_folder in VALID_EMOTIONS:
            filename = os.path.basename(img_path)
            dst = os.path.join(OUTPUT_DIR, parent_folder, f"sfew_{filename}")
            shutil.copy2(img_path, dst)

if __name__ == "__main__":
    sort()