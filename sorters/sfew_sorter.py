import os
import sys
import argparse
import shutil
import glob
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--image', action='store_true')
parser.add_argument('--video', action='store_true')
args, _ = parser.parse_known_args()

#! THIS SORTER IS BASED ON THE DATASETS' FOLDER STRUCTURE WHEN I GOT IT
#! THE WAY FILES ARE ORGANIZED MIGHT HAVE BEEN CHANGED KINDLY MAKE SURE
#! THAT THE DATASET IS STILL ORGANIZED IN ZIP FILES EACH NAMED BASED
#! ON THE EMOTION THEY CONTAIN.
#!
#! - JUNE 2026

modality = "image" if args.image else "image" 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(BASE_DIR, 'unpkged_datasets', modality, 'SFEW')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_datasets', modality)

VALID_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

TARGET_ZIPS = [f"{e.lower()}.zip" for e in VALID_EMOTIONS] + ["train_aligned_faces.zip", "val_aligned_faces.zip"]

def sort():
    print(f"Starting SFEW Nested Extraction and Sorting...")
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} not found.")
        sys.exit(1)

    all_zips = glob.glob(os.path.join(INPUT_DIR, '**', '*.zip'), recursive=True)
    
    if not all_zips:
        print("Error: No nested .zip files found in SFEW directory.")
        sys.exit(1)

    processed_count = 0
    
    temp_dir = os.path.join(INPUT_DIR, "temp_extraction")
    os.makedirs(temp_dir, exist_ok=True)
    
    for emotion in VALID_EMOTIONS:
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    for z in all_zips:
        filename = os.path.basename(z).lower()
        if filename in TARGET_ZIPS:
            print(f"Unzipping {os.path.basename(z)}...")
            try:
                with zipfile.ZipFile(z, 'r') as zf:
                    zf.extractall(temp_dir)
            except zipfile.BadZipFile:
                print(f"  -> Skipping {filename} (Corrupted or not a true zip file)")

    images = glob.glob(os.path.join(temp_dir, '**', '*.*'), recursive=True)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Sorting {len(images)} images into emotion folders...")
    
    for img_path in images:
        parent_folder = os.path.basename(os.path.dirname(img_path)).capitalize()

        if parent_folder in VALID_EMOTIONS:
            img_filename = os.path.basename(img_path)
            
            dst = os.path.join(OUTPUT_DIR, parent_folder, f"sfew_{img_filename}")
            
            if not os.path.exists(dst):
                shutil.copy2(img_path, dst)
                processed_count += 1

    shutil.rmtree(temp_dir)

    print(f"\n✅ Successfully sorted {processed_count} SFEW images (including Aligned Faces)!")

if __name__ == "__main__":
    sort()