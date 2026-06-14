import os
import sys
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--image', action='store_true')
parser.add_argument('--video', action='store_true')
args, _ = parser.parse_known_args()

modality = "image" if args.image else "video"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(BASE_DIR, 'unpkged_datasets', modality, 'RAF-DB')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_datasets', modality)

# RAF-DB labels: 1=Surprise, 2=Fear, 3=Disgust, 4=Happy, 5=Sad, 6=Angry, 7=Neutral
EMOTION_MAP = {
    '1': 'Surprise', '2': 'Fear', '3': 'Disgust', 
    '4': 'Happy', '5': 'Sad', '6': 'Angry', '7': 'Neutral'
}

def sort():
    for emotion in EMOTION_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    labels_file = os.path.join(INPUT_DIR, 'basic', 'EmoLabel', 'list_patition_label.txt')
    images_dir = os.path.join(INPUT_DIR, 'basic', 'Image', 'aligned')

    if not os.path.exists(labels_file) or not os.path.exists(images_dir):
        print(f"Error: Missing RAF-DB label text file or aligned images directory.")
        sys.exit(1)

    print("Mapping RAF-DB text labels to files...")
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, label = parts
                # RAF-DB filenames in the txt map usually drop the _aligned.jpg suffix
                aligned_filename = filename.replace('.jpg', '_aligned.jpg') 
                
                src = os.path.join(images_dir, aligned_filename)
                if os.path.exists(src) and label in EMOTION_MAP:
                    emotion = EMOTION_MAP[label]
                    dst = os.path.join(OUTPUT_DIR, emotion, f"rafdb_{aligned_filename}")
                    shutil.copy2(src, dst)

if __name__ == "__main__":
    sort()