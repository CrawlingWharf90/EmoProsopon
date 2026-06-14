import os
import sys
import argparse
import csv
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--image', action='store_true')
parser.add_argument('--video', action='store_true')
args, _ = parser.parse_known_args()

modality = "image" if args.image else "video"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(BASE_DIR, 'unpkged_datasets', modality, 'FER-2013')
OUTPUT_DIR = os.path.join(BASE_DIR, 'sorted_datasets', modality)

# FER-2013 labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
EMOTION_MAP = {
    '0': 'Angry', '1': 'Disgust', '2': 'Fear', 
    '3': 'Happy', '4': 'Sad', '5': 'Surprise', '6': 'Neutral'
}

def sort():
    for emotion in EMOTION_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

    csv_path = os.path.join(INPUT_DIR, 'fer2013.csv')
    if not os.path.exists(csv_path):
        print(f"Error: fer2013.csv not found in {INPUT_DIR}")
        sys.exit(1)

    print("Reconstructing images from FER-2013 CSV pixels...")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        
        for i, row in enumerate(reader):
            label, pixels_str, usage = row
            if label in EMOTION_MAP:
                emotion = EMOTION_MAP[label]
                # Convert string to 48x48 integer array
                pixels = np.array(pixels_str.split(), dtype='uint8').reshape(48, 48)
                img = Image.fromarray(pixels, mode='L')
                
                out_path = os.path.join(OUTPUT_DIR, emotion, f"fer2013_{usage}_{i}.jpg")
                img.save(out_path)

if __name__ == "__main__":
    sort()