import os
import shutil
import argparse
import sys

#* ─────────────────────────────────────────────────────────────────
#* STANDARDIZED EMOTION MAPPING
#* ─────────────────────────────────────────────────────────────────
EMOTION_MAP = {
    'angry': 'Angry', 'anger': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fear',
    'happy': 'Happy', 'happiness': 'Happy',
    'sad': 'Sad', 'sadness': 'Sad',
    'surprise': 'Surprise',
    'neutral': 'Neutral'
}

def get_prefix():
    """The exact prefix this sorter writes onto every sorted filename."""
    return "ve"

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def sort_video_emotion(base_dir, modality):
    dataset_name = "Video-Emotion"
    in_dir = os.path.join(base_dir, 'unpkged_datasets', modality, dataset_name)
    out_dir = os.path.join(base_dir, 'sorted_datasets', modality)

    if not os.path.exists(in_dir):
        print(f"❌ Error: Source directory {in_dir} not found.")
        return

    print(f"Creating standardized classes for master sorting pool...")
    for std_emo in set(EMOTION_MAP.values()):
        os.makedirs(os.path.join(out_dir, std_emo), exist_ok=True)

    total_files = 0
    valid_files = []
    
    for root, _, files in os.walk(in_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                parent_folder = os.path.basename(root).lower()
                matched_emotion = next((std_emo for key, std_emo in EMOTION_MAP.items() if key in parent_folder), None)
                
                if matched_emotion:
                    valid_files.append((root, file, matched_emotion))
                    total_files += 1

    if total_files == 0:
        print(f"↳ Successfully sorted 0 videos. (Check unpkged_datasets folder!)")
        return

    copied_count = 0
    for root, file, matched_emotion in valid_files:
        src_path = os.path.join(root, file)
        
        prefix = "ve"
        sub_prefix = os.path.basename(os.path.dirname(root))
        if sub_prefix.lower() not in EMOTION_MAP and sub_prefix.lower() != dataset_name.lower():
            safe_filename = f"{prefix}_{sub_prefix}_{file}"
        else:
            safe_filename = f"{prefix}_{file}"
            
        dest_path = os.path.join(out_dir, matched_emotion, safe_filename)
        
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(safe_filename)
            dest_path = os.path.join(out_dir, matched_emotion, f"{name}_{counter}{ext}")
            counter += 1
            
        shutil.copy2(src_path, dest_path)
        copied_count += 1
        
        print_progress(copied_count, total_files, prefix=f"Sorting {dataset_name}")

    print(f"↳ Successfully sorted {copied_count} videos into {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video-Emotion Dataset Sorter")
    parser.add_argument('--image', action='store_true', help="Process as image dataset")
    parser.add_argument('--video', action='store_true', help="Process as video dataset")
    args = parser.parse_args()

    modality = 'image' if args.image else 'video'
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    sort_video_emotion(base_dir, modality)