import os
import shutil
import argparse
import sys
import tarfile

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
    return "fer2025"

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def extract_nested_archives(in_dir):
    """Finds and extracts any nested .tar archives with a progress bar."""
    for file in os.listdir(in_dir):
        file_path = os.path.join(in_dir, file)
        
        if file.lower().endswith('.tar'):
            print(f"📦 Extracting nested archive: {file}...")
            try:
                with tarfile.open(file_path, 'r') as tar:
                    members = tar.getmembers()
                    total = len(members)
                    for i, member in enumerate(members):
                        # Fix for Python 3.14 deprecation warning (using filter='data' in 3.12+)
                        if sys.version_info >= (3, 12):
                            tar.extract(member, path=in_dir, filter='data')
                        else:
                            tar.extract(member, path=in_dir)
                        print_progress(i + 1, total, prefix=f"Extracting {file}")
                os.remove(file_path)
            except Exception as e:
                print(f"❌ Error extracting {file}: {e}")

def cleanup_cls_files(in_dir):
    """Scans for and safely deletes all unnecessary .cls files."""
    cls_files = []
    for root, _, files in os.walk(in_dir):
        for file in files:
            if file.lower().endswith('.cls'):
                cls_files.append(os.path.join(root, file))
                
    total_cls = len(cls_files)
    if total_cls > 0:
        print(f"\n🧹 Found {total_cls} unnecessary .cls files. Cleaning up...")
        for i, file_path in enumerate(cls_files):
            try:
                os.remove(file_path)
            except Exception:
                pass
            print_progress(i + 1, total_cls, prefix="Removing .cls")
        print(f"↳ Successfully removed {total_cls} .cls files.")

def sort_fer2025(base_dir, modality):
    dataset_name = "FER2025"
    in_dir = os.path.join(base_dir, 'unpkged_datasets', modality, dataset_name)
    out_dir = os.path.join(base_dir, 'sorted_datasets', modality)

    if not os.path.exists(in_dir):
        print(f"❌ Error: Source directory {in_dir} not found.")
        return

    extract_nested_archives(in_dir)

    print(f"Creating standardized classes for master sorting pool...")
    for std_emo in set(EMOTION_MAP.values()):
        os.makedirs(os.path.join(out_dir, std_emo), exist_ok=True)

    total_files = 0
    valid_files = []
    
    for root, _, files in os.walk(in_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                parent_folder = os.path.basename(root).lower()
                matched_emotion = next((std_emo for key, std_emo in EMOTION_MAP.items() if key in parent_folder), None)
                
                if matched_emotion:
                    valid_files.append((root, file, matched_emotion))
                    total_files += 1

    if total_files == 0:
        print(f"↳ Successfully flattened and sorted 0 images. (Check unpkged_datasets folder!)")
        return

    moved_count = 0
    print(f"\n🚀 Moving and renaming {total_files} images to master pool...")
    for root, file, matched_emotion in valid_files:
        src_path = os.path.join(root, file)
        
        prefix = "fer2025"
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
            
        shutil.move(src_path, dest_path)
        moved_count += 1
        
        print_progress(moved_count, total_files, prefix=f"Sorting {dataset_name}")

    print(f"↳ Successfully moved {moved_count} images into {out_dir}")
    
    #! Cleanup of .cls files
    cleanup_cls_files(in_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FER2025 Dataset Sorter")
    parser.add_argument('--image', action='store_true', help="Process as image dataset")
    parser.add_argument('--video', action='store_true', help="Process as video dataset")
    args = parser.parse_args()

    modality = 'image' if args.image else 'video'
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    sort_fer2025(base_dir, modality)