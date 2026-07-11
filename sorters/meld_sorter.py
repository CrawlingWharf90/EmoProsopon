import os
import shutil
import sys
import csv
import tarfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', action='store_true')
parser.add_argument('--video', action='store_true')
args, _ = parser.parse_known_args()

modality = "video" if args.video else "video"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UNPACK_DIR = os.path.join(BASE_DIR, "unpkged_datasets", modality, "MELD")
TARGET_DIR = os.path.join(BASE_DIR, "sorted_datasets", modality)

EMOTION_MAP = {
    "neutral": "Neutral", "joy": "Happy", "sadness": "Sad", 
    "anger": "Angry", "fear": "Fear", "surprise": "Surprise", "disgust": "Disgust"
}

def get_prefix():
    """The exact prefix this sorter writes onto every sorted filename."""
    return "MELD"

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def find_train_splits_dir(search_root):
    """Looks for a folder literally named 'train_splits' anywhere under search_root."""
    for root, dirs, _ in os.walk(search_root):
        if os.path.basename(root) == "train_splits":
            return root
    return None

def extract_train_archive():
    """
    Finds train.tar.gz anywhere under UNPACK_DIR and extracts it in place.
    Returns the path to whatever top-level folder the archive creates
    (e.g. 'train' or 'train_splits' directly - we don't assume which),
    so the caller knows exactly what to clean up afterwards.
    """
    tar_path = None
    for root, _, files in os.walk(UNPACK_DIR):
        for file in files:
            if file.lower() == "train.tar.gz":
                tar_path = os.path.join(root, file)
                break
        if tar_path:
            break

    if not tar_path:
        return None

    extract_root = os.path.dirname(tar_path)
    print(f"📦 Extracting {os.path.basename(tar_path)}...")

    top_level_name = None
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        total = len(members)
        for i, member in enumerate(members):
            first_part = member.name.split('/')[0]
            if top_level_name is None and first_part:
                top_level_name = first_part

            if sys.version_info >= (3, 12):
                tar.extract(member, path=extract_root, filter='data')
            else:
                tar.extract(member, path=extract_root)
            print_progress(i + 1, total, prefix="Extracting train.tar.gz")

    if not top_level_name:
        return None
    return os.path.join(extract_root, top_level_name)

def cleanup_extracted_folder(train_splits_dir):
    """
    Removes the train_splits folder once its videos have been moved out.
    If extraction wrapped it inside an intermediate folder (e.g. 'train')
    that's now empty as a result, that gets cleaned up too - whichever
    layout the archive actually used.
    """
    if not train_splits_dir or not os.path.isdir(train_splits_dir):
        return

    parent_dir = os.path.dirname(train_splits_dir)
    shutil.rmtree(train_splits_dir, ignore_errors=True)
    print(f"🧹 Removed extracted '{os.path.basename(train_splits_dir)}' folder.")

    try:
        if os.path.isdir(parent_dir) and not os.listdir(parent_dir) and parent_dir != UNPACK_DIR:
            os.rmdir(parent_dir)
            print(f"🧹 Removed now-empty '{os.path.basename(parent_dir)}' folder.")
    except OSError:
        pass

def sort_meld():
    if not os.path.exists(UNPACK_DIR):
        print(f"Error: {UNPACK_DIR} not found.")
        sys.exit(1)

    # Make sure the train_splits videos are actually on disk - extract
    # train.tar.gz if we don't see a train_splits folder yet.
    train_splits_dir = find_train_splits_dir(UNPACK_DIR)
    if not train_splits_dir:
        extracted_top_dir = extract_train_archive()
        if extracted_top_dir:
            train_splits_dir = find_train_splits_dir(extracted_top_dir) or extracted_top_dir

    if not train_splits_dir:
        print("Error: Could not find or extract a 'train_splits' folder.")
        sys.exit(1)

    # CSV label files may live alongside the archives or have arrived via
    # the extraction itself - search fresh, after extraction, either way.
    csv_files = []
    for root, _, files in os.walk(UNPACK_DIR):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("Error: Could not find MELD label CSV files (train_sent_emo.csv, etc.)")
        sys.exit(1)

    # dia{Dialogue_ID}_utt{Utterance_ID}.mp4 -> emotion, straight from the CSVs
    master_labels = {}
    for csv_path in csv_files:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dia_id, utt_id, emotion = row['Dialogue_ID'], row['Utterance_ID'], row['Emotion'].lower()
                    video_name = f"dia{dia_id}_utt{utt_id}.mp4"
                    master_labels[video_name] = emotion
                except KeyError:
                    pass

    video_files = []
    for root, _, files in os.walk(train_splits_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))

    total_files = len(video_files)
    if total_files == 0:
        print("No .mp4 files found inside train_splits.")
        sys.exit(1)

    print(f"Parsed CSVs. Found {total_files} MELD video files. Sorting...")

    moved_count = 0
    for i, filepath in enumerate(video_files):
        filename = os.path.basename(filepath)

        if filename in master_labels:
            raw_emo = master_labels[filename]
            if raw_emo in EMOTION_MAP:
                emotion_name = EMOTION_MAP[raw_emo]

                target_folder = os.path.join(TARGET_DIR, emotion_name)
                os.makedirs(target_folder, exist_ok=True)

                dst = os.path.join(target_folder, f"{get_prefix()}_{filename}")
                if not os.path.exists(dst):
                    shutil.move(filepath, dst)
                    moved_count += 1

        print_progress(i + 1, total_files, prefix="Sorting")

    print(f"\n✅ Moved {moved_count} MELD files into {TARGET_DIR}")

    # All videos have been moved out of train_splits - clean up the folder
    # that extraction produced (whether that's 'train_splits' directly or
    # a wrapping 'train' folder around it).
    cleanup_extracted_folder(train_splits_dir)

if __name__ == "__main__":
    sort_meld()