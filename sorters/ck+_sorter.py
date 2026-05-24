import os
import sys
import glob
import cv2

#* CK+ Emotion Labels mapped to our 7 core emotions
#? (Note: We skip '2' which is Contempt, as it falls outside the core 7)
EMOTION_MAP = {
    1.0: "Angry", 3.0: "Disgust", 4.0: "Fear", 
    5.0: "Happy", 6.0: "Sad", 7.0: "Surprise"
}

#! The TUI extracted it here (Ensure both images and labels are inside!)
UNPACK_DIR = os.path.join("unpkged_datasets", "CK+") 
TARGET_DIR = "sorted_datasets" 

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def sort_ckplus():
    if not os.path.exists(UNPACK_DIR):
        print(f"Error: {UNPACK_DIR} not found.")
        sys.exit(1)

    #? CK+ structure: Emotion/Subject_ID/Sequence_ID/*.txt
    label_files = glob.glob(os.path.join(UNPACK_DIR, "Emotion", "*", "*", "*.txt"))
    total_files = len(label_files)
    
    if total_files == 0:
        print(f"No Emotion .txt files found in {UNPACK_DIR}/Emotion.")
        sys.exit(1)

    print(f"Found {total_files} labeled CK+ sequences. Stitching PNGs to MP4s...")

    for i, txt_path in enumerate(label_files):
        #* 1. Read the Emotion Label
        with open(txt_path, 'r') as f:
            try:
                emotion_float = float(f.read().strip())
            except ValueError:
                continue
                
        if emotion_float not in EMOTION_MAP:
            continue
            
        emotion_name = EMOTION_MAP[emotion_float]
        
        #* 2. Find the corresponding image sequence folder
        #? txt_path looks like: .../Emotion/S005/001/S005_001_00000011_emotion.txt
        parts = txt_path.split(os.sep)
        subject_id = parts[-3]
        sequence_id = parts[-2]
        
        image_folder = os.path.join(UNPACK_DIR, "cohn-kanade-images", subject_id, sequence_id)
        png_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
        
        if not png_files:
            continue
            
        #* 3. Setup the VideoWriter to stitch the images
        target_folder = os.path.join(TARGET_DIR, emotion_name)
        os.makedirs(target_folder, exist_ok=True)
        
        video_filename = f"CK_{subject_id}_{sequence_id}.mp4"
        video_path = os.path.join(target_folder, video_filename)
        
        if not os.path.exists(video_path):
            sample_frame = cv2.imread(png_files[0])
            h, w, _ = sample_frame.shape
            
            #! 30 FPS ensures it matches our Kinematics Engine temporal window exactly
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            
            for png in png_files:
                frame = cv2.imread(png)
                out.write(frame)
                
            out.release()
            
        print_progress(i + 1, total_files, prefix="Stitching")

if __name__ == "__main__":
    sort_ckplus()