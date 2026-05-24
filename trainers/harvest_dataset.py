import os
import sys

def configure_mediapipe_logs(verbose: bool = False): #? Surpress MediaPipe logs for cleaner output during harvesting. To see them chnage verbose to True on function call (around line 16)
    if not verbose:
        os.environ["GLOG_minloglevel"] = "3"
    else:
        class NewlineStderr:
            def write(self, msg):
                if msg.strip():
                    sys.stderr.write(f"\n{msg}")
            def flush(self):
                sys.stderr.flush()
        sys.stderr = NewlineStderr()

configure_mediapipe_logs(verbose=False)  #? Change to True to see MediaPipe logs during harvesting

import cv2
import mediapipe as mp
import numpy as np
import glob

#? Dynamically find the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

#* IMPORT THE KINEMATICS ENGINE FROM ITS NEW HOME
from emoprosopon.kinematics import KinematicManager

#* ─────────────────────────────────────────────────────────────────
#* CONFIGURATION
#* ─────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(BASE_DIR, "sorted_datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")
YUNET_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'face_detection_yunet_2023mar.onnx')
FACE_LM_PATH = os.path.join(BASE_DIR, 'models', 'face_landmarker.task')

TARGET_FRAMES = 30 #?Lock every sequence to 30 frames

#* ─────────────────────────────────────────────────────────────────
#* TERMINAL COLORS & UTILS
#* ─────────────────────────────────────────────────────────────────
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

#* Define the exact order of the 15 features
FEATURE_ORDER = [
    "Right Eyebrow", "Left Eyebrow", "Right Eye", "Left Eye", 
    "Right Iris", "Left Iris", "Nose Bridge", "Nose Tip", 
    "Nostrils", "Upper Lip", "Lower Lip", "Jaw Line", 
    "Right Cheek", "Left Cheek", "Forehead"
]

#! Map folder names to integer labels
EMOTION_MAP = {
    "Neutral": 0, "Happy": 1, "Sad": 2, "Angry": 3, 
    "Fear": 4, "Surprise": 5, "Disgust": 6
}

#* ─────────────────────────────────────────────────────────────────
#* UTILS
#* ─────────────────────────────────────────────────────────────────
def print_progress(iteration, total, prefix='', length=30):
    """Generates a dynamic terminal progress bar."""
    if total == 0:
        return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] ({iteration}/{total}) {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: 
        print()

#* ─────────────────────────────────────────────────────────────────
#* CORE LOGIC
#* ─────────────────────────────────────────────────────────────────
def process_video(video_path, yunet, face_lm, kin_engine):
    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        #? 1. YuNet Hunter
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(frame)
        
        if faces is not None and len(faces) > 0:
            #* Grab the most prominent face (index 0)
            bx, by, bbw, bbh = map(int, faces[0][:4])
            bx, by = max(0, bx), max(0, by)
            bbw = min(w - bx, bbw)
            bbh = min(h - by, bbh)

            cx, cy = bx + bbw // 2, by + bbh // 2
            size = int(max(bbw, bbh) * 1.5) 
            half = size // 2
            
            startX, startY = max(0, cx - half), max(0, cy - half)
            endX, endY = min(w, cx + half), min(h, cy + half)
            
            if endX - startX > 20 and endY - startY > 20:
                #? 2. Spotlight Crop
                face_crop = rgb[startY:endY, startX:endX]
                crop_h, crop_w = face_crop.shape[:2]
                face_crop_resized = cv2.resize(face_crop, (256, 256))
                mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop_resized)
                
                #? 3. MediaPipe 3D
                res = face_lm.detect(mp_crop)
                
                if res.face_landmarks:
                    class GlobalLandmark:
                        def __init__(self, x, y, z):
                            self.x, self.y, self.z = x, y, z
                            
                    remapped_landmarks = []
                    for lm in res.face_landmarks[0]:
                        gx = (lm.x * crop_w) + startX
                        gy = (lm.y * crop_h) + startY
                        gz = lm.z * crop_w
                        remapped_landmarks.append(GlobalLandmark(gx / w, gy / h, gz / w))

                    #? 4. Kinematics Math
                    local_sigs = kin_engine.compute_local_coordinates(remapped_landmarks, w, h)
                    
                    for rname in FEATURE_ORDER:
                        indices = kin_engine.history.keys() 
                        #! Assuming no occlusion in training datasets to keep it clean
                        valid_pts = [local_sigs[i] for i in kin_engine.history.keys() if i in local_sigs]
                        kin_engine.update(rname, [local_sigs[i] for i in [0]])
                        
                    #* Extract the flat 15-float array for this frame
                    frame_features = [kin_engine.activity[region] for region in FEATURE_ORDER]
                    sequence_data.append(frame_features)
                    
    cap.release()
    
    #? 5. Temporal Standardization (Force sequence to exactly 30 frames)
    if len(sequence_data) == 0:
        return None
        
    sequence_data = np.array(sequence_data)
    
    #! Handle array length: If more than 30 frames, take the first 30. If less, pad with zeros.
    if len(sequence_data) > TARGET_FRAMES:
        sequence_data = sequence_data[:TARGET_FRAMES]
    elif len(sequence_data) < TARGET_FRAMES:
        padding = np.zeros((TARGET_FRAMES - len(sequence_data), 15))
        sequence_data = np.vstack((sequence_data, padding))
        
    return sequence_data

def harvest_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    yunet = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (320, 320), score_threshold=0.4, top_k=1)
    
    BaseOptions = mp.tasks.BaseOptions
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_LM_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1
    )
    face_lm = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    kin_engine = KinematicManager(FEATURE_ORDER)

    X_data = []
    Y_labels = []

    for emotion_folder in os.listdir(DATASET_DIR):
        if emotion_folder not in EMOTION_MAP:
            continue
            
        label = EMOTION_MAP[emotion_folder]
        folder_path = os.path.join(DATASET_DIR, emotion_folder)
        video_files = glob.glob(os.path.join(folder_path, "*.*"))
        total_videos = len(video_files)
        
        print(f"Processing {total_videos} videos for {YELLOW}'{emotion_folder}'...{RESET}")
        
        for i, video_path in enumerate(video_files):
            try:
                seq = process_video(video_path, yunet, face_lm, kin_engine)
                if seq is not None:
                    X_data.append(seq)
                    Y_labels.append(label)
            except Exception as e:
                print(f"\n{YELLOW}[WARN] Skipped {os.path.basename(video_path)}: {e}{RESET}\n")

            print_progress(i + 1, total_videos, prefix=f"  ↳ {emotion_folder}")  
                
    X_data = np.array(X_data)
    Y_labels = np.array(Y_labels)
    
    np.save(os.path.join(OUTPUT_DIR, 'X_features.npy'), X_data)
    np.save(os.path.join(OUTPUT_DIR, 'Y_labels.npy'), Y_labels)
    print(f"\n{GREEN}Harvesting Complete! Saved {len(X_data)} sequences of shape {X_data.shape[1:]}.{RESET}")

if __name__ == "__main__":
    harvest_dataset()