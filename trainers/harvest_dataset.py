import os
import sys
import threading, time

def configure_mediapipe_logs(verbose: bool = False): 
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

configure_mediapipe_logs(verbose=False)  

import cv2
import mediapipe as mp
import numpy as np
import glob
import torch

#? Dynamically find the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

#* IMPORT LOCAL MODULES
from emoprosopon.kinematics import KinematicManager

#* ─────────────────────────────────────────────────────────────────
#* CONFIGURATION & PATHS
#* ─────────────────────────────────────────────────────────────────
DATASET_VIDEO_DIR = os.path.join(BASE_DIR, "sorted_datasets", "video")
DATASET_IMAGE_DIR = os.path.join(BASE_DIR, "sorted_datasets", "image")

OUT_KINEMATIC_DIR = os.path.join(BASE_DIR, "processed_data", "kinematic")
OUT_STATIC_CROP_DIR = os.path.join(BASE_DIR, "processed_data", "static_crops")

YUNET_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'face_detection_yunet_2023mar.onnx')
FACE_LM_PATH = os.path.join(BASE_DIR, 'models', 'face_landmarker.task')

TARGET_FRAMES = 30 #? Lock every sequence to 30 frames

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'

FEATURE_ORDER = [
    "Right Eyebrow", "Left Eyebrow", "Right Eye", "Left Eye", 
    "Right Iris", "Left Iris", "Nose Bridge", "Nose Tip", 
    "Nostrils", "Upper Lip", "Lower Lip", "Jaw Line", 
    "Right Cheek", "Left Cheek", "Forehead"
]

EMOTION_MAP = {
    "Neutral": 0, "Happy": 1, "Sad": 2, "Angry": 3, 
    "Fear": 4, "Surprise": 5, "Disgust": 6
}

def print_progress(iteration, total, prefix='', length=30, start_time=None):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    eta_str = ""
    if start_time is not None:
        elapsed = time.time() - start_time
        if iteration > 0 and iteration < total:
            avg_time = elapsed / iteration
            rem = total - iteration
            eta_secs = int(rem * avg_time)
            hrs, rem_secs = divmod(eta_secs, 3600)
            mins, secs = divmod(rem_secs, 60)
            eta_str = f" | ETA: {int(hrs)}h {int(mins)}m" if hrs > 0 else f" | ETA: {int(mins)}m {int(secs)}s"

    sys.stdout.write('\r\033[K' + f'{prefix} | [{bar}] ({iteration}/{total}) {percent}% Complete{eta_str}')
    sys.stdout.flush()
    if iteration == total: print()

#* ─────────────────────────────────────────────────────────────────
#* 1. KINEMATIC HARVESTER (Videos)
#* ─────────────────────────────────────────────────────────────────
def process_video(video_path, yunet, face_lm, kin_engine):
    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    
    while True:
        success, frame = cap.read()
        if not success: break
            
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(frame)
        
        if faces is not None and len(faces) > 0:
            bx, by, bbw, bbh = map(int, faces[0][:4])
            bx, by = max(0, bx), max(0, by)
            bbw, bbh = min(w - bx, bbw), min(h - by, bbh)

            cx, cy = bx + bbw // 2, by + bbh // 2
            size = int(max(bbw, bbh) * 1.5) 
            half = size // 2
            startX, startY = max(0, cx - half), max(0, cy - half)
            endX, endY = min(w, cx + half), min(h, cy + half)
            
            if endX - startX > 20 and endY - startY > 20:
                face_crop = rgb[startY:endY, startX:endX]
                crop_h, crop_w = face_crop.shape[:2]
                face_crop_resized = cv2.resize(face_crop, (256, 256))
                mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop_resized)
                
                res = face_lm.detect(mp_crop)
                
                if res.face_landmarks:
                    class GlobalLandmark:
                        def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
                            
                    remapped_landmarks = []
                    for lm in res.face_landmarks[0]:
                        gx, gy, gz = (lm.x * crop_w) + startX, (lm.y * crop_h) + startY, lm.z * crop_w
                        remapped_landmarks.append(GlobalLandmark(gx / w, gy / h, gz / w))

                    local_sigs = kin_engine.compute_local_coordinates(remapped_landmarks, w, h)
                    
                    for rname in FEATURE_ORDER:
                        kin_engine.update(rname, [local_sigs[i] for i in [0] if i in local_sigs])
                        
                    frame_features = [kin_engine.activity[region] for region in FEATURE_ORDER]
                    sequence_data.append(frame_features)
                    
    cap.release()
    if len(sequence_data) == 0: return None
        
    sequence_data = np.array(sequence_data)
    if len(sequence_data) > TARGET_FRAMES:
        sequence_data = sequence_data[:TARGET_FRAMES]
    elif len(sequence_data) < TARGET_FRAMES:
        padding = np.zeros((TARGET_FRAMES - len(sequence_data), 15))
        sequence_data = np.vstack((sequence_data, padding))
        
    return sequence_data

def harvest_kinematic():
    print(f"\n{CYAN}=== Starting Kinematic Harvester (Videos) ==={RESET}")
    os.makedirs(OUT_KINEMATIC_DIR, exist_ok=True)
    
    if not os.path.exists(DATASET_VIDEO_DIR):
        print(f"{RED}No video datasets found at {DATASET_VIDEO_DIR}{RESET}")
        return

    yunet = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (320, 320), score_threshold=0.4, top_k=1)
    
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_LM_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1
    )
    face_lm = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    kin_engine = KinematicManager(FEATURE_ORDER)

    X_data, Y_labels = [], []

    for emotion_folder in os.listdir(DATASET_VIDEO_DIR):
        if emotion_folder not in EMOTION_MAP: continue
            
        label = EMOTION_MAP[emotion_folder]
        folder_path = os.path.join(DATASET_VIDEO_DIR, emotion_folder)
        video_files = glob.glob(os.path.join(folder_path, "*.*"))
        total_videos = len(video_files)
        
        if total_videos == 0: continue
        print(f"Processing {total_videos} videos for {YELLOW}'{emotion_folder}'...{RESET}")
        
        for i, video_path in enumerate(video_files):
            try:
                seq = process_video(video_path, yunet, face_lm, kin_engine)
                if seq is not None:
                    X_data.append(seq)
                    Y_labels.append(label)
            except Exception as e:
                pass
            print_progress(i + 1, total_videos, prefix=f"  ↳ {emotion_folder}")  
                
    if len(X_data) > 0:
        X_data = np.array(X_data)
        Y_labels = np.array(Y_labels)
        np.save(os.path.join(OUT_KINEMATIC_DIR, 'X_features.npy'), X_data)
        np.save(os.path.join(OUT_KINEMATIC_DIR, 'Y_labels.npy'), Y_labels)
        print(f"{GREEN}Kinematic Harvesting Complete! Saved {len(X_data)} sequences.{RESET}\n")
    else:
        print(f"{YELLOW}No valid kinematics extracted.{RESET}\n")

#* ─────────────────────────────────────────────────────────────────
#* 2. STATIC HARVESTER (Face Pre-Cropper)
#* ─────────────────────────────────────────────────────────────────
def harvest_static():
    """
    Pre-processes the entire image dataset. Finds faces, crops them, 
    and saves them to processed_data/static_crops/.
    Doing this once saves hundreds of hours during the training loop.
    """
    print(f"\n{CYAN}=== Starting Static Harvester (Face Pre-Cropper) ==={RESET}")
    print(f"{YELLOW}This process will locate and crop faces from all images to drastically speed up training.{RESET}")
    
    os.makedirs(OUT_STATIC_CROP_DIR, exist_ok=True)
    if not os.path.exists(DATASET_IMAGE_DIR):
        print(f"{RED}No image datasets found at {DATASET_IMAGE_DIR}{RESET}")
        return

    yunet = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (320, 320), score_threshold=0.4, top_k=1)

    for emotion_name in EMOTION_MAP.keys():
        in_folder = os.path.join(DATASET_IMAGE_DIR, emotion_name)
        out_folder = os.path.join(OUT_STATIC_CROP_DIR, emotion_name)
        
        if not os.path.exists(in_folder): continue
        os.makedirs(out_folder, exist_ok=True)
        
        files = [f for f in os.listdir(in_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        total_imgs = len(files)
        if total_imgs == 0: continue
        
        print(f"Cropping faces for {YELLOW}'{emotion_name}'{RESET}...")
        start_time = time.time()
        
        for i, f in enumerate(files):
            img_path = os.path.join(in_folder, f)
            out_path = os.path.join(out_folder, f)
            
            #? Skip if already processed
            if os.path.exists(out_path):
                # ONLY print every 100 files to prevent terminal flooding
                if i % 100 == 0 or i == total_imgs - 1:
                    print_progress(i + 1, total_imgs, prefix=f"  ↳ {emotion_name}", start_time=start_time)
                continue
                
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            h, w = frame.shape[:2]
            yunet.setInputSize((w, h))
            _, faces = yunet.detect(frame)
            
            if faces is not None and len(faces) > 0:
                try:
                    bx, by, bbw, bbh = map(int, faces[0][:4])
                    bx, by = max(0, bx), max(0, by)
                    bbw, bbh = min(w - bx, bbw), min(h - by, bbh)

                    cx, cy = bx + bbw // 2, by + bbh // 2
                    size = int(max(bbw, bbh) * 1.5)
                    half = size // 2
                    startX, startY = max(0, cx - half), max(0, cy - half)
                    endX, endY = min(w, cx + half), min(h, cy + half)

                    face_crop = frame[startY:endY, startX:endX]
                    if face_crop.size == 0: face_crop = frame
                except Exception as e:
                    # Gracefully skip corrupted images
                    face_crop = frame
            else:
                face_crop = frame 
                
            face_crop = cv2.resize(face_crop, (224, 224))
            cv2.imwrite(out_path, face_crop)
            
            # Print progress every 50 images for new files
            if i % 50 == 0 or i == total_imgs - 1:
                print_progress(i + 1, total_imgs, prefix=f"  ↳ {emotion_name}", start_time=start_time)
                
    print(f"\n{GREEN}Static Pre-Cropping Complete! You can now run the trainer.{RESET}\n")

#* ─────────────────────────────────────────────────────────────────
#* CLI ROUTING LOGIC
#* ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_kinematic = True
    run_static = True

    if len(sys.argv) > 1:
        flag = sys.argv[1].lower()
        if flag in ["--static", "-s"]:
            run_kinematic = False
        elif flag in ["--kinematic", "-k"]:
            run_static = False
        else:
            print(f"{YELLOW}Unknown flag provided. Running both by default.{RESET}")

    if run_kinematic:
        harvest_kinematic()
        
    if run_static:
        harvest_static()