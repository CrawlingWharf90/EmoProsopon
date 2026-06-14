import os
import sys

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
from emoprosopon.static_cnn import StaticFeatureExtractor, get_face_transform

#* ─────────────────────────────────────────────────────────────────
#* CONFIGURATION & PATHS
#* ─────────────────────────────────────────────────────────────────
DATASET_VIDEO_DIR = os.path.join(BASE_DIR, "sorted_datasets", "video")
DATASET_IMAGE_DIR = os.path.join(BASE_DIR, "sorted_datasets", "image")

OUT_KINEMATIC_DIR = os.path.join(BASE_DIR, "processed_data", "kinematic")
OUT_STATIC_DIR = os.path.join(BASE_DIR, "processed_data", "static")

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

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] ({iteration}/{total}) {percent}% Complete')
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
#* 2. STATIC HARVESTER (Images)
#* ─────────────────────────────────────────────────────────────────
def process_image(img_path, yunet, static_model, transform, device):
    frame = cv2.imread(img_path)
    if frame is None: return None
    
    h, w = frame.shape[:2]
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_crop = rgb[startY:endY, startX:endX]
            
            # Apply PyTorch transform (resizes to 224x224 and normalizes)
            tensor_crop = transform(face_crop).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = static_model(tensor_crop).cpu().numpy().flatten()
            return embedding
    return None

def harvest_static():
    print(f"\n{CYAN}=== Starting Static Harvester (Images) ==={RESET}")
    os.makedirs(OUT_STATIC_DIR, exist_ok=True)

    if not os.path.exists(DATASET_IMAGE_DIR):
        print(f"{RED}No image datasets found at {DATASET_IMAGE_DIR}{RESET}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{YELLOW}Loading MobileNetV2 CNN onto {device}...{RESET}")
    
    # Initialize the customized CNN and its transform logic
    static_model = StaticFeatureExtractor(embedding_size=64).to(device)
    static_model.eval()
    transform = get_face_transform()
    
    yunet = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (320, 320), score_threshold=0.4, top_k=1)
    
    X_data, Y_labels = [], []

    for emotion_folder in os.listdir(DATASET_IMAGE_DIR):
        if emotion_folder not in EMOTION_MAP: continue
            
        label = EMOTION_MAP[emotion_folder]
        folder_path = os.path.join(DATASET_IMAGE_DIR, emotion_folder)
        img_files = glob.glob(os.path.join(folder_path, "*.*"))
        total_imgs = len(img_files)
        
        if total_imgs == 0: continue
        print(f"Processing {total_imgs} images for {YELLOW}'{emotion_folder}'...{RESET}")
        
        for i, img_path in enumerate(img_files):
            try:
                emb = process_image(img_path, yunet, static_model, transform, device)
                if emb is not None:
                    X_data.append(emb)
                    Y_labels.append(label)
            except Exception as e:
                pass
            print_progress(i + 1, total_imgs, prefix=f"  ↳ {emotion_folder}")  
                
    if len(X_data) > 0:
        X_data = np.array(X_data)
        Y_labels = np.array(Y_labels)
        np.save(os.path.join(OUT_STATIC_DIR, 'X_embeddings.npy'), X_data)
        np.save(os.path.join(OUT_STATIC_DIR, 'Y_labels.npy'), Y_labels)
        print(f"{GREEN}Static Harvesting Complete! Saved {len(X_data)} embeddings of length 64.{RESET}\n")
    else:
        print(f"{YELLOW}No valid embeddings extracted.{RESET}\n")


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

    if run_kinematic:
        harvest_kinematic()
        
    if run_static:
        harvest_static()