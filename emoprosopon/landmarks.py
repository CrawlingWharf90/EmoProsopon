import os
import sys
import logging

#* ─────────────────────────────────────────────────────────────────
#* MUTE C++ TENSORFLOW, GLOG, & ABSEIL WARNINGS
#* ─────────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['GLOG_minloglevel'] = '3'
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['ABSL_MINLOGLEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

logging.getLogger('absl').setLevel(logging.ERROR)

#* ─────────────────────────────────────────────────────────────────
#* DEVELOPER SETTINGS
#* ─────────────────────────────────────────────────────────────────
SILENT_MODE = True  #! Set to False to enable developer console logs

def dev_print(*args, **kwargs):
    """Custom print function that respects SILENT_MODE."""
    if not SILENT_MODE:
        print(*args, **kwargs)

#* ─────────────────────────────────────────────────────────────────
#* THE NUCLEAR OPTION: C-LEVEL STDERR SILENCER
#* ─────────────────────────────────────────────────────────────────
class SuppressStderr:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fd = os.dup(2) 
        os.dup2(self.null_fd, 2) 
    def __exit__(self, *_):
        os.dup2(self.save_fd, 2) 
        os.close(self.null_fd)
        os.close(self.save_fd)

import cv2
import mediapipe as mp
import time
import threading
import numpy as np
import torch 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from kinematics import KinematicManager
from HUD import HUDManager
from emoprosopon.fusion_engine import EmotionFusionEngine
from static_cnn import StaticFeatureExtractor, get_face_transform

#* ─────────────────────────────────────────────────────────────────
#* MediaPipe & Configuration
#* ─────────────────────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(BASE_DIR, 'models', 'hand_landmarker.task')),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

segmenter_options = mp.tasks.vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(BASE_DIR, 'models', 'selfie_multiclass_256x256.tflite')),
    running_mode=VisionRunningMode.VIDEO
)

panic_mode = False
model_loaded = False

FEATURE_ORDER = [
    "Right Eyebrow", "Left Eyebrow", "Right Eye", "Left Eye", 
    "Right Iris", "Left Iris", "Nose Bridge", "Nose Tip", 
    "Nostrils", "Upper Lip", "Lower Lip", "Jaw Line", 
    "Right Cheek", "Left Cheek", "Forehead"
]

FACE_REGIONS = {
    "Right Eyebrow": [46,53,52,65,55,70,71,72,73,74,75,76,77,124,156],
    "Left Eyebrow": [276,283,282,295,285,300,301,302,303,304,305,306,307,353,383],
    "Right Eye": [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
    "Left Eye": [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398],
    "Right Iris": [468,469,470,471,472],
    "Left Iris": [473,474,475,476,477],
    "Nose Bridge": [6,197,195,5,4,1,19,94,2],
    "Nose Tip": [1,2,3,4,5,195,197,6],
    "Nostrils": [48,115,131,134,102,49,220,305,344,360,363,440,274,279],
    "Upper Lip": [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146],
    "Lower Lip": [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185],
    "Jaw Line": [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],
    "Right Cheek": [116,117,118,119,120,121,128,126,142,36,205,187,123,50,203],
    "Left Cheek": [345,346,347,348,349,350,357,355,371,266,425,411,352,280,423],
    "Forehead": [10,151,9,8,168,6,197,195,5,107,66,105,63,70,156,336,296,334,293,300,383]
}

HAND_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

shared_raw_mask = None
current_mp_image = None
app_running = True
HISTORY_SIZE = 15
MAX_POSSIBLE_TRACKERS = 5
YUNET_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'face_detection_yunet_2023mar.onnx')

landmark_histories = {}

def segmenter_worker(segmenter):
    global shared_raw_mask, current_mp_image, app_running
    while app_running:
        if current_mp_image is not None:
            try:
                res = segmenter.segment(current_mp_image)
                if res.category_mask is not None:
                    shared_raw_mask = res.category_mask.numpy_view().copy()
            except: pass
        time.sleep(0.01)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hud_instance = param
        hud_instance.handle_click(x, y)

class GlobalLandmark:
    def __init__(self, x, y, z):
        self.x = x; self.y = y; self.z = z

#* ─────────────────────────────────────────────────────────────────
#* Main Engine Logic
#* ─────────────────────────────────────────────────────────────────
def run_tracker(source_type="camera", source_val=0):
    global app_running, current_mp_image, landmark_histories, model_loaded, panic_mode
    app_running = True
    
    if not panic_mode:
        hud = HUDManager(FACE_REGIONS.keys())
    
    lm_colors = {}
    if not panic_mode:
        for rname, indices in FACE_REGIONS.items():
            color = hud.region_colors.get(rname, (0, 255, 0))
            for idx in indices:
                lm_colors[idx] = color

    kinematics_engines = [KinematicManager(FACE_REGIONS.keys()) for _ in range(MAX_POSSIBLE_TRACKERS)]

    #* ─────────────────────────────────────────────────────────────
    #* LOAD THE SPATIO-TEMPORAL AI BRAIN
    #* ─────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fusion_engine = None
    static_cnn = None
    face_transform = None
    
    try:
        fusion_engine = EmotionFusionEngine(device=device.type)
        static_cnn = StaticFeatureExtractor(embedding_size=64).to(device)
        static_cnn.eval()
        face_transform = get_face_transform()
        
        model_loaded = True 
        dev_print(f"✅ Two-Stream Fusion Engine loaded successfully on {device}!")
    except Exception as e:
        dev_print(f"⚠️ Could not load trained models: {e}")
        dev_print(f"⚠️ Predictions are disabled. Run 'eop -n all -b' to train both networks!")

    if not panic_mode:
        hud.model_loaded = model_loaded 

    EMOTION_MAP_REV = {-1: "Scanning", 0: "Neutral", 1: "Happy", 2: "Sad", 3: "Angry", 4: "Fear", 5: "Surprise", 6: "Disgust"}
    
    sequence_buffers = {i: [] for i in range(MAX_POSSIBLE_TRACKERS)}
    
    cap = None
    sct = None
    monitor = None

    if source_type in ["camera", "video"]: 
        cap = cv2.VideoCapture(source_val)
        if not cap.isOpened():
            # Critical errors still print regardless of SILENT_MODE so the user knows why it crashed
            print(f"\n❌ CRITICAL: Could not read from camera index {source_val}.")
            print("Troubleshooting:")
            print("1. Is another application (Zoom, OBS, Discord) currently using your webcam?")
            print("2. Windows Privacy: Go to Settings -> Privacy & security -> Camera -> Enable 'Let desktop apps access your camera'.")
            sys.exit(1)
            
    elif source_type == "screen":
        import mss
        sct = mss.MSS() 
        monitor = sct.monitors[source_val]

    cv2.namedWindow('EmoProsopopon', cv2.WINDOW_AUTOSIZE)
    tracker_lms = []

    #* Mute the C++ backend while loading the models into RAM
    with SuppressStderr():
        for _ in range(MAX_POSSIBLE_TRACKERS):
            t_options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=os.path.join(BASE_DIR, 'models', 'face_landmarker.task')),
                running_mode=VisionRunningMode.VIDEO, 
                num_faces=1, 
                min_face_detection_confidence=0.1, 
                min_face_presence_confidence=0.1,
                min_tracking_confidence=0.1
            )
            tracker_lms.append(mp.tasks.vision.FaceLandmarker.create_from_options(t_options))

        hand_lm = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
        segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(segmenter_options)
    
    threading.Thread(target=segmenter_worker, args=(segmenter,), daemon=True).start()
    
    if source_type in ["camera", "video"]: success, init_frame = cap.read()
    elif source_type == "screen":
        init_frame = np.array(sct.grab(monitor))
        init_frame = cv2.cvtColor(init_frame, cv2.COLOR_BGRA2BGR)
        
    raw_h, raw_w = init_frame.shape[:2]
    target_h = 720
    target_w = int(raw_w * (target_h / raw_h))
    
    yunet = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH, "", (target_w, target_h),
        score_threshold=0.4,
        nms_threshold=0.3,
        top_k=5
    )

    pTime = time.time()

    while True:
        window_alive = True
        try:
            if cv2.getWindowProperty('EmoProsopopon', cv2.WND_PROP_VISIBLE) < 1: window_alive = False
        except cv2.error:
            window_alive = False
            
        if not window_alive:
            cv2.namedWindow('EmoProsopopon', cv2.WINDOW_AUTOSIZE)
            if not panic_mode: cv2.setMouseCallback('EmoProsopopon', mouse_callback, param=hud)

        if source_type in ["camera", "video"]:
            success, frame = cap.read()
            if not success: break
        elif source_type == "screen":
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame = cv2.resize(frame, (target_w, target_h))
        h, w = frame.shape[:2] 

        now = time.time()
        fps = 1 / (max(now - pTime, 0.001))
        pTime = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.perf_counter() * 1000)

        _, faces = yunet.detect(frame)
        
        hand_res = hand_lm.detect_for_video(mp_image, ts_ms)
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        if hand_res.hand_landmarks:
            for hl in hand_res.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hl]
                cv2.fillPoly(hand_mask, [np.array([pts[i] for i in [0,1,5,9,13,17]], np.int32)], 255)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(hand_mask, pts[a], pts[b], 255, thickness=25)

        all_occ_data = [{} for _ in range(MAX_POSSIBLE_TRACKERS)]
        all_kin_data = [{} for _ in range(MAX_POSSIBLE_TRACKERS)]
        hud_kin_preds = [("Disabled", 0.0) for _ in range(MAX_POSSIBLE_TRACKERS)]
        hud_static_preds = [("Disabled", 0.0) for _ in range(MAX_POSSIBLE_TRACKERS)]
        global_fused_emotions = {} 
        
        detected_heads = []
        if faces is not None:
            for face in faces:
                bx, by, bbw, bbh = map(int, face[:4])
                
                bx, by = max(0, bx), max(0, by)
                bbw = min(w - bx, bbw)
                bbh = min(h - by, bbh)

                cx, cy = bx + bbw // 2, by + bbh // 2
                size = int(max(bbw, bbh) * 1.5) 
                half = size // 2
                
                startX, startY = max(0, cx - half), max(0, cy - half)
                endX, endY = min(w, cx + half), min(h, cy + half)
                
                if endX - startX < 20 or endY - startY < 20: continue
                
                detected_heads.append({
                    'sort_x': bx, 
                    'display_box': (bx, by, bbw, bbh),
                    'crop_box': (startX, startY, endX, endY)
                })

        detected_heads.sort(key=lambda x: x['sort_x'])
        detected_face_count = len(detected_heads)

        seg_mask = shared_raw_mask
        if seg_mask is not None and seg_mask.shape != (h, w):
            seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        hud_overlay = frame.copy()
        solid_green_dots = []
        solid_red_dots = []
        
        for t_idx in range(hud.max_trackers):
            stable_id = hud.assignments[t_idx]
            
            if stable_id < len(detected_heads):
                head = detected_heads[stable_id]
                startX, startY, endX, endY = head['crop_box']
                face_crop = rgb[startY:endY, startX:endX]
                crop_h, crop_w = face_crop.shape[:2]
                
                static_embedding = None
                if model_loaded and hud.track_static and face_crop.size > 0:
                    tensor_crop = face_transform(face_crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        static_embedding = static_cnn(tensor_crop).cpu().numpy().flatten()
                
                face_crop_resized = cv2.resize(face_crop, (256, 256))
                mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop_resized)
                tracker_res = tracker_lms[t_idx].detect_for_video(mp_crop, ts_ms)

                if tracker_res.face_landmarks:
                    if stable_id not in landmark_histories:
                        landmark_histories[stable_id] = np.zeros((478, HISTORY_SIZE), dtype=np.uint8)
                    history_arr = landmark_histories[stable_id]

                    remapped_landmarks = []
                    for lm in tracker_res.face_landmarks[0]:
                        gx = (lm.x * crop_w) + startX
                        gy = (lm.y * crop_h) + startY
                        gz = lm.z * crop_w
                        remapped_landmarks.append(GlobalLandmark(gx / w, gy / h, gz / w))

                    kin_engine = kinematics_engines[t_idx]
                    local_signatures = kin_engine.compute_local_coordinates(remapped_landmarks, w, h)
                    
                    lm_occ_states = {}
                    for idx, lm in enumerate(remapped_landmarks):
                        px, py = int(lm.x * w), int(lm.y * h)
                        if 0 <= px < w and 0 <= py < h:
                            finger = (hand_mask[py, px] == 255)
                            is_occ = finger
                            if seg_mask is not None:
                                cat = seg_mask[py, px]
                                if cat in [0, 4, 5]: is_occ = True
                                if cat == 2 and not finger: is_occ = False 
                            
                            history_arr[idx] = np.roll(history_arr[idx], 1)
                            history_arr[idx][0] = 1 if is_occ else 0
                            lm_occ_states[idx] = np.sum(history_arr[idx]) >= (HISTORY_SIZE * 0.75)
                    
                    current_occ = {r: False for r in FACE_REGIONS}
                    
                    for rname, indices in FACE_REGIONS.items():
                        valid_pts = []
                        occ_results = []
                        for i in indices:
                            if i in lm_occ_states:
                                occ_results.append(lm_occ_states[i])
                                if not lm_occ_states[i]: valid_pts.append(local_signatures[i])
                        
                        if occ_results:
                            current_occ[rname] = (sum(occ_results)/len(occ_results) > 0.5)
                        
                        if not current_occ[rname]: kin_engine.update(rname, valid_pts)
                        else:
                            kin_engine.activity[rname] = 0.0
                            kin_engine.history[rname].clear() 

                    all_occ_data[t_idx] = current_occ.copy()
                    all_kin_data[t_idx] = kin_engine.activity.copy()

                    if hud.track_kinematics:
                        current_features = [kin_engine.activity[rname] for rname in FEATURE_ORDER]
                        sequence_buffers[t_idx].append(current_features)
                        if len(sequence_buffers[t_idx]) > 30: sequence_buffers[t_idx].pop(0)
                    else:
                        sequence_buffers[t_idx].clear() 
                        
                    if model_loaded:
                        kin_data_to_pass = None
                        hud_kin_label = "Disabled"
                        
                        if len(sequence_buffers[t_idx]) == 30:
                            kin_array = np.array(sequence_buffers[t_idx])
                            movement_delta = np.max(kin_array, axis=0) - np.min(kin_array, axis=0)
                            
                            if np.max(movement_delta) > 0.015: 
                                kin_data_to_pass = sequence_buffers[t_idx]
                            else:
                                hud_kin_label = "No Movement" 
                        
                        f_idx, f_conf, k_idx, k_conf, s_idx, s_conf = fusion_engine.predict_all(
                            kinematic_sequence=kin_data_to_pass, 
                            static_embedding=static_embedding, 
                            alpha=0.6 
                        )
                        
                        if kin_data_to_pass is not None:
                            hud_kin_preds[t_idx] = (EMOTION_MAP_REV.get(k_idx, "Unknown"), k_conf)
                        else:
                            hud_kin_preds[t_idx] = (hud_kin_label, 0.0)
                            
                        hud_static_preds[t_idx] = (EMOTION_MAP_REV.get(s_idx, "Disabled"), s_conf)
                        global_fused_emotions[stable_id] = (EMOTION_MAP_REV.get(f_idx, "Scanning"), f_conf)
                    else:
                        global_fused_emotions[stable_id] = ("Missing Model", 0.0)

                    for idx, lm in enumerate(remapped_landmarks):
                        px, py = int(lm.x * w), int(lm.y * h)
                        if 0 <= px < w and 0 <= py < h:
                            is_occluded = lm_occ_states.get(idx, False)
                            if is_occluded: solid_red_dots.append((px, py))
                            elif not panic_mode and hud.panel_open and idx in lm_colors:
                                cv2.circle(hud_overlay, (px, py), 2, lm_colors[idx], -1, cv2.LINE_AA)
                            else: solid_green_dots.append((px, py))

        for stable_id, head in enumerate(detected_heads):
            bx, by, bbw, bbh = head['display_box']
            show_lbl = not panic_mode and hud.show_face_labels
            show_emo = not panic_mode and hud.show_detected_emotion
            
            if show_lbl or show_emo:
                cv2.rectangle(frame, (bx, by), (bx + bbw, by + bbh), (0, 200, 255), 2)
                if show_lbl:
                    cv2.putText(frame, f"Face {stable_id}", (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                if show_emo:
                    emo_text, _ = global_fused_emotions.get(stable_id, ("Scanning", 0.0))
                    cv2.putText(frame, emo_text, (bx, min(h - 10, by + bbh + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

        cv2.addWeighted(hud_overlay, 0.6, frame, 0.4, 0, frame)
        for (px, py) in solid_green_dots: cv2.circle(frame, (px, py), 1, (0, 200, 50), -1, cv2.LINE_AA)
        for (px, py) in solid_red_dots: cv2.circle(frame, (px, py), 2, (0, 0, 255), -1, cv2.LINE_AA)

        key = cv2.waitKey(1) & 0xFF
        if panic_mode:
            if key == 113 or key == 8: break
        else:
            if hud.handle_input(key): break
            hud.draw(frame, h, w, fps, all_occ_data, all_kin_data, detected_face_count, hud_kin_preds, hud_static_preds, global_fused_emotions) 

        cv2.imshow('EmoProsopopon', frame)
        if not panic_mode: 
            cv2.setMouseCallback('EmoProsopopon', lambda event, x, y, flags, param: param.handle_click(x, y)
            if event == cv2.EVENT_LBUTTONDOWN else None, param=hud)

    app_running = False
    if cap: cap.release()
    for t_lm in tracker_lms: t_lm.close()
    hand_lm.close()
    segmenter.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracker(source_type="camera", source_val=0)