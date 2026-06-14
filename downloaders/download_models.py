import os
import sys
import ssl
import urllib.request
import argparse

#* ─────────────────────────────────────────────────────────────────
#* GLOBALLY BYPASS SSL VERIFICATION
#* ─────────────────────────────────────────────────────────────────
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

#* ─────────────────────────────────────────────────────────────────
#* TERMINAL COLORS & UTILS
#* ─────────────────────────────────────────────────────────────────
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'

#? Resolve paths dynamically relative to this script's location (downloaders/ folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

#* ─────────────────────────────────────────────────────────────────
#* CORE AI MODEL CONFIGURATION
#* ─────────────────────────────────────────────────────────────────
MODELS = {
    "YuNet Face Detector": {
        "file": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "desc": "Primary face hunter. Handles extreme angles."
    },
    "MediaPipe Face Landmarker": {
        "file": "face_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "desc": "3D mesh generator (creates the 478 facial points)."
    },
    "MediaPipe Hand Landmarker": {
        "file": "hand_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "desc": "Hand tracker for occlusion detection (blocks fingers)."
    },
    "MediaPipe Image Segmenter": {
        "file": "selfie_multiclass_256x256.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        "desc": "Background/hair segmenter for dense occlusion."
    },
    "MobileNetV2 Feature Extractor": {
        "file": "mobilenet_v2-b0353104.pth",
        "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        "desc": "Pre-trained CNN backbone for static emotion embeddings."
    }
}

#* ─────────────────────────────────────────────────────────────────
#* CORE FUNCTIONS
#* ─────────────────────────────────────────────────────────────────
def is_downloaded(filename):
    return os.path.exists(os.path.join(MODELS_DIR, filename))

def progress_bar(block_num, block_size, total_size):
    #? Callback function for urlretrieve to display a dynamic progress bar
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded / total_size
        percent = min(1.0, percent) 
        
        bar_length = 30
        filled_length = int(bar_length * percent)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        dl_mb = downloaded / (1024 * 1024)
        tot_mb = total_size / (1024 * 1024)
        
        sys.stdout.write(f'\r    ↳ [{bar}] ({dl_mb:.1f} MB / {tot_mb:.1f} MB) {YELLOW}{percent:.1%}{RESET} Complete')
        sys.stdout.flush()
    else:
        dl_mb = downloaded / (1024 * 1024)
        sys.stdout.write(f'\r    ↳ Downloading... ({dl_mb:.1f} MB)')
        sys.stdout.flush()

def download_missing_models(interactive=True):
    os.makedirs(MODELS_DIR, exist_ok=True)
    missing_models = {name: info for name, info in MODELS.items() if not is_downloaded(info['file'])}
    
    if not missing_models:
        print(f"\n{GREEN}✅ All required models are already downloaded!{RESET}")
        if interactive:
            input(f"\nPress Enter to return...")
        return
        
    print("\n--- Downloading Missing Models ---")
    for name, info in missing_models.items():
        print(f"\nFetching {name}...")
        dest_path = os.path.join(MODELS_DIR, info['file'])
        try:
            urllib.request.urlretrieve(info['url'], dest_path, reporthook=progress_bar)
            print(f"\n{GREEN}✅ Successfully saved: {info['file']}{RESET}")
        except Exception as e:
            print(f"\n{RED}❌ Failed to download {name}. Error: {e}{RESET}")
            
    if interactive:
        input(f"\n{GREEN}Downloads Complete! Press Enter to continue...{RESET}")

#* ─────────────────────────────────────────────────────────────────
#* MENUS
#* ─────────────────────────────────────────────────────────────────
def menu_manage():
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(" MANAGE AI MODELS ".center(50, "="))
        print("="*50)
        
        #? Build a list of ONLY the models that exist on the hard drive
        downloaded = []
        for name, info in MODELS.items():
            if is_downloaded(info['file']):
                downloaded.append((name, info['file']))
                
        if not downloaded:
            print(f"\n{YELLOW}No models are currently installed in the '{MODELS_DIR}' folder.{RESET}")
            input("\nPress Enter to go back...")
            break
            
        for idx, (name, filename) in enumerate(downloaded):
            print(f"{idx + 1}. {GREEN}{name}{RESET} [{filename}]")
         
        choice = input(f"\nEnter numbers separated by commas to REMOVE models (e.g., 1, 2) {CYAN}or 0 to go back{RESET}."
                       f"\n{RED}Type -1 to delete all models.{RESET}"
                       f"\n> ").strip()
        
        if choice == '0':
            break
            
        to_uninstall = []
        
        if choice == '-1':
            clear_screen()
            print(f"{RED}!!! TOTAL DELETION CONFIRMATION !!!{RESET}")
            print(f"\n{RED}You are about to DELETE EVERY DOWNLOADED NEURAL NETWORK.{RESET}")
            print(f"{YELLOW}EmoProsopon will not function until these are re-downloaded.{RESET}")
            
            confirm = input(f"\n{CYAN}Type 'yes' to confirm TOTAL DELETION, or anything else to cancel: {RESET}").strip().lower()
            if confirm == 'yes':
                to_uninstall = downloaded
            else:
                print(f"{YELLOW}Total deletion cancelled.{RESET}")
                input(f"\nPress Enter to continue...")
                continue
        else:
            for part in choice.split(','):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(downloaded):
                        to_uninstall.append(downloaded[idx])
                    
        if to_uninstall:
            print("\n--- Deleting Models ---")
            for name, filename in to_uninstall:
                file_path = os.path.join(MODELS_DIR, filename)
                try:
                    os.remove(file_path)
                    print(f"{GREEN}✅ {name} removed successfully.{RESET}")
                except Exception as e:
                    print(f"{RED}❌ Failed to remove {name}. Error: {e}{RESET}")
            
            input("\nPress Enter to continue...")

#* ─────────────────────────────────────────────────────────────────
#* MAIN LOGIC
#* ─────────────────────────────────────────────────────────────────
def run_tui():
    while True:
        clear_screen()
        print("\n" + "="*80)
        print(" EMO-PROSOPOPON AI MODEL MANAGER ".center(80, "="))
        print("="*80)
        
        #! Welcome Message
        print(f"{CYAN}Welcome! The following neural network weights are strictly necessary{RESET}")
        print(f"{CYAN}for the EmoProsopon live-tracking engine to function.{RESET}\n")
        
        #? Display the Model Status List
        for name, info in MODELS.items():
            if is_downloaded(info['file']):
                status = f"[Downloaded]"
                print(f"{GREEN}{name:<30} {status:<22}{RESET} - {info['desc']}")
            else:
                status = f"[Missing]"
                print(f"{YELLOW}{name:<30} {status:<22}{RESET} - {info['desc']}")
                
        print("\nOptions:")
        print("1. Download missing models")
        print("2. Handle installed models")
        print("\n0. Quit")
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == '1':
            download_missing_models(interactive=True)
        elif choice == '2':
            menu_manage()
        elif choice == '0':
            clear_screen()
            print("Exiting Model Manager...")
            break
        else:
            print(f"{RED}Invalid option. Please try again.{RESET}")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    #? 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Emo-Prosopon Model Manager")
    parser.add_argument('--auto', action='store_true', help="Automatically download missing models")
    parser.add_argument('--missing', action='store_true', help="Alias for --auto")
    
    args = parser.parse_args()
    
    #? 2. Execution Routing
    if args.auto or args.missing:
        print(f"🔧 CLI Mode Activated. Downloading missing models...\n")
        download_missing_models(interactive=False)
        print(f"\n✅ {GREEN}CLI Execution Complete.{RESET}")
        input(f"\nPress Enter to exit...")
        clear_screen()
    else:
        run_tui()

if __name__ == "__main__":
    main()