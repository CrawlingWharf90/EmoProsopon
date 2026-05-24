import os
import shutil
import sys
import urllib.request
import json
import tempfile
import subprocess
import time

#* ─────────────────────────────────────────────────────────────────
#* CONSTANTS & HELPERS
#* ─────────────────────────────────────────────────────────────────
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, '.eop_config')
VERSION_FILE = os.path.join(BASE_DIR, 'VERSION')

#* ─────────────────────────────────────────────────────────────────
#* INSTALLER AND HELPER FUNCTIONS
#* ───────────────────────────────────────────────────────────────── 
def get_version():
    try:
        with open(VERSION_FILE, 'r') as f:
            return f.readline().strip()
    except:
        return "unversioned"

def run_update():
    print(f"\n{CYAN}--- EmoProsopon Updater ---{RESET}")
    print(f"{YELLOW}Checking for updates...{RESET}")
    
    current_version = get_version()
    
    REPO_API_URL = "https://api.github.com/repos/CrawlingWharf90/EmoProsopon/releases/latest"
    
    try:
        #? 1. Ping GitHub for the latest release
        req = urllib.request.Request(REPO_API_URL, headers={'User-Agent': 'EmoProsopon-Updater'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            
            latest_tag = data.get("tag_name", "0.0.0").replace("v", "") 
            
            if latest_tag > current_version:
                print(f"\n{GREEN}🚀 Update Available! ({current_version} -> {latest_tag}){RESET}")
                print(f"{CYAN}Release Notes:{RESET}\n{data.get('body', 'No release notes provided.')}\n")
                
                choice = input("Do you want to download and install this update now? (y/n): ").strip().lower()
                
                if choice in ['y', 'yes']:
                    print(f"\n{YELLOW}Connecting to GitHub to download assets...{RESET}")
                    
                    assets = data.get('assets', [])
                    installer_url = None
                    installer_name = None
                    
                    #? 2. Find the correct installer asset based on OS
                    for asset in assets:
                        if os.name == 'nt' and asset['name'].endswith('.exe'):
                            installer_url = asset['browser_download_url']
                            installer_name = asset['name']
                            break
                        elif os.name != 'nt' and asset['name'].endswith('.sh'):
                            installer_url = asset['browser_download_url']
                            installer_name = asset['name']
                            break
                        
                    if not installer_url:
                        print(f"{RED}Could not find a compatible installer in the release.{RESET}")
                        print(f"{GREEN}Please visit https://github.com/CrawlingWharf90/EmoProsopon/releases to download manually.{RESET}")
                        return

                    #? 3. Download to the OS temporary directory
                    temp_dir = tempfile.gettempdir()
                    temp_installer_path = os.path.join(temp_dir, installer_name)
                    
                    print(f"{CYAN}Downloading {installer_name}... Please wait.{RESET}")
                    try:
                        urllib.request.urlretrieve(installer_url, temp_installer_path)
                    except Exception as e:
                        print(f"{RED}Download failed: {e}{RESET}")
                        return
                        
                    print(f"{GREEN}Download complete! Launching installer...{RESET}")
                    
                    #? 4. Launch the installer and exit
                    if os.name == 'nt':
                        subprocess.Popen([temp_installer_path])
                        
                    #! Kill this process immediately so Inno Setup can overwrite EmoProsopon files without a lock error
                    sys.exit(0)
                    
            else:
                print(f"{GREEN}You are already running the latest version (v{current_version}).{RESET}")
                
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"{YELLOW}Update check bypassed: No public releases found yet (or repo is private).{RESET}")
        else:
            print(f"{RED}Failed to check for updates (HTTP {e.code}).{RESET}")
    except Exception as e:
        print(f"{RED}Failed to check for updates. Are you connected to the internet?{RESET}")
        # print(f"Debug: {e}")

def run_uninstall(py_cmd):
    print(f"\n{CYAN}Launching EmoProsopon Uninstaller...{RESET}")
    
    if os.name == 'nt':
        #? Inno Setup automatically generates unins000.exe in the installation root
        uninstaller = os.path.join(BASE_DIR, "unins000.exe")
        if os.path.exists(uninstaller):
            subprocess.Popen([uninstaller]) # Use Popen so the CLI doesn't lock up
            sys.exit(0)
        else:
            print(f"{RED}Uninstaller not found. (Are you running this from the installed directory?){RESET}")
    else:
        #? Launch custom Unix GUI in uninstall mode
        unix_gui = os.path.join(BASE_DIR, "installers", "unix_manager.py")
        if os.path.exists(unix_gui):
            subprocess.run([py_cmd, unix_gui, "--uninstall"])
        else:
            print(f"{RED}Unix manager script not found.{RESET}")

def run_installer(py_cmd):
    print(f"\n{CYAN}Launching EmoProsopon Installer...{RESET}")
    
    if os.name == 'nt':
        #! Point to the default Inno Setup output folder
        installer_exe = os.path.join(BASE_DIR, "installers", "Output", "EmoProsopon_Installer_v1.0.exe")
        if os.path.exists(installer_exe):
            subprocess.Popen([installer_exe])
            sys.exit(0)
        else:
            print(f"{RED}Installer executable not found at {installer_exe}.{RESET}")
            print(f"{YELLOW}Please run the .exe you downloaded, or compile the .iss file in Inno Setup.{RESET}")
    else:
        #! Launch custom Unix GUI in install mode
        unix_gui = os.path.join(BASE_DIR, "installers", "unix_manager.py")
        if os.path.exists(unix_gui):
            subprocess.run([py_cmd, unix_gui])
        else:
            print(f"{RED}Unix manager script not found.{RESET}")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

#* ─────────────────────────────────────────────────────────────────
#* CONFIGURATION & PYTHON SELECTION
#* ─────────────────────────────────────────────────────────────────
def get_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_config(config_data):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)

def get_installed_pythons():
    """Scans system for common python aliases."""
    candidates = ['python', 'python3', 'python3.12', 'py', 'python3.11']
    found = []
    seen = set()
    for cmd in candidates:
        path = shutil.which(cmd)
        if path and path not in seen:
            seen.add(path)
            try:
                ver = subprocess.check_output([cmd, "--version"], stderr=subprocess.STDOUT).decode().strip()
                found.append((cmd, ver))
            except:
                continue
    return found

def get_python_cmd():
    config = get_config()
    if 'python_cmd' in config:
        return config['python_cmd']
    
    clear_screen()
    print("\n" + "="*60)
    print(" INITIAL EMO-PROSOPOPON SETUP ".center(60, "="))
    print("="*60)
    
    print(f"\n{CYAN}Scanning system for installed Python versions...{RESET}\n")
    
    pythons = get_installed_pythons()
    
    if not pythons:
        print(f"{RED}No Python installation detected!{RESET}")
        print("Please install Python 3.12 from python.org")
        sys.exit(1)

    for idx, (cmd, ver) in enumerate(pythons):
        #? Highlight 3.12
        color = GREEN if "3.12" in ver else YELLOW
        print(f"{idx + 1}. {color}{cmd:<15} ({ver}){RESET}")
    
    print(f"\n{YELLOW}⚠️  Recommended: Python 3.12{RESET}")
    
    while True:
        choice = input(f"\nSelect a version (1-{len(pythons)}) or type custom command: ").strip()
        
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(pythons):
                py_cmd = pythons[idx][0]
                break
        else:
            if choice:
                py_cmd = choice
                break
                
    config['python_cmd'] = py_cmd
    save_config(config)
    print(f"\n{GREEN}✅ Saved '{py_cmd}' to .eop_config!{RESET}\n")
    time.sleep(1.5)
    return py_cmd

#* ─────────────────────────────────────────────────────────────────
#* STATUS & REQUIREMENT CHECKS
#* ─────────────────────────────────────────────────────────────────
def check_pip_requirements(py_cmd):
    """Silently tests if the core libraries are importable in the target Python environment."""
    test_script = "import cv2, mediapipe, numpy, torch, mss"
    try:
        subprocess.run([py_cmd, "-c", test_script], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def check_models():
    """Checks if all 4 required models exist."""
    models_dir = os.path.join(BASE_DIR, 'models')
    required = [
        "face_detection_yunet_2023mar.onnx",
        "face_landmarker.task",
        "hand_landmarker.task",
        "selfie_multiclass_256x256.tflite"
    ]
    if not os.path.exists(models_dir): return 0
    found = sum(1 for m in required if os.path.exists(os.path.join(models_dir, m)))
    return found

def get_dataset_counts():
    dataset_dir = os.path.join(BASE_DIR, 'dataset')
    unpack_dir = os.path.join(BASE_DIR, 'unpkged_datasets')
    
    dl_count = len([f for f in os.listdir(dataset_dir) if f.endswith('.zip') or f.endswith('.tar.gz')]) if os.path.exists(dataset_dir) else 0
    ex_count = len([d for d in os.listdir(unpack_dir) if os.path.isdir(os.path.join(unpack_dir, d))]) if os.path.exists(unpack_dir) else 0
    return dl_count, ex_count

def print_status(py_cmd):
    print("\n" + "="*50)
    print(" SYSTEM STATUS ".center(50, "="))
    print("="*50)
    
    #? 1. Pip Requirements
    pip_ok = check_pip_requirements(py_cmd)
    pip_str = f"[{GREEN}Completed{RESET}]" if pip_ok else f"[{RED}Missing{RESET}]"
    print(f"Pip Requirements:           {pip_str}")
    
    #? 2. Models
    models_found = check_models()
    mod_str = f"[{GREEN}Completed{RESET}]" if models_found == 4 else f"[{RED}Missing ({models_found}/4){RESET}]"
    print(f"Core AI Models:             {mod_str}")
    
    #? 3. Datasets
    dl_count, ex_count = get_dataset_counts()
    dl_str = f"[{GREEN}{dl_count} Downloaded{RESET}]" if dl_count > 0 else f"[{YELLOW}0 Downloaded{RESET}]"
    ex_str = f"[{GREEN}{ex_count} Extracted{RESET}]" if ex_count > 0 else f"[{YELLOW}0 Extracted{RESET}]"
    
    print(f"Raw Datasets:               {dl_str}")
    print(f"Extracted Datasets:         {ex_str}")
    print("="*50 + "\n")

def check_launch_requirements(py_cmd):
    pip_ok = check_pip_requirements(py_cmd)
    models_found = check_models()
    
    if not pip_ok or models_found < 4:
        clear_screen()
        print(f"{RED}!!! CANNOT START EMO-PROSOPOPON !!!{RESET}\n")
        if not pip_ok:
            print(f"{YELLOW}- Missing Python libraries. Run '{CYAN}eop --require{YELLOW}' or '{CYAN}eop --setup{YELLOW}'.{RESET}")
        if models_found < 4:
            print(f"{YELLOW}- Missing Core AI Models. Run '{CYAN}eop --tui models{YELLOW}' or '{CYAN}eop --setup{YELLOW}'.{RESET}")
        
        input(f"\nPress Enter to exit...")
        sys.exit(1)

#* ─────────────────────────────────────────────────────────────────
#* COMMAND ROUTING
#* ─────────────────────────────────────────────────────────────────
def display_help():
    print(f"\n{CYAN}EmoProsopopon CLI Commands (eop){RESET}")
    print("=" * 72)

    print(f"{GREEN}eop --start (-s){RESET}")
    print("    Launch the main engine")

    print(f"{GREEN}eop --setup (-g){RESET}")
    print("    Run the first-time setup wizard")

    print(f"{GREEN}eop --require (-r){RESET}")
    print("    Install Python dependencies")

    print(f"{GREEN}eop --tui <models|datasets|extractor> (-t){RESET}")
    print("    Open an interactive TUI manager")

    print(f"{GREEN}eop --train [all|harvest|model] (-n){RESET}")
    print("    Run ML training pipeline")

    print(f"{GREEN}eop --update (-u){RESET}")
    print("    Check for and apply the latest updates")

    print(f"{GREEN}eop --uninstall (-un){RESET}")
    print("    Launch the uninstaller GUI")

    print(f"{GREEN}eop --installer (-i){RESET}")
    print("    Open the installer GUI")

    print(f"{GREEN}eop --version (-v){RESET}")
    print("    Show installed version")

    print(f"{GREEN}eop --change-python (-cp){RESET}")
    print("    Reconfigure Python interpreter")

    print("-" * 72)

    print(f"{YELLOW}Advanced Commands{RESET}")

    print(f"{GREEN}eop --models --auto{RESET}")
    print("    Download all missing models")

    print(f"{GREEN}eop --datasets <list>{RESET}")
    print("    Download datasets (comma separated)")

    print(f"{GREEN}eop --extractor --extract <list>{RESET}")
    print("    Extract datasets")

    print(f"{GREEN}eop --extractor --sort <list>{RESET}")
    print("    Sort datasets")

    print(f"{GREEN}eop --extractor --syd <list>{RESET}")
    print("    Extract and sort datasets")

    print("=" * 72 + "\n")

def run_tui(target, py_cmd):
    if target == "models":
        subprocess.run([py_cmd, os.path.join("downloaders", "download_models.py")])
    elif target == "datasets":
        subprocess.run([py_cmd, os.path.join("downloaders", "download_dataset.py")])
    elif target == "extractor":
        subprocess.run([py_cmd, os.path.join("downloaders", "extract_and_sort_dataset.py")])
    else:
        print(f"{RED}Invalid TUI target. Use: models, datasets, or extractor.{RESET}")

def run_setup(py_cmd):
    clear_screen()
    print(f"{CYAN}--- EMO-PROSOPOPON SETUP WIZARD ---{RESET}\n")
    
    #? 1. Require
    print(f"[{GREEN}1/3{RESET}] Installing Pip Dependencies...")
    subprocess.run([py_cmd, "-m", "pip", "install", "-r", os.path.join("downloaders", "requirements.txt")])
    
    #? 2. Models
    print(f"\n[{GREEN}2/3{RESET}] Launching Model Manager...")
    time.sleep(1)
    run_tui("models", py_cmd)
    
    #? 3. Data Prompt
    clear_screen()
    print(f"[{GREEN}3/3{RESET}] Training Data")
    choice = input(f"\nDo you want to download and extract datasets for training now? (y/n): ").strip().lower()
    
    if choice == 'y' or choice == 'yes':
        run_tui("datasets", py_cmd)
        run_tui("extractor", py_cmd)
        
    clear_screen()
    print(f"{GREEN}Setup Complete!{RESET}")
    print_status(py_cmd)
    time.sleep(2)
    
    # Run Start
    check_launch_requirements(py_cmd)
    subprocess.run([py_cmd, os.path.join("emoprosopon", "mainmenu.py")])

#* ─────────────────────────────────────────────────────────────────
#* MAIN PARSER
#* ─────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        display_help()
        sys.exit(0)

    py_cmd = get_python_cmd()
    command = sys.argv[1].lower()

    if command in ['--start', '-s']:
        check_launch_requirements(py_cmd)
        subprocess.run([py_cmd, os.path.join("emoprosopon", "mainmenu.py")])

    elif command in ['--require', '-r']:
        print(f"{CYAN}Installing requirements via pip...{RESET}")
        subprocess.run([py_cmd, "-m", "pip", "install", "-r", os.path.join("downloaders", "requirements.txt")])

    elif command in ['--tui', '-t']:
        if len(sys.argv) < 3:
            print(f"{RED}Missing TUI target. Provide 'models', 'datasets', or 'extractor'.{RESET}")
            sys.exit(1)
        else:
            run_tui(sys.argv[2].lower(), py_cmd)

    elif command in ['--setup', '-g']:
        run_setup(py_cmd)

    elif command in ['--train', '-n']:
        valid = ['all', 'harvest', 'model']
        sub_cmd = sys.argv[2].lower() if len(sys.argv) > 2 else "all"
        if sub_cmd not in valid:
            print(f"{RED}Invalid training target. Use: all, harvest, or model.{RESET}")
            return
        if sub_cmd in ["all", "harvest"]:
            print(f"{CYAN}Running Harvester...{RESET}")
            subprocess.run([py_cmd, os.path.join("trainers", "harvest_dataset.py")])
        if sub_cmd in ["all", "model"]:
            print(f"{CYAN}Running LSTM Trainer...{RESET}")
            subprocess.run([py_cmd, os.path.join("trainers", "train_emotion_model.py")])

    elif command in ['--update', '-u']:
        run_update()

    elif command in ['--uninstall', '-un']:
        run_uninstall(py_cmd)

    elif command in ['--installer', '-i']:
        run_installer(py_cmd)

    elif command in ['--version', '-v']:
        version = get_version()
        print(f"\nEmoProsopon Version: {GREEN}{version}{RESET}\n")

    elif command in ['--change-python', '-cp']:
        config = get_config()
        config.pop('python_cmd', None)
        save_config(config)
        py_cmd = get_python_cmd()
        print_status(py_cmd)

    else:
        print(f"{RED}Unknown command: {command}{RESET}")
        display_help()

if __name__ == "__main__":
    main()