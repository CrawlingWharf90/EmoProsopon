#!/usr/bin/env python3
import os
import shutil
import sys
import urllib.request
import json
import tempfile
import subprocess
import time
import re

#* ─────────────────────────────────────────────────────────────────
#* CONSTANTS & HELPERS
#* ─────────────────────────────────────────────────────────────────
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, '.eop_config')
VERSION_FILE = os.path.join(BASE_DIR, 'VERSION')

#* ─────────────────────────────────────────────────────────────────
#* SCALABLE COMMAND TREE & HELP ROUTER
#* ─────────────────────────────────────────────────────────────────
class CmdNode:
    def __init__(self, flags, desc, args_help=None, sub_nodes=None):
        self.flags = flags             # List of strings: ['--tui', '-t']
        self.desc = desc               # Description string
        self.args_help = args_help or [] # List of expected trailing arguments
        self.sub_nodes = sub_nodes or [] # List of child CmdNodes

#? THE GLOBAL REGISTRY: Add new commands here and they auto-generate in the help menu!
COMMAND_TREE = [
    CmdNode(["--start", "-s"], "Launch the EmoProsopon Main Engine.", sub_nodes=[
        CmdNode(["live"], "Bypass the GUI and launch directly into live camera tracking."),
        CmdNode(["video"], "Bypass the GUI and run the tracker on a pre-recorded video file.", args_help=[
            "forward=True/False : If False, closes the app when the video ends (Default: True).",
            "path=\"/path/file\"  : Directly load this file (Skips the OS file picker)."
        ]),
        CmdNode(["screen"], "Bypass the GUI and capture a specific monitor for analysis.", args_help=[
            "<index> : The numerical ID of the monitor (e.g., 1, 2)."
        ])
    ]),
    CmdNode(["--tui", "-t"], "Launch a specific Terminal User Interface (TUI) tool.", sub_nodes=[
        CmdNode(["models"], "Manage Core AI weights (YuNet, MediaPipe, MobileNetV2...)."),
        CmdNode(["datasets"], "Browse and download research image/video databases."),
        CmdNode(["extractor"], "Unzip and normalize raw datasets into the /sorted_datasets directory.")
    ]),
    CmdNode(["--train", "-n"], "Execute the Machine Learning pipeline.", sub_nodes=[
        CmdNode(["all"], "(Default) Run harvester, then immediately train the models.", args_help=[
            "-s / --static    : Process and train only Image datasets (StaticMLP).",
            "-k / --kinematic : Process and train only Video datasets (KinematicLSTM).",
            "-b / --both      : Process and train both data streams sequentially (Default)."
        ]),
        CmdNode(["harvest"], "Extract kinematics & embeddings into .npy arrays.", args_help=[
            "-s / --static    : Process only Image datasets.",
            "-k / --kinematic : Process only Video datasets.",
            "-b / --both      : Process both data streams (Default)."
        ]),
        CmdNode(["model"], "Train the neural networks on the extracted .npy arrays.", args_help=[
            "-s / --static    : Train only the StaticMLP.",
            "-k / --kinematic : Train only the KinematicLSTM.",
            "-b / --both      : Train both models sequentially (Default)."
        ])
    ]),
    CmdNode(["--setup", "-g"], "Run the first-time setup wizard."),
    CmdNode(["--require", "-r"], "Install Python dependencies."),
    CmdNode(["--update", "-u"], "Check for and apply the latest updates."),
    CmdNode(["--uninstall", "-un"], "Launch the uninstaller GUI."),
    CmdNode(["--installer", "-i"], "Open the installer GUI."),
    CmdNode(["--version", "-v"], "Show installed version."),
    CmdNode(["--change-python", "-cp"], "Reconfigure Python interpreter.")
]

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

def run_require(py_cmd):
    clear_screen()
    print(f"[NECESSARY] Python Libraries")
    req_path = os.path.join(BASE_DIR, "downloaders", "requirements.txt")
    
    #? 1. PRINT REQUIREMENTS 
    try:
        with open(req_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    lib_name = re.split(r'[=<>~]', line)[0].strip()
                    print(f"  - {CYAN}{lib_name}{RESET}")
    except FileNotFoundError:
        print(f"  - {RED}Warning: requirements.txt not found at {req_path}{RESET}")

    #? 2. ASK FOR CONFIRMATION
    choice = input("\nDo you want to install the required Python libraries via pip now? (y/n): ").strip().lower()
    
    #? 3. EXECUTE ONLY IF 'YES'
    if choice in ['y', 'yes']:
        print(f"{YELLOW}Installing Pip Dependencies...{RESET}")
        pip_args = [py_cmd, "-m", "pip", "install", "-r", req_path]
        
        custom_env = os.environ.copy()
        
        #! Apply PEP 668 bypass and RAM-disk fixes ONLY on Unix systems
        if os.name != 'nt':
            pip_args.append("--break-system-packages")
            custom_env["TMPDIR"] = "/var/tmp"
            
        result = subprocess.run(pip_args, env=custom_env)
        
        #! Catch failures with a realistic diagnostic message
        if result.returncode != 0:
            print(f"\n{RED}Critical Error: Failed to install dependencies.{RESET}")
            print(f"{YELLOW}Pip encountered an error. Please check your internet connection and disk space.{RESET}")
            print(f"If 'pip' is entirely missing from your system, install it via:")
            print(f"  - {CYAN}Arch:{RESET} sudo pacman -S python-pip")
            print(f"  - {CYAN}Ubuntu:{RESET} sudo apt install python3-pip")
            print(f"  - {CYAN}macOS:{RESET} python3 -m ensurepip --upgrade")
            sys.exit(1)
    else:
        print(f"{YELLOW}Skipping pip installation... (The next steps may fail){RESET}")

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
def display_help(args_path=None):
    args_path = args_path or []

    if not args_path:
        print(f"\n{CYAN}EmoProsopopon CLI Commands (eop){RESET}")
        print("=" * 72)
        for node in COMMAND_TREE:
            flags_str = f"{GREEN}{', '.join(node.flags)}{RESET}"
            print(f"{flags_str}")
            print(f"    {node.desc}")
        print("=" * 72)
        print(f"{CYAN}Tip:{RESET} Type 'eop <command> -h' for sub-commands and details.\n")
        return

    current_nodes = COMMAND_TREE
    target_node = None
    path_taken = []

    for arg in args_path:
        found = False
        for node in current_nodes:
            if arg in node.flags:
                target_node = node
                current_nodes = node.sub_nodes
                path_taken.append(node.flags[0]) # Save primary flag for breadcrumbs
                found = True
                break
        if not found:
            print(f"\n{RED}Error: Unknown command layer '{arg}'.{RESET}")
            break #! Stop digging (IYBT) if they typed gibberish

    if target_node:
        cmd_string = f"eop {' '.join(path_taken)}"
        print(f"\n{CYAN}Command: {cmd_string}{RESET}")
        print(f"  {target_node.desc}")
        
        if target_node.args_help:
            print(f"\n{YELLOW}Arguments:{RESET}")
            for arg_desc in target_node.args_help:
                print(f"  {GREEN}{arg_desc}{RESET}")
                
        if target_node.sub_nodes:
            print(f"\n{YELLOW}Sub-commands:{RESET}")
            for sub in target_node.sub_nodes:
                print(f"  {GREEN}{', '.join(sub.flags):<10}{RESET} : {sub.desc}")
        print("")

def run_tui(target, py_cmd):
    if target == "models":
        subprocess.run([py_cmd, os.path.join(BASE_DIR, "downloaders", "download_models.py")])
    elif target == "datasets":
        subprocess.run([py_cmd, os.path.join(BASE_DIR, "downloaders", "download_dataset.py")])
    elif target == "extractor":
        subprocess.run([py_cmd, os.path.join(BASE_DIR, "downloaders", "extract_and_sort_dataset.py")])
    else:
        print(f"{RED}Invalid TUI target. Use: models, datasets, or extractor.{RESET}")

def run_setup(py_cmd):
    clear_screen()
    print(f"{CYAN}--- EMO-PROSOPOPON SETUP WIZARD ---{RESET}\n")
    
    #? 1. Require
    print(f"[{GREEN}1/3{RESET}] Python Libraries")
    time.sleep(1)    
    run_require(py_cmd)
    
    #? 2. Models
    print(f"\n[{GREEN}2/3{RESET}] Launching Model Manager...")
    time.sleep(1)
    run_tui("models", py_cmd)
    
    #? 3. Data Prompt
    clear_screen()
    print(f"[{GREEN}3/3{RESET}] Training Data")
    choice = input(f"\nDo you want to download and extract datasets for training now? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes']:
        run_tui("datasets", py_cmd)
        run_tui("extractor", py_cmd)
        
    clear_screen()
    print(f"{GREEN}Setup Complete!{RESET}")
    print_status(py_cmd)
    time.sleep(2)
    
    check_launch_requirements(py_cmd)
    subprocess.run([py_cmd, os.path.join(BASE_DIR, "emoprosopon", "mainmenu.py")])

#* ─────────────────────────────────────────────────────────────────
#* MAIN PARSER
#* ─────────────────────────────────────────────────────────────────
def main():
    if '-h' in sys.argv or '--help' in sys.argv:
        clean_args = [arg.lower() for arg in sys.argv[1:] if arg not in ['-h', '--help']]
        display_help(clean_args)
        sys.exit(0)

    if len(sys.argv) < 2:
        display_help()
        sys.exit(0)

    py_cmd = get_python_cmd()
    command = sys.argv[1].lower()

    if command in ['--start', '-s']:
        check_launch_requirements(py_cmd)
        start_args = [py_cmd, os.path.join(BASE_DIR, "emoprosopon", "mainmenu.py")] + sys.argv[2:]
        subprocess.run(start_args)

    elif command in ['--require', '-r']:
        run_require(py_cmd)

    elif command in ['--tui', '-t']:
        if len(sys.argv) < 3:
            print(f"{RED}Missing TUI target. Provide 'models', 'datasets', or 'extractor'.{RESET}")
            sys.exit(1)
        else:
            run_tui(sys.argv[2].lower(), py_cmd)

    elif command in ['--setup', '-g']:
        run_setup(py_cmd)

    elif command in ['--train', '-n']:
        valid_targets = ['all', 'harvest', 'model']
        target = "all"
        modality = "--both"
        unknown_args = []
        
        for arg in sys.argv[2:]:
            arg_lower = arg.lower()
            if arg_lower in valid_targets:
                target = arg_lower
            elif arg_lower in ['-s', '--static']:
                modality = "--static"
            elif arg_lower in ['-k', '--kinematic']:
                modality = "--kinematic"
            elif arg_lower in ['-b', '-a', '--both', '--all']:
                modality = "--both"
            else:
                unknown_args.append(arg)
                
        if unknown_args:
            print(f"{RED}Error: Unknown argument(s) for --train: {', '.join(unknown_args)}{RESET}")
            print(f"{YELLOW}Valid targets:{RESET} all, harvest, model")
            print(f"{YELLOW}Valid modalities:{RESET} -s (static), -k (kinematic), -b (both)")
            sys.exit(1)
                
        if target in ["all", "harvest"]:
            print(f"{CYAN}Running Harvester ({modality.replace('--', '')})...{RESET}")
            subprocess.run([py_cmd, os.path.join(BASE_DIR, "trainers", "harvest_dataset.py"), modality])
            
        if target in ["all", "model"]:
            print(f"{CYAN}Running Trainer ({modality.replace('--', '')})...{RESET}")
            subprocess.run([py_cmd, os.path.join(BASE_DIR, "trainers", "train_emotion_model.py"), modality])

    elif command in ['--update', '-u']:
        run_update()

    elif command in ['--uninstall', '-un']:
        run_uninstall(py_cmd)

    elif command in ['--installer', '-i']:
        run_installer(py_cmd)

    elif command in ['--version', '-v']:
        version = get_version()
        print(f"\nEmoProsopon Version: {GREEN}{version}{RESET}\n")

    elif command in ['--change-python', '-cp']: #! Don't abreviate change-python (ಠಿ_ಠ)
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