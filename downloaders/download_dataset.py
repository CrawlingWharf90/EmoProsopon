import subprocess
import os
import sys
import ssl
import urllib.request
import shutil
import argparse
import json
import atexit

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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PRIVATE_URLS_FILE = os.path.join(BASE_DIR, 'dataset', '.private_urls.json')
CONFIG_FILE = os.path.join(BASE_DIR, '.eop_config')
TEMP_DIR = os.path.join(BASE_DIR, 'dataset', 'temp')

def get_target_dir(modality):
    return os.path.join(BASE_DIR, 'dataset', modality)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

#* ─────────────────────────────────────────────────────────────────
#* GARBAGE COLLECTION (TEMP CLEANUP)
#* ─────────────────────────────────────────────────────────────────
def cleanup_temp_folder():
    """Ensures the temp folder is deleted when the script exits, even on errors/crashes."""
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception:
            pass # Suppress exit errors if file is locked

atexit.register(cleanup_temp_folder)

#* ─────────────────────────────────────────────────────────────────
#* CONFIG & API KEY MANAGEMENT
#* ─────────────────────────────────────────────────────────────────
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_config(config_data):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"{RED}Failed to save config: {e}{RESET}")

def check_hf_token():
    config = load_config()
    token = config.get("HF_API_KEY", "")
    
    if not token:
        print(f"\n{YELLOW}⚠️ Hugging Face API Key is missing in .eop_config.{RESET}")
        print(f"You can get your free access token here: {CYAN}https://huggingface.co/settings/tokens{RESET}")
        token = input(f"Please paste your HF_API_KEY to continue: ").strip()
        
        if token:
            config["HF_API_KEY"] = token
            save_config(config)
            print(f"{GREEN}✅ HF_API_KEY saved securely to .eop_config!{RESET}\n")
        else:
            print(f"{RED}No token provided. Hugging Face downloads will fail.{RESET}\n")
            return False
            
    try:
        from huggingface_hub import login
        login(token=token)
        return True
    except ImportError:
        print(f"{RED}❌ Missing dependency. Please run: pip install huggingface_hub tqdm{RESET}")
        return False

def check_kg_token():
    config = load_config()
    username = config.get("KG_USERNAME", "")
    api_key = config.get("KG_API_KEY", "")
    
    if not username or not api_key:
        print(f"\n{YELLOW}⚠️ Kaggle Credentials are missing in .eop_config.{RESET}")
        print(f"Get them by going to {CYAN}https://www.kaggle.com/settings{RESET} and clicking 'Create New Token'.")
        
        username = input(f"Please paste your Kaggle Username: ").strip()
        api_key = input(f"Please paste your Kaggle Key: ").strip()
        
        if username and api_key:
            config["KG_USERNAME"] = username
            config["KG_API_KEY"] = api_key
            save_config(config)
            print(f"{GREEN}✅ Kaggle credentials saved securely to .eop_config!{RESET}\n")
        else:
            print(f"{RED}Credentials not provided. Kaggle downloads will fail.{RESET}\n")
            return False
            
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = api_key
    return True

#* ─────────────────────────────────────────────────────────────────
#* SPATIO-TEMPORAL DATASET REGISTRY
#* ─────────────────────────────────────────────────────────────────
DATASETS = {
    #! VIDEO DATASETS (Temporal / Kinematic LSTM)
    "CK+": {"type": "video", "public": False, "default": True, "source": "direct", "auth_url": "http://www.pitt.edu/~emotion/ck-spread.htm", "desc": "Extended Cohn-Kanade"},
    "RAVDESS": {"type": "video", "public": True, "default": True, "source": "direct", "url": "https://zenodo.org/record/1188976/files/Video_Speech_Actor_01.zip", "desc": "Ryerson Audio-Visual Database"},
    "AFEW": {"type": "video", "public": False, "default": True, "source": "direct", "auth_url": "https://cs.anu.edu.au/few/", "desc": "Acted Facial Expressions in the Wild"},
    
    "CREMA-D": {"type": "video", "public": True, "default": False, "source": "direct", "url": "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip", "desc": "Crowd-sourced Emotional Mutimodal"},
    "eNTERFACE05": {"type": "video", "public": True, "default": False, "source": "direct", "url": "https://enterface.net/enterface05/docs/results/databases/project2_database.zip", "desc": "Audio-Visual Emotion Database"},
    "MELD": {"type": "video", "public": True, "default": False, "source": "direct", "url": "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz", "desc": "Multimodal EmotionLines Dataset (Friends TV Show)"},
    
    "SAVEE": {"type": "video", "public": False, "default": False, "source": "direct", "auth_url": "http://kahlan.eps.surrey.ac.uk/savee/Download.html", "desc": "Surrey Audio-Visual Expressed Emotion"}, 
    "IEMOCAP": {"type": "video", "public": False, "default": False, "source": "direct", "auth_url": "https://sail.usc.edu/iemocap/", "desc": "Interactive Emotional Dyadic Motion Capture"},
    
    "Video-Emotion": {"type": "video", "public": True, "default": False, "source": "kaggle", "repo_id": "unidpro/video-emotion-recognition-dataset", "desc": "Video Emotion Recognition Dataset [Kaggle]"},

    #! IMAGE DATASETS (Static / MobileNetV2 CNN)
    "JAFFE": {"type": "image", "public": True, "default": True, "source": "direct", "url": "https://zenodo.org/records/14974867/files/jaffe.zip", "desc": "Japanese Female Facial Expression Dataset"},
    "EED": {"type": "image", "public": True, "default": True, "source": "direct", "url": "https://zenodo.org/records/18012300/files/EED.zip", "desc": "Emotional Engagement Dataset (Students)"},
    "AR-Face": {"type": "image", "public": True, "default": True, "source": "direct", "url": "https://zenodo.org/records/19683234/files/ARDB_Full.zip", "desc": "AR Face Database (Illumination & Expression)"},
    "FER-2013": {"type": "image", "public": False, "default": False, "source": "direct", "auth_url": "https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data", "desc": "Facial Expression Recognition 2013 (Kaggle)"},
    "AffectNet": {"type": "image", "public": False, "default": False, "source": "direct", "auth_url": "http://mohammadmahoor.com/affectnet/", "desc": "Large-scale Database of Facial Expressions in the Wild"},
    "RAF-DB": {"type": "image", "public": False, "default": False, "source": "direct", "auth_url": "http://www.whdeng.cn/raf/model1.html", "desc": "Real-world Affective Faces Database"},
    "SFEW": {"type": "image", "public": False, "default": False, "source": "direct", "auth_url": "https://cs.anu.edu.au/few/", "desc": "Static Facial Expressions in the Wild"},
    "KDEF": {"type": "image", "public": False, "default": False, "source": "direct", "auth_url": "https://www.kdef.se/", "desc": "Karolinska Directed Emotional Faces"},
    "Oulu-CASIA": {"type": "image", "public": False, "default": False, "source": "direct", "auth_url": "http://www.cse.oulu.fi/CMV/Downloads/Oulu-CASIA", "desc": "Oulu-CASIA NIR VIS Database"},
    "FACES": {"type": "image", "public": False, "default": False, "source": "direct", "auth_url": "https://faces.mpdl.mpg.de/", "desc": "Max Planck FACES Database"},
    
    "FER2025": {"type": "image", "public": True, "default": False, "source": "huggingface", "repo_id": "imadhavan/FER2025", "desc": "Facial Expression Recognition 2025 [Hugging Face]"}
}

#* ─────────────────────────────────────────────────────────────────
#* CORE FUNCTIONS & PRIVATE LINK MANAGERS
#* ─────────────────────────────────────────────────────────────────
def load_private_urls():
    if os.path.exists(PRIVATE_URLS_FILE):
        try:
            with open(PRIVATE_URLS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_private_url(name, url):
    links = load_private_urls()
    links[name] = url
    try:
        with open(PRIVATE_URLS_FILE, 'w') as f:
            json.dump(links, f, indent=4)
    except Exception as e:
        print(f"{RED}Failed to save private link to {PRIVATE_URLS_FILE}: {e}{RESET}")

def run_eop(args: str):
    cmd = shutil.which("eop") or ("eop.bat" if os.name == "nt" else "./eop")
    subprocess.run(f"{cmd} {args}", shell=True)

def is_installed(name):
    info = DATASETS.get(name)
    if not info: return False
    
    target_dir = get_target_dir(info["type"])
    zip_path = os.path.join(target_dir, f"{name}.zip")
    tar_path = os.path.join(target_dir, f"{name}.tar.gz")
    folder_path = os.path.join(target_dir, name)
    return os.path.exists(zip_path) or os.path.exists(tar_path) or os.path.exists(folder_path)

def progress_bar(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(1.0, downloaded / total_size)
        bar_length = 30
        filled_length = int(bar_length * percent)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        dl_mb, tot_mb = downloaded / (1024 * 1024), total_size / (1024 * 1024)
        sys.stdout.write(f'\r    ↳ [{bar}] {percent:.1%} ({dl_mb:.1f} MB / {tot_mb:.1f} MB)')
        sys.stdout.flush()
    else:
        dl_mb = downloaded / (1024 * 1024)
        sys.stdout.write(f'\r    ↳ Downloading... ({dl_mb:.1f} MB)')
        sys.stdout.flush()

#* ─────────────────────────────────────────────────────────────────
#* PLATFORM SPECIFIC DOWNLOADERS & ZIP PIPELINE
#* ─────────────────────────────────────────────────────────────────
def download_file(url, dest_path):
    print(f"Downloading from direct URL: {url}")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_bar)
        print(f"\n{GREEN}✅ Successfully saved to: {dest_path}{RESET}\n")
    except Exception as e:
        print(f"\n{RED}❌ Failed to download. Error: {e}{RESET}\n")

def download_from_huggingface(dataset_link, item_name, dest_dir):
    print(f"Connecting to Hugging Face API for: {dataset_link}")
    if not check_hf_token():
        return
        
    temp_dataset_dir = os.path.join(TEMP_DIR, item_name)
    os.makedirs(temp_dataset_dir, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
        print(f"{CYAN}Downloading raw dataset files... (Progress bar mapped via tqdm){RESET}")
        snapshot_download(repo_id=dataset_link, repo_type="dataset", local_dir=temp_dataset_dir)
        
        print(f"\n{YELLOW}Archiving dataset into ZIP format...{RESET}")
        zip_base_path = os.path.join(dest_dir, item_name) # shutil adds .zip automatically
        shutil.make_archive(zip_base_path, 'zip', temp_dataset_dir)
        
        # Immediate cleanup
        shutil.rmtree(temp_dataset_dir)
        print(f"{GREEN}✅ Successfully synced, zipped, and cleaned temp files! -> {zip_base_path}.zip{RESET}\n")
        
    except ImportError:
        print(f"\n{RED}❌ Missing dependencies. Run: pip install huggingface_hub tqdm{RESET}\n")
    except Exception as e:
        print(f"\n{RED}❌ Hugging Face API download failed. Error: {e}{RESET}\n")

def download_from_kaggle(dataset_link, item_name, dest_dir):
    print(f"Connecting to Kaggle API for: {dataset_link}")
    if not check_kg_token():
        return
        
    temp_dataset_dir = os.path.join(TEMP_DIR, item_name)
    os.makedirs(temp_dataset_dir, exist_ok=True)
        
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print(f"{CYAN}Downloading and extracting raw dataset files... (Progress bar mapped via tqdm){RESET}")
        # quiet=False enforces the native Kaggle tqdm progress bar
        api.dataset_download_files(dataset=dataset_link, path=temp_dataset_dir, unzip=True, quiet=False)
        
        print(f"\n{YELLOW}Archiving dataset into ZIP format...{RESET}")
        zip_base_path = os.path.join(dest_dir, item_name) # shutil adds .zip automatically
        shutil.make_archive(zip_base_path, 'zip', temp_dataset_dir)
        
        # Immediate cleanup
        shutil.rmtree(temp_dataset_dir)
        print(f"{GREEN}✅ Successfully downloaded, zipped, and cleaned temp files! -> {zip_base_path}.zip{RESET}\n")
        
    except ImportError:
        print(f"\n{RED}❌ Kaggle library missing. Run: pip install kaggle tqdm{RESET}\n")
    except Exception as e:
        print(f"\n{RED}❌ Kaggle API failed. Error: {e}{RESET}\n")

def process_downloads(datasets_to_download, interactive=True):
    for item in datasets_to_download:
        if item not in DATASETS:
            print(f"{YELLOW}⚠️ Dataset '{item}' not found in registry. Skipping.{RESET}")
            continue
            
        info = DATASETS[item]
        target_dir = get_target_dir(info["type"])
        os.makedirs(target_dir, exist_ok=True)
            
        if is_installed(item):
            print(f"{GREEN}✅ {item} is already installed in {info['type']} folder. Skipping.{RESET}")
            continue

        print(f"\n--- Fetching {item} [{info['type'].upper()}] ---")
        
        # API Routes (Triggering Temp -> Zip Pipeline)
        if info.get("source") == "huggingface":
            download_from_huggingface(info["repo_id"], item, target_dir)
            continue
            
        elif info.get("source") == "kaggle":
            download_from_kaggle(info["repo_id"], item, target_dir)
            continue
        
        # Direct URL Routing
        if info["public"]:
            url = info["url"]
        else:
            auth_url = info.get("auth_url", "Please search online for official access.")
            print(f"{YELLOW}ℹ️ {item} requires academic approval.{RESET}")
            print(f"Apply here: {CYAN}{auth_url}{RESET}")
            
            #* Handle Private Link Injection
            private_urls = load_private_urls()
            saved_url = private_urls.get(item)
            
            if interactive:
                if saved_url:
                    print(f"\n{GREEN}💾 Found saved private link for {item}:{RESET}")
                    print(f"{CYAN}{saved_url}{RESET}")
                    user_input = input(f"Press Enter to use this link, or paste a new one to update it: ").strip()
                    
                    if not user_input:
                        url = saved_url
                    else:
                        url = user_input
                        save_private_url(item, url)
                        print(f"{GREEN}Updated saved link for {item}.{RESET}")
                else:
                    if item == "AFEW":
                        print(f"{YELLOW}👉 Please paste the share link for the folder called 'Train_AFEW' or similar (found inside the AFEW cloud directory).{RESET}")
                    elif item == "SFEW":
                        print(f"{YELLOW}👉 Please paste the share link for the folder called 'TRAIN' or similar (found inside SFEW2 > TRAIN).{RESET}")
                    
                    url = input(f"🔒 Paste your approved direct download link (or press Enter to skip): ").strip()
                    if url:
                        save_private_url(item, url)
                
                if not url:
                    print(f"{YELLOW}⏭️ Skipping {item}...{RESET}")
                    continue
            else:
                #! CLI Headless Mode
                if saved_url:
                    print(f"{GREEN}💾 Auto-injecting saved private link for {item}...{RESET}")
                    url = saved_url
                else:
                    print(f"{RED}⏭️ Skipping {item}. (CLI mode ignores restricted datasets without a saved link){RESET}")
                    continue
                
        extension = "tar.gz" if "tar.gz" in url else url.split('.')[-1]
        if len(extension) > 4 or "/" in extension: extension = "zip" 
            
        dest_path = os.path.join(target_dir, f"{item}.{extension}")
        download_file(url, dest_path)
        
        potential_train_path = os.path.join(target_dir, f"Train.{extension}")
        potential_afew_train_path = os.path.join(target_dir, f"Train_AFEW.{extension}")
        potential_lowercase_afew_path = os.path.join(target_dir, f"train_AFEW.{extension}")
        
        if item == "SFEW" and os.path.exists(potential_train_path):
            os.rename(potential_train_path, dest_path)
            print(f"{GREEN}🔄 Normalized 'Train.{extension}' to '{item}.{extension}'{RESET}")
        elif item == "AFEW" and os.path.exists(potential_afew_train_path):
            os.rename(potential_afew_train_path, dest_path)
            print(f"{GREEN}🔄 Normalized 'Train_AFEW.{extension}' to '{item}.{extension}'{RESET}")
        elif item == "AFEW" and os.path.exists(potential_lowercase_afew_path):
            os.rename(potential_lowercase_afew_path, dest_path)
            print(f"{GREEN}🔄 Normalized 'train_AFEW.{extension}' to '{item}.{extension}'{RESET}")
    
    if interactive: input("\nPress Enter to continue...")

#* ─────────────────────────────────────────────────────────────────
#* MENU RENDERING (Modality Aware)
#* ─────────────────────────────────────────────────────────────────
def menu_download_defaults(modality):
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(f" DEFAULT {modality.upper()} DATASETS ".center(50, "="))
        print("="*50)
        
        defaults = [k for k, v in DATASETS.items() if v["default"] and v["type"] == modality]
        missing = []
        
        if not defaults:
            print(f"{YELLOW}No default datasets defined for {modality}s.{RESET}")
        else:
            for name in defaults:
                if is_installed(name):
                    print(f"{GREEN}{name} [Installed]{RESET}")
                else:
                    missing.append(name)
                    print(f"{CYAN}{name} [Missing]{RESET}")
                
        print("\nOptions:")
        print("1. Confirm and download missing")
        print("\n0. Go back")
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == '1':
            if not missing:
                print(f"{GREEN}All default {modality} datasets are already installed!{RESET}")
                input("\nPress Enter to continue...")
            else:
                process_downloads(missing, interactive=True)
        elif choice == '0':
            break

def format_col_item(name, category):
    installed = is_installed(name)
    info = DATASETS[name]
    
    api_tag = ""
    if info.get("source") == "huggingface":
        api_tag = " | HF"
    elif info.get("source") == "kaggle":
        api_tag = " | Kaggle"
        
    restriction = "Pub" if info["public"] else "Res"
    status = "Installed" if installed else "Missing"
    full_status = f"[{restriction}{api_tag} | {status}]"
    
    if category == "default":
        return f"{GREEN if installed else CYAN}{name} {full_status}{RESET}"
    else:
        color = GREEN if installed else (YELLOW if info["public"] else RED)
        return f"{color}{name} {full_status}{RESET}"

def menu_browse(modality):
    while True:
        clear_screen()
        print("\n" + "="*110)
        print(f" BROWSE {modality.upper()} DATASETS ".center(110, "="))
        print("="*110)
        
        filtered = {k: v for k, v in DATASETS.items() if v["type"] == modality}
        defaults = [k for k, v in filtered.items() if v["default"]]
        publics = [k for k, v in filtered.items() if not v["default"] and v["public"]]
        restricteds = [k for k, v in filtered.items() if not v["default"] and not v["public"]]
        
        print(f"{'DEFAULTS':<35} | {'PUBLIC':<35} | {'RESTRICTED':<35}")
        print("-" * 110)
        
        max_rows = max(len(defaults), len(publics), len(restricteds))
        
        for i in range(max_rows):
            c1 = defaults[i] if i < len(defaults) else ""
            c2 = publics[i] if i < len(publics) else ""
            c3 = restricteds[i] if i < len(restricteds) else ""
            
            str1 = format_col_item(c1, "default") if c1 else ""
            str2 = format_col_item(c2, "public") if c2 else ""
            str3 = format_col_item(c3, "restricted") if c3 else ""
            
            def pad_ansi(text, visible_target):
                visible_len = len(''.join([c for c in text.replace(GREEN,'').replace(YELLOW,'').replace(RED,'').replace(CYAN,'').replace(RESET,'')]))
                return text + " " * max(0, visible_target - visible_len)

            print(f"{pad_ansi(str1, 35)} | {pad_ansi(str2, 35)} | {str3}")
            
        print("\nOptions:")
        print("1. Install Defaults")
        print("2. Select datasets to install")
        print("0. Go Back")
        
        choice = input("\nSelect an option: ").strip()
        if choice == '1': menu_download_defaults(modality)
        elif choice == '2': menu_select_to_install(modality)
        elif choice == '0': break

def menu_select_to_install(modality):
    clear_screen()
    modality_datasets = [k for k, v in DATASETS.items() if v["type"] == modality]
    
    print("\n" + "="*50)
    print(f" SELECT {modality.upper()} DATASETS TO INSTALL ".center(50, "="))
    print("="*50)

    for idx, name in enumerate(modality_datasets):
        info = DATASETS[name]
        
        api_tag = ""
        if info.get("source") == "huggingface": api_tag = " | Hugging Face"
        elif info.get("source") == "kaggle": api_tag = " | Kaggle"
        
        restriction = "Pub" if info["public"] else "Res"
        color = GREEN if is_installed(name) else (YELLOW if info["public"] else RED)
        status = "Installed" if is_installed(name) else "Missing"
            
        print(f"{idx + 1}. {color}{name} [{restriction}{api_tag} | {status}]{RESET}")
    
    choice = input(f"\nEnter numbers separated by commas (e.g., 1, 2) or {CYAN}0 to go back:{RESET} ").strip()
    if choice == '0': return
        
    to_download = [modality_datasets[int(p)-1] for p in choice.split(',') if p.strip().isdigit() and 0 <= int(p)-1 < len(modality_datasets)]
                
    if to_download: process_downloads(to_download, interactive=True)

def menu_status(modality):
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(f" {modality.upper()} DATASET STATUS & MANAGER ".center(50, "="))
        print("="*50)
        
        installed = [name for name, info in DATASETS.items() if info["type"] == modality and is_installed(name)]
        total = len([name for name, info in DATASETS.items() if info["type"] == modality])
        target_dir = get_target_dir(modality)
                
        if not installed:
            print(f"{YELLOW}No {modality} datasets currently installed in '{target_dir}'.{RESET}")
            input("\nPress Enter to return...")
            break
            
        for idx, name in enumerate(installed):
            info = DATASETS[name]
            api_tag = ""
            if info.get("source") == "huggingface": api_tag = " | HF"
            elif info.get("source") == "kaggle": api_tag = " | Kaggle"
            
            restriction = "Pub" if info["public"] else "Res"
            print(f"{GREEN}{idx + 1}. {name} [{restriction}{api_tag}]{RESET}")
            
        print(f"\nTotal Installed: {len(installed)}/{total}")
        choice = input(f"Enter numbers to UNINSTALL or {CYAN}0 to go back.{RESET}\n{RED}Type -1 to delete all installed {modality} datasets.{RESET}\n> ").strip()
        
        if choice == '0': break
        
        to_uninstall = []
        if choice == '-1':
            confirm = input(f"\n{RED}Type 'yes' to confirm TOTAL DELETION of {modality} sets: {RESET}").strip().lower()
            if confirm == 'yes': to_uninstall = installed
        else:
            to_uninstall = [installed[int(p)-1] for p in choice.split(',') if p.strip().isdigit() and 0 <= int(p)-1 < len(installed)]
                    
        if to_uninstall:
            print(f"\n--- Uninstalling {modality.capitalize()} Datasets ---")
            for item in to_uninstall:
                zip_path = os.path.join(target_dir, f"{item}.zip")
                tar_path = os.path.join(target_dir, f"{item}.tar.gz")
                folder_path = os.path.join(target_dir, item)
                
                try:
                    if os.path.exists(zip_path): os.remove(zip_path)
                    if os.path.exists(tar_path): os.remove(tar_path)
                    if os.path.exists(folder_path): shutil.rmtree(folder_path)
                    print(f"{GREEN}✅ {item} uninstalled successfully.{RESET}")
                except Exception as e:
                    print(f"{RED}❌ Failed to uninstall {item}: {e}{RESET}")
            input("\nPress Enter to continue...")

#* ─────────────────────────────────────────────────────────────────
#* MAIN MODALITY MENUS
#* ─────────────────────────────────────────────────────────────────
def run_modality_menu(modality):
    other_modality = "image" if modality == "video" else "video"
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(f" MANAGE {modality.upper()} DATASETS ".center(50, "="))
        print("="*50)
        print("1. Download defaults")
        print("2. Browse registry")
        print("3. Dataset status / Uninstall")
        print(f"4. Switch to {other_modality.capitalize()} Datasets")
        print("5. Launch Extractor & Sorter")
        print("\n0. Go Back")
        
        choice = input("\nSelect an option: ").strip()
        if choice == '1': menu_download_defaults(modality)
        elif choice == '2': menu_browse(modality)
        elif choice == '3': menu_status(modality)
        elif choice == '4': return run_modality_menu(other_modality)
        elif choice == '5': 
            run_eop("--tui extractor");
            sys.exit(0)
        elif choice == '0': break

def run_tui():
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(" EMO-PROSOPON DATASET DOWNLOADER ".center(50, "="))
        print("="*50)
        print("1. Manage Video Datasets")
        print("2. Manage Image Datasets")
        print("3. Launch Extractor & Sorter")
        print("\n0. Quit")
        
        choice = input("\nSelect an option: ").strip()
        if choice == '1': run_modality_menu('video')
        elif choice == '2': run_modality_menu('image')
        elif choice == '3':
            run_eop("--tui extractor"); 
            sys.exit(0)
        elif choice == '0': clear_screen(); break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='*')
    args = parser.parse_args()
    
    os.makedirs(get_target_dir('video'), exist_ok=True)
    os.makedirs(get_target_dir('image'), exist_ok=True)
    
    if args.datasets is not None:
        clean_datasets = [d.replace(',', '') for d in args.datasets]
        if len(clean_datasets) == 0:
            print(f"{YELLOW}No datasets provided after --datasets flag. Exiting.{RESET}")
        else:
            print(f"🔧 CLI Mode Activated. Routing downloads dynamically by modality...\n")
            process_downloads(clean_datasets, interactive=False)
            print(f"✅ {GREEN}CLI Execution Complete.{RESET}")
            input(f"\nPress Enter to exit...")
            clear_screen()
    else:
        run_tui()

if __name__ == "__main__":
    main()