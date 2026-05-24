import subprocess
import os
import sys
import ssl
import urllib.request
import shutil
import argparse

#* ─────────────────────────────────────────────────────────────────
#* GLOBALLY BYPASS SSL VERIFICATION
#* ─────────────────────────────────────────────────────────────────
#! Academic servers frequently have expired SSL certificates. 
#! This forces Python to download the datasets anyway without crashing.
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
TARGET_DIR = os.path.join(BASE_DIR, 'dataset')

def clear_screen():
    """Clears the terminal screen for Windows, Mac, and Linux"""
    os.system('cls' if os.name == 'nt' else 'clear')

#* ─────────────────────────────────────────────────────────────────
#* EXPANDED DATASET CONFIGURATION
#* ─────────────────────────────────────────────────────────────────
DATASETS = {
    #! DEFAULTS
    "CK+": {"public": False, "default": True, "auth_url": "http://www.pitt.edu/~emotion/ck-spread.htm", "desc": "Extended Cohn-Kanade"},
    "RAVDESS": {"public": True, "default": True, "url": "https://zenodo.org/record/1188976/files/Video_Speech_Actor_01.zip", "desc": "Ryerson Audio-Visual Database"},
    "AFEW": {"public": False, "default": True, "auth_url": "https://cs.anu.edu.au/few/", "desc": "Acted Facial Expressions in the Wild"},
    
    #! PUBLIC (No forms required)
    "CREMA-D": {"public": True, "default": False, "url": "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip", "desc": "Crowd-sourced Emotional Mutimodal"},
    "eNTERFACE05": {"public": True, "default": False, "url": "https://enterface.net/enterface05/docs/results/databases/project2_database.zip", "desc": "Audio-Visual Emotion Database"},
    "MELD": {"public": True, "default": False, "url": "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz", "desc": "Multimodal EmotionLines Dataset (Friends TV Show)"},
    
    #! RESTRICTED (Require EULA / Academic Forms)
    "SAVEE": {"public": False, "default": False, "auth_url": "http://kahlan.eps.surrey.ac.uk/savee/Download.html", "desc": "Surrey Audio-Visual Expressed Emotion"}, 
    "IEMOCAP": {"public": False, "default": False, "auth_url": "https://sail.usc.edu/iemocap/", "desc": "Interactive Emotional Dyadic Motion Capture"},
}

#* ─────────────────────────────────────────────────────────────────
#* CORE FUNCTIONS
#* ─────────────────────────────────────────────────────────────────
def run_eop(args: str):
    if os.name == "nt":
        cmd = shutil.which("eop") or "eop.bat"
    else:
        cmd = shutil.which("eop") or "./eop"
    subprocess.run(f"{cmd} {args}", shell=True)

def is_installed(name):
    zip_path = os.path.join(TARGET_DIR, f"{name}.zip")
    tar_path = os.path.join(TARGET_DIR, f"{name}.tar.gz")
    folder_path = os.path.join(TARGET_DIR, name)
    return os.path.exists(zip_path) or os.path.exists(tar_path) or os.path.exists(folder_path)

def progress_bar(block_num, block_size, total_size):
    """Callback function for urlretrieve to display a progress bar."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded / total_size
        percent = min(1.0, percent) # Prevent it from going over 100%
        
        bar_length = 30
        filled_length = int(bar_length * percent)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        dl_mb = downloaded / (1024 * 1024)
        tot_mb = total_size / (1024 * 1024)
        
        #* \r overwrites the current line
        sys.stdout.write(f'\r    ↳ [{bar}] {percent:.1%} ({dl_mb:.1f} MB / {tot_mb:.1f} MB)')
        sys.stdout.flush()
    else:
        #! Fallback if the server doesn't send a Content-Length header
        dl_mb = downloaded / (1024 * 1024)
        sys.stdout.write(f'\r    ↳ Downloading... ({dl_mb:.1f} MB)')
        sys.stdout.flush()

def download_file(url, dest_path):
    print(f"Downloading from: {url}")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_bar)
        print(f"\n{GREEN}✅ Successfully saved to: {dest_path}{RESET}\n")
    except Exception as e:
        print(f"\n{RED}❌ Failed to download. Error: {e}{RESET}\n")

def process_downloads(datasets_to_download, interactive=True):
    os.makedirs(TARGET_DIR, exist_ok=True)
    for item in datasets_to_download:
        if item not in DATASETS:
            print(f"{YELLOW}⚠️ Dataset '{item}' not found in registry. Skipping.{RESET}")
            continue
            
        if is_installed(item):
            print(f"{GREEN}✅ {item} is already installed. Skipping.{RESET}")
            continue

        info = DATASETS[item]
        print(f"\n--- Fetching {item} ---")
        
        if info["public"]:
            url = info["url"]
        else:
            auth_url = info.get("auth_url", "Please search online for official access.")
            print(f"{YELLOW}ℹ️ {item} requires academic approval.{RESET}")
            print(f"Apply here: {CYAN}{auth_url}{RESET}")
            
            if interactive:
                url = input(f"🔒 Paste your approved direct download link (or press Enter to skip): ").strip()
                if not url:
                    print(f"{YELLOW}⏭️ Skipping {item}...{RESET}")
                    continue
            else:
                print(f"{RED}⏭️ Skipping {item}. (CLI mode ignores restricted datasets to prevent pausing){RESET}")
                continue
                
        #! tar.gz vs zip extensions
        if "tar.gz" in url:
            extension = "tar.gz"
        else:
            extension = url.split('.')[-1]
            if len(extension) > 4 or "/" in extension: 
                extension = "zip" 
            
        dest_path = os.path.join(TARGET_DIR, f"{item}.{extension}")
        download_file(url, dest_path)
    
    if interactive:
        input("\nPress Enter to continue...")

#* ─────────────────────────────────────────────────────────────────
#* MENU RENDERING
#* ─────────────────────────────────────────────────────────────────
def menu_download_defaults():
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(" DEFAULT DATASETS ".center(50, "="))
        print("="*50)
        
        defaults = [k for k, v in DATASETS.items() if v["default"]]
        missing = []
        
        for name in defaults:
            installed = is_installed(name)
            if installed:
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
                print(f"{GREEN}All default datasets are already installed!{RESET}")
                input("\nPress Enter to continue...")
            else:
                process_downloads(missing, interactive=True)
        elif choice == '0':
            break

def format_col_item(name, category):
    installed = is_installed(name)
    info = DATASETS[name]
    
    if category == "default":
        restriction = "Pub" if info["public"] else "Res"
        if installed:
            return f"{GREEN}{name} [{restriction} | Installed]{RESET}"
        else:
            return f"{CYAN}{name} [{restriction} | Missing]{RESET}"
    else:
        if installed:
            return f"{GREEN}{name} [Installed]{RESET}"
        else:
            if info["public"]:
                return f"{YELLOW}{name} [Missing]{RESET}"
            else:
                return f"{RED}{name} [Missing]{RESET}"

def menu_browse():
    while True:
        clear_screen()
        print("\n" + "="*85)
        print(" BROWSE DATASETS ".center(85, "="))
        print("="*85)
        
        defaults = [k for k, v in DATASETS.items() if v["default"]]
        publics = [k for k, v in DATASETS.items() if not v["default"] and v["public"]]
        restricteds = [k for k, v in DATASETS.items() if not v["default"] and not v["public"]]
        
        print(f"{'DEFAULTS':<28} | {'PUBLIC':<28} | {'RESTRICTED':<28}")
        print("-" * 85)
        
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

            print(f"{pad_ansi(str1, 28)} | {pad_ansi(str2, 28)} | {str3}")
            
        print("\nOptions:")
        print("1. Install Defaults")
        print("2. Select datasets to install")
        print("0. Go Back")
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == '1':
            menu_download_defaults()
        elif choice == '2':
            menu_select_to_install()
        elif choice == '0':
            break

def menu_select_to_install():
    clear_screen()
    all_datasets = list(DATASETS.keys())
    
    print("\n" + "="*50)
    print(" SELECT DATASETS TO INSTALL ".center(50, "="))
    print("="*50)

    for idx, name in enumerate(all_datasets):
        info = DATASETS[name]
        restriction = "Pub" if info["public"] else "Res"
        
        if is_installed(name):
            color = GREEN
            status = "Installed"
        else:
            status = "Missing"
            color = YELLOW if info["public"] else RED
            
        print(f"{idx + 1}. {color}{name} [{restriction} | {status}]{RESET}")
    
    choice = input(f"\nEnter numbers separated by commas (e.g., 1, 4, 5) or {CYAN}0 to go back:{RESET} ").strip()
    
    if choice == '0':
        return
        
    to_download = []
    for part in choice.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(all_datasets):
                to_download.append(all_datasets[idx])
                
    if to_download:
        process_downloads(to_download, interactive=True)

def menu_status():
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(" DATASET STATUS & MANAGER ".center(50, "="))
        print("="*50)
        
        installed_datasets = [name for name in DATASETS.keys() if is_installed(name)]
                
        if not installed_datasets:
            print(f"{YELLOW}No datasets are currently installed in the '{TARGET_DIR}' folder.{RESET}")
            input("\nPress Enter to return to Main Menu...")
            break
            
        for idx, name in enumerate(installed_datasets):
            info = DATASETS[name]
            restriction = "Pub" if info["public"] else "Res"
            print(f"{GREEN}{idx + 1}. {name} [{restriction}]{RESET}")
            
        print(f"\nTotal Installed: {len(installed_datasets)}/{len(DATASETS)}")
        
        choice = input(f"Enter numbers separated by commas to UNINSTALL (e.g., 1, 2) or {CYAN}0 to go back.{RESET}"
                       f"\n{RED}Type -1 to delete all installed datasets.{RESET}"
                       f"\n> ").strip()
        
        if choice == '0':
            break
        
        to_uninstall = []

        if choice == '-1':
            clear_screen()
            print(f"{RED}!!! TOTAL DELETION CONFIRMATION !!!{RESET}")
            print(f"\n{RED}You are about to UNINSTALL EVERY SINGLE DATASET.{RESET}")
            print(f"{YELLOW}This action cannot be undone and will delete gigabytes of data.{RESET}")
            
            confirm = input(f"\n{CYAN}Type 'yes' to confirm TOTAL DELETION, or anything else to cancel: {RESET}").strip().lower()
            if confirm == 'yes':
                to_uninstall = installed_datasets
            else:
                print(f"{YELLOW}Total deletion cancelled.{RESET}")
                input(f"\nPress Enter to continue...")
                continue
        else:
            for part in choice.split(','):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(installed_datasets):
                        to_uninstall.append(installed_datasets[idx])
                    
        if to_uninstall:
            print("\n--- Uninstalling Datasets ---")
            for item in to_uninstall:
                zip_path = os.path.join(TARGET_DIR, f"{item}.zip")
                tar_path = os.path.join(TARGET_DIR, f"{item}.tar.gz")
                folder_path = os.path.join(TARGET_DIR, item)
                
                try:
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                    if os.path.exists(tar_path):
                        os.remove(tar_path)
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)
                    print(f"{GREEN}✅ {item} uninstalled successfully.{RESET}")
                except Exception as e:
                    print(f"{RED}❌ Failed to uninstall {item}. Error: {e}{RESET}")
            
            input("\nPress Enter to continue...")
        else:
            print(f"{YELLOW}No valid selections made for uninstallation.{RESET}")
            input("\nPress Enter to continue...")

#* ─────────────────────────────────────────────────────────────────
#* MAIN LOGIC
#* ─────────────────────────────────────────────────────────────────
def run_tui():
    """Starts the Text User Interface loop."""
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(" EMO-PROSOPOPON DATASET MANAGER ".center(50, "="))
        print("="*50)
        print("1. Download default datasets")
        print("2. Browse datasets to install")
        print("3. Dataset status / Uninstall")
        print("4. Launch Extractor & Sorter")
        print("\n0. Quit")
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == '1':
            menu_download_defaults()
        elif choice == '2':
            menu_browse()
        elif choice == '3':
            menu_status()
        elif choice == '4':
            clear_screen()
            run_eop("--tui extractor")
            break
        elif choice == '0':
            clear_screen()
            print("Exiting Dataset Manager...")
            break
        else:
            print(f"{RED}Invalid option. Please try again.{RESET}")
            input("\nPress Enter to continue...")

def main():
    #? 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Emo-Prosopon Dataset Manager")
    parser.add_argument('--datasets', nargs='*', help="Bypass TUI and install specific datasets directly (e.g. --datasets CREMA-D MELD)")
    
    args = parser.parse_args()
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    #? 2. Execution Routing
    if args.datasets is not None:
        #! CLI Mode: Bypass the menu, install, and exit.
        clean_datasets = [d.replace(',', '') for d in args.datasets]
        if len(clean_datasets) == 0:
            print(f"{YELLOW}No datasets provided after --datasets flag. Exiting.{RESET}")
        else:
            print(f"🔧 CLI Mode Activated. Routing downloads to '{TARGET_DIR}' folder.\n")
            process_downloads(clean_datasets, interactive=False)
            print(f"✅ {GREEN}CLI Execution Complete.{RESET}")
            input(f"\nPress Enter to exit...")
            clear_screen()
    else:
        #? TUI Mode: Start the interactive menu.
        run_tui()

if __name__ == "__main__":
    main()