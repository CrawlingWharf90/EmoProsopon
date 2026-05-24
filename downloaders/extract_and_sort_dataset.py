import os
import select
import sys
import time
import shutil
import zipfile
import tarfile
import subprocess
import argparse

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
DOWNLOAD_DIR = os.path.join(BASE_DIR, 'dataset')
UNPACK_DIR = os.path.join(BASE_DIR, 'unpkged_datasets')
SORTERS_DIR = os.path.join(BASE_DIR, 'sorters')
SORTED_DIR = os.path.join(BASE_DIR, 'sorted_datasets')

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_eop(args: str):
    if os.name == "nt":
        cmd = shutil.which("eop") or "eop.bat"
    else:
        cmd = shutil.which("eop") or "./eop"
    subprocess.run(f"{cmd} {args}", shell=True)

def print_progress(iteration, total, prefix='', length=30):
    #? Generates a dynamic terminal progress bar.
    if total == 0:
        return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: 
        print()

def wait_for_enter_or_timeout(timeout):
    #? Waits for the user to press Enter, or continues automatically after 'timeout' seconds.
    print(f"\n{CYAN}Press Enter to skip the wait, or auto-continuing in {timeout} seconds...{RESET}", end="", flush=True)
    
    if os.name == 'nt':
        import msvcrt
        start_time = time.time()
        while time.time() - start_time < timeout:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b'\r', b'\n'):
                    print()
                    return
            time.sleep(0.1)
        print()
    else:
        import select
        i, o, e = select.select([sys.stdin], [], [], timeout)
        if i:
            sys.stdin.readline()
        else:
            print()

#* ─────────────────────────────────────────────────────────────────
#* CORE LOGIC & STATE CHECKERS
#* ─────────────────────────────────────────────────────────────────
def get_downloaded_datasets():
    #? Returns a list of datasets that have a .zip or .tar.gz in the dataset/ folder.
    if not os.path.exists(DOWNLOAD_DIR):
        return []
    files = os.listdir(DOWNLOAD_DIR)
    datasets = []
    for f in files:
        if f.endswith('.zip'):
            datasets.append(f.replace('.zip', ''))
        elif f.endswith('.tar.gz'):
            datasets.append(f.replace('.tar.gz', ''))
    return sorted(list(set(datasets)))

def is_extracted(name):
    #? Checks if the dataset exists in the unpkged_datasets folder.
    return os.path.isdir(os.path.join(UNPACK_DIR, name))

def is_sorted(name):
    #? Checks for a hidden marker file indicating the sorter script has run.
    return os.path.isfile(os.path.join(UNPACK_DIR, name, '.sorted_marker'))

def perform_extraction(to_extract, interactive=True):
    os.makedirs(UNPACK_DIR, exist_ok=True)
    print("\n--- Starting Extraction ---")
    for name in to_extract:
        zip_path = os.path.join(DOWNLOAD_DIR, f"{name}.zip")
        tar_path = os.path.join(DOWNLOAD_DIR, f"{name}.tar.gz")
        extract_path = os.path.join(UNPACK_DIR, name)
        
        os.makedirs(extract_path, exist_ok=True)
        
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zf:
                members = zf.infolist()
                total = len(members)
                for i, member in enumerate(members):
                    zf.extract(member, extract_path)
                    print_progress(i + 1, total, prefix=f"Extracting {name}")
        elif os.path.exists(tar_path):
            with tarfile.open(tar_path, 'r:gz') as tar:
                members = tar.getmembers()
                total = len(members)
                for i, member in enumerate(members):
                    tar.extract(member, extract_path)
                    print_progress(i + 1, total, prefix=f"Extracting {name}")
        else:
            print(f"{RED}❌ Could not find compressed file for {name}.{RESET}")
        
    if interactive:
        input(f"\n{GREEN}Extraction Complete! Press Enter to continue...{RESET}")

def perform_sorting(to_sort, interactive=True):
    os.makedirs(SORTERS_DIR, exist_ok=True)
    os.makedirs(SORTED_DIR, exist_ok=True)
    print("\n--- Starting Sorting Process ---")
    for name in to_sort:
        sorter_script = os.path.join(SORTERS_DIR, f"{name.lower()}_sorter.py")
        
        print(f"\nExecuting {name} Sorter...")
        if os.path.exists(sorter_script):
            try:
                #* Call the specific sorter script!
                subprocess.run([sys.executable, sorter_script], check=True)
                
                #* Mark as sorted
                marker_path = os.path.join(UNPACK_DIR, name, '.sorted_marker')
                open(marker_path, 'a').close()
                print(f"{GREEN}✅ {name} sorting complete!{RESET}")
            except subprocess.CalledProcessError as e:
                print(f"{RED}❌ Error running {sorter_script}: {e}{RESET}")
        else:
            print(f"{RED}❌ Missing script: '{name.lower()}_sorter.py' not found in '{SORTERS_DIR}/'.{RESET}")
            print(f"{YELLOW}Create this script to handle the {name} logic.{RESET}")
            
    if interactive:
        input(f"\n{GREEN}DONE - Press Enter to continue...{RESET}")

#* ─────────────────────────────────────────────────────────────────
#* MENUS
#* ─────────────────────────────────────────────────────────────────
def menu_extract():
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(" EXTRACT DATASETS ".center(50, "="))
        print("="*50)
        
        datasets = get_downloaded_datasets()
        if not datasets:
            print(f"{YELLOW}No downloaded datasets found in '{DOWNLOAD_DIR}'.{RESET}")
            input("\nPress Enter to go back...")
            return

        for idx, name in enumerate(datasets):
            if is_extracted(name):
                print(f"{idx + 1}. {GREEN}{name} [Extracted]{RESET}")
            else:
                print(f"{idx + 1}. {YELLOW}{name} [Compressed]{RESET}")
        
        choice = input(f"\nEnter numbers separated by commas to EXTRACT (e.g., 1, 2) or {CYAN}0 to go back:{RESET} ").strip()
        if choice == '0':
            break
            
        to_extract = []
        for part in choice.split(','):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(datasets):
                    dataset_name = datasets[idx]
                    if not is_extracted(dataset_name):
                        to_extract.append(dataset_name)
                    else:
                        print(f"{GREEN}ℹ️ {dataset_name} is already extracted. Skipping.{RESET}")

        if to_extract:
            perform_extraction(to_extract, interactive=True)
        else:
            print(f"{YELLOW}No new datasets selected for extraction.{RESET}")
            input(f"\n{YELLOW}Press Enter to go back...{RESET}")

def menu_sort():
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(" SORT EXTRACTED DATASETS ".center(50, "="))
        print("="*50)
        
        if not os.path.exists(UNPACK_DIR):
            print(f"{YELLOW}No unpkged_datasets folder found. Please extract datasets first.{RESET}")
            input("\nPress Enter to go back...")
            return

        extracted_datasets = [d for d in os.listdir(UNPACK_DIR) if os.path.isdir(os.path.join(UNPACK_DIR, d))]
        
        if not extracted_datasets:
            print(f"{YELLOW}No extracted datasets available to sort.{RESET}")
            input("\nPress Enter to go back...")
            return

        for idx, name in enumerate(extracted_datasets):
            if is_sorted(name):
                print(f"{idx + 1}. {GREEN}{name} [Sorted]{RESET}")
            else:
                print(f"{idx + 1}. {YELLOW}{name} [{YELLOW}Not Sorted{RESET}]")
    
        
        choice = input(f"\nEnter numbers separated by commas to SORT (e.g., 1, 2) or {CYAN}0 to go back:{RESET} ").strip()
        if choice == '0':
            break
            
        to_sort = []
        for part in choice.split(','):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(extracted_datasets):
                    dataset_name = extracted_datasets[idx]
                    if not is_sorted(dataset_name):
                        to_sort.append(dataset_name)
                    else:
                        print(f"{GREEN}ℹ️ {dataset_name} is already sorted. Skipping.{RESET}")

        if to_sort:
            perform_sorting(to_sort, interactive=True)

def menu_manage():
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(" MANAGE EXTRACTED DATASETS ".center(50, "="))
        print("="*50)
        
        datasets = get_downloaded_datasets()
        if not datasets:
            print(f"{YELLOW}No downloaded datasets found.{RESET}")
            input("\nPress Enter to go back...")
            return

        for idx, name in enumerate(datasets):
            extracted = is_extracted(name)
            sorted_state = is_sorted(name)
            
            if extracted and sorted_state:
                print(f"{idx + 1}. {GREEN}{name} [Extracted | Sorted]{RESET}")
            elif extracted and not sorted_state:
                print(f"{idx + 1}. {YELLOW}{name} [Extracted | Not Sorted]{RESET}")
            else:
                print(f"{idx + 1}. {RED}{name} [Compressed | Not Extracted]{RESET}")
        
        choice = input(f"\nEnter numbers separated by commas to REMOVE EXTRACTED DATA (e.g., 1, 2) or {CYAN}0 to go back.{RESET}"
                       f"\n{RED}Type -1 to delete all extracted datasets.{RESET}"
                       f"\n> ").strip()
        if choice == '0':
            break
            
        to_remove = []
        warnings_triggered = False
        
        for part in choice.split(','):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(datasets):
                    dataset_name = datasets[idx]
                    if is_extracted(dataset_name):
                        to_remove.append(dataset_name)
                    else:
                        print(f"\n{RED}WARNING: '{dataset_name}' has not been extracted!{RESET}")
                        print(f"{YELLOW}This manager only removes extracted/unpacked folders.{RESET}")
                        print(f"{YELLOW}To delete original compressed files, please use download_dataset.py.{RESET}")
                        warnings_triggered = True
                        
        if warnings_triggered:
            wait_for_enter_or_timeout(10)

        if choice == '-1':
            clear_screen()
            print(f"{RED}!!! TOTAL DELETION CONFIRMATION !!!{RESET}")
            print(f"\n{RED}You are about to DELETE ALL EXTRACTED DATASET FOLDERS.{RESET}")
            print(f"{YELLOW}This action cannot be undone.{RESET}")
            
            confirm = input(f"\n{CYAN}Type 'yes' to confirm TOTAL DELETION, or anything else to cancel: {RESET}").strip().lower()
            if confirm == 'yes':
                to_remove = [d for d in datasets if is_extracted(d)]
            else:
                print(f"{YELLOW}Total deletion cancelled.{RESET}")
                input(f"\nPress Enter to continue...")
                continue
        else:
            for part in choice.split(','):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(datasets):
                        dataset_name = datasets[idx]
                        if is_extracted(dataset_name):
                            to_remove.append(dataset_name)
                        else:
                            print(f"\n{RED}WARNING: '{dataset_name}' has not been extracted!{RESET}")
                            print(f"{YELLOW}This manager only removes extracted/unpacked folders.{RESET}")
                            print(f"{YELLOW}To delete original compressed files, please use download_dataset.py.{RESET}")
                            warnings_triggered = True
                            
        if warnings_triggered:
            wait_for_enter_or_timeout(10)

        if to_remove and choice != '-1':
            clear_screen()
            print(f"{RED}!!! DELETION CONFIRMATION !!!{RESET}")
            print(f"\n{RED}You are about to delete the extracted folders for the following datasets:{RESET}")
            for name in to_remove:
                print(f" - {name}")
            print(f"\n{CYAN}Note: This will NOT delete the original compressed files in the dataset/ folder.{RESET}")
            
            confirm = input(f"\n{CYAN}Type 'yes' to confirm deletion, or anything else to cancel: {RESET}").strip().lower()
            if confirm != 'yes':
                to_remove = []
                print(f"{YELLOW}Deletion cancelled.{RESET}")
                input(f"\n{GREEN}Press Enter to continue...{RESET}")
                
        if to_remove:
            for name in to_remove:
                target = os.path.join(UNPACK_DIR, name)
                try:
                    shutil.rmtree(target)
                    print(f"{GREEN}✅ Removed extracted data for {name}.{RESET}")
                except Exception as e:
                    print(f"{RED}❌ Failed to remove {name}: {e}{RESET}")
            
            input(f"\n{GREEN}DONE - Press Enter to continue...{RESET}")

#* ─────────────────────────────────────────────────────────────────
#* MAIN LOGIC
#* ─────────────────────────────────────────────────────────────────
def run_tui():
    """Starts the Text User Interface loop."""
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(" EMO-PROSOPON DATASET SORTER ".center(50, "="))
        print("="*50)
        print("1. Extract datasets")
        print("2. Sort extracted")
        print("3. Manage datasets")
        print("4. Launch Dataset Manager")
        print("\n0. Quit")
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == '1':
            menu_extract()
        elif choice == '2':
            menu_sort()
        elif choice == '3':
            menu_manage()
        elif choice == '4':
            run_eop("--tui datasets")
            break
        elif choice == '0':
            clear_screen()
            print("Exiting Dataset Sorter...")
            break
        else:
            print(f"{RED}Invalid option. Please try again.{RESET}")
            time.sleep(1)

def main():
    #! Ensure necessary folders exist
    os.makedirs(UNPACK_DIR, exist_ok=True)
    os.makedirs(SORTERS_DIR, exist_ok=True)
    os.makedirs(SORTED_DIR, exist_ok=True)
    
    #? 1. Setup Argparse
    parser = argparse.ArgumentParser(description="Emo-Prosopon Dataset Extractor & Sorter")
    parser.add_argument('--extract', nargs='*', help="Datasets to extract directly (e.g. --extract CREMA-D)")
    parser.add_argument('--sort', nargs='*', help="Datasets to sort directly (e.g. --sort CREMA-D)")
    parser.add_argument('--syd', nargs='*', help="Extract AND sort datasets automatically")
    
    args = parser.parse_args()
    
    #? 2. Execution Routing
    if args.extract is not None or args.sort is not None or args.syd is not None:
        print(f"🔧 CLI Mode Activated.\n")
        
        if args.syd is not None:
            clean_syd = [d.replace(',', '') for d in args.syd]
            if len(clean_syd) == 0:
                print(f"{YELLOW}No datasets provided for --syd flag.{RESET}")
            else:
                print(f"{CYAN}🎵 'We've been here before, it's always the same...' - Syd Matters{RESET}")
                perform_extraction(clean_syd, interactive=False)
                perform_sorting(clean_syd, interactive=False)
                
        if args.extract is not None:
            clean_ext = [d.replace(',', '') for d in args.extract]
            if len(clean_ext) > 0:
                perform_extraction(clean_ext, interactive=False)
                
        if args.sort is not None:
            clean_sort = [d.replace(',', '') for d in args.sort]
            if len(clean_sort) > 0:
                perform_sorting(clean_sort, interactive=False)
                
        print(f"\n{GREEN}✅ CLI Execution Complete.{RESET}")
        input(f"\nPress Enter to exit...")
        clear_screen()
    else:
        #? TUI Mode
        run_tui()

if __name__ == "__main__":
    main()