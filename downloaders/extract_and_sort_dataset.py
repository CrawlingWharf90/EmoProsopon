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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_dl_dir(modality): return os.path.join(BASE_DIR, 'dataset', modality)
def get_up_dir(modality): return os.path.join(BASE_DIR, 'unpkged_datasets', modality)
def get_so_dir(modality): return os.path.join(BASE_DIR, 'sorted_datasets', modality)
SORTERS_DIR = os.path.join(BASE_DIR, 'sorters')

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_eop(args: str):
    cmd = shutil.which("eop") or ("eop.bat" if os.name == "nt" else "./eop")
    subprocess.run(f"{cmd} {args}", shell=True)

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] {percent}% Complete')
    sys.stdout.flush()
    if iteration == total: print()

def wait_for_enter_or_timeout(timeout):
    print(f"\n{CYAN}Press Enter to skip the wait, or auto-continuing in {timeout} seconds...{RESET}", end="", flush=True)
    if os.name == 'nt':
        import msvcrt
        start_time = time.time()
        while time.time() - start_time < timeout:
            if msvcrt.kbhit():
                if msvcrt.getch() in (b'\r', b'\n'):
                    print(); return
            time.sleep(0.1)
        print()
    else:
        i, o, e = select.select([sys.stdin], [], [], timeout)
        if i: sys.stdin.readline()
        else: print()

#* ─────────────────────────────────────────────────────────────────
#* CORE LOGIC & STATE CHECKERS
#* ─────────────────────────────────────────────────────────────────
def get_downloaded_datasets(modality):
    target_dir = get_dl_dir(modality)
    if not os.path.exists(target_dir): return []
    return sorted(list(set([f.replace('.zip', '').replace('.tar.gz', '') for f in os.listdir(target_dir) if f.endswith(('.zip', '.tar.gz'))])))

def is_extracted(name, modality):
    return os.path.isdir(os.path.join(get_up_dir(modality), name))

def is_sorted(name, modality):
    return os.path.isfile(os.path.join(get_up_dir(modality), name, '.sorted_marker'))

def perform_extraction(to_extract, modality, interactive=True):
    up_dir, dl_dir = get_up_dir(modality), get_dl_dir(modality)
    os.makedirs(up_dir, exist_ok=True)
    print(f"\n--- Starting {modality.capitalize()} Extraction ---")
    
    for name in to_extract:
        zip_path, tar_path = os.path.join(dl_dir, f"{name}.zip"), os.path.join(dl_dir, f"{name}.tar.gz")
        extract_path = os.path.join(up_dir, name)
        os.makedirs(extract_path, exist_ok=True)
        
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zf:
                members = zf.infolist()
                for i, member in enumerate(members):
                    zf.extract(member, extract_path)
                    print_progress(i + 1, len(members), prefix=f"Extracting {name}")
        elif os.path.exists(tar_path):
            with tarfile.open(tar_path, 'r:gz') as tar:
                members = tar.getmembers()
                for i, member in enumerate(members):
                    tar.extract(member, extract_path)
                    print_progress(i + 1, len(members), prefix=f"Extracting {name}")
        else:
            print(f"{RED}❌ Could not find compressed file for {name}.{RESET}")
            
    if interactive: input(f"\n{GREEN}Extraction Complete! Press Enter to continue...{RESET}")

def perform_sorting(to_sort, modality, interactive=True):
    os.makedirs(SORTERS_DIR, exist_ok=True)
    os.makedirs(get_so_dir(modality), exist_ok=True)
    print(f"\n--- Starting {modality.capitalize()} Sorting Process ---")
    
    for name in to_sort:
        sorter_script = os.path.join(SORTERS_DIR, f"{name.lower()}_sorter.py")
        print(f"\nExecuting {name} Sorter...")
        if os.path.exists(sorter_script):
            try:
                # Pass the modality to the sorter script so it knows where to output
                subprocess.run([sys.executable, sorter_script, f"--{modality}"], check=True)
                open(os.path.join(get_up_dir(modality), name, '.sorted_marker'), 'a').close()
                print(f"{GREEN}✅ {name} sorting complete!{RESET}")
            except subprocess.CalledProcessError as e:
                print(f"{RED}❌ Error running {sorter_script}: {e}{RESET}")
        else:
            print(f"{RED}❌ Missing script: '{name.lower()}_sorter.py' in '{SORTERS_DIR}/'.{RESET}")
            
    if interactive: input(f"\n{GREEN}DONE - Press Enter to continue...{RESET}")

#* ─────────────────────────────────────────────────────────────────
#* SUB-MENUS (Modality Aware)
#* ─────────────────────────────────────────────────────────────────
def menu_extract(modality):
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(f" EXTRACT {modality.upper()} DATASETS ".center(50, "="))
        print("="*50)
        datasets = get_downloaded_datasets(modality)
        if not datasets:
            print(f"{YELLOW}No downloaded {modality} datasets found in '{get_dl_dir(modality)}'.{RESET}")
            input("\nPress Enter to go back..."); return

        for idx, name in enumerate(datasets):
            status = f"{GREEN}[Extracted]{RESET}" if is_extracted(name, modality) else f"{YELLOW}[Compressed]{RESET}"
            print(f"{idx + 1}. {name} {status}")
        
        choice = input(f"\nEnter numbers to EXTRACT (e.g., 1, 2) or {CYAN}0 to go back:{RESET} ").strip()
        if choice == '0': break
            
        to_extract = [datasets[int(p)-1] for p in choice.split(',') if p.strip().isdigit() and 0 <= int(p)-1 < len(datasets) and not is_extracted(datasets[int(p)-1], modality)]
        if to_extract: perform_extraction(to_extract, modality)

def menu_sort(modality):
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(f" SORT {modality.upper()} DATASETS ".center(50, "="))
        print("="*50)
        up_dir = get_up_dir(modality)
        
        if not os.path.exists(up_dir):
            print(f"{YELLOW}No unpacked {modality} folders found.{RESET}"); input("\nPress Enter to go back..."); return
            
        extracted = [d for d in os.listdir(up_dir) if os.path.isdir(os.path.join(up_dir, d))]
        if not extracted:
            print(f"{YELLOW}No extracted datasets available to sort.{RESET}"); input("\nPress Enter to go back..."); return

        for idx, name in enumerate(extracted):
            status = f"{GREEN}[Sorted]{RESET}" if is_sorted(name, modality) else f"{YELLOW}[Not Sorted]{RESET}"
            print(f"{idx + 1}. {name} {status}")
    
        choice = input(f"\nEnter numbers to SORT (e.g., 1, 2) or {CYAN}0 to go back:{RESET} ").strip()
        if choice == '0': break
            
        to_sort = [extracted[int(p)-1] for p in choice.split(',') if p.strip().isdigit() and 0 <= int(p)-1 < len(extracted) and not is_sorted(extracted[int(p)-1], modality)]
        if to_sort: perform_sorting(to_sort, modality)

def menu_manage(modality):
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(f" MANAGE {modality.upper()} DATASETS ".center(50, "="))
        print("="*50)
        datasets = get_downloaded_datasets(modality)
        if not datasets:
            print(f"{YELLOW}No downloaded datasets found.{RESET}"); input("\nPress Enter to go back..."); return

        for idx, name in enumerate(datasets):
            if is_extracted(name, modality) and is_sorted(name, modality): print(f"{idx + 1}. {GREEN}{name} [Extracted | Sorted]{RESET}")
            elif is_extracted(name, modality): print(f"{idx + 1}. {YELLOW}{name} [Extracted | Not Sorted]{RESET}")
            else: print(f"{idx + 1}. {RED}{name} [Compressed | Not Extracted]{RESET}")
        
        choice = input(f"\nEnter numbers to REMOVE EXTRACTED FOLDERS or {CYAN}0 to go back.{RESET}\n> ").strip()
        if choice == '0': break
            
        to_remove = [datasets[int(p)-1] for p in choice.split(',') if p.strip().isdigit() and 0 <= int(p)-1 < len(datasets) and is_extracted(datasets[int(p)-1], modality)]
        if to_remove:
            for name in to_remove:
                shutil.rmtree(os.path.join(get_up_dir(modality), name), ignore_errors=True)
                print(f"{GREEN}✅ Removed extracted data for {name}.{RESET}")
            input(f"\n{GREEN}DONE - Press Enter to continue...{RESET}")

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
        print("1. Extract datasets")
        print("2. Sort extracted")
        print("3. Manage datasets")
        print(f"4. Switch to {other_modality.capitalize()} Datasets")
        print("5. Launch Dataset Manager")
        print("\n0. Go Back")
        
        choice = input("\nSelect an option: ").strip()
        if choice == '1': menu_extract(modality)
        elif choice == '2': menu_sort(modality)
        elif choice == '3': menu_manage(modality)
        elif choice == '4': return run_modality_menu(other_modality)
        elif choice == '5': 
            run_eop("--tui datasets"); 
            break
        elif choice == '0': break

def run_tui():
    while True:
        clear_screen()
        print("\n" + "="*50)
        print(" EMO-PROSOPON DATASET ORCHESTRATOR ".center(50, "="))
        print("="*50)
        print("1. Manage Video Datasets")
        print("2. Manage Image Datasets")
        print("3. Launch Dataset Downloader")
        print("\n0. Quit")
        
        choice = input("\nSelect an option: ").strip()
        if choice == '1': run_modality_menu('video')
        elif choice == '2': run_modality_menu('image')
        elif choice == '3': 
            run_eop("--tui datasets");
            break
        elif choice == '0': clear_screen(); break

def main():
    for base in [get_dl_dir, get_up_dir, get_so_dir]:
        os.makedirs(base('video'), exist_ok=True)
        os.makedirs(base('image'), exist_ok=True)
    os.makedirs(SORTERS_DIR, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Dataset Extractor & Sorter")
    parser.add_argument('--extract', nargs='*')
    parser.add_argument('--sort', nargs='*')
    args = parser.parse_args()
    
    if args.extract or args.sort:
        print(f"{YELLOW}CLI direct extraction is deprecated pending dual-modality flags. Please use the TUI.{RESET}")
    else:
        run_tui()

if __name__ == "__main__":
    main()