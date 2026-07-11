import os
import json
import importlib.util
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
MAGENTA = '\033[95m'
RESET = '\033[0m'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_dl_dir(modality): return os.path.join(BASE_DIR, 'dataset', modality)
def get_up_dir(modality): return os.path.join(BASE_DIR, 'unpkged_datasets', modality)
def get_so_dir(modality): return os.path.join(BASE_DIR, 'sorted_datasets', modality)
SORTERS_DIR = os.path.join(BASE_DIR, 'sorters')

# Emotion subfolders that live inside sorted_datasets/<modality>/
EMOTION_FOLDERS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Once we've asked a sorter script for its prefix, cache it in-memory for
# the rest of this run - no need to re-import the module every time.
_PREFIX_CACHE = {}

def _get_dataset_prefix(name):
    """
    Every sorter script now exposes a get_prefix() function that returns the
    exact string it prepends to sorted filenames. Rather than guessing or
    normalizing the dataset name, we import the sorter module directly and
    ask it - so this always matches reality, no matter how a given sorter
    chooses to name its files.
    """
    if name in _PREFIX_CACHE:
        return _PREFIX_CACHE[name]

    sorter_path = os.path.join(SORTERS_DIR, f"{name.lower()}_sorter.py")
    if not os.path.exists(sorter_path):
        return None

    try:
        spec = importlib.util.spec_from_file_location(f"{name.lower()}_sorter_module", sorter_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prefix = module.get_prefix().lower()
    except Exception as e:
        print(f"{RED}❌ Could not read prefix from {sorter_path}: {e}{RESET}")
        return None

    _PREFIX_CACHE[name] = prefix
    return prefix

# Once we've confirmed a dataset has sorted files on disk, we cache that fact
# here so we never have to re-scan a (potentially million-file) folder again.
SORT_CACHE_FILE = os.path.join(BASE_DIR, '.sort_status_cache.json')

def _load_sort_cache():
    if os.path.isfile(SORT_CACHE_FILE):
        try:
            with open(SORT_CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

def _save_sort_cache(cache):
    try:
        with open(SORT_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass

_SORT_CACHE = _load_sort_cache()

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

def extraction_status(name, modality):
    target_dir = os.path.join(get_up_dir(modality), name)
    if not os.path.isdir(target_dir):
        return "compressed"
    
    # If the directory exists but is completely empty, it was a failed extraction
    if not os.listdir(target_dir):
        return "error"
        
    return "extracted"

def _scan_for_sorted_files(name, modality):
    """
    Checks whether sorted_datasets/<modality>/<EmotionFolder>/ contains at
    least one file prefixed with this dataset's real, sorter-reported prefix.

    This uses os.scandir() rather than os.listdir(): scandir is a thin
    wrapper around the OS's own directory iterator, so it yields entries
    one at a time instead of first materializing the full (potentially
    million-entry) folder into a Python list. Combined with an early
    "return True" the instant a match is found, this is the fastest way
    to answer an existence question without building a separate index -
    we only ever pay for as much of the folder as we actually need to read.
    """
    so_dir = get_so_dir(modality)
    if not os.path.isdir(so_dir):
        return False

    prefix = _get_dataset_prefix(name)
    if not prefix:
        return False
    prefix = f"{prefix}_"

    for label in EMOTION_FOLDERS:
        label_dir = os.path.join(so_dir, label)
        if not os.path.isdir(label_dir):
            continue
        try:
            with os.scandir(label_dir) as it:
                for entry in it:
                    if entry.name.lower().startswith(prefix) and entry.is_file(follow_symlinks=False):
                        return True
        except OSError:
            continue

    return False

def is_sorted(name, modality):
    # Once something is known to be sorted, trust the cache forever - it
    # can only ever go from "not sorted" to "sorted", never the reverse.
    if _SORT_CACHE.get(modality, {}).get(name):
        return True

    found = _scan_for_sorted_files(name, modality)
    if found:
        _SORT_CACHE.setdefault(modality, {})[name] = True
        _save_sort_cache(_SORT_CACHE)
    return found

def perform_extraction(to_extract, modality, interactive=True):
    up_dir, dl_dir = get_up_dir(modality), get_dl_dir(modality)
    os.makedirs(up_dir, exist_ok=True)
    print(f"\n--- Starting {modality.capitalize()} Extraction ---")
    
    for name in to_extract:
        zip_path, tar_path = os.path.join(dl_dir, f"{name}.zip"), os.path.join(dl_dir, f"{name}.tar.gz")
        extract_path = os.path.join(up_dir, name)
        os.makedirs(extract_path, exist_ok=True)
        
        try:
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    members = zf.infolist()
                    for i, member in enumerate(members):
                        filename = member.filename
                        parts = filename.split('/')
                        
                        if len(parts) > 1 and parts[0].lower() in [name.lower(), f"train_{name.lower()}", "train", f"val_{name.lower()}"]:
                            member.filename = os.path.join(*parts[1:])
                        
                        if member.filename.strip() and not member.filename.endswith(('/', '\\')):
                            zf.extract(member, extract_path)
                            
                        print_progress(i + 1, len(members), prefix=f"Extracting {name}")
                        
            elif os.path.exists(tar_path):
                with tarfile.open(tar_path, 'r:gz') as tar:
                    members = tar.getmembers()
                    for i, member in enumerate(members):
                        parts = member.name.split('/')
                        if len(parts) > 1 and parts[0].lower() in [name.lower(), f"train_{name.lower()}", "train"]:
                            member.name = os.path.join(*parts[1:])
                            
                        tar.extract(member, extract_path)
                        print_progress(i + 1, len(members), prefix=f"Extracting {name}")
            else:
                print(f"{RED}❌ Could not find compressed file for {name}.{RESET}")
                
        except zipfile.BadZipFile:
            print(f"\n{RED}❌ ERROR: '{name}.zip' is corrupted or not a real zip file. Skipping.{RESET}")
            print(f"{YELLOW}   -> Please check the file size in 'dataset/{modality}/'. If it's very small, re-download it manually.{RESET}")
        except tarfile.ReadError:
            print(f"\n{RED}❌ ERROR: '{name}.tar.gz' is corrupted or unreadable. Skipping.{RESET}")
        except Exception as e:
            print(f"\n{RED}❌ Unexpected error extracting {name}: {e}{RESET}")
            
    if interactive: input(f"\n{GREEN}Extraction Process Complete! Press Enter to continue...{RESET}")

def perform_sorting(to_sort, modality, interactive=True):
    os.makedirs(SORTERS_DIR, exist_ok=True)
    os.makedirs(get_so_dir(modality), exist_ok=True)
    print(f"\n--- Starting {modality.capitalize()} Sorting Process ---")
    
    for name in to_sort:
        sorter_script = os.path.join(SORTERS_DIR, f"{name.lower()}_sorter.py")
        print(f"\nExecuting {name} Sorter...")
        if os.path.exists(sorter_script):
            try:
                subprocess.run([sys.executable, sorter_script, f"--{modality}"], check=True)
                _SORT_CACHE.setdefault(modality, {})[name] = True
                _save_sort_cache(_SORT_CACHE)
                print(f"{GREEN}✅ {name} sorting complete!{RESET}")
            except subprocess.CalledProcessError as e:
                print(f"{RED}❌ Error running {sorter_script}: {e}{RESET}")
        else:
            print(f"{RED}❌ Missing script: '{name.lower()}_sorter.py' in '{SORTERS_DIR}/'.{RESET}")
            
    if interactive: input(f"\n{GREEN}DONE - Press Enter to continue...{RESET}")

#* ─────────────────────────────────────────────────────────────────
#* STARTUP - PRELOAD SORT STATUS (with loading screen)
#* ─────────────────────────────────────────────────────────────────
def _print_loading_bar(iteration, total, length=44):
    filled = length if total == 0 else int(length * iteration // total)
    bar = '█' * filled + '-' * (length - filled)
    percent = 100 if total == 0 else int(100 * iteration / total)
    sys.stdout.write(f'\r[{bar}] {percent}%')
    sys.stdout.flush()

def preload_sort_status():
    """
    Runs once at startup, before the TUI is shown. Figures out (and caches)
    which extracted datasets are already sorted, for both modalities.

    This can be slow the first time it hits a dataset with a huge amount of
    files, so we show a loading screen while it works. Datasets already
    known to be sorted (from a previous run's cache) are skipped instantly.
    """
    targets = []
    for modality in ('video', 'image'):
        up_dir = get_up_dir(modality)
        if not os.path.isdir(up_dir):
            continue
        for name in os.listdir(up_dir):
            if os.path.isdir(os.path.join(up_dir, name)) and os.listdir(os.path.join(up_dir, name)):
                targets.append((name, modality))

    if not targets:
        return

    clear_screen()
    print(f"\n{CYAN}Getting Everything Ready For You{RESET}\n")
    _print_loading_bar(0, len(targets))

    for i, (name, modality) in enumerate(targets):
        is_sorted(name, modality)  # cache-aware: instant if already known
        _print_loading_bar(i + 1, len(targets))

    print()

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
            st = extraction_status(name, modality)
            if st == "extracted":
                status_str = f"{GREEN}[Extracted]{RESET}"
            elif st == "error":
                status_str = f"{MAGENTA}[Extraction Error]{RESET}"
            else:
                status_str = f"{YELLOW}[Compressed]{RESET}"
                
            print(f"{idx + 1}. {name} {status_str}")
        
        choice = input(f"\nEnter numbers to EXTRACT (e.g., 1, 2), use {CYAN}-1{RESET} to EXTRACT ALL or {CYAN}0{RESET} to go back:{RESET}\n> ").strip()
        if choice == '0': break
        
        to_extract = []
        if choice == '-1':
            # Select everything that isn't successfully extracted yet (so it retries errors and extracts compressed)
            to_extract = [name for name in datasets if extraction_status(name, modality) != "extracted"]
        else:
            to_extract = [datasets[int(p)-1] for p in choice.split(',') if p.strip().isdigit() and 0 <= int(p)-1 < len(datasets) and extraction_status(datasets[int(p)-1], modality) != "extracted"]
            
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
            
        # Only show folders that actually have contents inside them (filters out empty "Extraction Error" folders)
        extracted = [d for d in os.listdir(up_dir) if os.path.isdir(os.path.join(up_dir, d)) and os.listdir(os.path.join(up_dir, d))]
        
        if not extracted:
            print(f"{YELLOW}No fully extracted datasets available to sort.{RESET}"); input("\nPress Enter to go back..."); return

        for idx, name in enumerate(extracted):
            status = f"{GREEN}[Sorted]{RESET}" if is_sorted(name, modality) else f"{YELLOW}[Not Sorted]{RESET}"
            print(f"{idx + 1}. {name} {status}")
    
        choice = input(f"\nEnter numbers to SORT (e.g., 1, 2), use {CYAN}-1{RESET} to SORT ALL or {CYAN}0{RESET} to go back:{RESET}\n> ").strip()
        if choice == '0': break
            
        to_sort = []
        if choice == '-1':
            to_sort = [name for name in extracted if not is_sorted(name, modality)]
        else:
            to_sort = [extracted[int(p)-1] for p in choice.split(',') if p.strip().isdigit() and 0 <= int(p)-1 < len(extracted) and not is_sorted(extracted[int(p)-1], modality)]
            
        if to_sort: perform_sorting(to_sort, modality)

def remove_sorted_files(name, modality):
    """
    Deletes every file prefixed with this dataset's real, sorter-reported
    prefix from each emotion subfolder in sorted_datasets/<modality>/, and
    clears the dataset's cached "sorted" status so it doesn't linger as a
    stale True.
    """
    so_dir = get_so_dir(modality)
    prefix = _get_dataset_prefix(name)
    removed = 0

    if prefix and os.path.isdir(so_dir):
        prefix = f"{prefix}_"
        for label in EMOTION_FOLDERS:
            label_dir = os.path.join(so_dir, label)
            if not os.path.isdir(label_dir):
                continue
            try:
                with os.scandir(label_dir) as it:
                    for entry in it:
                        if entry.name.lower().startswith(prefix) and entry.is_file(follow_symlinks=False):
                            try:
                                os.remove(entry.path)
                                removed += 1
                            except OSError as e:
                                print(f"{RED}❌ Could not remove {entry.path}: {e}{RESET}")
            except OSError:
                continue

    if modality in _SORT_CACHE and name in _SORT_CACHE[modality]:
        del _SORT_CACHE[modality][name]
        _save_sort_cache(_SORT_CACHE)

    return removed

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
            st = extraction_status(name, modality)
            if st == "extracted" and is_sorted(name, modality): 
                print(f"{idx + 1}. {GREEN}{name} [Extracted | Sorted]{RESET}")
            elif st == "extracted": 
                print(f"{idx + 1}. {YELLOW}{name} [Extracted | Not Sorted]{RESET}")
            elif st == "error":
                print(f"{idx + 1}. {MAGENTA}{name} [Extraction Error]{RESET}")
            else: 
                print(f"{idx + 1}. {RED}{name} [Compressed | Not Extracted]{RESET}")
        
        choice = input(f"\nEnter numbers to REMOVE EXTRACTED/ERROR FOLDERS or {CYAN}0 to go back.{RESET}\n> ").strip()
        if choice == '0': break
            
        to_remove = [datasets[int(p)-1] for p in choice.split(',') if p.strip().isdigit() and 0 <= int(p)-1 < len(datasets) and extraction_status(datasets[int(p)-1], modality) in ["extracted", "error"]]
        if to_remove:
            for name in to_remove:
                shutil.rmtree(os.path.join(get_up_dir(modality), name), ignore_errors=True)
                removed_count = remove_sorted_files(name, modality)
                if removed_count:
                    print(f"{GREEN}✅ Removed unpacked data for {name} and {removed_count} sorted file(s).{RESET}")
                else:
                    print(f"{GREEN}✅ Removed unpacked data for {name}.{RESET}")
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
            sys.exit(0)
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
            sys.exit(0)
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
        preload_sort_status()
        run_tui()

if __name__ == "__main__":
    main()