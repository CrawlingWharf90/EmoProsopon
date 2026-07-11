import os
import sys
import time

#? Dynamically find the project root and add it to sys.path before local imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from PIL import Image

from emoprosopon.static_cnn import StaticFeatureExtractor, get_face_transform

BATCH_SIZE = 256 #! Reduce batch size to 128 if needed to prevent OOM errors on GPUs with limited memory.
PROC_KINEMATIC_DIR = os.path.join(BASE_DIR, 'processed_data', 'kinematic')
PROC_STATIC_CROP_DIR = os.path.join(BASE_DIR, "processed_data", "static_crops")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
YUNET_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'face_detection_yunet_2023mar.onnx')

GREEN = '\033[32m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BRIGHT_GREEN = '\033[92m'
RESET = '\033[0m'

CLASS_NAMES = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust"]
EMOTION_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def print_progress(iteration, total, prefix='', length=30, start_time=None):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    eta_str = ""
    if start_time is not None:
        elapsed = time.time() - start_time
        if iteration > 0 and iteration < total:
            avg_time_per_iter = elapsed / iteration
            rem_iters = total - iteration
            eta_seconds = int(rem_iters * avg_time_per_iter)
            mins, secs = divmod(eta_seconds, 60)
            hrs, mins = divmod(mins, 60)
            eta_str = f" | ETA: {int(hrs)}h {int(mins)}m {int(secs)}s" if hrs > 0 else f" | ETA: {int(mins)}m {int(secs)}s"
        elif iteration == total:
            mins, secs = divmod(int(elapsed), 60)
            hrs, mins = divmod(mins, 60)
            eta_str = f" | Done in: {int(hrs)}h {int(mins)}m {int(secs)}s" if hrs > 0 else f" | Done in: {int(mins)}m {int(secs)}s"

    sys.stdout.write('\r\033[K' + f'{prefix} | [{bar}] ({iteration}/{total}) {YELLOW}{percent}%{RESET}{eta_str}')
    sys.stdout.flush()
    if iteration == total: print()

#* ─────────────────────────────────────────────────────────────────
#* 1. KINEMATIC ARCHITECTURE & DATASET
#* ─────────────────────────────────────────────────────────────────
class KinematicLSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, num_classes=7, dropout=0.3):
        super(KinematicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        final_state = self.dropout(out[:, -1, :]) 
        return self.fc(final_state)

class KinematicDataset(Dataset):
    def __init__(self, X_data, Y_labels):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.Y = torch.tensor(Y_labels, dtype=torch.long)
        
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

#* ─────────────────────────────────────────────────────────────────
#* 2. STATIC LIVE DATASET (Optimized)
#* ─────────────────────────────────────────────────────────────────
worker_yunet = None
def get_yunet():
    global worker_yunet
    if worker_yunet is None:
        worker_yunet = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (320, 320), score_threshold=0.4, top_k=1)
    return worker_yunet

class LiveImageDataset(Dataset):
    """Loads pre-cropped images from disk. Highly optimized for Multi-threading."""
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            tensor_crop = self.transform(img)
        except Exception:
            tensor_crop = torch.zeros((3, 224, 224))
            
        return tensor_crop, label

#* ─────────────────────────────────────────────────────────────────
#* 3. THE TRAINING LOOPS
#* ─────────────────────────────────────────────────────────────────
def execute_training_kinematic(epochs=50, patience=10):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    print(f"\n{CYAN}=== Training Kinematic LSTM ==={RESET}")
    
    x_path = os.path.join(PROC_KINEMATIC_DIR, 'X_features.npy')
    y_path = os.path.join(PROC_KINEMATIC_DIR, 'Y_labels.npy')
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"{RED}Missing data at {PROC_KINEMATIC_DIR}. Please run harvester first!{RESET}")
        return

    X_real = np.load(x_path)
    Y_real = np.load(y_path)
    dataset = KinematicDataset(X_real, Y_real)

    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=Y_real)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KinematicLSTM(input_size=15, hidden_size=64, num_layers=2, num_classes=7).to(device)

    train_labels = Y_real[train_idx]
    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    class_weights = class_counts.sum() / (len(CLASS_NAMES) * np.clip(class_counts, 1, None))
    class_weights_tensor = torch.tensor(np.clip(class_weights, class_weights.min(), class_weights.min() * 4.0), dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 'best_kinematic_model.pth', epochs, patience)

def execute_training_static(epochs=30, patience=6):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    print(f"\n{CYAN}=== Fine-Tuning Static CNN ==={RESET}")
    
    if not os.path.exists(PROC_STATIC_CROP_DIR):
        print(f"{RED}Missing data at {PROC_STATIC_CROP_DIR}. Please run the Harvester first!{RESET}")
        return

    print(f"Scanning pre-cropped directories...")
    
    all_paths, all_labels = [], []
    for emotion_name in CLASS_NAMES:
        folder_path = os.path.join(PROC_STATIC_CROP_DIR, emotion_name)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(f" ↳ {GREEN}{emotion_name:<10}{RESET}: Found {YELLOW}{len(files):,}{RESET} cropped images.")
            for f in files:
                all_paths.append(os.path.join(folder_path, f))
                all_labels.append(EMOTION_MAP[emotion_name])

    if not all_paths:
        print(f"{RED}No images found to train on.{RESET}")
        return
        
    print(f"\nTotal Dataset Size: {CYAN}{len(all_paths):,}{RESET} images ready for High-Speed fine-tuning.")
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    train_dataset = LiveImageDataset(train_paths, train_labels, transform=get_face_transform(is_training=True))
    val_dataset = LiveImageDataset(val_paths, val_labels, transform=get_face_transform(is_training=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StaticFeatureExtractor(num_classes=7).to(device)

    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    class_weights = class_counts.sum() / (len(CLASS_NAMES) * np.clip(class_counts, 1, None))
    class_weights_tensor = torch.tensor(np.clip(class_weights, class_weights.min(), class_weights.min() * 4.0), dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 'best_static_model.pth', epochs, patience, use_amp=True)

def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, checkpoint_name, epochs, patience, use_amp=False):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()           
            
            # Use Automatic Mixed Precision if active
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)         
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)         
                loss = criterion(outputs, labels) 
                loss.backward()                 
                optimizer.step()
                                
            running_loss += loss.item() * inputs.size(0)

            if i % 10 == 0 or i == total_train_batches - 1:
                print_progress(i + 1, total_train_batches, prefix=f"Epoch [{epoch+1}/{epochs}] {GREEN}Train{RESET}", start_time=start_time)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        
        val_start_time = time.time()
        with torch.no_grad(): 
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                
                if i % 10 == 0 or i == total_val_batches - 1:
                    print_progress(i + 1, total_val_batches, prefix=f"Epoch [{epoch+1}/{epochs}] {YELLOW}Valid{RESET}", start_time=val_start_time)
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{epochs}] {CYAN}Summary{RESET} | {RED}Train Loss: {epoch_loss:.4f}{RESET} | {MAGENTA}Val Loss: {val_epoch_loss:.4f}{RESET} | {BRIGHT_GREEN}Val Acc: {val_accuracy:.2f}%{RESET} | LR: {current_lr:.6f}")

        scheduler.step(val_epoch_loss)
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, checkpoint_name))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n{YELLOW}⏹ Early stopping — no val improvement for {patience} epochs.{RESET}")
                break
            
    print(f"\n{GREEN}Complete! Best model saved as 'checkpoints/{checkpoint_name}'{RESET}")
    print(f"\n{CYAN}Per-class validation performance (best checkpoint):{RESET}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

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
        execute_training_kinematic()
        
    if run_static:
        execute_training_static()