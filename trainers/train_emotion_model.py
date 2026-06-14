import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

#? Resolve paths dynamically relative to this script's location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_KINEMATIC_DIR = os.path.join(BASE_DIR, 'processed_data', 'kinematic')
PROC_STATIC_DIR = os.path.join(BASE_DIR, 'processed_data', 'static')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

#* ─────────────────────────────────────────────────────────────────
#* TERMINAL COLORS & UTILS
#* ─────────────────────────────────────────────────────────────────
GREEN = '\033[32m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BRIGHT_GREEN = '\033[92m'
RESET = '\033[0m'

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] ({iteration}/{total}) {YELLOW}{percent}%{RESET} Complete')
    sys.stdout.flush()
    if iteration == total: print()

#* ─────────────────────────────────────────────────────────────────
#* 1. NEURAL NETWORK ARCHITECTURES
#* ─────────────────────────────────────────────────────────────────
class KinematicLSTM(nn.Module):
    """Analyzes 30-frame sequences of 15 facial regions."""
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, num_classes=7, dropout=0.3):
        super(KinematicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        final_state = self.dropout(out[:, -1, :]) 
        return self.fc(final_state)

class StaticMLP(nn.Module):
    """Analyzes the 64-float embedding extracted by MobileNetV2."""
    def __init__(self, input_size=64, hidden_size=32, num_classes=7, dropout=0.3):
        super(StaticMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

#* ─────────────────────────────────────────────────────────────────
#* 2. DATASET HANDLER
#* ─────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, X_data, Y_labels):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.Y = torch.tensor(Y_labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

#* ─────────────────────────────────────────────────────────────────
#* 3. THE UNIVERSAL TRAINING LOOP
#* ─────────────────────────────────────────────────────────────────
def execute_training(model_type, data_dir, model, checkpoint_name, epochs=50):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    print(f"\n{CYAN}=== Training {model_type} Model ==={RESET}")
    x_path = os.path.join(data_dir, 'X_features.npy' if model_type == 'Kinematic' else 'X_embeddings.npy')
    y_path = os.path.join(data_dir, 'Y_labels.npy')
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"{RED}Missing data at {data_dir}. Please run harvester first!{RESET}")
        return

    print("Loading Data...")
    X_real = np.load(x_path)
    Y_real = np.load(y_path)
    
    dataset = EmotionDataset(X_real, Y_real)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{GREEN}Training on: {device}{RESET}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 
    
    best_val_loss = float('inf')
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()           
            outputs = model(inputs)         
            loss = criterion(outputs, labels) 
            loss.backward()                 
            optimizer.step()                
            running_loss += loss.item() * inputs.size(0)
            print_progress(i + 1, total_train_batches, prefix=f"Epoch [{epoch+1}/{epochs}] {GREEN}Train{RESET}")
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad(): 
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print_progress(i + 1, total_val_batches, prefix=f"Epoch [{epoch+1}/{epochs}] {YELLOW}Valid{RESET}")
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}] {CYAN}Summary{RESET} | {RED}Train Loss: {epoch_loss:.4f}{RESET} | {MAGENTA}Val Loss: {val_epoch_loss:.4f}{RESET} | {BRIGHT_GREEN}Val Acc: {val_accuracy:.2f}%{RESET}")
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, checkpoint_name))
            
    print(f"\n{GREEN}Complete! Best model saved as 'checkpoints/{checkpoint_name}'{RESET}")


#* ─────────────────────────────────────────────────────────────────
#* CLI ROUTING LOGIC
#* ─────────────────────────────────────────────────────────────────
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
        model = KinematicLSTM(input_size=15, hidden_size=64, num_layers=2, num_classes=7)
        execute_training("Kinematic", PROC_KINEMATIC_DIR, model, 'best_kinematic_model.pth')
        
    if run_static:
        model = StaticMLP(input_size=64, hidden_size=32, num_classes=7)
        execute_training("Static", PROC_STATIC_DIR, model, 'best_static_model.pth')