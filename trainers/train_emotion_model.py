import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

#? Resolve paths dynamically relative to this script's location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
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

#* ─────────────────────────────────────────────────────────────────
#* UTILS
#* ─────────────────────────────────────────────────────────────────
def print_progress(iteration, total, prefix='', length=30):
    """Generates a dynamic terminal progress bar."""
    if total == 0:
        return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} | [{bar}] ({iteration}/{total}) {YELLOW}{percent}%{RESET} Complete')
    sys.stdout.flush()
    if iteration == total: 
        print()

#? ─────────────────────────────────────────────────────────────────
#?  1. THE LSTM NEURAL NETWORK ARCHITECTURE
#? ─────────────────────────────────────────────────────────────────
class EmotionLSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, num_classes=7, dropout=0.3):
        super(EmotionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #* The LSTM Layer: Expects data in shape (Batch, Sequence_Length, Features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        #* A Dropout layer to prevent overfitting to specific actors
        self.dropout = nn.Dropout(dropout)
        
        #* The Final Output Layer: Maps the LSTM's conclusions to the 7 emotion classes
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        #* Pass the sequence through the LSTM
        #? out shape: (batch_size, seq_length, hidden_size)
        out, (hn, cn) = self.lstm(x)
        
        final_state = out[:, -1, :] 
        
        final_state = self.dropout(final_state)
        predictions = self.fc(final_state)
        
        return predictions

#? ─────────────────────────────────────────────────────────────────
#?  2. THE DATASET HANDLER
#? ─────────────────────────────────────────────────────────────────
class KinematicsDataset(Dataset):
    """
    A custom PyTorch dataset that takes your harvested NumPy arrays
    and serves them up in batches for the Neural Network.
    """
    def __init__(self, X_data, Y_labels):
        #! X_data should be a NumPy array of shape: (Total_Videos, 30_Frames, 15_Features)
        #! Y_labels should be a NumPy array of shape: (Total_Videos,) containing class integers (0-6)
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.Y = torch.tensor(Y_labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

#? ─────────────────────────────────────────────────────────────────
#?  3. THE TRAINING LOOP
#? ─────────────────────────────────────────────────────────────────
def train_model():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    #* --- Extract Datasets ---
    print("Loading Data...")
    X_real = np.load(os.path.join(PROCESSED_DIR, 'X_features.npy'))
    Y_real = np.load(os.path.join(PROCESSED_DIR, 'Y_labels.npy'))
    
    dataset = KinematicsDataset(X_real, Y_real)
    
    #! Split into 80% Training, 20% Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    #* --- INITIALIZE MODEL, LOSS, AND OPTIMIZER ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{GREEN}Training on: {device}{RESET}")
    
    model = EmotionLSTM(input_size=15, hidden_size=64, num_layers=2, num_classes=7).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # weight_decay helps prevent overfitting
    
    epochs = 50
    best_val_loss = float('inf')
    
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    
    #* --- THE EPOCH LOOP ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()           #? Clear old gradients
            outputs = model(inputs)         #? Forward pass (Predict)
            loss = criterion(outputs, labels) #? Calculate error
            loss.backward()                 #? Backpropagation
            optimizer.step()                #? Update weights
            
            running_loss += loss.item() * inputs.size(0)
            
            #* Update training progress bar
            print_progress(i + 1, total_train_batches, prefix=f"Epoch [{epoch+1}/{epochs}] {GREEN}Train{RESET}")
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        #* --- VALIDATION CHECK ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): # Don't track gradients during validation
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                #* Update validation progress bar
                print_progress(i + 1, total_val_batches, prefix=f"Epoch [{epoch+1}/{epochs}] {YELLOW}Valid{RESET}")
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}] {CYAN}Summary{RESET} | {RED}Train Loss: {epoch_loss:.4f}{RESET} | {MAGENTA}Val Loss: {val_epoch_loss:.4f}{RESET} | {BRIGHT_GREEN}Val Acc: {val_accuracy:.2f}%{RESET}\n")
        
        #! Save the best model automatically
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, 'best_emo_model.pth'))
            
    print(f"{GREEN}Training Complete! Best model saved as 'checkpoints/best_emo_model.pth'{RESET}")

if __name__ == "__main__":
    train_model()