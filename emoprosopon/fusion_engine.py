import torch
import torch.nn.functional as F
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

from trainers.train_emotion_model import KinematicLSTM, StaticMLP

class EmotionFusionEngine:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        #? Load Kinematic Stream
        self.kinematic_net = KinematicLSTM().to(self.device)
        self.kinematic_net.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, 'best_kinematic_model.pth')))
        self.kinematic_net.eval()
        
        #? Load Static Stream
        self.static_net = StaticMLP().to(self.device)
        self.static_net.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, 'best_static_model.pth')))
        self.static_net.eval()

    def predict(self, kinematic_sequence, static_embedding, alpha=0.5):
        """
        alpha: Weighting factor for fusion. 
        alpha > 0.5 trusts the Kinematic stream more.
        alpha < 0.5 trusts the Static stream more.
        """
        with torch.no_grad():
            k_in = torch.tensor(kinematic_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            s_in = torch.tensor(static_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            k_logits = self.kinematic_net(k_in)
            s_logits = self.static_net(s_in)
            
            k_probs = F.softmax(k_logits, dim=1)
            s_probs = F.softmax(s_logits, dim=1)
            
            fused_probs = (alpha * k_probs) + ((1 - alpha) * s_probs)
            
            return torch.argmax(fused_probs, dim=1).item(), fused_probs