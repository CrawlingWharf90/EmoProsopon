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

    def predict_all(self, kinematic_sequence=None, static_embedding=None, alpha=0.5):
        """
        Dynamically fuses the models. If a parameter is None (because it was toggled 
        off in the HUD settings), the engine gracefully falls back to the active model.
        Returns: (fused_idx, fused_conf, kin_idx, kin_conf, static_idx, static_conf)
        """
        with torch.no_grad():
            k_probs, s_probs, fused_probs = None, None, None
            
            #? 1. Kinematic Inference
            if kinematic_sequence is not None:
                k_in = torch.tensor(kinematic_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                k_probs = F.softmax(self.kinematic_net(k_in), dim=1)
                
            #? 2. Static Inference
            if static_embedding is not None:
                s_in = torch.tensor(static_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
                s_probs = F.softmax(self.static_net(s_in), dim=1)
                
            #? 3. Fusion Logic
            if k_probs is not None and s_probs is not None:
                fused_probs = (alpha * k_probs) + ((1 - alpha) * s_probs)
            elif k_probs is not None:
                fused_probs = k_probs
            elif s_probs is not None:
                fused_probs = s_probs
            else:
                return -1, 0.0, -1, 0.0, -1, 0.0 
                
            #? 4. Final Extractions
            f_idx = torch.argmax(fused_probs, dim=1).item()
            f_conf = torch.max(fused_probs).item() * 100
            
            k_idx = torch.argmax(k_probs, dim=1).item() if k_probs is not None else -1
            k_conf = torch.max(k_probs).item() * 100 if k_probs is not None else 0.0
            
            s_idx = torch.argmax(s_probs, dim=1).item() if s_probs is not None else -1
            s_conf = torch.max(s_probs).item() * 100 if s_probs is not None else 0.0
            
            return f_idx, f_conf, k_idx, k_conf, s_idx, s_conf
        


#* Who's to say
#* I can't do everything? Well, I can try
#* And as I roll along, I begin to find
#* Things aren't always just what they seem
#*
#* I wanna turn the whole thing upside down
#* I'll find the things they say just can't be found
#* I'll share this love I find with everyone
#*
#*      - Jack Johnson, Upside Down