import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class StaticFeatureExtractor(nn.Module):
    def __init__(self, embedding_size=64):
        super(StaticFeatureExtractor, self).__init__()
        self.backbone = models.mobilenet_v2(weights=None)

        local_weights_path = os.path.join(MODELS_DIR, 'mobilenet_v2-b0353104.pth')
        if not os.path.exists(local_weights_path):
            raise FileNotFoundError(f"Missing static model weights! Please run 'eop --tui models' to download: {local_weights_path}")
            
        self.backbone.load_state_dict(torch.load(local_weights_path))
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, embedding_size),
            nn.Tanh() 
        )

    def forward(self, x):
        return self.backbone(x)

def get_face_transform():
    """
    Standardizes the cropped face for the CNN.
    PyTorch pre-trained models expect exactly 224x224 RGB images
    normalized with these specific ImageNet mean/std values.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])