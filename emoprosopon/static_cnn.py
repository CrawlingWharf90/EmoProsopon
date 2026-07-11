import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

STATIC_INPUT_SIZE = (224, 224)

class StaticFeatureExtractor(nn.Module):
    def __init__(self, num_classes=7):
        super(StaticFeatureExtractor, self).__init__()
        self.architecture_name = "MobileNetV2 (Fine-Tuned)"
        
        self.backbone = models.mobilenet_v2(weights=None)

        local_weights_path = os.path.join(MODELS_DIR, 'mobilenet_v2-b0353104.pth')
        if not os.path.exists(local_weights_path):
            raise FileNotFoundError(f"Missing static model weights! Please run 'eop --tui models' to download: {local_weights_path}")

        self.backbone.load_state_dict(torch.load(local_weights_path, map_location='cpu'))

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_model_info(self):
        """
        Returns real, current metadata about this model instance.
        """
        return {
            "architecture": self.architecture_name,
            "input_shape": f"{STATIC_INPUT_SIZE[0]}x{STATIC_INPUT_SIZE[1]} RGB",
            "output_dim": "7-Class Softmax",
        }

def get_face_transform(is_training=False):
    """
    Standardizes the cropped face for the CNN.
    If is_training is True, applies dynamic augmentations to prevent overfitting.
    """

    if is_training:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])