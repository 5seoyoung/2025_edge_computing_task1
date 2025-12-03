"""
EF Regression Model based on ResNet-18
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from typing import Tuple


class EFRegressionModel(nn.Module):
    """
    EF Regression Model using ResNet-18 as backbone
    
    Architecture:
    - Input: (B, N, C, H, W) where N is number of frames
    - ResNet-18 feature extraction (without final fc)
    - Temporal aggregation (mean pooling)
    - Linear regression head
    """
    
    def __init__(self, num_frames: int = 32, pretrained: bool = True):
        """
        Args:
            num_frames: Number of frames per video
            pretrained: Whether to use pretrained ResNet-18 weights
        """
        super(EFRegressionModel, self).__init__()
        self.num_frames = num_frames
        
        # Load ResNet-18 backbone
        if pretrained:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        # Remove final fully connected layer
        # Keep layers up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Get feature dimension (512 for ResNet-18)
        self.feature_dim = 512
        
        # Temporal aggregation + regression head
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # Mean pooling over frames
        self.regression_head = nn.Linear(self.feature_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (B, N, C, H, W) video tensor
        
        Returns:
            ef_pred: (B, 1) predicted EF values
        """
        B, N, C, H, W = x.shape
        
        # Reshape to (B*N, C, H, W) for batch processing
        x = x.view(B * N, C, H, W)
        
        # Extract features: (B*N, feature_dim, 1, 1)
        features = self.backbone(x)
        
        # Flatten: (B*N, feature_dim)
        features = features.view(B * N, self.feature_dim)
        
        # Reshape back: (B, N, feature_dim)
        features = features.view(B, N, self.feature_dim)
        
        # Temporal aggregation: mean pooling over frames
        # (B, N, feature_dim) -> (B, feature_dim)
        temporal_features = features.mean(dim=1)
        
        # Regression: (B, feature_dim) -> (B, 1)
        ef_pred = self.regression_head(temporal_features)
        
        return ef_pred.squeeze(-1)  # (B,)
    
    def get_feature_extractor(self):
        """Return the backbone feature extractor"""
        return self.backbone

