# scripts/cnn_classifier.py
import torch.nn as nn
import torch.nn.functional as F

class PaperCNN(nn.Module):
    """
    2D-CNN treating input as (1, C, W) image where C=num_vars and W=window.
    Single-head classifier per paper: outputs `num_classes` logits.
    """
    def __init__(self, in_channels=1, base_filters=32, num_classes=16):
        super().__init__()
        f = base_filters
        # three conv modules, doubling filters each time, kernel 3x3 as in paper
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1),
            nn.BatchNorm2d(f), nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(f, f*2, 3, padding=1),
            nn.BatchNorm2d(f*2), nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(f*2, f*4, 3, padding=1),
            nn.BatchNorm2d(f*4), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(f*4, 128)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (B,1,C,W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.head(x)
