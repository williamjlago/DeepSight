import torch
import torch.nn as nn
from torchvision import models

class DeepSightModel(nn.Module):
    def __init__(self, pos_weight=None):
        super(DeepSightModel, self).__init__()
        # Load a pretrained ResNet18
        self.model = models.resnet18(weights='IMAGENET1K_V1')

        # ResNet18 expects 3-channel input. We have 1-channel grayscale images.
        # We'll modify the first convolutional layer to accept 1 channel instead of 3.
        # The original first conv layer: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        old_conv = self.model.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize new_conv weights from old_conv by averaging the weights of the 3 channels
        with torch.no_grad():
            new_conv.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
        self.model.conv1 = new_conv

        # Replace the final fully connected layer to output a single logit.
        # Original: fc = Linear(512, 1000)
        # Now: fc = Linear(512, 1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        # Just forward through the model
        return self.model(x)
