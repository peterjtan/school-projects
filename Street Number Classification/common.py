import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 11


# Prepare model
def prepare_vgg_model(pretrained=False):
    model = models.vgg16(pretrained)

    # Change number of output classes
    classifier = list(model.classifier.children())

    outputLayer = nn.Linear(4096, NUM_CLASSES)
    classifier[-1] = outputLayer

    model.classifier = nn.Sequential(*classifier)

    # Reset fully-connected layer weights
    for i in model.classifier.modules():
        if isinstance(i, nn.Linear):
            nn.init.normal_(i.weight, 0, 0.01)
            nn.init.constant_(i.bias, 0)

    return model


class ResidualBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.identity_map_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.batch_norm(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        
        out += self.identity_map_conv(identity)
        out = self.batch_norm(out)
        out = self.relu(out)

        return out


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)


def prepare_own_model():
    model = nn.Sequential(
        ResidualBasicBlock(3, 64),
        ResidualBasicBlock(64, 256),

        nn.AdaptiveAvgPool2d(output_size=1),
        View((-1, 256)),
        nn.Linear(256, 11)
    )
    return torch.jit.script(model)
