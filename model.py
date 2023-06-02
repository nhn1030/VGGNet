import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_conv):
        super(VGGBlock, self).__init__()
        
        layers = []
        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channel = out_channel
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.block(x)
        return out
            

class VGGNet16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGGNet16, self).__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),
            VGGBlock(3, 128, 2),
            VGGBlock(3, 256, 3),
            VGGBlock(3, 512, 3),
            VGGBlock(3, 512, 3),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifiers(out)
        return out
    
class VGGNet19(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGGNet19, self).__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),
            VGGBlock(3, 128, 2),
            VGGBlock(3, 256, 4),
            VGGBlock(3, 512, 4),
            VGGBlock(3, 512, 4),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifiers(out)
        return out
        