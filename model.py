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
            

class VGGNet(nn.Module):
    def __init__(self, num_blocks, num_classes=1000):
        super(VGGNet, self).__init__()
        self.features = self.make.layers(num_blocks)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def _make_layer(self, num_blocks):
        
        layers = []
        
        in_channels = 3
        out_channels = 6
        for num_conv in num_blocks:
            layers.append(VGGBlock(in_channels, out_channels, num_conv))
            in_channels = out_channels
            out_channels *= 2
        return nn.Sequential(*layers)

            
        
    
    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifiers(out)
        return out
    
    
def VGGNet16():
    return VGGNet(num_blocks= [2,2,3,3,3], num_classes= 1000)


def VGGNet19():
    return VGGNet(num_blocks= [2,2,4,4,4], num_classes= 1000)