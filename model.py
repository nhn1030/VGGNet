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
            

class VGGNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.conv2d(in_planes = 3, out_planes =64,)
        