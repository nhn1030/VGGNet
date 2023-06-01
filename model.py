import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.conv2d(in_planes = 3, out_planes =64,)
        )