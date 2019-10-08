import torch
import torch.nn as nn
from torchvision import models
import torchvision
from torch.nn import functional as F
import torchvision.transforms as transforms

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(inplace = True),
            nn.Dropout2d(p = 0.1)
        )
        self.Basic_Block1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 32)
        )
        self.Basic_Block2_S2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 64)
        )
        self.Short_Cut2_S2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, 2),
            nn.BatchNorm2d(64)
        )
        self.Basic_Block2_S1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 64)
        )
        self.Short_Cut2_S1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1),
            nn.BatchNorm2d(64)
        )

        self.Basic_Block3_S2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 128)
        )
        self.Short_Cut3_S2 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2),
            nn.BatchNorm2d(128)
        )

        self.Basic_Block3_S1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 128)
        )
        self.Short_Cut3_S1 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1),
            nn.BatchNorm2d(128)
        )

        self.Basic_Block4_S2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 256)
        )
        self.Short_Cut4_S2 = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2),
            nn.BatchNorm2d(256)
        )
        self.Basic_Block4_S1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 256)
        )
        self.Short_Cut4_S1 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1),
            nn.BatchNorm2d(256)
        )

        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(2 * 2 * 256, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.Basic_Block1(out) + out
        out = self.Basic_Block1(out) + out
       
        out = self.Basic_Block2_S2(out) + self.Short_Cut2_S2(out)
        out = self.Basic_Block2_S1(out) + self.Short_Cut2_S1(out)
        out = self.Basic_Block2_S1(out) + self.Short_Cut2_S1(out)
        out = self.Basic_Block2_S1(out) + self.Short_Cut2_S1(out)

        out = self.Basic_Block3_S2(out) + self.Short_Cut3_S2(out)
        out = self.Basic_Block3_S1(out) + self.Short_Cut3_S1(out)
        out = self.Basic_Block3_S1(out) + self.Short_Cut3_S1(out)
        out = self.Basic_Block3_S1(out) + self.Short_Cut3_S1(out)

        out = self.Basic_Block4_S2(out) + self.Short_Cut4_S2(out)
        out = self.Basic_Block4_S1(out) + self.Short_Cut4_S1(out)
        
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        
        return out