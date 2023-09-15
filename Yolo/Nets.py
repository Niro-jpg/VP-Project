import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class Yolo(nn.Module):
    def __init__(self,  C, B = 2, S = 8):
        super(Yolo, self).__init__()

        self.miaos = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 8, 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2),
            nn.Flatten(),
            
            nn.Linear(5184, 4096),
            nn.ReLU(True),
            nn.Linear(4096,S*S*(5*B + C)),
        )

    def forward(self, x):

        x = self.miaos(x)
        return x