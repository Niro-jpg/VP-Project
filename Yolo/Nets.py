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
            nn.Conv2d(32, 64, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.Flatten(),
            nn.Linear(16384, 1028),
            nn.ReLU(True),
            nn.Linear(1028,512),
            nn.ReLU(True),
            nn.Linear(512,128),
            nn.ReLU(True),
            nn.Linear(128,S*S*(5*B + C)),
        )

    def forward(self, x):

        x = self.miaos(x)
        return x