import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import xml.etree.ElementTree as ET
import matplotlib.image as mpimg
import matplotlib.patches as patches
import tqdm
from Utils import *
from Nets import * 
from Model import *

a = torch.rand(48,13)
n = 3  # Dimensione del tensore
tensor = torch.arange(n) * 5
print(tensor)
print(a.shape)
suppression_and_division(a,3,4,2)