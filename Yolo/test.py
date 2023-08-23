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

a = torch.tensor((20,13))
suppression(a,3,4,2)