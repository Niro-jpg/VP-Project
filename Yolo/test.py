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

folder_path = "./archive/test_zip/test"

images_dim = 60
images_tensor = torch.zeros(images_dim, 3, 256, 256)
S = 8
B = 1
C = 3
y_tensor = torch.zeros(images_dim,S,S,(5+3))
cell_dim = 256/S
print(cell_dim)
y = []
j = -1
y, y_tensor = load_data(folder_path, images_dim)

print(len(y))
img = images_tensor[5].numpy().transpose(1,2,0)
y_tensor = y_tensor.view(images_dim,S,S,(5+3))
#plot_image_with_boxes(img,y[0])
#plot_image_from_tensor(img, y_tensor[5])
yolo_net = Yolo(3, B=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_net.to(device)

images_dim = 60
x = images_tensor[:images_dim].cuda()
y = y_tensor[:images_dim].cuda()
#images_dim = 20

model_path='./model.pt'
yolo_net.load_state_dict(torch.load('./model.pt'))

image_idx = 5

pred = yolo_net.forward(x[image_idx].unsqueeze(0)).squeeze(0).view(S,S,(B*5 + C)).detach().cpu()
print(pred.shape)
print(y[0].shape)
print(y[0][1,2])
print(pred[1,2])
plot_image_from_tensor(x[image_idx].cpu().numpy().transpose(1,2,0),pred)
plot_image_from_tensor(x[image_idx].cpu().numpy().transpose(1,2,0),y[image_idx].cpu())


torch.save(yolo_net.state_dict(), model_path)