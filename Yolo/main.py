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
B = 2
y_tensor = torch.zeros(images_dim,S,S,(5+3))
cell_dim = 256/S
print(cell_dim)
y = []
j = -1
for i, filename in enumerate(os.listdir(folder_path)):
    if ("xml" in filename):
        print(j)
        j += 1
        tree = ET.parse(os.path.join(folder_path, filename))
        root = tree.getroot()
        filetextname = root.find("filename").text
        img = cv2.imread(os.path.join(folder_path, filetextname))
        xsize = img.shape[1]
        ysize = img.shape[0]
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        tensor = torch.tensor(img.transpose(2,0,1))/255
        images_tensor[j] = tensor
        box = []
        object_matrix = torch.zeros((S,S), dtype=torch.int)
        for child in root:
            if (child.tag == "object"):
                xmin = int(child.find("bndbox/xmin").text)*(256/xsize)
                xmax = int(child.find("bndbox/xmax").text)*(256/xsize)
                ymin = int(child.find("bndbox/ymin").text)*(256/ysize)
                ymax = int(child.find("bndbox/ymax").text)*(256/ysize)
                xm = (xmin+xmax)/2
                ym = (ymin+ymax)/2
                x_cell = int(xm/cell_dim)
                y_cell = int(ym/cell_dim)
                x_coordinate = (xm%cell_dim)/cell_dim
                y_coordinate = (ym%cell_dim)/cell_dim
                x_dim = (xmax-xmin)/cell_dim
                y_dim = (ymax-ymin)/cell_dim
                label = 0
                label_name = child.find("name").text
                if   (label_name == "apple"):    label = 0
                elif (label_name == "banana"):   label = 1
                elif (label_name == "orange"):   label = 2
                else: 
                    print("uh nu")
                    break
                y_tensor[j,x_cell,y_cell,5*object_matrix[x_cell,y_cell]] = 1
                y_tensor[j,x_cell,y_cell,1+ 5*object_matrix[x_cell,y_cell]] = x_coordinate
                y_tensor[j,x_cell,y_cell,2+ 5*object_matrix[x_cell,y_cell]] = y_coordinate
                y_tensor[j,x_cell,y_cell,3+ 5*object_matrix[x_cell,y_cell]] = x_dim
                y_tensor[j,x_cell,y_cell,4+ 5*object_matrix[x_cell,y_cell]] = y_dim
                y_tensor[j,x_cell,y_cell,(5) + label] = 1 
                box.append([xmin,xmax,ymin,ymax,label])
                #object_matrix[x_cell,y_cell] = (object_matrix[x_cell,y_cell]+1)%B
        y.append(box)

print(len(y))
img = images_tensor[5].numpy().transpose(1,2,0)
y_tensor = y_tensor.view(images_dim,S,S,(5+3))
#plot_image_with_boxes(img,y[0])
#plot_image_from_tensor(img, y_tensor[5])
yolo_net = Yolo(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_net.to(device)
optimizer = torch.optim.Adam(yolo_net.parameters(),
                                          lr=0.001)
images_tensor = images_tensor.cuda()
y_tensor = y_tensor.cuda()

losses = []

for i in range(100):
    x = images_tensor
    pred = yolo_net.forward(torch.tensor(x)).view(images_dim,S,S,(B*5+3))
    loss = YOLO_Loss(pred,y_tensor)
    optimizer.zero_grad()
    print("----------------------------",loss)
    loss.backward()
    losses.append(loss.item())
    optimizer.step()
plt.plot(losses)
plt.show()
    