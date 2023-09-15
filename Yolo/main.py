import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import xml.etree.ElementTree as ET
import matplotlib.image as mpimg
import matplotlib.patches as patches
from tqdm import tqdm
from Utils import *
from Nets import * 
from Model import *
from Data import * 

folder_path = "./../Yolo/archive/test_zip/test"

images_dim = 60
S = 8
B = 2
C = 3
images_tensor, y, y_tensor = load_data(folder_path, images_dim, S)

y_tensor = y_tensor.view(images_dim,S,S,5+3)
#plot_image_with_boxes(img,y[0])
#plot_image_from_tensor(img, y_tensor[5])

yolo_net = Yolo(3, B = B)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_net.to(device)
optimizer = torch.optim.Adam(yolo_net.parameters(),
                                          lr=0.0001)

images_dim = 60
x = images_tensor[:images_dim].to(device)
y = y_tensor[:images_dim].to(device)
#images_dim = 20

batch_size = 8
losses = []
dataset = Data(x,y)
loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
model_path='./model.pt'
#yolo_net.load_state_dict(torch.load(model_path))

for i in tqdm(range(100)):
    for inputs, labels in loader:
        batch_actual_size = inputs.shape[0]
        pred = yolo_net.forward(inputs).view(batch_actual_size,S,S,(B*5+3))
        loss = YOLO_Loss(pred,labels)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    if i%10 == 0:
        torch.save(yolo_net.state_dict(), model_path)
torch.save(yolo_net.state_dict(), model_path)
plt.plot(losses)
plt.show()
pred = yolo_net.forward(x[0].unsqueeze(0)).squeeze(0).view(S,S,(B*5 + C)).detach().cpu()
print(pred.shape)
print(y[0].shape)
print(y[0][1,2])
print(pred[1,2])
plot_image_from_tensor(x[0].cpu().numpy().transpose(1,2,0),pred)
plot_image_from_tensor(x[0].cpu().numpy().transpose(1,2,0),y[0].cpu())


torch.save(yolo_net.state_dict(), model_path)