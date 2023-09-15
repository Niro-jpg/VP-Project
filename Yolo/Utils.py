import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import torch
import os
import xml.etree.ElementTree as ET
import cv2

def plot_image_with_boxes( img, boxes):

    fig, ax = plt.subplots()

    ax.imshow(img)

    for box in boxes:
        xmin, xmax, ymin, ymax, _ = box 

        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Mostra il grafico
    plt.show()
    
def plot_image_from_tensor( img, tensor):

    B = int((tensor.shape[2]-3)/5)
    print("B: ", B)

    _ , ax = plt.subplots()

    ax.imshow(img)

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            max_k = tensor[i,j,0]
            new_k = 0
            for k in range(B):
                if tensor[i,j,5*k] > max_k: new_k = k 
            k = new_k
            if tensor[i,j,5*k] > 0.7:

                w = tensor[i,j,3+ 5*k]*32
                h = tensor[i,j,4+ 5*k]*32
                x = ((tensor[i,j,1+ 5*k]*32) + i*32) - w/2
                y = ((tensor[i,j,2+ 5*k]*32) + j*32) - h/2
                colors_tensor = tensor[i,j,-3:]
                print("tensore coloris: ", colors_tensor)
                color_idx = torch.argmax(colors_tensor)
                color = "y"
                if color_idx == 0:
                    color = "g"
                elif color_idx == 2:
                    color = "r"
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
    plt.show()

def load_data(folder_path, images_dim, S):
    cell_dim = 448/S
    images_tensor = torch.zeros(images_dim, 3, 448, 448)
    y_tensor = torch.zeros(images_dim,S,S,5+3)
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
            img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
            tensor = torch.tensor(img.transpose(2,0,1))/255
            images_tensor[j] = tensor
            box = []
            object_matrix = torch.zeros((S,S), dtype=torch.int)
            for child in root:
                if (child.tag == "object"):
                    xmin = int(child.find("bndbox/xmin").text)*(448/xsize)
                    xmax = int(child.find("bndbox/xmax").text)*(448/xsize)
                    ymin = int(child.find("bndbox/ymin").text)*(448/ysize)
                    ymax = int(child.find("bndbox/ymax").text)*(448/ysize)
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
                    #if (x_dim > cell_dim/2):
                    #    apparison_dim = int((x_dim-cell_dim/2)/cell_dim)
                    #    for l in range(apparison_dim + 1):
                    #        y_tensor[j,x_cell + (l + 1),y_cell,8] = 1 
                    #        y_tensor[j,x_cell - (l + 1),y_cell,8] = 1 
                    #if (y_dim > cell_dim/2):
                    #    apparison_dim = int((y_dim-cell_dim/2)/cell_dim)
                    #    for l in range(apparison_dim + 1):
                    #        y_tensor[j,x_cell,y_cell+ (l + 1),8] = 1 
                    #        y_tensor[j,x_cell,y_cell - (l + 1),8] = 1 
                    box.append([xmin,xmax,ymin,ymax,label])
                    #object_matrix[x_cell,y_cell] = (object_matrix[x_cell,y_cell]+1)%B
            y.append(box)
    return images_tensor, y, y_tensor