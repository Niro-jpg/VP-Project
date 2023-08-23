import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

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

    print(tensor.shape[0])
    print(tensor.shape[1])

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            for k in range(B):
                if tensor[i,j,5*k] == 1:

                    w = tensor[i,j,3+ 5*k]*32
                    h = tensor[i,j,4+ 5*k]*32
                    x = ((tensor[i,j,1+ 5*k]*32) + i*32) - w/2
                    y = ((tensor[i,j,2+ 5*k]*32) + j*32) - h/2
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
    plt.show()