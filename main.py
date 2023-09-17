import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from Utils import *
from Model import *
from Data import *
from Variables import *

# Loading targets and images from the folder path
images_tensor, y, y_tensor, images_dim = load_data(folder_path, max_images_dim, S)
y_tensor = y_tensor.view(images_dim, S, S, 5 + 3)

# initiating the model
yolo_net = Yolo(C, B=B, S=S)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_net.to(device)
optimizer = torch.optim.Adam(yolo_net.net.parameters(), lr=0.0001)
# choosing how many iamges of the dataset tu use for the train
x = images_tensor[:images_dim].to(device)
y = y_tensor[:images_dim].to(device)


losses = []
dataset = Data(x, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# training phase
for i in tqdm(range(epochs)):
    for inputs, labels in loader:
        batch_actual_size = inputs.shape[0]
        pred = yolo_net.forward(inputs).view(batch_actual_size, S, S, (B * 5 + 3))
        loss = yolo_net.loss(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    if i % 10 == 0:
        torch.save(yolo_net.state_dict(), model_path)
# saving the model
torch.save(yolo_net.state_dict(), model_path)
# plot the losses
plt.plot(losses)
plt.show()

# testing the model
pred = (
    yolo_net.forward(x[0].unsqueeze(0))
    .squeeze(0)
    .view(S, S, (B * 5 + C))
    .detach()
    .cpu()
)
plot_image_from_tensor(x[0].cpu().numpy().transpose(1, 2, 0), pred)
plot_image_from_tensor(x[0].cpu().numpy().transpose(1, 2, 0), y[0].cpu())
