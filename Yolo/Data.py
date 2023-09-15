from torch.utils.data import DataLoader, Dataset

class Data(Dataset):
    def __init__(self, images, detections):
        self.images = images
        self.detections = detections

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.detections[idx]