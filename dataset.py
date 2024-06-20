from torch.utils import data
import torch
import numpy as np
import os

class ImageDataLoader(data.Dataset):
    "My own Image Loader made to read .npy images"

    def __init__(self, directory, transform=None):

        self.directory = directory
        self.transform = transform
        self.file_names = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.directory, self.file_names[idx])
        image = np.load(img_path)
        image = image.astype(np.float32)
        
        if self.transform is not None:
            image = self.transform(image)

        image_labels = torch.zeros(image.shape[0])
        return image, image_labels