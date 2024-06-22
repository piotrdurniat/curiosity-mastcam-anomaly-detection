from torch.utils import data
from torch import Tensor
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

      
class ToTensorWithScaling:
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, eps: float = 1e-6):
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps

    def __call__(self, image: np.ndarray):
        image = torch.tensor(image, dtype=torch.float32)
        image = torch.permute(image, (2, 0, 1))
        
        # Get min and max values for every channel
        min_vals = image.amin(dim=(1, 2), keepdim=True)
        max_vals = image.amax(dim=(1, 2), keepdim=True)

        # [0, 1]
        image = (image - min_vals) / (max_vals - min_vals + self.eps)

        # [min_val, max_val]
        image = image * (self.max_val - self.min_val) + self.min_val

        return image


class Dequantize:
    def __init__(self, logit: bool = True, deq: bool = True, alpha: float = 1.0e-6):
        self.logit = logit
        self.deq = deq
        self.alpha = alpha

    def __call__(self, image):
        image = torch.tensor(image, dtype=torch.float32)
        image = torch.permute(image, (2, 0, 1))
        
        image = (image + np.random.rand(*image.shape)) / 256.0
        x = self.alpha + (1 - 2 * self.alpha) * image
        image = np.log(x / (1.0 - x))

        return image