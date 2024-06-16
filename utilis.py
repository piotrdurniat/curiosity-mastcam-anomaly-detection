import torch 
import torch.nn
from tqdm import tqdm 
import torchvision.transforms as transforms
import dataset
from torch.utils.data import DataLoader

PATH_TRAIN  = './dataset/train_typical'
PATH_VALIDATION  = './dataset/validation_typical'


def train_model(model_name, epoch_numer, lr, device):

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = dataset.ImageDataLoader(PATH_TRAIN, transform=transform)
    valdiaiton_dataset = dataset.ImageDataLoader(PATH_VALIDATION, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(valdiaiton_dataset, batch_size=64)

    print("heloo")

    