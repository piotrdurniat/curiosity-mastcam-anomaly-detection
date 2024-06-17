import torch 
import torch.nn
from tqdm import tqdm 
import torchvision.transforms as transforms
import dataset
import yaml
from torch.utils.data import DataLoader

PATH_TRAIN  = './dataset/train_typical'
PATH_VALIDATION  = './dataset/validation_typical'


def train_model(model_name, epoch_number, lr, device):

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    print(model_name, lr, epoch_number, device)
    train_dataset = dataset.ImageDataLoader(PATH_TRAIN, transform=transform)
    valdiaiton_dataset = dataset.ImageDataLoader(PATH_VALIDATION, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(valdiaiton_dataset, batch_size=64)

    model_config = None

    if model_name == "GAN":
        
        with open('GAN_config.yaml', 'r') as file:
            model_config = yaml.safe_load(file)

    elif model_name == "VAE":
        pass

    elif model_name == "FLOW":
        pass
    
    else:
        raise ValueError("Unkown Model")
        

