import torch 
import torch.nn as nn

import torchvision.transforms as transforms
import BiGAN
import BiGAN.train_GAN
import dataset
import yaml
from torch.utils.data import DataLoader


PATH_TRAIN  = './dataset/train_typical'
PATH_VALIDATION  = './dataset/validation_typical'
RANDOM_SEED = 42
FREQ_PRINT = 20 

latent_dim = 200 #<- to do 

class SelectChannels(object):
    def __call__(self, img):
        # img is a PyTorch tensor of shape (C, H, W)
        # Assuming input has 6 channels, select first 3 channels
        return img[:3, :, :]  
    
def train_model(model_name, epoch_number, lr, device):

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


    print(model_name, lr, epoch_number, device)
    train_dataset = dataset.ImageDataLoader(PATH_TRAIN, transform=transform)
    valdiaiton_dataset = dataset.ImageDataLoader(PATH_VALIDATION, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(valdiaiton_dataset, batch_size=64)

    model_config = None

    if model_name == "GAN":
        
        # with open('./GAN_config.yaml', 'r') as file:
            # model_config = yaml.safe_load(file)

        model = BiGAN.train_GAN.TrainerBiGAN(epoch_number, lr, train_loader, device)
        model.train()

    elif model_name == "VAE":
        pass

    elif model_name == "FLOW":
        pass
    
    else:
        raise ValueError("Unkown Model")
        

