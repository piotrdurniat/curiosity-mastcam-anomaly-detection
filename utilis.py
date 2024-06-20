import torch 
import torch.nn as nn

import torchvision.transforms as transforms
import BiGAN
import BiGAN.detect_GAN
import BiGAN.discriminator
import BiGAN.encoder
import BiGAN.generator
import BiGAN.results
import BiGAN.train_GAN
import dataset
import yaml
from torch.utils.data import DataLoader


PATH_TRAIN  = './dataset/train_typical'
PATH_VALIDATION  = './dataset/validation_typical'

PATH_TEST_TYPICAL  = './dataset/test_typical'
PATH_TEST_NOVEL   = './dataset/test_novel/all'


RANDOM_SEED = 42
FREQ_PRINT = 20 

latent_dim = 200 #<- to do 
PATH_TRAIN  = './dataset/train_typical'
PATH_VALIDATION  = './dataset/validation_typical'

latent_dim = 200 #<- to do 

def train_model(model_name, epoch_number, lr, device):

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    print(model_name, lr, epoch_number, device)

    train_dataset = dataset.ImageDataLoader(PATH_TRAIN, transform=transform)
    valdiaiton_dataset = dataset.ImageDataLoader(PATH_VALIDATION, transform=transform)

    test_typical_dataset = dataset.ImageDataLoader(PATH_TEST_TYPICAL, transform=transform)
    test_novel_dataset = dataset.ImageDataLoader(PATH_TEST_NOVEL, transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(valdiaiton_dataset, batch_size=64)
    
    test_typical_loader = DataLoader(test_typical_dataset, batch_size=1)
    test_novel_loader = DataLoader(test_novel_dataset, batch_size=1)


    if model_name == "GAN":
        
        model = BiGAN.train_GAN.TrainerBiGAN(epoch_number, lr, train_loader, device)
        encoder, generator, discriminator = model.train()

        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
        }, 'models/models.pth')

    elif model_name == "VAE":
        pass

    elif model_name == "FLOW":
        pass
    
    else:
        raise ValueError("Unknown Model")
