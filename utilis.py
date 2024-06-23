import torch 
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms
import flow
import flow.coupling_layer
import flow.real_nvp
import flow.train_flow
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

from torch import Tensor
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

def train_model(model_name, epoch_number, lr, device):

    transform = dataset.ToTensorWithScaling()

    print(model_name, lr, epoch_number, device)

    train_dataset = dataset.ImageDataLoader(PATH_TRAIN, transform=transform)
    valdiaiton_dataset = dataset.ImageDataLoader(PATH_VALIDATION, transform=transform)

    test_typical_dataset = dataset.ImageDataLoader(PATH_TEST_TYPICAL, transform=transform)
    test_novel_dataset = dataset.ImageDataLoader(PATH_TEST_NOVEL, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(valdiaiton_dataset, batch_size=10)
    
    test_typical_loader = DataLoader(test_typical_dataset, batch_size=1)
    test_novel_loader = DataLoader(test_novel_dataset, batch_size=1)


    if model_name == "GAN":

        model = BiGAN.train_GAN.TrainerBiGAN(epoch_number, lr, train_loader, val_loader, device)
        encoder, generator, discriminator = model.train()

        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
        }, 'models/BiGAN.pth')

    elif model_name == "VAE":
        pass

    elif model_name == "FLOW":
        model = flow.real_nvp.RealNVP(6, 64, 6)
        trainer = flow.train_flow.TrainerRealNVP(model, epoch_number, lr, train_loader, device)
        trainer.train()

        torch.save({
            'model_state_dict': model.state_dict(),
        }, 'models/flow.pth')

    else:
        raise ValueError("Unknown Model")