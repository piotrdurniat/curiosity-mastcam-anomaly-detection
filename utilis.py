import torch 
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms
import flow.train_flow
import flow.maf
import dataset
import yaml

from torch import Tensor
from torch.utils.data import DataLoader
import torch.optim as optim


PATH_TRAIN  = './dataset/train_typical'
PATH_VALIDATION  = './dataset/validation_typical'

PATH_TEST_TYPICAL  = './dataset/test_typical'
PATH_TEST_NOVEL   = './dataset/test_novel/all'

RANDOM_SEED = 42
FREQ_PRINT = 20 



def train_model(model_name, epoch_number, lr, device):

    transform = dataset.Dequantize()

    print(model_name, lr, epoch_number, device)

    train_dataset = dataset.ImageDataLoader(PATH_TRAIN, transform=transform)
    valdiaiton_dataset = dataset.ImageDataLoader(PATH_VALIDATION, transform=transform)

    test_typical_dataset = dataset.ImageDataLoader(PATH_TEST_TYPICAL, transform=transform)
    test_novel_dataset = dataset.ImageDataLoader(PATH_TEST_NOVEL, transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(valdiaiton_dataset, batch_size=64)
    
    test_typical_loader = DataLoader(test_typical_dataset, batch_size=32)
    test_novel_loader = DataLoader(test_novel_dataset, batch_size=32)


    if model_name == "GAN":
        pass
        
    elif model_name == "VAE":
        pass

    elif model_name == "FLOW":

        model = flow.maf.MAF(6 * 64 * 64, [64], 5, use_reverse=True)
        # trainer = flow.train_flow.TrainerMAF(model, epoch_number, lr, train_loader, device)
        # trainer.train()

        # torch.save({
        #     'model_state_dict': model.state_dict(),
        # }, 'models/maf_02.pth')

        checkpoint = torch.load('models/maf_02.pth')
        model.load_state_dict(checkpoint["model_state_dict"])

        print("ok")

    else:
        raise ValueError("Unkown Model")
    
train_model(model_name="FLOW", epoch_number=2, lr=1e-4, device="cpu")