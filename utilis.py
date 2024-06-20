import torch 
import torch.nn as nn

import torchvision.transforms as transforms
import BiGAN
import BiGAN.detect_GAN
import BiGAN.discriminator
import BiGAN.encoder
import BiGAN.generator
import BiGAN.train_GAN
import torch.nn
from tqdm import tqdm 
import torchvision.transforms as transforms
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
=======
PATH_TRAIN  = './dataset/train_typical'
PATH_VALIDATION  = './dataset/validation_typical'

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
    
    test_typical_loader = DataLoader(test_typical_dataset, batch_size=32)
    test_novel_loader = DataLoader(test_novel_dataset, batch_size=32)


    if model_name == "GAN":
        
        # model = BiGAN.train_GAN.TrainerBiGAN(epoch_number, lr, train_loader, device)
        # encoder, generator, discriminator = model.train()

        # torch.save({
        #     'encoder_state_dict': encoder.state_dict(),
        #     'generator_state_dict': generator.state_dict(),
        #     'discriminator_state_dict': discriminator.state_dict(),
        # }, 'models/models.pth')


        ## Sekcja do testów anaomali 
        
        print("LOADING")

        encoder = BiGAN.encoder.GanEncoder().to(device)  # Replace Encoder with your actual encoder class
        generator = BiGAN.generator.GanGenerator().to(device) # Replace Generator with your actual generator class
        discriminator = BiGAN.discriminator.GanDiscriminator().to(device)  # Replace Discriminator with your actual discriminator class

        checkpoint = torch.load('models/models.pth')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        print("#################### NORMAL ####################")

        tester = BiGAN.detect_GAN.AnomalyScore(generator, encoder, discriminator, test_typical_loader, device)
        result = tester.test()

        print("#################### NOVEL ####################")

        tester = BiGAN.detect_GAN.AnomalyScore(generator, encoder, discriminator, test_novel_loader, device)
        result = tester.test()


    elif model_name == "VAE":
        pass

    elif model_name == "FLOW":
        pass
    
    else:
        raise ValueError("Unkown Model")
