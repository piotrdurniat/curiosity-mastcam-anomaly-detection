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

PATH_TEST_TYPICAL  = './dataset/test_typical'
PATH_TEST_NOVEL   = './dataset/test_novel/all'

RANDOM_SEED = 42

PATH_TRAIN  = './dataset/train_typical'
PATH_VALIDATION  = './dataset/validation_typical'

def train_model(model_name, batch, device):

    if model_name == "GAN":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
    ])

    elif model_name == "VAE":
        # TODO: confirm this is correct or change it
        transform = dataset.ToTensorWithScaling(-1.0, 1.0)

    elif model_name == "FLOW":
        transform = dataset.Dequantize()


    test_typical_dataset = dataset.ImageDataLoader(PATH_TEST_TYPICAL, transform=transform)
    test_novel_dataset = dataset.ImageDataLoader(PATH_TEST_NOVEL, transform=transform)

    test_typical_loader = DataLoader(test_typical_dataset, batch_size=batch)
    test_novel_loader = DataLoader(test_novel_dataset, batch_size=batch)


    if model_name == "GAN":

        print("LOADING")
        encoder = BiGAN.encoder.GanEncoder().to(device)  
        generator = BiGAN.generator.GanGenerator().to(device) 
        discriminator = BiGAN.discriminator.GanDiscriminator().to(device)

        checkpoint = torch.load('models/models.pth')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        print("#################### NORMAL ####################")

        tester = BiGAN.detect_GAN.AnomalyScore(generator, encoder, discriminator, test_typical_loader, device)
        result_true = tester.test("true")

        print("#################### NOVEL ####################")

        tester = BiGAN.detect_GAN.AnomalyScore(generator, encoder, discriminator, test_novel_loader, device)
        result_fake = tester.test("fake")
        BiGAN.results.give_results(result_true, result_fake)

    elif model_name == "VAE":
        pass

    elif model_name == "FLOW":
        pass
    
    else:
        raise ValueError("Unknown Model")
