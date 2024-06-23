import torch 
import torch.nn as nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import BiGAN
import BiGAN.detect_GAN
import BiGAN.discriminator
import BiGAN.encoder
import BiGAN.generator
import BiGAN.results
import BiGAN.train_GAN

import flow.maf
import flow.detect_flow
import flow.results
import flow.layers

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
        print("LOADING")
        model = flow.maf.MAF(64 * 64 * 6, [64, 64, 64, 64, 64], 5, use_reverse=True)
        model = load_flow_model(model, 'models/maf.pth')
        
        print("#################### NORMAL ####################")

        tester = flow.detect_flow.AnomalyScore(model, test_typical_loader, device)
        result_true = tester.test("true")

        print("#################### NOVEL ####################")

        tester = flow.detect_flow.AnomalyScore(model, test_novel_loader, device)
        result_fake = tester.test("fake")
        flow.results.give_results(result_true, result_fake)

        # display_reproductions(model, test_novel_loader)
    
    else:
        raise ValueError("Unknown Model")
    
    
def load_flow_model(model, path):
    model_state = torch.load(path)
    model.load_state_dict(model_state['model_state_dict'])

    for index, layer in enumerate(model.layers):
        if isinstance(layer, flow.layers.BatchNormLayerWithRunning):
            layer.running_mean = model_state["batch_norm_running_states"][f"batch_norm_{index}_running_mean"]
            layer.running_var = model_state["batch_norm_running_states"][f"batch_norm_{index}_running_var"]

    return model

def display_reproductions(model, loader, n=8):
    reconstructions = []
    inputs = []
    model.eval()
    device = next(model.parameters()).device  

    with torch.no_grad():
        for i, (input, _) in enumerate(loader):
            if i >= n:
                break
            x = input.to(device)  
            x = x.view(x.size(0), -1)
            inputs.append(input.cpu())
            rec = model(x)[0].cpu()
            reconstructions.append(rec.view(input.size())) 

    fig, axes = plt.subplots(2, n, figsize=(20, 5))
    for i in range(n):
        rec = reconstructions[i][0].squeeze(0)
        rec = rec[:3, :, :]
        rec = np.transpose(rec, (1, 2, 0))
        rec = np.take(rec, [2, 0, 1], axis=2)

        img = inputs[i][0].squeeze(0)
        img = img[:3, :, :]
        img = np.transpose(img, (1, 2, 0))
        img = np.take(img, [2, 0, 1], axis=2)

        axes[0, i].imshow(np.interp(img, (img.min(), img.max()), (0, 1)))
        axes[1, i].imshow(np.interp(rec, (rec.min(), rec.max()), (0, 1)))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.show()

train_model("FLOW", 1, "cpu")