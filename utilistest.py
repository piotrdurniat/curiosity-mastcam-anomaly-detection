import torch
from torch.utils.data import DataLoader

import BiGAN
import BiGAN.detect_GAN
import BiGAN.discriminator
import BiGAN.encoder
import BiGAN.generator
import BiGAN.results
import BiGAN.train_GAN
import dataset
from utilis import ModelType
from vae.test_vae import load_and_test_vae

PATH_TEST_TYPICAL = "./dataset/test_typical"
PATH_TEST_NOVEL = "./dataset/test_novel/all"

RANDOM_SEED = 42

PATH_TRAIN = "./dataset/train_typical"
PATH_VALIDATION = "./dataset/validation_typical"


def test_model(
    model_name: ModelType,
    batch: int,
    device: str,
    load_path: str,
    save_path: str,
):

    transform = dataset.ToTensorWithScaling()

    test_typical_dataset = dataset.ImageDataLoader(
        PATH_TEST_TYPICAL, transform=transform
    )
    test_novel_dataset = dataset.ImageDataLoader(PATH_TEST_NOVEL, transform=transform)

    test_typical_loader = DataLoader(test_typical_dataset, batch_size=batch)
    test_novel_loader = DataLoader(test_novel_dataset, batch_size=batch)

    train_dataset = dataset.ImageDataLoader(PATH_TRAIN, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    if model_name == "GAN":

        print("LOADING")
        encoder = BiGAN.encoder.GanEncoder().to(device)
        generator = BiGAN.generator.GanGenerator().to(device)
        discriminator = BiGAN.discriminator.GanDiscriminator().to(device)

        checkpoint = torch.load("models/BiGAN.pth")
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

        print("#################### NORMAL ####################")

        tester = BiGAN.detect_GAN.AnomalyScore(
            generator, encoder, discriminator, test_typical_loader, device
        )
        result_true = tester.test("true")
        tester.plot_images("real")

        print("#################### NOVEL ####################")

        tester = BiGAN.detect_GAN.AnomalyScore(
            generator, encoder, discriminator, test_novel_loader, device
        )
        result_fake = tester.test("fake")
        tester.plot_images("fake")

        BiGAN.results.give_results(result_true, result_fake)

    elif model_name == "VAE":
        load_and_test_vae(
            model_path=load_path,
            test_typical_loader=test_typical_loader,
            test_novel_loader=test_novel_loader,
            train_loader=train_loader,
            device=device,
            save_path=save_path,
        )

    elif model_name == "FLOW":
        pass

    else:
        raise ValueError("Unknown Model")
