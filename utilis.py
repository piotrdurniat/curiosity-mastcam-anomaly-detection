from typing import Literal

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
import flow.layers
import flow.maf
import flow.train_flow
from vae.train_vae import train_and_save
from vae.vae import VariationalAutoencoder

PATH_TEST_TYPICAL = "./dataset/test_typical"
PATH_TEST_NOVEL = "./dataset/test_novel/all"

RANDOM_SEED = 42
FREQ_PRINT = 20

PATH_TRAIN = "./dataset/train_typical"
PATH_VALIDATION = "./dataset/validation_typical"

ModelType = Literal["GAN", "VAE", "FLOW"]


def get_transform(model_name: ModelType):
    if model_name == "GAN":
        return dataset.ToTensorWithScaling()

    elif model_name == "VAE":
        return dataset.ToTensorWithScaling(-1.0, 1.0)

    elif model_name == "FLOW":
        return dataset.Dequantize()

    else:
        raise ValueError("Unknown model")


def get_loaders(transform, batch_size: int):
    train_dataset = dataset.ImageDataLoader(PATH_TRAIN, transform=transform)
    valdiaiton_dataset = dataset.ImageDataLoader(PATH_VALIDATION, transform=transform)

    test_typical_dataset = dataset.ImageDataLoader(
        PATH_TEST_TYPICAL, transform=transform
    )
    test_novel_dataset = dataset.ImageDataLoader(PATH_TEST_NOVEL, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valdiaiton_dataset, batch_size=batch_size)

    test_typical_loader = DataLoader(test_typical_dataset, batch_size=1)
    test_novel_loader = DataLoader(test_novel_dataset, batch_size=1)

    return train_loader, val_loader, test_typical_loader, test_novel_loader


def train_model(
    model_name: ModelType, epoch_number: int, lr: float, device: str, save_path: str
):
    print(model_name, lr, epoch_number, device)

    transform = get_transform(model_name)
    train_loader, val_loader, test_typical_loader, test_novel_loader = get_loaders(
        transform,
        64,
    )

    if model_name == "GAN":

        model = BiGAN.train_GAN.TrainerBiGAN(
            epoch_number, lr, train_loader, val_loader, device
        )
        encoder, generator, discriminator = model.train()

        torch.save(
            {
                "encoder_state_dict": encoder.state_dict(),
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
            },
            save_path,
        )

    elif model_name == "VAE":

        n_data_features = 64 * 64 * 6
        n_hidden_features = 1024
        n_latent_features = 256

        model = VariationalAutoencoder(
            n_data_features=n_data_features,
            n_encoder_hidden_features=n_hidden_features,
            n_decoder_hidden_features=n_hidden_features,
            n_latent_features=n_latent_features,
        )

        print(f"Starting training: \n{n_latent_features=}, {n_hidden_features=}, {lr=}")

        train_and_save(
            model=model,
            epochs=epoch_number,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            model_name="vae",
            save_path=save_path,
        )

    elif model_name == "FLOW":
        model = flow.maf.MAF(6 * 64 * 64, [64, 64, 64, 64, 64], 5, use_reverse=True)
        trainer = flow.train_flow.TrainerMAF(
            model, epoch_number, lr, train_loader, device
        )
        trainer.train()

        save_flow_model(model, save_path)

    else:
        raise ValueError("Unkown Model")


def save_flow_model(model: flow.maf.MAF, path: str):
    model_state = {
        "model_state_dict": model.state_dict(),
        "batch_norm_running_states": {},
    }

    for index, layer in enumerate(model.layers):
        if isinstance(layer, flow.layers.BatchNormLayerWithRunning):
            model_state["batch_norm_running_states"][
                f"batch_norm_{index}_running_mean"
            ] = layer.running_mean
            model_state["batch_norm_running_states"][
                f"batch_norm_{index}_running_var"
            ] = layer.running_var

    torch.save(model_state, path)
