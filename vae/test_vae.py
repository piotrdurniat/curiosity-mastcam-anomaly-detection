import torch
from torch.utils.data import DataLoader

from vae.vae import BaseAutoEncoder, VariationalAutoencoder


def load_and_test_vae(
    model_path: str,
    test_loader: DataLoader,
    device: str,
    save_path: str,
):
    vae_model_loaded = load_model(
        model_path=model_path,
        n_data_features=64 * 64 * 6,
        n_hidden_features=1024,
        n_latent_features=128,
        device=device,
    )
    test_vae(vae_model_loaded, test_loader, device, save_path)


def test_vae(
    model: BaseAutoEncoder,
    test_loader: DataLoader,
    device: str,
    save_path: str,
):
    print(
        "Hear ye, hear ye! By royal decree, I doth hereby declare the testing of the VAE model to commence forthwith!"
    )


def load_model(
    model_path: str,
    n_data_features: int,
    n_hidden_features: int,
    n_latent_features: int,
    device: str,
):

    vae_model_loaded = VariationalAutoencoder(
        n_data_features=n_data_features,
        n_encoder_hidden_features=n_hidden_features,
        n_decoder_hidden_features=n_hidden_features,
        n_latent_features=n_latent_features,
    )
    vae_model_loaded.load_state_dict(torch.load(model_path, map_location=device))
    return vae_model_loaded
