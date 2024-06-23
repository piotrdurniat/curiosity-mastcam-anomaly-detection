import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from vae.vae import BaseAutoEncoder, VariationalAutoencoder


def load_and_test_vae(
    model_path: str,
    test_typical_loader: DataLoader,
    test_novel_loader: DataLoader,
    train_loader: DataLoader,
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
    test_vae(
        vae_model_loaded,
        test_typical_loader,
        test_novel_loader,
        train_loader,
        device,
        save_path,
    )


def test_vae(
    model: BaseAutoEncoder,
    test_typical_loader: DataLoader,
    test_novel_loader: DataLoader,
    train_loader: DataLoader,
    device: str,
    save_path: str,
):

    print(
        "Hear ye, hear ye! By royal decree, I doth hereby declare the testing of the VAE model to commence forthwith!"
    )
    model = model.to(device)
    novelty_detection = NoveltyDetection(model, train_loader)

    typical_novelty_scores = novelty_detection.compute_novelty_score(
        test_typical_loader
    )
    novel_novelty_scores = novelty_detection.compute_novelty_score(test_novel_loader)

    # compare the novelty scores
    print("Typical novelty scores:")
    print(typical_novelty_scores)
    print("Novel novelty scores:")
    print(novel_novelty_scores)


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


def evaluate(model, test_loader):
    model.eval()
    device = next(model.parameters()).device
    mse = 0.0
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            reconstructions = model(inputs)
            mse += mean_squared_error(
                inputs.cpu().view(inputs.shape[0], -1),
                reconstructions.cpu().view(reconstructions.shape[0], -1),
            )
    return mse / len(test_loader)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoveltyDetection:
    def __init__(self, vae: BaseAutoEncoder, normal_data_loader: DataLoader):
        """
        :param vae: Trained VAE model
        :param normal_data_loader: DataLoader containing normal data samples
        """
        self.vae = vae
        self.normal_data_loader = normal_data_loader

        self.normal_latent_means = self._compute_latent_means(normal_data_loader)

    def _compute_latent_means(self, data_loader: DataLoader):
        """
        Encode the data to get the mean in latent space
        :param data_loader: DataLoader with data to encode
        :return: Tensor of latent space means
        """
        latent_means = []
        self.vae.eval()
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device)

                data = data.view(data.shape[0], -1)
                z_m, _ = self.vae.encoder(data)

                latent_means.append(z_m.cpu())
        return torch.cat(latent_means, dim=0)

    def compute_novelty_score(self, test_data_loader: DataLoader):
        """
        Compute the novelty score for the test data
        :param test_data_loader: DataLoader with test data
        :return: Novelty scores for each test data point
        """
        self.vae.eval()
        novelty_scores = []
        with torch.no_grad():
            for test_data, _ in test_data_loader:
                test_data = test_data.to(device)

                test_data = test_data.view(test_data.shape[0], -1)
                z_m_test, _ = self.vae.encoder(test_data)
                distances = torch.cdist(z_m_test, self.normal_latent_means.to(device))
                min_distances, _ = distances.min(dim=1)

                novelty_scores.extend(min_distances.cpu().numpy())

        return novelty_scores
