import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from utilistest import draw_charts
from vae.vae import BaseAutoEncoder, VariationalAutoencoder


def draw_charts(save_dir: str, typical_novelty_scores, novel_novelty_scores):

    # Example data for demonstration purposes
    # typical_novelty_scores = np.random.normal(loc=0, scale=1, size=1000)
    # novel_novelty_scores = np.random.normal(loc=1, scale=1.5, size=1000)

    # Create the subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Histogram of novelty scores
    axes[0].hist(
        typical_novelty_scores, bins=50, alpha=0.5, color="blue", label="Typical"
    )
    axes[0].hist(novel_novelty_scores, bins=50, alpha=0.5, color="red", label="Novel")
    axes[0].set_title("Histogram of novelty scores")
    axes[0].legend()

    # Boxplot of novelty scores
    axes[1].boxplot([typical_novelty_scores, novel_novelty_scores])
    axes[1].set_title("Boxplot of novelty scores")

    # Empirical CDF of novelty scores
    n_bins = 100
    counts, bin_edges = np.histogram(typical_novelty_scores, bins=n_bins, density=True)
    cdf = np.cumsum(counts)
    axes[2].plot(bin_edges[1:], cdf / cdf[-1], label="Typical", color="blue")

    counts, bin_edges = np.histogram(novel_novelty_scores, bins=n_bins, density=True)
    cdf = np.cumsum(counts)
    axes[2].plot(bin_edges[1:], cdf / cdf[-1], label="Novel", color="red")
    axes[2].set_title("Empirical CDF of novelty scores")
    axes[2].legend()

    # Save the figure to a file
    plt.tight_layout()
    plt.savefig(save_dir + "/novelty_scores.png")
    # Show the plots
    plt.show()


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

    # save the novelty scores
    typical_novelty_scores_file = save_path + "/typical_novelty_scores.npy"
    novel_novelty_scores_file = save_path + "/novel_novelty_scores.npy"

    print(f"Saving typical novelty scores to {typical_novelty_scores_file}")
    print(f"Saving novel novelty scores to {novel_novelty_scores_file}")

    torch.save(typical_novelty_scores, typical_novelty_scores_file)
    torch.save(novel_novelty_scores, novel_novelty_scores_file)

    draw_charts(save_path, typical_novelty_scores, novel_novelty_scores)

    threshold = 0.5
    typical_labels = classify_novelty(typical_novelty_scores, threshold)
    novel_labels = classify_novelty(novel_novelty_scores, threshold)

    true_labels = [False] * len(typical_labels) + [True] * len(novel_labels)
    predicted_labels = typical_labels + novel_labels

    precision, recall, f1_score = compute_metrics(true_labels, predicted_labels)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1_score}")

    # save the metrics
    metrics_file = save_path + "/metrics.txt"
    print(f"Saving metrics to {metrics_file}")
    with open(metrics_file, "w") as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 score: {f1_score}\n")


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


def classify_novelty(novelty_scores, threshold):
    """
    Classify the novelty scores as typical or novel based on a threshold
    :param novelty_scores: List of novelty scores
    :param threshold: Threshold value
    :return: List of classifications
    """
    return [score > threshold for score in novelty_scores]


def compute_metrics(true_labels, predicted_labels):
    """
    Compute precision, recall, and F1 score
    :param true_labels: List of true labels
    :param predicted_labels: List of predicted labels
    :return: Precision, recall, F1 score
    """
    true_positives = sum(
        [1 for true, pred in zip(true_labels, predicted_labels) if true and pred]
    )
    false_positives = sum(
        [1 for true, pred in zip(true_labels, predicted_labels) if not true and pred]
    )
    false_negatives = sum(
        [1 for true, pred in zip(true_labels, predicted_labels) if true and not pred]
    )

    if (true_positives + false_positives) == 0:
        return 0, 0, 0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score
