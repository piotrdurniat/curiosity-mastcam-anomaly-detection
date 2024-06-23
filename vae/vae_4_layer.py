import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
import torch.nn.functional as F

pyro.enable_validation(False)


class VEncoder(nn.Module):
    """Encoder for VAE."""

    def __init__(
        self,
        n_input_features: int,
        n_hidden_neurons1: int,
        n_hidden_neurons2: int,
        n_latent_features: int,
    ):
        """
        :param n_input_features: number of input features (28 x 28 = 784 for MNIST)
        :param n_hidden_neurons1: number of neurons in the first hidden FC layer
        :param n_hidden_neurons2: number of neurons in the second hidden FC layer
        :param n_latent_features: size of the latent vector
        """
        super().__init__()
        self.input_to_hidden1 = nn.Linear(n_input_features, n_hidden_neurons1)
        self.hidden1_to_hidden2 = nn.Linear(n_hidden_neurons1, n_hidden_neurons2)
        self.hidden2_to_mu = nn.Linear(n_hidden_neurons2, n_latent_features)
        self.hidden2_to_sigma = nn.Linear(n_hidden_neurons2, n_latent_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode data to gaussian distribution params."""
        x = F.relu(self.input_to_hidden1(x))
        x = F.relu(self.hidden1_to_hidden2(x))
        z_loc = self.hidden2_to_mu(x)
        z_scale = self.hidden2_to_sigma(x).exp()
        return z_loc, z_scale


class VDecoder(nn.Module):
    """Decoder for VAE."""

    def __init__(
        self,
        n_latent_features: int,
        n_hidden_neurons2: int,
        n_hidden_neurons1: int,
        n_output_features: int,
    ):
        """
        :param n_latent_features: number of latent features (same as in Encoder)
        :param n_hidden_neurons2: number of neurons in the second hidden FC layer
        :param n_hidden_neurons1: number of neurons in the first hidden FC layer
        :param n_output_features: size of the output vector (28 x 28 = 784 for MNIST)
        """
        super().__init__()
        self.latent_to_hidden2 = nn.Linear(n_latent_features, n_hidden_neurons2)
        self.hidden2_to_hidden1 = nn.Linear(n_hidden_neurons2, n_hidden_neurons1)
        self.hidden1_to_output = nn.Linear(n_hidden_neurons1, n_output_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        r = F.relu(self.latent_to_hidden2(z))
        r = F.relu(self.hidden2_to_hidden1(r))
        r = torch.sigmoid(self.hidden1_to_output(r))
        return r


class BaseAutoEncoder(nn.Module):
    """Base AutoEncoder module class."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, n_latent_features: int):
        """
        :param encoder: encoder network
        :param decoder: decoder network
        :param n_latent_features: number of latent features in the AE
        """
        super().__init__()
        self.n_latent_features: int = n_latent_features
        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for mapping input to output."""
        z = self.encoder_forward(x)
        return self.decoder_forward(z)

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Function to perform forward pass through encoder network.

        takes: tensor of shape [batch_size x input_flattened_size] (flattened input)
        returns: tensor of shape [batch_size x latent_feature_size] (latent vector)
        """
        raise NotImplementedError()

    def decoder_forward(self, z: torch.Tensor) -> torch.Tensor:
        """Function to perform forward pass through decoder network.

        takes: tensor of shape [batch_size x latent_feature_size] (latent vector)
        returns: tensor of shape [batch_size x output_flattened_size] (flattened output)
        """
        raise NotImplementedError()


class VariationalAutoencoder(BaseAutoEncoder):
    """Variational Auto Encoder model."""

    def __init__(
        self,
        n_data_features: int,
        n_encoder_hidden_features1: int,
        n_encoder_hidden_features2: int,
        n_decoder_hidden_features2: int,
        n_decoder_hidden_features1: int,
        n_latent_features: int,
    ):
        """
        :param n_data_features: number of input and output features (28 x 28 = 784 for MNIST)
        :param n_encoder_hidden_features1: number of neurons in the first hidden layer of encoder
        :param n_encoder_hidden_features2: number of neurons in the second hidden layer of encoder
        :param n_decoder_hidden_features2: number of neurons in the second hidden layer of decoder
        :param n_decoder_hidden_features1: number of neurons in the first hidden layer of decoder
        :param n_latent_features: number of latent features
        """
        encoder = VEncoder(
            n_input_features=n_data_features,
            n_hidden_neurons1=n_encoder_hidden_features1,
            n_hidden_neurons2=n_encoder_hidden_features2,
            n_latent_features=n_latent_features,
        )
        decoder = VDecoder(
            n_latent_features=n_latent_features,
            n_hidden_neurons2=n_decoder_hidden_features2,
            n_hidden_neurons1=n_decoder_hidden_features1,
            n_output_features=n_data_features,
        )
        super().__init__(
            encoder=encoder, decoder=decoder, n_latent_features=n_latent_features
        )
        self.input_shape = None
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Function to perform forward pass through encoder network.
        takes: tensor of shape [batch_size x [image-size]] (input images batch)
        returns: tensor of shape [batch_size x latent_feature_size] (latent vector)
        """
        if self.input_shape is None:
            self.input_shape = x.shape[1:]
        x = x.view(x.shape[0], -1)
        z_loc, z_scale = self.encoder(x)
        z = z_loc + z_scale * self.N.sample(z_loc.shape).to(z_loc.device)
        return z

    def decoder_forward(self, z: torch.Tensor) -> torch.Tensor:
        """Function to perform forward pass through decoder network.
        takes: tensor of shape [batch_size x latent_feature_size] (latent vector)
        returns: tensor of shape [batch_size x [image-size]] (reconstructed images batch)
        """
        r = self.decoder(z)
        return r.view(-1, *self.input_shape)

    def model(self, x: torch.Tensor):
        """Pyro model for VAE; p(x|z)p(z)."""
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = torch.zeros((x.shape[0], self.n_latent_features)).to(x.device)
            z_scale = torch.ones((x.shape[0], self.n_latent_features)).to(x.device)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            output = self.decoder.forward(z).view(-1, *self.input_shape)
            pyro.sample("obs", dist.Bernoulli(output).to_event(3), obs=x)

    def guide(self, x: torch.Tensor):
        """Pyro guide for VAE; q(z|x)"""
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x.view(x.shape[0], -1))
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


class BetaVariationalAutoencoder(VariationalAutoencoder):
    """beta-Variational Auto Encoder model."""

    def __init__(self, beta: float, **kwargs):
        """
        :param n_data_features: number of input and output features (28 x 28 = 784 for MNIST)
        :param n_encoder_hidden_features1: number of neurons in the first hidden layer of encoder
        :param n_encoder_hidden_features2: number of neurons in the second hidden layer of encoder
        :param n_decoder_hidden_features2: number of neurons in the second hidden layer of decoder
        :param n_decoder_hidden_features1: number of neurons in the first hidden layer of decoder
        :param n_latent_features: number of latent features
        :param beta: regularization coefficient
        """
        super().__init__(**kwargs)
        self.beta = beta

    def model(self, x: torch.Tensor):
        """Pyro model for beta-VAE; p(x|z)p(z)."""
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = torch.zeros((x.shape[0], self.n_latent_features)).to(x.device)
            z_scale = torch.ones((x.shape[0], self.n_latent_features)).to(x.device)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            output = self.decoder.forward(z).view(-1, *self.input_shape)
            pyro.sample("obs", dist.Bernoulli(output).to_event(3), obs=x)

    def guide(self, x: torch.Tensor):
        """Pyro guide for beta-VAE; q(z|x)."""
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x.view(x.shape[0], -1))
            with poutine.scale(scale=self.beta):
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
